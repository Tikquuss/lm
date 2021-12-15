import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoModelForMaskedLM
)

import os
import click
from loguru import logger

MODELS_CLASS = {
    "clm" : AutoModelForCausalLM,
    "mlm" : AutoModelForMaskedLM
}
__None__ = '__None__'

from .dataset import LMLightningDataModule
from .language_modelling import LMLightningModule
from .tokenizing import load_tokenizer
from .utils import str2dic

def get_mode(validation_metrics) :
        return "min" if 'loss' in validation_metrics \
                or "ppl" in validation_metrics else 'max'

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("=========== Training is started! ========")
    def on_train_end(self, trainer, pl_module):
        print("======== Training is done. ======== ")

def to_none(a):
    return None if a == __None__ else a

def get_data_files(data_files):
    d = {}
    for v, k in zip(data_files.split(","), ["train", 'validation', 'test']) :
        d[k] = v
    return d

@click.command()
# General
@click.argument("model_name", type=str)
@click.argument("from_pretrained", type=bool)
@click.argument("task", type=click.Choice(["mlm", "clm"]))
@click.argument("log_dir", type=click.Path())
@click.option("--tokenizer_params", type=str, default=__None__, help="tokenizer_folder=,t_class=,t_type=")
# Dataset
@click.option("--dataset_name", type=str, default=__None__)
@click.option("--data_files", type=str, default=__None__)
@click.option("--text_column", type=str, default="text")
@click.option("--label_column", type=str, default=__None__)
@click.option("--mlm_probability", type=float, default=0.15)
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=4)
@click.option("--max_length", type=int, default=512)
@click.option("--limit_train_batches", type=float, default=1.)
@click.option("--limit_val_batches", type=float, default=1.)
@click.option("--limit_test_batches", type=float, default=1.)
# Optimizer
@click.option("--learning_rate", type=float, default=1e-5)
@click.option("--lr_factor", type=float, default=0.1)
@click.option("--lr_patience", type=int, default=4)
@click.option("--early_stopping_patience", type=int, default=5)
@click.option("--validation_metrics", type=str, default="val_loss", help="Validation metrics : val_acc, val_loss ...")
@click.option("--optimizer_name", type=str, default="Adam")
@click.option("--max_epochs", type=int, default=10)
@click.option("--val_check_interval", type=float, default=0.25)
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--save_top_k", type=int, default=5)
@click.option("--strategy", type=str, default="ddp", help="ddp, ddp_spawn ...")
@click.option("--random_seed", type=int, default=2021)
# Training
@click.option("--checkpoint_path", type=str, default=__None__, help="Reload a checkpoint")
@click.option("--eval_only", type=bool, default=False, help="Only run evaluations")
@click.option("--eval_split", type=click.Choice(["train", "validation", "test"]), default="test")
@click.option("--auto_scale_batch_size", type=str, default=None, # "binsearch" 
            help="Automatically tries to find the largest batch size that fits into memory, before any training")
@click.option("--auto_lr_find", type=bool, default=False, help="runs a learning rate finder algorithm")
@click.option("--deterministic", type=bool, default=False, help='ensures reproducibility')
@click.option("--freeze_transformer", type=bool, default=False, help="")
def main(
    model_name: str,
    from_pretrained : bool,
    task : str,
    log_dir: str,
    tokenizer_params : str,
    dataset_name: str,
    data_files : str,
    text_column : str, 
    label_column  : str, 
    mlm_probability : float, 
    batch_size: int,
    num_workers: int,
    max_length: int,
    limit_train_batches : float, 
    limit_val_batches : float,
    limit_test_batches : float,
    learning_rate: float,
    lr_factor: float,
    lr_patience: int,
    early_stopping_patience: int,
    validation_metrics : str,
    optimizer_name: str,
    max_epochs: int,
    val_check_interval: float,
    accumulate_grad_batches: int,
    save_top_k: int,
    strategy: str,
    random_seed: int,
    checkpoint_path : str,
    eval_only : bool,
    eval_split : str,
    auto_scale_batch_size : str,
    auto_lr_find : bool,
    deterministic : bool,
    freeze_transformer : bool
):
    
    pl.seed_everything(random_seed, workers=True)
    
    resume_from_checkpoint = checkpoint_path if os.path.isfile(checkpoint_path) else None
    assert not eval_only or os.path.isfile(resume_from_checkpoint if resume_from_checkpoint else "")
    root_dir = os.path.join(log_dir, task)
    os.makedirs(root_dir, exist_ok=True)
    
    tokenizer_params = str2dic(to_none(tokenizer_params))
    dataset_name = to_none(dataset_name)
    data_files = to_none(data_files)
    label_column = to_none(label_column)
    checkpoint_path = to_none(checkpoint_path)
        
    logger.info("Tokenizer...")
    tokenizer_folder=getattr(tokenizer_params, 'tokenizer_folder', None)
    tokenizer = load_tokenizer(
        model_name= model_name if tokenizer_folder is None else None, 
        tokenizer_folder=tokenizer_folder, 
        t_class = getattr(tokenizer_params, 't_class', None), 
        t_type = getattr(tokenizer_params, 't_type', None), 
        task = task, 
        MAX_LEN = max_length
    )
    
    num_proc=num_workers
    clm = task == "clm"
    mlm = task == "mlm"
    
    logger.info(f"{dataset_name} lightning data module creation...")
    data_files = get_data_files(data_files) if data_files is not None else None
    pl_data_module = LMLightningDataModule(
            tokenizer,
            batch_size,
            num_workers,
            max_length,
            dataset_name,
            num_proc = num_proc,
            data_files = data_files,
            text_column = text_column,
            label_column = label_column,
            clm = clm,
            mlm = mlm, 
            mlm_probability = mlm_probability
        )

    logger.info("model building...")
    if from_pretrained :
        model = MODELS_CLASS[task].from_pretrained(model_name)
    else :
        config = AutoConfig.from_pretrained(model_name)
        model = MODELS_CLASS[task].from_config(config)
    
    model.resize_token_embeddings(len(tokenizer))
    
    if freeze_transformer :
        for param in model.parameters():
            param.requires_grad = False

    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(f"Training new model - Total size={n_params/2**20:.2f}M params")
    
    trainer_config = {
        "max_epochs": max_epochs,
        
        "default_root_dir" : root_dir,
        #"log_every_n_steps" : max(len(pl_data_module.train) // batch_size, 0),
        "resume_from_checkpoint" : resume_from_checkpoint,
        #"weights_save_path" : dump_path,
        "auto_scale_batch_size":auto_scale_batch_size, # None
        "auto_select_gpus" : True,
        "auto_lr_find":auto_lr_find,
        "benchmark" : False,
        "deterministic" : deterministic,
        
        "val_check_interval": val_check_interval,
        "accumulate_grad_batches": accumulate_grad_batches,
        "strategy": strategy,
        "limit_train_batches":limit_train_batches, 
        "limit_val_batches" :limit_val_batches,
        "limit_test_batches":limit_test_batches
    }
    if not eval_only :
        trainer_config["log_every_n_steps"] = max(len(pl_data_module.train) // batch_size, 1)
    
    if torch.cuda.is_available():
        trainer_config["gpus"] = -1
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, verbose=False, strict=True,
        mode = get_mode(validation_metrics)
    )
    model_checkpoint_callback = ModelCheckpoint(
            dirpath=root_dir,
            filename="{epoch}-{%s:.4f}"%validation_metrics,
            monitor="val_loss",
            save_top_k=save_top_k,
    )
    trainer_config["callbacks"] = [
        early_stopping_callback, 
        model_checkpoint_callback, 
        PrintCallback()
    ]
    
    pl_trainer = pl.Trainer(**trainer_config)
    
    if label_column is not None :
        d_model = model.config.to_dict().get('hidden_size', None)
        if d_model is None :
            # https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1044
            # TODO : This is just for gpt2, ...
            d_model = model.lm_head.in_features
            def bert(input_ids, attention_mask = None, token_type_ids = None):
                return model.transformer(
                                input_ids = input_ids, 
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids)
            model.bert = bert
        clf_params = {
            "label_column" : label_column,
            "d_model": d_model,
            "n_labels" : 2, 
            "n_layers" : 1, 
            "intermediate_dim": 100, 
            "criterion": None, 
            "dropout" : 0.1, 
            "multi_label":False
        }
    else :
        clf_params = {}
    decoder_start_token_id = tokenizer.pad_token_id
    if not eval_only :
        pl_model = LMLightningModule(
            model=model,
            task=task,
            learning_rate=learning_rate,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            optimizer_name=optimizer_name,
            decoder_start_token_id=decoder_start_token_id,
            clf_params=clf_params
        )
    
        logger.info("Training starts...")
        pl_model.train()
        pl_trainer.fit(pl_model, datamodule=pl_data_module)
        logger.info("Training completed.")

        logger.info("Testing starts....")
    else :
        pl_model = LMLightningModule.load_from_checkpoint(
            checkpoint_path=resume_from_checkpoint,
            model=model,
            task = task,
            learning_rate=learning_rate,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            optimizer_name=optimizer_name,
            decoder_start_token_id=decoder_start_token_id,
            clf_params=clf_params
        )
        
        logger.info("Evaluation starts....")
        if eval_split == "train":
            pl_data_module.test_dataloader = pl_data_module.train_dataloader
        elif eval_split == "validation" :
            pl_data_module.test_dataloader = pl_data_module.val_dataloader
        
    pl_model.eval()
    pl_trainer.test(pl_model, datamodule=pl_data_module)

if __name__ == "__main__":
    main()