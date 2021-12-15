#!/bin/bash

None="__None__"

# clm : gpt2, facebook/bart-large ... (https://huggingface.co/models?filter=causal-lm)
# mlm : bert-base-uncased, roberta-base, ...
#model_name="gpt2"
model_name="bert-base-uncased"
from_pretrained=True
#task=clm
task=mlm

#log_dir="../log_files"
log_dir="log_files"

# TO_CHANGE is not using pretrained tokenizer
#tokenizer_params="tokenizer_folder=/content,t_class=bert_tokenizer,t_type=bert_word_piece"
tokenizer_params=$None

# TO_CHANGE
dataset_name=$None
data_files=$None
#datapath=/content
#data_files=${datapath}/data_train.csv,${datapath}/data_val.csv,${datapath}/data_test.csv
# TO_CHANGE is classification
text_column=text
label_column=$None

batch_size=32
validation_metrics=val_loss

#strategy="ddp_spawn"
strategy="ddp"

# TO_CHANGE
checkpoint_path=$None
#checkpoint_path=/content/log_files/mlm/epoch=1-val_loss=2.0445.ckpt

eval_only=False
#eval_split="train"
#eval_split="validation"
eval_split="test"

auto_scale_batch_size=True
auto_lr_find=False

python3 -m src.trainer \
		$model_name \
		$from_pretrained \
		$task \
		$log_dir \
		--tokenizer_params $tokenizer_params \
		--dataset_name $dataset_name \
		--data_files $data_files \
		--text_column $text_column \
		--label_column $label_column \
		--mlm_probability 0.15 \
		--batch_size $batch_size \
		--num_workers 4 \
		--max_length 512 \
		--limit_train_batches 1.0 \
		--limit_val_batches 1.0 \
		--limit_test_batches 1.0 \
		--learning_rate 0.00001 \
		--lr_factor 0.1 \
		--lr_patience 4 \
		--early_stopping_patience 5 \
		--validation_metrics $validation_metrics \
		--optimizer_name Adam \
		--max_epochs 10 \
		--val_check_interval 0.25 \
		--accumulate_grad_batches 1 \
		--save_top_k 5 \
		--strategy $strategy \
		--random_seed 2021 \
		--checkpoint_path $checkpoint_path \
		--eval_only $eval_only \
		--eval_split $eval_split \
		--auto_scale_batch_size $auto_scale_batch_size \
		--auto_lr_find $auto_lr_find \
		--deterministic False \
		--freeze_transformer False