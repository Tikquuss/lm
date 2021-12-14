import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, AutoTokenizer

from datasets import load_dataset, arrow_dataset

from loguru import logger
from typing import List, Union, Dict
from functools import partial
import os
import nltk
import pandas as pd
from pandas.io.parsers import ParserError
import tqdm
import random

"""
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L552
input_ids : torch.LongTensor
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L570
attention_mask : torch.FloatTensor
"""

def mlm_collate_fn(features, mask_token_id : int, mlm_collator : DataCollatorForLanguageModeling, attn_pad_token_id = 0):
    #keys = [k for k in features[0].keys() if k != label_column]
    keys = features[0].keys()
    L = len(features)
    features = {k : [features[i][k] for i in range(L)] for k in keys}
    x_mlm = mlm_collator(features["input_ids"])
    
    #https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L922
    # TODO : check the good option, avoid using for
    if False :
        padding_mask  = pad_sequence(
            [torch.tensor(a_m).int() for a_m in features["attention_mask"]], 
            padding_value=attn_pad_token_id,
            batch_first=True
        )
        mlm_mask = (x_mlm['input_ids'] != mask_token_id).int() # 1 if token == mask_token_id else 0
        features["attention_mask"] = (mlm_mask & padding_mask).float()
    else :
        padding_mask  = pad_sequence(
            [torch.FloatTensor(a_m) for a_m in features["attention_mask"]], 
            padding_value=attn_pad_token_id,
            batch_first=True
        )
        features["attention_mask"] = padding_mask 
    features['input_ids'] = torch.LongTensor(x_mlm['input_ids'])
    features['labels'] = torch.LongTensor(x_mlm['labels']) 
    #features["token_type_ids"] = torch.LongTensor(features["token_type_ids"])
    features["token_type_ids"] = pad_sequence(
        [torch.LongTensor(x_i) for x_i in features["token_type_ids"]], 
        padding_value=0,
        batch_first=True
    )
    return features

def clm_collate_fn(features, pad_token_id : int, attn_pad_token_id : int = 0):
    #keys = [k for k in features[0].keys() if k != label_column]
    keys = features[0].keys()
    L = len(features)
    features = {k : [features[i][k] for i in range(L)] for k in keys}
    
    #padding_collator = DataCollatorWithPadding(tokenizer, padding = True, max_length=512)
    #features = padding_collator(features)
    
    #https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L922
    # TODO : avoid using for
    features["input_ids"]  = pad_sequence(
        [torch.LongTensor(x_i) for x_i in features["input_ids"]], 
        padding_value=pad_token_id,
        batch_first=True
    )
    padding_mask  = pad_sequence(
        [torch.FloatTensor(a_m) for a_m in features["attention_mask"]], 
        padding_value=attn_pad_token_id,
        batch_first=True
    )
    features["attention_mask"] = padding_mask 
    """
    The labels is the same as the inputs, shifted to the left.
    We duplicate the inputs for our labels. This is because the model of the 🤗 Transformers library 
    apply the shifting to the right, so we don't need to do it manually.
    """
    #features["labels"] = features["input_ids"].copy()
    features["labels"] = features["input_ids"].clone()
    return features

class LMLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        batch_size: int,
        num_workers: int,
        max_length: int,
        dataset_name : str,
        num_proc : int = 4,
        data_files : dict = None,
        text_column : str = 'text',
        label_column : str = '',
        clm : bool = False,
        mlm : bool = False, 
        mlm_probability : float=0.15
    ):
        super(LMLightningDataModule, self).__init__()
        assert clm ^ mlm
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        assert data_files is None or all([k in data_files.keys() for k in ["train", "validation", "test"]])
        self.data_files = data_files
        self.dataset_name = dataset_name
        self.num_proc = num_proc
        self.text_column = text_column
        self.label_column = label_column
        if clm :
            if not tokenizer.pad_token_id :
                tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            self.collate_fn =  partial(clm_collate_fn, pad_token_id = tokenizer.pad_token_id)  
        if mlm :
            mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
            )
            mask_token_id = tokenizer.mask_token_id 
            self.collate_fn = partial(mlm_collate_fn, mask_token_id=mask_token_id, mlm_collator = mlm_collator)
        
        self.prepare_data()
        
    def prepare_data(self):
        logger.info(f"Dataset {self.dataset_name} loading....")
        #https://github.com/huggingface/datasets/blob/master/src/datasets/load.py#L1503
        if self.data_files is None :
            self.dataset = load_dataset(self.dataset_name)
        else :
            self.dataset = load_dataset(
                path = os.path.abspath(os.getcwd()), 
                name = self.dataset_name, 
                data_files = self.data_files
            )
        logger.info(f"Loading of {self.dataset_name} datasets completed.")
        
        to_remove_column = [x for x in self.dataset.column_names["train"] 
                            if x != self.text_column and x != self.label_column]
        self.dataset = self.dataset.remove_columns(to_remove_column)

        def tokenize_function(examples):
            return self.tokenizer(examples[self.text_column], truncation=True, max_length=self.max_length)
        self.dataset = self.dataset.map(tokenize_function, batched=True, num_proc=self.num_proc, remove_columns=[self.text_column])

        self.train, self.validation, self.test = (
            self.dataset["train"],
            self.dataset["validation"],
            self.dataset["test"],
        )

        #self.train = self._data_processing(self.train, "Training")
        #self.validation = self._data_processing(self.validation, "Validation")
        #self.test = self._data_processing(self.test, "Testing")

    def _data_processing(self, dataset: arrow_dataset.Dataset, name: str):
        logger.info(f"{name} data transformation...")
        # TODO
        logger.info(f"{name} data transformation completed.")
        return dataset

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

def load_from_csv(file_list, text_column, label_column, if_shuffle=True, n_samples = None):
    data_label_pairs = []
    #n_samples = float("inf") if n_samples is None or n_samples < 0 else n_samples
    #n = 0
    for _index in range(len(file_list)):
    #    flag = False
        file_item = file_list[_index]
        try :
            df = pd.read_csv(file_item)
        except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
            df = pd.read_csv(file_item, lineterminator='\n')
        #for row in df.iterrows() : 
        for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
            row = row[1]
            text = row[text_column].strip()
            label = [row[label_column]]
            data_label_pairs.append([text, label])

    if if_shuffle:
        random.shuffle(data_label_pairs)
        
    data_label_pairs = data_label_pairs[:n_samples]

    return data_label_pairs

def buid_dict_file_from_csv(file_list, dict_file, text_column):
    word_to_id = {}
    for file_item in file_list:
        try :
            df = pd.read_csv(file_item)
        except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
            df = pd.read_csv(file_item, lineterminator='\n')
        for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
            text = row[1][text_column].strip()
            word_list = nltk.word_tokenize(text)
            for word in word_list:
                word = word.lower()
                word_to_id[word] = word_to_id.get(word, 0) + 1

    print("Get word_dict success: %d words" % len(word_to_id))
    # write word_to_id to file
    word_dict_list = sorted(word_to_id.items(), key=lambda d: d[1], reverse=True)
    with open(dict_file, 'w') as f:
        #f.write("%s\n"%UNK_WORD)
        #f.write("%s\n"%SEP_WORD)
        #f.write("%s\n"%PAD_WORD)
        #f.write("%s\n"%CLS_WORD)
        #f.write("%s\n"%MASK_WORD)
        for ii in word_dict_list:
            #f.write("%s\t%d\n" % (str(ii[0]), ii[1]))
            f.write("%s\n" % str(ii[0]))
    print("build dict finished!") 
    

if __name__ == '__main__':
    
    import argparse
    from .utils import file_path
    nltk.download('punkt')
    
    # parse parameters
    parser = argparse.ArgumentParser(description="Dataset")
    
    parser.add_argument('-p','--paths', type=str, help="path_to_file1,path_to_file1,...") 
    parser.add_argument('-df','--dict_file', type=file_path, help="") 
    parser.add_argument('-tc','--text_column', type=str, default="", help="") 

    # generate parser / parse parameters
    args = parser.parse_args()
    args.paths = [file_path(p) for p in args.paths.split(',')]
    
    buid_dict_file_from_csv(
        file_list = args.paths, 
        dict_file = args.dict_files, 
        text_column = args.text_column
    )