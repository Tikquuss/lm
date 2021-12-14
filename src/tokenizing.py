"""
https://medium.com/analytics-vidhya/create-a-tokenizer-and-train-a-huggingface-roberta-model-from-scratch-f3ed1138180c
https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
"""
import torch

from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, RobertaTokenizerFast, BertTokenizerFast, GPT2TokenizerFast, AlbertTokenizerFast
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, AlbertTokenizer
from tokenizers.processors import BertProcessing
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from datasets import load_dataset

import argparse
import os

from .utils import bool_flag, dir_path, file_path, csv2txt

#BOS_WORD = '[BOS]'
#EOS_WORD = "[EOS]"
MASK_WORD = "[MASK]"
#MASK_WORD = "<mask>"
SEP_WORD = "[SEP]"
#SEP_WORD = "</s>"
CLS_WORD = "[CLS]"
#CLS_WORD = "<s>"
PAD_WORD = "[PAD]"
#PAD_WORD = "<pad>"
UNK_WORD = '[UNK]'
#UNK_WORD = '<unk>'
special_tokens=[CLS_WORD, PAD_WORD, SEP_WORD, UNK_WORD, MASK_WORD]

TOKENIZERS_TYPE = {
    "byte_level_bpe" : ByteLevelBPETokenizer,
    "bert_word_piece" : None, # BERT
    "gpt2_bpe" : None, # GPT2
    "albert_unigram_model": None, # Albert, T5
}

TOKENIZERS_CLASS = {
    "roberta_tokenizer" : RobertaTokenizer,
    "roberta_tokenizer_fast": RobertaTokenizerFast, 
    "bert_tokenizer" : BertTokenizer,
    "bert_tokenizer_fast" : BertTokenizerFast,
    "gpt2_tokenizer" : GPT2Tokenizer, 
    "gpt2_tokenizer_fast": GPT2TokenizerFast,
    "albert_tokenizer" : AlbertTokenizer,
    "albert_tokenizer_fast": AlbertTokenizerFast
}

def train_from_existing(args) :
    """ model_name : gpt2, bert-base-uncased, ... """
    split="train"
    dataset = load_dataset(path = os.path.abspath(os.getcwd()), name="tmp", data_files={split : args.files})
    dataset = dataset[split]
    #all_texts = [dataset[i : i + batch_size]["text"] for i in range(0, len(dataset), batch_size)]
    def batch_iterator():
        for i in range(0, len(dataset), args.batch_size):
            yield dataset[i : i + args.batch_size]["text"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    assert tokenizer.is_fast
    tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), args.vocab_size, new_special_tokens=special_tokens)
    #tokenizer.model.save(args.save_to)
    torch.save(tokenizer, os.path.join(args.save_to, "tokenizer.pt"))

def general(args) :
    tokenizer = TOKENIZERS_TYPE[args.type](lowercase=args.lowercase)
    # Customize training
    tokenizer.train(
        files = args.files,
        vocab_size=args.vocab_size, 
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=special_tokens
    )
    #Save the Tokenizer to disk
    tokenizer.save_model(args.save_to)
    #tokenizer.save_pretrained("my-tokenizer")
    
def bert_word_piece(args) :
    tokenizer = Tokenizer(models.WordPiece(unl_token=UNK_WORD))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=args.lowercase)
    #tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(
        vocab_size=args.vocab_size, 
        min_frequency=args.min_frequency, 
        show_progress=args.show_progress, 
        special_tokens=special_tokens
    )
    # tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.train(files = args.files, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id(CLS_WORD)),
            ("[SEP]", tokenizer.token_to_id(SEP_WORD)),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.model.save(args.save_to)
    torch.save(tokenizer, os.path.join(args.save_to, "tokenizer.pt"))
    
def gpt2_bpe(args) :
    tokenizer = Tokenizer(models.BPE(lowercase=args.lowercase))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, 
        min_frequency=args.min_frequency, 
        show_progress=args.show_progress, 
        special_tokens=["<|endoftext|>"]#+special_tokens
    )
    tokenizer.train(files = args.files, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.model.save(args.save_to)
    torch.save(tokenizer, os.path.join(args.save_to, "tokenizer.pt"))
    
def albert_unigram_model(args) :
    if args.lowercase :
        pass # TODO
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Replace("``", '"'), normalizers.Replace("''", '"'), normalizers.Lowercase()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    trainer = trainers.UnigramTrainer(
        vocab_size=args.vocab_size, 
        min_frequency=args.min_frequency, 
        show_progress=args.show_progress, 
        special_tokens=special_tokens
    )
    tokenizer.train(files = args.files, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id(CLS_WORD)),
            ("[SEP]", tokenizer.token_to_id(SEP_WORD)),
        ],
    )
    tokenizer.decoder = decoders.Metaspace()
    tokenizer.model.save(args.save_to)
    torch.save(tokenizer, os.path.join(args.save_to, "tokenizer.pt"))
    
def train_tokenizer(args) :
    if args.from_existing != "" :
        args.model_name = args.from_existing
        args.batch_size = 1000
        train_from_existing(args) 
    elif args.type in ["byte_level_bpe"] :
        general(args)
    elif args.type == "bert_word_piece" :
        bert_word_piece(args)
    elif args.type == "gpt2_bpe":
        gpt2_bpe(args)
    elif args.type == "albert_unigram_model": 
        albert_unigram_model(args)
        
def load_tokenizer(model_name=None, tokenizer_folder="", t_class = None, t_type = None, task = None, MAX_LEN = 512) :
    if model_name is not None :
        return AutoTokenizer.from_pretrained(model_name)
    if os.path.exists(os.path.join(tokenizer_folder, "tokenizer.pt")) :
        tokenizer = torch.load(os.path.join(tokenizer_folder, "tokenizer.pt"))
        if t_class is not None :
            tokenizer = TOKENIZERS_CLASS[t_class](tokenizer_object=tokenizer)
        return tokenizer
    if t_type is not None :    
        assert task is None or task in ["clm", "mlm"]
        tokenizer = TOKENIZERS_TYPE[t_type](
            os.path.abspath(os.path.join(tokenizer_folder,'vocab.json')),
            os.path.abspath(os.path.join(tokenizer_folder,'merges.txt'))
        )
        # Prepare the tokenizer
        if task == "mlm" :
            tokenizer._tokenizer.post_processor = BertProcessing(
                (SEP_WORD, tokenizer.token_to_id(SEP_WORD)),
                (CLS_WORD, tokenizer.token_to_id(CLS_WORD)),
            )
        tokenizer.enable_truncation(max_length=MAX_LEN)
        if t_class is not None :
            tokenizer = TOKENIZERS_CLASS[t_class](tokenizer_object=tokenizer)
        return tokenizer
    
    if t_class is not None :
        assert task is None or task in ["clm", "mlm"]
        # Create the tokenizer from a trained one
        tokenizer = TOKENIZERS_CLASS[t_class].from_pretrained(tokenizer_folder, max_len=MAX_LEN)
        # Prepare the tokenizer
        if task == "mlm" :
            tokenizer._tokenizer.post_processor = BertProcessing(
                (SEP_WORD, tokenizer.convert_tokens_to_ids(SEP_WORD)),
                (CLS_WORD, tokenizer.convert_tokens_to_ids(CLS_WORD)),
            )
        return tokenizer
    
def build_tokenizer_from_vocab(vocab_file, t_class1, t_class2 : None) :
    assert os.path.isfile(vocab_file)
    tokenizer = TOKENIZERS_CLASS[t_class1](
            vocab_file, 
            do_lower_case=True, 
            do_basic_tokenize=True, 
            never_split=None, 
            unk_token=UNK_WORD, 
            sep_token=SEP_WORD, 
            pad_token=PAD_WORD, 
            cls_token=CLS_WORD, 
            mask_token=MASK_WORD, 
            tokenize_chinese_chars=True, 
            #strip_accents=None
        )
    if t_class2 is not None :
        tokenizer = TOKENIZERS_CLASS[t_class2](tokenizer_object=tokenizer)
    return tokenizer
    
if __name__ == '__main__':
    
    # parse parameters
    parser = argparse.ArgumentParser(description="Tokenizer")
    
    parser.add_argument('-fe', '--from_existing', type=str, default="", help="gpt2, bert-base-uncased, ...")
    parser.add_argument('-t', '--type', type=str, default="byte_level_bpe", help="")
    parser.add_argument('-lc','--lowercase', type=bool_flag, default=True, help="") 
    parser.add_argument('-p','--paths', type=str, help="path_to_file1,path_to_file1,...") 
    parser.add_argument('-vs','--vocab_size', type=int, help="") 
    parser.add_argument('-mf','--min_frequency', type=int, default=2, help="") 
    parser.add_argument('-st','--save_to', type=dir_path, help="") 
    parser.add_argument('-tc','--text_column', type=str, default="", help="") 

    # generate parser / parse parameters
    args = parser.parse_args()
    args.paths = [file_path(p) for p in args.paths.split(',')]
    
    f1 = [f for f in args.paths if os.path.splitext(f)[1] == ".csv" ]
    f1 = csv2txt(f1, args.text_column, os.path.join(args.save_to, "files.txt"))
    f2 = [f for f in args.paths if os.path.splitext(f)[1] != ".csv" ]
    args.files = f1 + f2
    
    args.show_progress = True
    train_tokenizer(args)