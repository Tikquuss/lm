# LM (CLM & MLM) Training with 🤗 Transformers

## Setting
```bash
python3 -m install pip
pip install -r requirements.txt
```

## Build a tokenizer from scratch (Supports txt and csv)  

See [tokenizing.py](src/tokenizing.py)
```bash
datapath=/path/to/data
text_column=text

st=my/save/path
mkdir -p $st

python tokenizing.py -fe gpt2 -p ${datapath}/data_train.csv,${datapath}/data_val.csv,${datapath}/data_test.csv -vs 20000 -mf 2 -st $st -tc $text_column
```

## Train and/or eval a model (from scratch or from a pre-trained model and/or tokenizer)  
See [trainer.py](src/trainer.py) and [train.sh](train.sh)
```bash
. train.sh
```

### TensorBoard (loss, acc ... per epoch)
```
%load_ext tensorboard

%tensorboard --logdir ${my/log/dir}/${task}/lightning_logs
```

## Fine-tune a pre-trained model
```bash
. TODO
```