# Summarizer
The goal of the project is to create a bot that can create a summarization of a scientific paper.

## Setup
### Requirements

* Python 3.10

### Installation
`pip install -r requirements.txt`

## Run fine-tuning of a model on a chosen dataset

Example of fine-tuning of bigbird model on a xsum dataset:
```
python src/summarization_train.py 
    --model_name_or_path "google/bigbird-pegasus-large-pubmed"   # name of the model from a huggingface hub or path to saved model 
    --tokenizer_name "google/bigbird-pegasus-large-pubmed"   # name of the tokeinzer from a huggingface hub or path to saved tokenizer
    --use_auth_token True   # use or not authentification token to load model to Huggingface
    --dataset_name xsum   # name of the dataset from a huggingface
    --output_dir output/   # path to save some output
    --use_fast_tokenizer True # use or not fast tokenizer
    --cache_dir "/path/to/cache_directory/" # directory for the cache
    --overwrite_cache True # overwrite cache or not
```
N.B: Nvidia GPU is required for the fine-tuning


## Run inference on a chosen pdf paper
```
python inference.py 
    --input_url_pdf="https://arxiv.org/pdf/1908.08593.pdf"   # url of the pdf paper which will be used as input
    --input_url_tokenizer="google/bigbird-pegasus-large-pubmed"   # name of the tokenizer from a huggingface hub 
    --input_url_model="google/bigbird-pegasus-large-pubmed"   # name of the model from a huggingface hub 
    --output_path_pdf="wandb/1908.08593.pdf"   # path where the inputh .pdf paper will be saved
```
Summary of the chosen paper will be showed as an output
