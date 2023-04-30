# Federated Domain-adaptive Pre-training

This repository is  the implementation of Federated Domain-adaptive Pre-training (FDAPT) for language models. 



## Setup

1. Create python environment (e.g. `conda create -n fdapt` and `conda activate fdapt`)

2. Run the install: `pip install -r requirements.txt`

   

## Data Preparation

### Pre-training data

#### Download

```bash
cd data
wget https://huggingface.co/datasets/ccdv/pubmed-summarization/blob/main/train.zip
wget https://huggingface.co/datasets/ccdv/pubmed-summarization/blob/main/val.zip
unzip train.zip val.zip
```

#### Split

```bash
python3 split.py \
    --num_clients 8 \ 
    --output_dir 'noniid_voc/' \
    --noniid_type 'voc' \ # select from num, voc, len 
```

#### Pre-processing

```bash
python3 get_dataloader.py \
    --model_name_or_path "distilbert-base-cased" \
    --per_device_train_batch_size 8 \
    --output_dir "data_dir/"
    --cache_dir "cache/" \
    --num_clients 8 \
    --dataset_dir "noniid_voc/" \
```



## Federate Domain-adaptive Pre-training

```bash
python3 fdabert.py \
    --model_name_or_path "distilbert-base-cased" \
    --fed_dir_data 'data/data_dir/' \
    --checkpointing_steps=epoch \
    --output_dir "saved_models/" \
    --cache_dir "cache/" \
    --num_train_epochs 1 \
    --num_rounds 15 \
    --num_clients 8 \
    --do_freeze True \
```



## Evaluate on Downstream Tasks

```bash
# Download datasets
cd downstream-task
bash download.sh
```

### Named Entity Recognition

```bash
cd downstream-task/named-entity-recognition

python3 finetune.py \
    --model_name_or_path  "saved_models_path" \
    --tokenizer_name "distilbert-base-cased" \
    --dataset_name "ncbi_disease" \
    --output_dir "results/" \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 20 \
    --per_gpu_train_batch_size 8 \
    --cache_dir "cache/" \
    --seed 42 \
    --overwrite_output_dir \
    --save_strategy 'epoch' \
    --evaluation_strategy 'epoch' \
    --load_best_model_at_end \
    --metric_for_best_model 'eval_f1' \
    --greater_is_better True \
```

Note: `--dataset_name` is selected from "ncbi_disease", "ghadeermobasher/BC5CDR-Chemical-Disease", "drAbreu/bc4chemd_ner", "bc2gm_corpus", "linnaeus", "species_800". 

### Question Answering

```bash
cd downstream-task/question-answering

export DATA_DIR=../datasets/QA/BioASQ
export data_name=BioASQ-test-factoid-7b.json 
export gold_name=7B_golden.json
export OFFICIAL_DIR=./scripts/bioasq_eval

# Train
python3 run_factoid.py \
    --model_type distilbert \
    --tokenizer_name 'distilbert-base-cased' \
    --model_name_or_path  "saved_models_path" \
    --do_train \
    --train_file "${DATA_DIR}/BioASQ-train-factoid-7b.json" \
    --per_gpu_train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --seed 42 \
    --output_dir "results/" \
    --overwrite_output_dir \
    --cache_dir 'cache' \

# Evaluate 
python run_factoid.py \
    --model_type distilbert \
    --tokenizer_name 'distilbert-base-cased' \
    --model_name_or_path "saved_models_path" \
    --do_eval \
    --predict_file ${DATA_DIR}/${data_name[j]} \
    --golden_file ${DATA_DIR}/${gold_name[j]} \
    --per_gpu_eval_batch_size 8 \
    --seed 42 \
    --official_eval_dir ${OFFICIAL_DIR} \
    --output_dir "results/" \
    --overwrite_output_dir \
    --cache_dir 'cache' \
```



### Relation Extraction

```bash
cd downstream-task/relation-extraction
bash preprocess.sh
pip install scikit-learn
pip install pandas

export SAVE_DIR=./output
export DATA="euadr" # eudar or gad
export SPLIT="1" # 1,2,3,4,5,6,7,8,9,10
export DATA_DIR=../datasets/RE/${DATA}/${SPLIT}
export ENTITY=${DATA}-${SPLIT}

# Train 
python run_re.py \
    --task_name SST-2 \
    --config_name distilbert-base-cased \
    --tokenizer_name distilbert-base-cased \
    --data_dir ${DATA_DIR} \
    --model_name_or_path  "saved_models_path" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --seed 42 \
    --do_train \
    --do_predict \
    --learning_rate 5e-5 \
    --output_dir ${SAVE_DIR}/${ENTITY} \
    --overwrite_output_dir \
    --cache_dir "cache" \
    --overwrite_cache 
    
# Evaluate
python ./scripts/re_eval.py --output_path=${SAVE_DIR}/${ENTITY}/test_results.txt --answer_path=${DATA_DIR}/test.tsv
```



## Acknowledgement

We would like to thank the authors for providing the code access to [Flower](https://github.com/adap/flower), [Hugging Face]([Hugging Face (github.com)](https://github.com/huggingface)) and [BioBERT](https://github.com/dmis-lab/biobert-pytorch).

