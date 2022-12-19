#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:v100:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=prebert # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

srun python3 pretrain.py \
    --model_name_or_path "domainbert/" \
    --dataset_name ccdv/pubmed-summarization \
    --dataset_config_name document \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --save_steps=40000 \
    --num_train_epochs 7 \
    --output_dir "/nfs-share/lj408/FDABERT/domainbert" \
    --cache_dir "/nfs-share/lj408/FDABERT/cache/domainbert" \