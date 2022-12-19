#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:gtx1080:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=finetune # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

python3 finetune.py \
  --model_name_or_path "domainbert/" \
  --dataset_name ncbi_disease \
  --output_dir domain-ner \
  --do_train \
  --do_eval \
  --save_steps=4000 \
  --cache_dir "/nfs-share/lj408/FDABERT/cache/domainbert" \