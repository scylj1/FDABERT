#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:rtx2080:2 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=8 # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

srun python3 fdabert.py \
    --train_file "data/train.txt" \
    --validation_file "data/val.txt" \
    --model_name_or_path "fdabert/client8/" \
    --per_device_train_batch_size 8 \
    --checkpointing_steps=epoch \
    --output_dir "/nfs-share/lj408/FDABERT/fdabert-15/client8/" \
    --cache_dir "/nfs-share/lj408/FDABERT/cache/fdabert/" \
    --num_train_epochs 1 \
    --num_rounds 5 \
    --num_clients 8 \
    