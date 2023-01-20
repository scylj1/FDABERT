#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:rtx2080:0 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=data # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

srun python3 get_dataloader.py \
    --train_file "partition/train1.txt" \
    --validation_file "partition/val1.txt" \
    --model_name_or_path distilbert-base-cased \
    --per_device_train_batch_size 8 \
    --checkpointing_steps=20000 \
    --output_dir "/nfs-share/lj408/FDABERT/fdabert/" \
    --cache_dir "/nfs-share/lj408/FDABERT/cache/fdabert/" \
    --num_train_epochs 2 \
    --num_clients 2 \
    --dataset_dir "noniid_voc/"
    