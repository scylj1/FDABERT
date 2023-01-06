#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:gtx1080:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=finetune # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

declare -a data_name=( "ncbi_disease" "drAbreu/bc4chemd_ner" "bc2gm_corpus" "jnlpba" "species_800" )  # "tner/bc5cdr" "linnaeus" "ncbi_disease"  "bigbio/gad" "bigbio/euadr" "bigbio/chemprot"
for ((k=0; k<${#data_name[@]}; k++)); 
do
  srun python3 finetune.py \
    --model_name_or_path "fdabert/" \
    --tokenizer_name "distilbert-base-cased" \
    --dataset_name "${data_name[k]}" \
    --output_dir "fda-downstream/${data_name[k]}" \
    --do_train \
    --do_eval \
    --num_train_epochs 20 \
    --per_gpu_train_batch_size 32 \
    --save_steps=100000 \
    --cache_dir "/nfs-share/lj408/FDABERT/cache/ner" \
    --overwrite_output_dir
done 