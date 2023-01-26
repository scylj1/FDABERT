#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:v100:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=ft # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

declare -a data_name=( "ncbi_disease" "ghadeermobasher/BC5CDR-Chemical-Disease" "drAbreu/bc4chemd_ner" "bc2gm_corpus" "jnlpba" "linnaeus" "species_800" )  # "ghadeermobasher/BC5CDR-Chemical-Disease" "drAbreu/bc4chemd_ner" "bc2gm_corpus" "jnlpba" "linnaeus" "species_800"
declare -a seeds=(42 123 3407 54354 43534 ) # 123 3407 54354 43534

for ((k=0; k<${#data_name[@]}; k++)); 
do
  for ((j=0; j<${#seeds[@]}; j++)); 
  do
    srun python3 finetune.py \
      --model_name_or_path  "distilbert-base-cased" \
      --tokenizer_name "distilbert-base-cased" \
      --dataset_name "${data_name[k]}" \
      --output_dir "distilbert-ner-origin/${data_name[k]}" \
      --do_train \
      --do_eval \
      --do_predict \
      --num_train_epochs 20 \
      --per_gpu_train_batch_size 8 \
      --save_steps=100000 \
      --cache_dir "/nfs-share/lj408/FDABERT/cache/ner" \
      --seed ${seeds[j]} \
      --overwrite_output_dir \
      --save_strategy 'epoch' \
      --evaluation_strategy 'epoch' \
      --load_best_model_at_end \
      --metric_for_best_model 'eval_f1' \
      --greater_is_better True
      
  done
done 