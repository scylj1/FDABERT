#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:gtx1080:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=eval # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

declare -a model_path=( "domainbert/checkpoint-29760/" "domainbert/checkpoint-59520/" "domainbert/checkpoint-89280/" "domainbert/checkpoint-119040/" "domainbert/checkpoint-148800/" "domainbert/checkpoint-178560/" "domainbert/checkpoint-208320/" "domainbert/checkpoint-238080/" "domainbert/checkpoint-267840/" "domainbert/checkpoint-297600/" ) # "domainbert/checkpoint-29760/" "domainbert/checkpoint-59520/" "domainbert/checkpoint-89280/" "domainbert/checkpoint-119040/" "domainbert/checkpoint-148800/" "domainbert/checkpoint-178560/" "domainbert/checkpoint-208320/" "domainbert/checkpoint-238080/" "domainbert/checkpoint-267840/" "domainbert/checkpoint-297600/"
for ((k=0; k<${#model_path[@]}; k++)); # saved models
do
    srun python3 evalperplex.py \
        --model_name_or_path "${model_path[k]}" \
        --dataset_name ccdv/pubmed-summarization \
    
       
        --output_dir "/nfs-share/lj408/FDABERT/perplex" \
        --cache_dir "/nfs-share/lj408/FDABERT/cache/domainbert" \
        
done