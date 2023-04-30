#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:v100:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=finetune # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate test

DATA="euadr"

for SPLIT in {1..10}
do
  ENTITY=$DATA-$SPLIT

  echo "***** " $DATA " test score " $SPLIT " *****"
  python re_eval.py \
    --output_path=../output/$ENTITY/test_results.txt \
    --answer_path=../../datasets/RE/$DATA/$SPLIT/test.tsv
done
