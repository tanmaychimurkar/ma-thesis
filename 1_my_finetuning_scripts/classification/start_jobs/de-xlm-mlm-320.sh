#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --job-name="de_classification-xlm-mlm-320"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/scratch/yakram/job_logs/de/classification-xlm-mlm-320.out"
#SBATCH --error="/cluster/scratch/yakram/job_logs/de/classification-xlm-mlm-320.err"
#SBATCH --open-mode=truncate

python ../classification_finetuning.py --language=de \
--model_dir=xlm-roberta-base_mlm_description_correct \
--model_checkpoint=checkpoint-600 \
--epochs=5 \
--batch_size=16 \
--pretrained=false \
--outdir=classification-xlm-mlm-600