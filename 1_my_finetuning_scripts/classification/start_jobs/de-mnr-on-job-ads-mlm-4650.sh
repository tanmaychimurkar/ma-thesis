#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --job-name="de_classification-xlm-job-ads-mlm-4650"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/scratch/yakram/job_logs/de/classification-xlm-job-ads-mlm-4650.out"
#SBATCH --error="/cluster/scratch/yakram/job_logs/de/classification-xlm-job-ads-mlm-4650.err"
#SBATCH --open-mode=truncate

python ../classification_finetuning.py --language=de \
--model_dir=mnr-hardneg-xlm-finetuned-jobads-checkpoint-6450 \
--model_checkpoint=checkpoint/898 \
--epochs=5 \
--batch_size=16 \
--pretrained=false \
--outdir=classification-mnr-on-job-ads-mlm-6450
