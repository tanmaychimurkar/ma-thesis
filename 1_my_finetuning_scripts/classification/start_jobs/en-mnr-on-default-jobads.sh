#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --job-name="en_classification-mnr-on-xlm-default-jobads"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/scratch/yakram/job_logs/en/classification-mnr-on-xlm-default-jobads.out"
#SBATCH --error="/cluster/scratch/yakram/job_logs/en/classification-mnr-on-xlm-default-jobads.err"
#SBATCH --open-mode=truncate

python ../classification_finetuning.py --language=en \
--model_dir=mnr-hardneg-xlm-default-jobads-checkpoint \
--model_checkpoint=checkpoint/5469 \
--epochs=5 \
--batch_size=16 \
--pretrained=false \
--outdir=classification-mnr-on-default-jobads