#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --job-name="en_classification-xlm-mlm-320"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/scratch/yakram/job_logs/en/classification-xlm-mlm-320.out"
#SBATCH --error="/cluster/scratch/yakram/job_logs/en/classification-xlm-mlm-320.err"
#SBATCH --open-mode=truncate

python ../classification_finetuning.py --language=en \
--model_dir=mnr-hardneg-xlm-finetuned-320 \
--model_checkpoint=checkpoint/5469 \
--epochs=5 \
--batch_size=16 \
--pretrained=false \
--outdir=classification-mnr-on-mlm-320