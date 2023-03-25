#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --job-name="en_classification-xlm-out-of-box"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/scratch/yakram/job_logs/en/classification-xlm-out-of-box.out"
#SBATCH --error="/cluster/scratch/yakram/job_logs/en/classification-xlm-out-of-box.err"
#SBATCH --open-mode=truncate

python ../classification_finetuning.py --language=en \
--model_dir=. \
--model_checkpoint=checkpoint-4650 \
--epochs=3 \
--batch_size=16 \
--pretrained=true \
--outdir=classification-bert-multilingual-out-of-box