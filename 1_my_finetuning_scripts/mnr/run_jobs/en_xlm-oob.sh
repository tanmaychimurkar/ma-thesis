#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --job-name="xlm-oob-hard"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/scratch/yakram/job_logs/xlm-oob-hard.out"
#SBATCH --error="/cluster/scratch/yakram/job_logs/xlm-oob-hard.err"
#SBATCH --open-mode=truncate

python ../mnr_triplet_loss.py --language en \
--dataset_name=hard_neg_triplets \
--model_dir=. \
--model_checkpoint=checkpoint-320 \
--epochs=1 \
--batch_size=16 \
--pretrained_mode=True \
--pretrained_model=xlm-roberta-base \
--outdir=mnr-hardneg-xlm-out-of-box_rtx

