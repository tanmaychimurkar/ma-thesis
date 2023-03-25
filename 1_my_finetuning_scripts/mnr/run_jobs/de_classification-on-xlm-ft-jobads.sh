#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --job-name="de_xlm-ft-jobads-rand"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/scratch/yakram/job_logs/de_xlm-ft-jobads-rand.out"
#SBATCH --error="/cluster/scratch/yakram/job_logs/de_xlm-ft-jobads-rand.err"
#SBATCH --open-mode=truncate

python ../mnr_triplet_loss.py --language de \
--dataset_name=hard_neg_triplets \
--model_dir=classification-xlm-job-ads-mlm-6450 \
--model_checkpoint=checkpoint-3980 \
--epochs=2 \
--batch_size=16 \
--pretrained_mode=False \
--pretrained_model=bert-base-uncased \
--outdir=mnr-hardneg-classification-on-xlm-finetuned-jobads-checkpoint-6450
