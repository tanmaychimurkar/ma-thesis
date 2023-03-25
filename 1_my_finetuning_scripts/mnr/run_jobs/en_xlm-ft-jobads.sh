#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --job-name="xlm-ft-jobads-hard"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/scratch/yakram/job_logs/xlm-ft-jobads-hard.out"
#SBATCH --error="/cluster/scratch/yakram/job_logs/xlm-ft-jobads-hard.err"
#SBATCH --open-mode=truncate

python ../mnr_triplet_loss.py --language en \
--dataset_name=hard_neg_triplets \
--model_dir=xlm_roberta-job_ads_checkpoint_mlm_description_correct \
--model_checkpoint=checkpoint-4000 \
--epochs=1 \
--batch_size=16 \
--pretrained_mode=False \
--pretrained_model=bert-base-uncased \
--outdir=mnr-hardneg-xlm-finetuned-jobads-checkpoint-4650