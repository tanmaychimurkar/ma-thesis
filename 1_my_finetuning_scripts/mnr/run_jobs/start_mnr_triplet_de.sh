#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_2080_ti:4
#SBATCH --time=12:00:00
#SBATCH --job-name="de_default"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/home/yakram/tan/sbatch_out/de_default.out"
#SBATCH --error="/cluster/home/yakram/tan/sbatch_out/de_default.err"
#SBATCH --open-mode=truncate

accelerate launch ../mnr_triplet_loss.py --language de \
--dataset_name=skills_labels_triplet_pairs \
--model_dir=. \
--model_checkpoint=xlm-roberta-base-pretrained \
--epochs=15 \
--batch_size=16 \
--pretrained_mode=True \
--pretrained_model=xlm-roberta-base