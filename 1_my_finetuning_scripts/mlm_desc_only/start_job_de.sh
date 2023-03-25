#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=4:00:00
#SBATCH --job-name="2080_de"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/home/yakram/tan/sbatch_out/2080_de_mlm_train_tensorboard.out"
#SBATCH --error="/cluster/home/yakram/tan/sbatch_out/2080_de_mlm_train_tensorboard.err"
#SBATCH --open-mode=truncate

python mlm_description_only_checkpoint.py --language de