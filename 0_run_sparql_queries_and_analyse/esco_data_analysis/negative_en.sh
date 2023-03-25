#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --job-name="en_neg"
#SBATCH --mem-per-cpu=512
#SBATCH --output="/cluster/home/yakram/tan/sbatch_out/en_negative.out"
#SBATCH --error="/cluster/home/yakram/tan/sbatch_out/en_negative.err"
#SBATCH --open-mode=truncate

python mine_negatives_en.py