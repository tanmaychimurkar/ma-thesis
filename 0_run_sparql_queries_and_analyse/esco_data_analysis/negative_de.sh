#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --job-name="de_neg"
#SBATCH --mem-per-cpu=512
#SBATCH --output="/cluster/home/yakram/tan/sbatch_out/de_negative.out"
#SBATCH --error="/cluster/home/yakram/tan/sbatch_out/de_negative.err"
#SBATCH --open-mode=truncate

python mine_negatives_de.py