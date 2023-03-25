#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=9
#SBATCH --time=120:59:00
#SBATCH --gpus=1
#SBATCH --job-name="building_vectors"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/home/yakram/tan/sbatch_out/building_vectors.out"
#SBATCH --error="/cluster/home/yakram/tan/sbatch_out/building_vectors.err"
#SBATCH --open-mode=truncate

python build_vectors_and_index.py --language=en \
--checkpoint_directory=mnr-xlm-finetuned-jobads-checkpoint-4650_rtx/checkpoint \
--checkpoint_path=4020 \
--outdir=mnr-xlm_jobads_checkpoint_4650-4020_mnr \
--modelname=mnr-xlm_jobads_checkpoint_4650-4020_mnr \
--run_mode='euler'
