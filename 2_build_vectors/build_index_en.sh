#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=9
#SBATCH --time=24:00:00
#SBATCH --job-name="all_build"
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=72000
#SBATCH --output="/cluster/home/yakram/tan/sbatch_out/all_build_cpu.out"
#SBATCH --error="/cluster/home/yakram/tan/sbatch_out/all_build_cpu.err"
#SBATCH --open-mode=truncate

#python build_vectors_and_index.py --language=en \
#--checkpoint_directory=xlm-roberta-base \
#--checkpoint_path=xlm-roberta-base \
#--outdir=xlm-roberta-base \
#--modelname=xlm-roberta-base \
#--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=xlm-roberta-base_mlm_description_correct \
--checkpoint_path=checkpoint-400 \
--outdir=xlm-roberta_mlm_description_310 \
--modelname=xlm-roberta_mlm_description_310 \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=. \
--checkpoint_path=xlm_roberta-job_ads_checkpoint \
--outdir=xlm_roberta-job_ads_checkpoint \
--modelname=xlm_roberta-job_ads_checkpoint \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=xlm_roberta-job_ads_checkpoint_mlm_description_correct \
--checkpoint_path=checkpoint-4400 \
--outdir=xlm_roberta-job_ads_checkpoint-4400 \
--modelname=xlm_roberta-job_ads_checkpoint-4400 \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=mnr-hardneg-xlm-finetuned-320/checkpoint \
--checkpoint_path=4400 \
--outdir=mnr-hardneg-xlm-finetuned-320 \
--modelname=mnr-hardneg-xlm-finetuned-320 \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=mnr-hardneg-xlm-default-jobads-checkpoint/checkpoint \
--checkpoint_path=4400 \
--outdir=mnr-hardneg-xlm-default-jobads-checkpoint \
--modelname=mnr-hardneg-xlm-default-jobads-checkpoint \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=mnr-hardneg-xlm-finetuned-jobads-checkpoint-4650/checkpoint \
--checkpoint_path=4400 \
--outdir=mnr-hardneg-xlm-finetuned-jobads-checkpoint-4650 \
--modelname=mnr-hardneg-xlm-finetuned-jobads-checkpoint-4650 \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=mnr-xlm-out-of-box/checkpoint \
--checkpoint_path=4400 \
--outdir=mnr-xlm_out-of-box \
--modelname=mnr-xlm_out-of-box \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=classification-xlm-job-ads-mlm-4650 \
--checkpoint_path=checkpoint-40765 \
--outdir=classification-xlm-job-ads-mlm-4650 \
--modelname=classification-xlm-job-ads-mlm-4650 \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=classification-bert-multilingual-out-of-box \
--checkpoint_path=checkpoint-24459 \
--outdir=classification-bert-multilingual-out-of-box \
--modelname=classification-bert-multilingual-out-of-box \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=classification-xlm-mlm-320 \
--checkpoint_path=checkpoint-40765 \
--outdir=classification-xlm-mlm-320 \
--modelname=classification-xlm-mlm-320 \
--run_mode='euler'

python build_vectors_and_index.py --language=en \
--checkpoint_directory=classification-xlm-default-jobads \
--checkpoint_path=checkpoint-40765 \
--outdir=classification-xlm-default-jobads \
--modelname=classification-xlm-default-jobads \
--run_mode='euler'
