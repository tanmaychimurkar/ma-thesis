#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --job-name="de_all"
#SBATCH --mem-per-cpu=2048
#SBATCH --tmp=72000
#SBATCH --output="/cluster/home/yakram/tan/sbatch_out/de_all.out"
#SBATCH --error="/cluster/home/yakram/tan/sbatch_out/de_all.err"
#SBATCH --open-mode=truncate

#python build_vectors_and_index.py --language=en \
#--checkpoint_directory=xlm-roberta-base \
#--checkpoint_path=xlm-roberta-base \
#--outdir=xlm-roberta-base \
#--modelname=xlm-roberta-base \
#--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=xlm-roberta-base_mlm_description_correct \
--checkpoint_path=checkpoint-600 \
--outdir=xlm-roberta_mlm_description_600 \
--modelname=xlm-roberta_mlm_description_600 \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=. \
--checkpoint_path=xlm_roberta-job_ads_checkpoint \
--outdir=xlm_roberta-job_ads_checkpoint \
--modelname=xlm_roberta-job_ads_checkpoint \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=xlm_roberta-job_ads_checkpoint_mlm_description_correct \
--checkpoint_path=checkpoint-5000 \
--outdir=xlm_roberta-job_ads_checkpoint-6450 \
--modelname=xlm_roberta-job_ads_checkpoint-6450 \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=mnr-hardneg-xlm-finetuned-600/checkpoint \
--checkpoint_path=898 \
--outdir=mnr-hardneg-xlm-finetuned-600 \
--modelname=mnr-hardneg-xlm-finetuned-600 \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=mnr-xlm-out-of-box/checkpoint \
--checkpoint_path=898 \
--outdir=mnr-xlm_out-of-box \
--modelname=mnr-xlm_out-of-box \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=mnr-hardneg-xlm-default-jobads-checkpoint/checkpoint \
--checkpoint_path=898 \
--outdir=mnr-hardneg-xlm-default-jobads-checkpoint \
--modelname=mnr-hardneg-xlm-default-jobads-checkpoint \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=mnr-hardneg-xlm-finetuned-jobads-checkpoint-6450/checkpoint \
--checkpoint_path=898 \
--outdir=mnr-hardneg-xlm-finetuned-jobads-checkpoint-6450 \
--modelname=mnr-hardneg-xlm-finetuned-jobads-checkpoint-6450 \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=classification-xlm-mlm-600 \
--checkpoint_path=checkpoint-3980 \
--outdir=classification-xlm-mlm-600 \
--modelname=classification-xlm-mlm-600 \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=classification-xlm-job-ads-mlm-6450 \
--checkpoint_path=checkpoint-3980 \
--outdir=classification-xlm-job-ads-mlm-6450 \
--modelname=classification-xlm-job-ads-mlm-6450 \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=classification-xlm-default-jobads \
--checkpoint_path=checkpoint-3980 \
--outdir=classification-xlm-default-jobads \
--modelname=classification-xlm-default-jobads \
--run_mode='euler'

python build_vectors_and_index.py --language=de \
--checkpoint_directory=classification-bert-multilingual-out-of-box \
--checkpoint_path=checkpoint-2388 \
--outdir=classification-bert-multilingual-out-of-box \
--modelname=classification-bert-multilingual-out-of-box \
--run_mode='euler'
