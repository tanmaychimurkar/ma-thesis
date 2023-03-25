#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --time=120:59:00
#SBATCH --job-name="en_data_full"
#SBATCH --mem-per-cpu=6144
#SBATCH --tmp=300000
#SBATCH --output="../sbatch_logs/en_data_full.out"
#SBATCH --error="../sbatch_logs/en_data_full.err"
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=chimurkar.tanmay@gmail.com

python run_sparql_queries.py

