#!/bin/bash
#SBATCH --job-name=prepHubertPretrain
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/deeplearninglab/ss2024/ssl-2/logs/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
conda activate /mnt/stud/work/deeplearninglab/ss2024/ssl-2/conda/fairseq
cd /mnt/stud/work/deeplearninglab/ss2024/ssl-2/Code/
srun python prepHubertPretraining.py
