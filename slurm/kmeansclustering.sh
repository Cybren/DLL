#!/bin/bash
#SBATCH --job-name=kmeanclustering
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/deeplearninglab/ss2024/ssl-2/logs/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb

split="train"
n_shard=1
rank=0
feat_dir="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCL_wav/features"
km_path="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCL_wav/kmeans/XCL_train.km"

#source /mnt/stud/work/deeplearninglab/ss2024/ssl-2/venvs/DLLfairseq/bin/activate
conda activate /mnt/stud/work/deeplearninglab/ss2024/ssl-2/conda/fairseq
cd /mnt/stud/work/deeplearninglab/ss2024/ssl-2/Code/fairseq/examples/hubert/simple_kmeans/
srun python learn_kmeans.py $feat_dir $split 1 $km_path 250 --percent 0.5
