#!/bin/bash
#SBATCH --job-name=makeLabels
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/deeplearninglab/ss2024/ssl-2/logs/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb

split="valid"
n_shard=1
rank=0
feat_dir="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCL_wav/features"
km_path="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCL_wav/kmeans/XCL_train.km"
lab_dir="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCL_wav/labels"

#source /mnt/stud/work/deeplearninglab/ss2024/ssl-2/venvs/DLLfairseq/bin/activate
conda activate /mnt/stud/work/deeplearninglab/ss2024/ssl-2/conda/fairseq
cd /mnt/stud/work/deeplearninglab/ss2024/ssl-2/Code/fairseq/examples/hubert/simple_kmeans/
srun python dump_km_label.py $feat_dir $split $km_path 1 0 $lab_dir
