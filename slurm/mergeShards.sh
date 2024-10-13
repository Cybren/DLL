#!/bin/bash
#SBATCH --job-name=mergeShards
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/deeplearninglab/ss2024/ssl-2/logs/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb

split="valid"
n_shard=1
rank=0
feat_dir="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCM_wav/features"
km_path="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCM_wav/kmeans/XCM_train.km"
lab_dir="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCM_wav/labels"
for rank in $(seq 0 $((nshard - 1))); do
	cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km
