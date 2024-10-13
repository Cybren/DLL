#!/bin/bash
#SBATCH --job-name=trainHubert
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/deeplearninglab/ss2024/ssl-2/logs/%A_%a_%x.log
#SBATCH --error=/mnt/stud/work/deeplearninglab/ss2024/ssl-2/errors/%A_%a_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBTACH --gres=gpu:1

split="train"
n_shard=1
rank=0
feat_dir="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCM_wav/features"
km_path="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCM_wav/kmeans/XCL_train.km"
lab_dir="/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCM_wav/labels"

#source /mnt/stud/work/deeplearninglab/ss2024/ssl-2/venvs/DLLfairseq/bin/activate
conda activate /mnt/stud/work/deeplearninglab/ss2024/ssl-2/conda/fairseq
cd /mnt/stud/work/deeplearninglab/ss2024/ssl-2/Code/fairseq/
srun python fairseq_cli/hydra_train.py \
	--config-dir /mnt/stud/work/deeplearninglab/ss2024/ssl-2/Code/fairseq/examples/hubert/config/pretrain \
	--config-name hubert_base_librispeech \
	task.data=/mnt/stud/work/deeplearninglab/ss2024/ssl-2/Data/XCM_wav task.label_dir=$lab_dir task.labels='["km"]' model.label_rate=100
