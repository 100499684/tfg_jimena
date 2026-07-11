#!/bin/bash
#SBATCH --job-name=debug_gpu
#SBATCH --output=prueba1.out
#SBATCH --error=prueba1.err
#SBATCH --gres=gpu:1

echo "HOSTNAME:"
hostname

echo
echo "NVIDIA-SMI:"
nvidia-smi

echo
echo "NVIDIA-SMI -L:"
nvidia-smi -L

echo
echo "SLURM VARIABLES:"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo
echo "TENSORFLOW:"
python - << 'EOF'
import os
from collections import Counter

train_path = "./Training"
clases = os.listdir(train_path)
counts = {cls: len(os.listdir(os.path.join(train_path, cls))) for cls in clases}
print(counts)
EOF