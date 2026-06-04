#!/bin/bash
#SBATCH --job-name=debug_gpu
#SBATCH --output=debug_gpu.out
#SBATCH --error=debug_gpu.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a40

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
import tensorflow as tf

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        print(tf.config.experimental.get_device_details(gpu))
    except Exception as e:
        print(e)
EOF