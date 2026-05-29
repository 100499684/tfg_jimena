#!/bin/bash
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=100499684@alumnos.uc3m.es
#SBATCH --job-name=desbalance
#SBATCH --output=Estudios/terminal/process.Desbalance_clases.%j.out
#SBATCH --error=Estudios/terminal/process.Desbalance_clases.%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a40
#SBATCH --mem=64G
source ~/.bashrc

echo "🚀 Iniciando job: $SLURM_JOB_ID"
echo "📅 Fecha: $(date)"
echo "🖥️  Nodo: $SLURM_NODELIST"
echo "🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

#python EfficientNet_train.py
#python EfficientNet_test.py
python EfficientNet_train_aug.py
#python EfficientNet_train2.py

#python MobileNetV2_model.py
#python MobileNetV2_test.py
#python MobileNetV2_train_augmentation.py

#python evaluar_llm.py
#python Extras/imagenes_ia.py 
#python Extras/desbalance_clases.py


echo "Job finalizado: $(date)"
