#!/bin/bash
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=100499684@alumnos.uc3m.es
#SBATCH --job-name=augm_sin2
#SBATCH --output=Estudios/terminal/process.EfficientNetB3_sintetico_train.%j.out
#SBATCH --error=Estudios/terminal/process.EfficientNetB3_sintetico_train.%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a40
source ~/.bashrc


echo "🚀 Iniciando job: $SLURM_JOB_ID"
echo "📅 Fecha: $(date)"
echo "🖥️  Nodo: $SLURM_NODELIST"
echo "🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Rutas de datos:"
echo "TRAIN_PATH: /remote-repositorio/afrodita/repo-fast/tfg_jcabrera/Training"
echo "TEST_PATH: /remote-repositorio/afrodita/repo-fast/tfg_jcabrera/Testing"

#python EfficientNet_train.py
#python EfficientNet_test.py
#python EfficientNet_train_aug.py
python EfficientNet_train2.py
#python EfficientNet_test2.py
#python EfficientNet_aug_completo.py

#python MobileNetV2_model.py
#python MobileNetV2_test.py
#python MobileNetV2_train_augmentation.py

#python evaluar_llm.py
#python Extras/imagenes_ia.py 
#python Extras/desbalance_clases.py
#python Extras/separacion_train_test.py
#python Extras/eliminar_fotos.py


echo "Job finalizado: $(date)"
