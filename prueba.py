"""
Script de diagnóstico para encontrar el problema en el entrenamiento
Ejecutar ANTES de entrenar para verificar data generator, labels y modelo
"""

import os
import shutil

# ==============================================================================
# CONFIG
# ==============================================================================
ORIGIN_PATH = "../../../remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Training/texting"
DESTINATION_PATH = "./sint_texting"
print("Origen:", os.path.abspath(ORIGIN_PATH))
print("Destino:", os.path.abspath(DESTINATION_PATH))
count = 0
cara = 0

for foto in os.listdir(ORIGIN_PATH):
    if "cara" not in foto and "cuerpo" not in foto:
        print(f"Moviendo: {foto}")
        shutil.move(os.path.join(ORIGIN_PATH, foto), os.path.join(DESTINATION_PATH, foto))
        count += 1

print(f"Fotos = {count}.")