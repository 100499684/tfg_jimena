import shutil
import os
import glob

ruta_train = "../../../../remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Training"
ruta_test = "../../../../remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Testing"

for dir in os.listdir(ruta_test):
    if dir == "texting":
        r = os.path.join(ruta_train, "texting")
        for archivo in os.listdir(r):
            if "cara" in archivo:
                a = os.path.join(r, archivo)
                os.remove(a)
                print(f"El archivo {archivo} ha sido eliminado.")