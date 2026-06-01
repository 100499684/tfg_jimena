# imports
import pandas as pd
import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import time



# Borrar el contenido de train y test

import shutil
import os

# Rutas
ruta_train = "./Training"
ruta_test = "./Testing"

# Ver rutas reales
print("TRAIN →", os.path.abspath(ruta_train))
print("TEST  →", os.path.abspath(ruta_test))

print("⚠️  Borrando Training y Testing...")

# Borrar sin preguntar (DIRECTO)
if os.path.exists(ruta_train):
    shutil.rmtree(ruta_train)
    print("✅ Training borrado")
    
if os.path.exists(ruta_test):
    shutil.rmtree(ruta_test)
    print("✅ Testing borrado")

print("🎯 Listo!")



def procesar_csv(archivo):
    try:
        # Nombre de la foto es el del csv añadiendo el frame que es al final
        nombre_csv = os.path.basename(archivo)
        prefijo_video = nombre_csv.replace('.csv', '')
        num_sujeto = int(nombre_csv.split('_')[1]) # Extraemos el ID

        # Ruta carpeta video
        carpeta_video = nombre_csv.split(';')[0]
        
        # Determinamos si es para la carpeta de entrenamiento o test
        if num_sujeto in sujeto_train:
            dir_parc = ruta_train  
        else:
            dir_parc = ruta_test
        
        df = pd.read_csv(archivo)
        
        contador_copias = 0
        for index, row in df.iterrows():
            frame_num = int(row['frame'])
            accion = row['action']
            dir_final = os.path.join(dir_parc, accion)
            
            # Nombre exacto del archivo PNG
            nombre_img = f"{prefijo_video}{frame_num}.png"
            
            # Copiar cuerpo
            origen_cuerpo = os.path.join(ruta_cuerpo, carpeta_video, nombre_img)
            if os.path.exists(origen_cuerpo):
                destino_cuerpo = os.path.join(dir_final, f"cuerpo_{nombre_img}")
                shutil.copy2(origen_cuerpo, destino_cuerpo)
                contador_copias += 1

            # Copiar cara
            origen_cara = os.path.join(ruta_cara, carpeta_video, nombre_img)
            if os.path.exists(origen_cara):
                destino_cara = os.path.join(dir_final, f"cara_{nombre_img}")
                shutil.copy2(origen_cara, destino_cara)
                contador_copias += 1
                
        print(f"Completado: {prefijo_video}")
        return f"✓ {nombre_csv} ({contador_copias} archivos)"

    except Exception as e:
        return f"Error en {archivo}: {e}"




# Creamos una lista de los sujetos de training
# Miramos todos los csv de esos sujetos
# Cada frame de una accion determinada ira a su carpeta determinada


# Lista con el número de cada sujeto del training (80/20)
sujeto_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
sujeto_test = [24, 25, 26, 27, 28, 29]

# Rutas
ruta_test = "./Testing"
ruta_train = "./Training"
ruta_csv = "../repositorio/DMD/acciones/*.csv"
ruta_cara = "../repositorio/DMD/cara/"
ruta_cuerpo = "../repositorio/DMD/cuerpo/"

print("TRAIN →", os.path.abspath(ruta_train))
print("TEST  →", os.path.abspath(ruta_test))
print("CSV →", os.path.abspath(ruta_csv))
print("CARA  →", os.path.abspath(ruta_cara))
print("CUERPO →", os.path.abspath(ruta_cuerpo))


print("--- Iniciando copia de archivos ---")

# Encontrar todos los CSV
archivos = glob.glob(ruta_csv)

# Sacamos las acciones del fichero de acciones_DMD.txt
acciones = [
    "safe_drive",
    "phonecall_left",
    "phonecall_right", 
    "reach_side",
    "texting_right",
    "texting_left",
    "hair_and_makeup",
    "drinking",
    "radio",
    "talking_to_passenger",
    "change_gear",
    "reach_backseat"
]
print("Número de acciones a procesar: ", len(acciones), "\n")



# Creamos todas las carpetas de destino
print("Creando carpetas destino...")
for accion in acciones:
    os.makedirs(os.path.join(ruta_train, accion), exist_ok=True)
    os.makedirs(os.path.join(ruta_test, accion), exist_ok=True)
print("Carpetas creadas\n\n")



# Realizamos la separacion en train y test paralelamente
print("-"*20, "Realizando copia de archivos", "-"*20)
inicio = time.time()

# Creamos los hilos
NUM_PROCESOS = 8

with ThreadPoolExecutor(max_workers=NUM_PROCESOS) as executor:
    futures = []
    for csv in archivos:
        futures.append(executor.submit(procesar_csv, csv))

    # Sin barra de progreso, solo vamos contando
    completados = 0
    for future in futures:
        resultado = future.result()
        completados += 1
        if completados % 10 == 0:  # Muestra cada 5 CSVs
            print(f"  Procesados: {completados}/{len(archivos)} CSVs")
    

tiempo = time.time() - inicio

print("\n\n")
print("-"*20, "Proceso de organización terminado", "-"*20)
print(f"Tiempo total: {tiempo/60:.1f} minutos")



# Comprobamos la longitud de ambas carpetas (train y test)
size = 0

print("-"*10, "TEST", "-"*10)
for i in os.listdir(ruta_test):
    r = os.path.join(ruta_test, i)
    long = len(os.listdir(r))
    print(f"CARPETA: {i}, LONGITUD: {long}")
    size += long
print(f"TOTAL = {size} \n\n")

size = 0
print("-"*10, "TRAIN", "-"*10)
for i in os.listdir(ruta_train):
    r = os.path.join(ruta_train, i)
    long = len(os.listdir(r))
    print(f"CARPETA: {i}, LONGITUD: {long}")
    size += long
print(f"TOTAL = {size} \n\n")



