import os

ruta_train = "../Training/texting"
ruta_test = "../Testing/texting"

# Extensiones de imágenes comunes
extensiones = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")

for foto in os.listdir(ruta_train):
    if foto.lower().endswith(extensiones) and "cara" in foto.lower():
        ruta_completa = os.path.join(ruta_train, foto)
        
        try:
            os.remove(ruta_completa)
            print(f"Eliminado: {foto}")
        except Exception as e:
            print(f"Error al eliminar {foto}: {e}") 

for foto in os.listdir(ruta_test):
    if foto.lower().endswith(extensiones) and "cara" in foto.lower():
        ruta_completa = os.path.join(ruta_test, foto)
        
        try:
            os.remove(ruta_completa)
            print(f"Eliminado: {foto}")
        except Exception as e:
            print(f"Error al eliminar {foto}: {e}") 