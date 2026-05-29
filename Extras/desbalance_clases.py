from collections import Counter
import os

ruta_train = "Training"
clases = sorted(os.listdir(ruta_train))

print("Distribución de clases:")
conteos = {}
for clase in clases:
    n = len(os.listdir(os.path.join(ruta_train, clase)))
    conteos[clase] = n
    print(f"  {clase}: {n} imágenes")

maximo = max(conteos.values())
print("\nFactores sugeridos para equilibrar:")
for clase, n in conteos.items():
    factor = round(maximo / n)
    print(f"  {clase}: factor {factor}")