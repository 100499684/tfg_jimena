# ==============================================================================
# EVALUACIÓN DEL MEJOR MODELO EfficientNetB3 FT
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import numpy as np
import datetime

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

from keras import config
from Codigo_graficas.graficas import graficar_confianza_vs_acierto

from Codigo_graficas.m_confusion import generar_matriz_confusion

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

ruta_test = "/remote-repositorio/afrodita/repo-fast/tfg_jcabrera/Testing"

ruta_modelo = (
    "./Estudios/Modelo/"
    "mejor_modelo_efficientnetb3_principal_train_ft.keras"
)

ruta_output = "./Estudios"

IMG_SIZE = 300
BATCH_SIZE = 32

# ==============================================================================
# GPU
# ==============================================================================

gpus = tf.config.list_physical_devices("GPU")

print("GPUs disponibles:", gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ==============================================================================
# DATASET TEST
# ==============================================================================

AUTOTUNE = tf.data.AUTOTUNE

test_ds = tf.keras.utils.image_dataset_from_directory(
    ruta_test,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

test_ds = test_ds.map(
    lambda x, y: (
        preprocess_input(tf.cast(x, tf.float32)),
        y
    ),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

print("\nClases:")
print(class_names)

# ==============================================================================
# CARGAR MODELO
# ==============================================================================

print("\nCargando modelo...")

config.enable_unsafe_deserialization()

model = tf.keras.models.load_model(
    ruta_modelo,
    compile=False,
    safe_mode=False
)


print("✅ Modelo cargado")

# ==============================================================================
# EVALUACIÓN GLOBAL
# ==============================================================================

print("\n" + "="*60)
print("EVALUACIÓN")
print("="*60)

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nLoss     : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy*100:.2f}%")

# ==============================================================================
# MATRIZ DE CONFUSIÓN
# ==============================================================================

resultados = generar_matriz_confusion(
    model=model,
    test_datagen=test_ds,
    class_names=class_names,
    guardar=True,
    ruta_guardado=os.path.join(
        ruta_output,
        "Evaluacion"
    ),
    mostrar_reporte=True
)

# ==============================================================================
# REPORTE DE CLASIFICACIÓN
# ==============================================================================

print("\n" + "="*60)
print("REPORTE")
print("="*60)

print(resultados["classification_report"])

# ==============================================================================
# CONFIANZA DE PREDICCIÓN
# ==============================================================================

y_probs = model.predict(test_ds)

confianzas = np.max(y_probs, axis=1)

print("\nConfianza media :", np.mean(confianzas))
print("Confianza min   :", np.min(confianzas))
print("Confianza max   :", np.max(confianzas))
print("Desviación      :", np.std(confianzas))

# ==============================================================================
# GUARDAR RESULTADOS
# ==============================================================================

fecha = datetime.date.today()

ruta_txt = os.path.join(
    ruta_output,
    "Evaluacion",
    f"Evaluacion_EfficientNetB3_{fecha}.txt"
)

os.makedirs(
    os.path.dirname(ruta_txt),
    exist_ok=True
)

with open(ruta_txt, "w", encoding="utf-8") as f:

    f.write("EVALUACIÓN EfficientNetB3\n")
    f.write("="*60 + "\n\n")

    f.write(f"Loss: {loss:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")

    f.write(resultados["classification_report"])

print(f"\nResultados guardados en:")
print(ruta_txt)

print("\n✅ Evaluación completada")