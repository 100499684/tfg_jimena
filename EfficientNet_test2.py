import os
import tensorflow as tf
import numpy as np
import datetime

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix
from Codigo_graficas.graficas import (
    graficar_accuracy,
    graficar_loss,
    graficar_accuracy_por_clase,
    graficar_confianza_vs_acierto,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIG
# ==============================================================================

ruta_test = "./Testing"

ruta_modelo = "./Estudios/Modelo/mejor_modelo_efficientnetb3_principal_train_ft.keras"
ruta_output = "./Estudios"

IMG_SIZE = 300
BATCH_SIZE = 16   

fecha = datetime.date.today()

# ==============================================================================
# GPU
# ==============================================================================

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ==============================================================================
# DATASET
# ==============================================================================

AUTOTUNE = tf.data.AUTOTUNE

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    ruta_test,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds_raw.class_names
print("\nClases:", class_names)

test_ds = test_ds_raw.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# ==============================================================================
# MODELO (RECONSTRUIDO)
# ==============================================================================

print("\nReconstruyendo modelo...")

base = EfficientNetB3(
    weights=None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)

outputs = Dense(
    len(class_names),
    activation="softmax",
    dtype="float32"
)(x)

model = Model(base.input, outputs)

# ==============================================================================
# CARGA DE PESOS
# ==============================================================================

print("Cargando pesos...")

model.load_weights(ruta_modelo)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Modelo cargado correctamente")

# ==============================================================================
# EVALUACIÓN
# ==============================================================================

print("\n===================================")
print("EVALUACIÓN GLOBAL")
print("===================================")

loss, acc = model.evaluate(test_ds, verbose=1)

print("\nLOSS:", loss)
print("ACCURACY:", acc)

# ==============================================================================
# PREDICCIONES
# ==============================================================================

print("\nCalculando predicciones...")

y_probs = model.predict(test_ds)
y_pred = np.argmax(y_probs, axis=1)

# etiquetas reales
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# ==============================================================================
# REPORTE
# ==============================================================================

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==============================================================================
# MATRIZ DE CONFUSIÓN
# ==============================================================================

cm = confusion_matrix(y_true, y_pred)

print("\nMatriz de confusión lista:", cm.shape)

# Guardar matriz de confusión
matriz_dir = os.path.join(ruta_output, "Matriz confusion")
os.makedirs(matriz_dir, exist_ok=True)
ruta_matriz = os.path.join(matriz_dir, f"matriz_confusion_{fecha}_efficientnetb3_comb_text.png")
plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title(f"Matriz de Confusión — EfficientNetB3\nAccuracy: {acc:.4f}")
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig(ruta_matriz, dpi=300, bbox_inches='tight')
plt.close()
print("PNG de matriz guardado en:", ruta_matriz)

# Grafica accuracy por clase (barra)
ruta_acc_clase = os.path.join(matriz_dir, f"accuracy_por_clase_{fecha}_efficientnetb3_comb_text.png")
graficar_accuracy_por_clase(cm, class_names, titulo="Accuracy por clase — EfficientNetB3", guardar=True, ruta_guardado=ruta_acc_clase)

# ==============================================================================
# CONFIANZA
# ==============================================================================

conf = np.max(y_probs, axis=1)

print("\nConfianza media:", np.mean(conf))
print("Confianza min:", np.min(conf))
print("Confianza max:", np.max(conf))

# ==============================================================================
# GUARDAR RESULTADO
# ==============================================================================

os.makedirs(os.path.join(ruta_output, "Evaluacion"), exist_ok=True)

ruta_txt = os.path.join(
    ruta_output,
    "Evaluacion",
    f"evaluacion2_efficientnetb3_{fecha}.txt"
)

with open(ruta_txt, "w", encoding="utf-8") as f:
    f.write("EVALUACION EfficientNetB3\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Loss: {loss}\n")
    f.write(f"Accuracy: {acc}\n\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))

print("\nGuardado en:", ruta_txt)

print("\n EVALUACIÓN COMPLETADA")
