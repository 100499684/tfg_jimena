import os
import datetime
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from Codigo_graficas.graficas import graficar_accuracy_por_clase
except ImportError:
    graficar_accuracy_por_clase = None


# ==============================================================================
# CONFIG
# ==============================================================================

RUTA_TEST = "./Testing"
RUTA_MODELO = "./Estudios/Modelos/modelo_aug_final.h5"
RUTA_OUTPUT = "./Estudios"

NOMBRE_ARCHIVO = f"test_modelo_aug_final_{datetime.date.today()}"

IMG_SIZE = (300, 300)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

CLASSES = [
    "safe_drive",
    "phonecall",
    "hair_and_makeup",
    "texting",
    "drinking",
    "reach_side",
    "radio",
    "talking_to_passenger",
    "reach_backseat",
]


# ==============================================================================
# GPU
# ==============================================================================

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


# ==============================================================================
# VALIDACIONES
# ==============================================================================

if not os.path.exists(RUTA_MODELO):
    raise FileNotFoundError(f"No existe el modelo: {RUTA_MODELO}")

if not os.path.isdir(RUTA_TEST):
    raise FileNotFoundError(f"No existe la carpeta de test: {RUTA_TEST}")

missing_classes = [
    class_name
    for class_name in CLASSES
    if not os.path.isdir(os.path.join(RUTA_TEST, class_name))
]

if missing_classes:
    raise FileNotFoundError(
        "Faltan carpetas de clases en Testing: " + ", ".join(missing_classes)
    )


# ==============================================================================
# DATASET
# ==============================================================================

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    RUTA_TEST,
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nOrden de clases usado en el test:")
for i, class_name in enumerate(test_ds_raw.class_names):
    print(f"{i}: {class_name}")


def preparar_imagenes(images, labels):
    images = tf.cast(images, tf.float32)
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    return images, labels


test_ds = test_ds_raw.map(
    preparar_imagenes,
    num_parallel_calls=AUTOTUNE,
).prefetch(AUTOTUNE)


# ==============================================================================
# MODELO
# ==============================================================================

print("\nCargando modelo completo...")
model = tf.keras.models.load_model(RUTA_MODELO, compile=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Modelo cargado correctamente")


# ==============================================================================
# EVALUACION GLOBAL
# ==============================================================================
print("\nCalculando predicciones lote a lote...")

y_probs_batches = []
y_true_batches = []
losses = []

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

for step, (images, labels) in enumerate(test_ds, start=1):
    probs = model(images, training=False)
    batch_loss = loss_fn(labels, probs).numpy()

    y_probs_batches.append(probs.numpy())
    y_true_batches.append(labels.numpy())
    losses.append(batch_loss)

    if step % 100 == 0:
        print(f"Lotes procesados: {step}")

y_probs = np.concatenate(y_probs_batches, axis=0)
y_true = np.concatenate(y_true_batches, axis=0)
y_pred = np.argmax(y_probs, axis=1)

loss = float(np.mean(losses))
acc = float(np.mean(y_true == y_pred))

print("\nLOSS:", loss)
print("ACCURACY:", acc)

# ==============================================================================
# REPORTE
# ==============================================================================

print("\nCLASSIFICATION REPORT:")
reporte = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    zero_division=0,
)
print(reporte)


# ==============================================================================
# MATRIZ DE CONFUSION
# ==============================================================================

cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))

matriz_dir = os.path.join(RUTA_OUTPUT, "Matriz confusion")
eval_dir = os.path.join(RUTA_OUTPUT, "Evaluacion")
os.makedirs(matriz_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

ruta_matriz = os.path.join(
    matriz_dir,
    f"matriz_confusion_{NOMBRE_ARCHIVO}.png",
)

plt.figure(figsize=(14, 11))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    cbar=True,
)
plt.title(f"Matriz de confusion - modelo_aug_final\nAccuracy: {acc:.4f}")
plt.xlabel("Prediccion")
plt.ylabel("Valor real")
plt.tight_layout()
plt.savefig(ruta_matriz, dpi=300, bbox_inches="tight")
plt.close()

print("\nPNG de matriz guardado en:", ruta_matriz)


# ==============================================================================
# ACCURACY POR CLASE
# ==============================================================================

class_totals = cm.sum(axis=1)
class_correct = np.diag(cm)
class_accuracy = np.divide(
    class_correct,
    class_totals,
    out=np.zeros_like(class_correct, dtype=float),
    where=class_totals != 0,
)

print("\nACCURACY POR CLASE:")
for class_name, accuracy in zip(CLASSES, class_accuracy):
    print(f"{class_name}: {accuracy:.4f}")

ruta_acc_clase = os.path.join(
    matriz_dir,
    f"accuracy_por_clase_{NOMBRE_ARCHIVO}.png",
)

if graficar_accuracy_por_clase is not None:
    graficar_accuracy_por_clase(
        cm,
        CLASSES,
        titulo="Accuracy por clase - modelo_aug_final",
        guardar=True,
        ruta_guardado=ruta_acc_clase,
    )
else:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=CLASSES, y=class_accuracy)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Clase")
    plt.title("Accuracy por clase - modelo_aug_final")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(ruta_acc_clase, dpi=300, bbox_inches="tight")
    plt.close()

print("PNG de accuracy por clase guardado en:", ruta_acc_clase)


# ==============================================================================
# CONFIANZA
# ==============================================================================

conf = np.max(y_probs, axis=1)

print("\nCONFIANZA:")
print("Media:", np.mean(conf))
print("Min:", np.min(conf))
print("Max:", np.max(conf))


# ==============================================================================
# GUARDAR RESULTADO TXT
# ==============================================================================

ruta_txt = os.path.join(
    eval_dir,
    f"evaluacion_{NOMBRE_ARCHIVO}.txt",
)

with open(ruta_txt, "w", encoding="utf-8") as f:
    f.write("EVALUACION modelo_aug_final\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Modelo: {RUTA_MODELO}\n")
    f.write(f"Test: {RUTA_TEST}\n")
    f.write(f"Loss: {loss}\n")
    f.write(f"Accuracy: {acc}\n\n")
    f.write("Clases:\n")
    for i, class_name in enumerate(CLASSES):
        f.write(f"{i}: {class_name}\n")
    f.write("\n")
    f.write("Classification report:\n")
    f.write(reporte)
    f.write("\nAccuracy por clase:\n")
    for class_name, accuracy in zip(CLASSES, class_accuracy):
        f.write(f"{class_name}: {accuracy:.4f}\n")
    f.write("\n")
    f.write(f"Confianza media: {np.mean(conf)}\n")
    f.write(f"Confianza min: {np.min(conf)}\n")
    f.write(f"Confianza max: {np.max(conf)}\n")

print("\nTXT guardado en:", ruta_txt)
print("\nEVALUACION COMPLETADA")
