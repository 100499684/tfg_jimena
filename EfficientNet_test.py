
# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Codigo_graficas.graficas import graficar_accuracy, graficar_loss

tf.keras.backend.clear_session()

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
gpus = tf.config.list_physical_devices('GPU')
print("GPUs disponibles:", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.optimizer.set_jit(True)

ruta_train  = "/remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Training"
ruta_test   = "/remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Testing"
ruta_output = "./Estudios"
nombre_archivo = "modelo_efficientnetb3_augmentation1"
f_name = f"EfficientNetB3_{datetime.date.today()}_augmentation1.txt"

IMG_SIZE    = 300
BATCH_SIZE  = 32
NUM_CLASSES = 9
EPOCHS      = 20

# ==============================================================================
# CONFIGURACIÓN DE BALANCEO
# ==============================================================================
# Target de imágenes por clase. Se ha elegido 80_000 como equilibrio entre
# no desperdiciar datos de las clases grandes y no repetir en exceso las pequeñas.
TARGET     = 80_000
MAX_FACTOR = 8      # Cap máximo de repetición para evitar overfitting

# Conteos reales del dataset (obtenidos en el análisis previo)
CONTEOS_REALES = {
    'drinking':              51_408,
    'hair_and_makeup':       67_916,
    'phonecall':            350_710,
    'radio':                 31_736,
    'reach_backseat':        13_390,
    'reach_side':            83_919,
    'safe_drive':           540_526,
    'talking_to_passenger':  18_176,
    'texting':               69_008,
}

print("Configuración lista")


# ==============================================================================
# DATA AUGMENTATION
# Aplicado SOLO a imágenes de training (shuffle=True).
# Las capas RandomFlip/Rotation/etc. se desactivan automáticamente en inferencia.
#
# Intensidad moderada-alta para clases con pocos datos originales:
#   - RandomFlip horizontal:     reflejo espejo, muy seguro para conducción
#   - RandomRotation ±15°:       simula variaciones de cámara/postura
#   - RandomZoom ±15%:           simula distancias distintas al conductor
#   - RandomTranslation ±10%:    simula encuadres ligeramente desplazados
#   - RandomContrast ±20%:       simula condiciones de iluminación distintas
#   - RandomBrightness ±15%:     simula día/noche/túnel
#
# NO se usa RandomFlip vertical ni rotaciones >20° porque en imágenes de
# conducción el techo y el suelo tienen significado semántico claro.
# ==============================================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
    tf.keras.layers.RandomContrast(0.20),
    tf.keras.layers.RandomBrightness(0.15),
], name="data_augmentation")

print("Data augmentation configurado")


# ==============================================================================
# PIPELINE DE DATOS BALANCEADO
# Estrategia:
#   · Clases grandes (>TARGET): undersampling aleatorio hasta TARGET
#   · Clases pequeñas (<TARGET): oversampling por repetición de paths,
#     con factor = min(round(TARGET/n), MAX_FACTOR)
#   · El augmentation en cada época genera variantes distintas gracias a
#     reshuffle_each_iteration=True + transformaciones aleatorias
# ==============================================================================

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = preprocess_input(tf.cast(img, tf.float32))
    return img, label

def make_dataset_balanced(ruta, shuffle=False):
    class_names = sorted(os.listdir(ruta))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_paths  = []
    all_labels = []

    print(f"\n{'Clase':<25} {'Original':>10} {'Factor':>8} {'Final':>10}")
    print("-" * 57)

    for class_name in class_names:
        class_dir = os.path.join(ruta, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_idx = class_to_idx[class_name]
        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        n = len(files)

        if n > TARGET:
            # Undersampling: selección aleatoria reproducible
            rng = np.random.default_rng(seed=42)
            files = list(rng.choice(files, size=TARGET, replace=False))
            factor_str = "undersample"
            final = TARGET
        else:
            factor = min(round(TARGET / n), MAX_FACTOR)
            factor = max(factor, 1)
            files  = files * factor
            factor_str = f"×{factor}"
            final  = len(files)

        all_paths.extend(files)
        all_labels.extend([class_idx] * final)

        print(f"  {class_name:<23} {n:>10,} {factor_str:>8} {final:>10,}")

    print(f"\n  Total imágenes en dataset: {len(all_paths):,}")

    ds = tf.data.Dataset.from_tensor_slices((all_paths, all_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(all_paths), reshuffle_each_iteration=True)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentation SOLO en training
    if shuffle:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_datagen = make_dataset_balanced(ruta_train, shuffle=True)
val_datagen   = make_dataset_balanced(ruta_train, shuffle=False)  # sin augmentation

print("\nDatasets listos 🚀")


# ==============================================================================
# CLASS WEIGHTS — red de seguridad adicional sobre el balanceo
# Penaliza más los errores en clases con pocos datos originales
# ==============================================================================
labels_array = np.concatenate([
    np.full(n, i) for i, n in enumerate(CONTEOS_REALES.values())
])
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_array),
    y=labels_array
)
class_weight_dict = dict(enumerate(weights))

print("\nClass weights calculados:")
for idx, (clase, _) in enumerate(CONTEOS_REALES.items()):
    print(f"  [{idx}] {clase:<25}: {class_weight_dict[idx]:.4f}")


# ==============================================================================
# MODELO — EfficientNetB3 + cabeza personalizada
# El data_augmentation NO se mete dentro del modelo porque ya se aplica
# en el pipeline de datos, evitando duplicarlo en inferencia.
# ==============================================================================
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False    # Fase 1: solo entrenar la cabeza

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

x = Lambda(lambda t: tf.cast(t, tf.float32))(x)

predictions = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nResumen del modelo:")
model.summary()


# Guardar resumen del modelo en disco
os.makedirs(os.path.join(ruta_output, "Modelo"), exist_ok=True)
path_modelo = os.path.join(ruta_output, "Modelo", nombre_archivo)
with open(path_modelo, 'w', encoding='utf-8') as f:
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    f.write('\n'.join(summary_lines))


# ==============================================================================
# CALLBACKS — Fase 1
# ==============================================================================
os.makedirs(os.path.join(ruta_output, "Modelo"), exist_ok=True)

n = "mejor_" + nombre_archivo + ".keras"
ruta_mejor_modelo = os.path.join(ruta_output, "Modelo", n)

callbacks = [
    ModelCheckpoint(
        ruta_mejor_modelo,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("✅ Callbacks configurados")


# ==============================================================================
# ENTRENAMIENTO — Fase 1 (solo cabeza)
# ==============================================================================
print("\n" + "="*60)
print("🎯 FASE 1: ENTRENANDO CABEZA DE CLASIFICACIÓN")
print("="*60)

start_time = time.time()

history = model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,   # ← refuerzo adicional
    verbose=2
)

tiempo_total = time.time() - start_time
print(f"\n✅ Fase 1 completada en {tiempo_total/60:.2f} minutos")


# ==============================================================================
# ENTRENAMIENTO — Fase 2: Fine-tuning (últimas 30 capas)
# ==============================================================================
print("\n" + "="*60)
print("🎯 FASE 2: FINE-TUNING (últimas 30 capas)")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

n_ft = "mejor_" + nombre_archivo + "_ft.keras"
ruta_mejor_modelo_ft = os.path.join(ruta_output, "Modelo", n_ft)

callbacks_ft = [
    ModelCheckpoint(
        ruta_mejor_modelo_ft,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )
]

start_ft = time.time()

history_ft = model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=10,
    callbacks=callbacks_ft,
    class_weight=class_weight_dict,   # ← también en fase 2
    verbose=2
)

tiempo_ft = time.time() - start_ft
print(f"\n✅ Fase 2 completada en {tiempo_ft/60:.2f} minutos")


# ==============================================================================
# GUARDAR MÉTRICAS DE ENTRENAMIENTO
# ==============================================================================
os.makedirs(os.path.join(ruta_output, "Entrenamiento"), exist_ok=True)
ruta_archivo = os.path.join(ruta_output, "Entrenamiento", f_name)

with open(ruta_archivo, "w", encoding='utf-8') as f:
    f.write(f"Modelo: EfficientNetB3\n")
    f.write(f"Image size = {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write(f"Target por clase = {TARGET}\n")
    f.write(f"Max factor oversampling = {MAX_FACTOR}\n\n")

    f.write("Distribución de clases tras balanceo:\n")
    for clase, n_orig in CONTEOS_REALES.items():
        if n_orig > TARGET:
            f.write(f"  {clase}: {n_orig} → {TARGET} (undersampling)\n")
        else:
            factor = min(round(TARGET / n_orig), MAX_FACTOR)
            f.write(f"  {clase}: {n_orig} → {n_orig * factor} (×{factor})\n")

    f.write(f"\nEpochs fase 1 = {len(history.history['accuracy'])}\n")
    f.write(f"Epochs fase 2 = {len(history_ft.history['accuracy'])}\n")
    f.write(f"Tiempo fase 1 = {tiempo_total/60:.2f} min\n")
    f.write(f"Tiempo fase 2 = {tiempo_ft/60:.2f} min\n\n")

    best_acc_f1 = max(history.history['val_accuracy'])
    best_acc_f2 = max(history_ft.history['val_accuracy'])
    f.write(f"Mejor val_accuracy fase 1: {best_acc_f1:.4f}\n")
    f.write(f"Mejor val_accuracy fase 2: {best_acc_f2:.4f}\n")

    f.write("\nClass weights utilizados:\n")
    for idx, clase in enumerate(CONTEOS_REALES.keys()):
        f.write(f"  [{idx}] {clase}: {class_weight_dict[idx]:.4f}\n")

print(f"Métricas guardadas en: {ruta_archivo}")


# ==============================================================================
# GRÁFICAS DE ENTRENAMIENTO
# ==============================================================================
print("\n" + "="*60)
print("📊 GENERANDO GRÁFICAS DE ENTRENAMIENTO")
print("="*60)

os.makedirs(os.path.join(ruta_output, "Graficas"), exist_ok=True)

graficar_accuracy(history, titulo="Accuracy - Fase 1", guardar=True,
                  ruta_guardado=os.path.join(ruta_output, "Graficas", f"acc_fase1_{nombre_archivo}.png"))
graficar_loss(history, titulo="Loss - Fase 1", guardar=True,
              ruta_guardado=os.path.join(ruta_output, "Graficas", f"loss_fase1_{nombre_archivo}.png"))

graficar_accuracy(history_ft, titulo="Accuracy - Fase 2 (Fine-Tuning)", guardar=True,
                  ruta_guardado=os.path.join(ruta_output, "Graficas", f"acc_fase2_{nombre_archivo}.png"))
graficar_loss(history_ft, titulo="Loss - Fase 2 (Fine-Tuning)", guardar=True,
              ruta_guardado=os.path.join(ruta_output, "Graficas", f"loss_fase2_{nombre_archivo}.png"))

print("\n✅ Todo completado.")