import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time, datetime
import numpy as np

tf.keras.backend.clear_session()

# ==============================================================================
# GPU CONFIG
# ==============================================================================
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# ==============================================================================
# PATHS & HYPERPARAMS
# ==============================================================================
ruta_train  = "./Training"
ruta_output = "./Estudios"
nombre_archivo = "modelo_efficientnetb3_augm_eliminados_9clases_train"
f_name = f"{nombre_archivo}_{datetime.date.today()}.txt"

IMG_SIZE   = 300
BATCH_SIZE = 32
EPOCHS     = 20
AUTOTUNE   = tf.data.AUTOTUNE

# ==============================================================================
# CONTEO REAL DESDE DISCO (sin tocar el dataset tf.data)
# FIX: evita el doble pase y garantiza class_weight sobre datos reales
# ==============================================================================
class_dirs = sorted([
    d.name for d in os.scandir(ruta_train) if d.is_dir()
])
counts_real = {
    name: len(os.listdir(os.path.join(ruta_train, name)))
    for name in class_dirs
}
class_names  = class_dirs
NUM_CLASSES  = len(class_names)
total_real   = sum(counts_real.values())
mean_count   = total_real / NUM_CLASSES

print("Clases:", class_names)
print("Conteo por clase:", counts_real)

# class_weight calculado ANTES de cualquier augmentation
class_name_to_idx = {name: i for i, name in enumerate(class_names)}
class_weight = {
    class_name_to_idx[name]: total_real / (NUM_CLASSES * count)
    for name, count in counts_real.items()
}
print("Class weights:", class_weight)

# Clases minoritarias (< 75% de la media)
MINORITY_THRESHOLD = 0.75
minority_indices = tf.constant(
    [class_name_to_idx[n] for n, c in counts_real.items()
     if c < mean_count * MINORITY_THRESHOLD],
    dtype=tf.int32
)
print("Índices minoritarios:", minority_indices.numpy())

# ==============================================================================
# AUGMENTATION — instanciado UNA sola vez fuera del map()
# FIX: evita reinstanciar capas Keras por cada muestra
# ==============================================================================
augment_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")

@tf.function
def augment_if_minority(image, label):
    """Augmentation selectivo: solo si la clase es minoritaria."""
    is_minority = tf.reduce_any(tf.equal(minority_indices, tf.cast(label, tf.int32)))
    image = tf.cond(
        is_minority,
        lambda: augment_layer(tf.expand_dims(image, 0), training=True)[0],
        lambda: image
    )
    return image, label

# ==============================================================================
# DATASETS
# FIX: unbatch/batch solo en train; val nunca se augmenta
# ==============================================================================
def make_dataset(subset, shuffle=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        ruta_train,
        validation_split=0.1,
        subset=subset,
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='rgb'
    )
    # IMPORTANT: apply augmentation on raw images (float32 in 0-255 range),
    # then apply `preprocess_input` required by EfficientNet.
    if shuffle:
        ds = ds.shuffle(2000, reshuffle_each_iteration=True)
        # Unbatch to work sample-by-sample and cast to float32
        raw = ds.unbatch().map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)

        # Create per-class repeated datasets (infinite) and sample from them
        # with weights proportional to inverse frequency (class_weight) so
        # minoritaries are oversampled.
        per_class = []
        weights_raw = []
        for name, i in class_name_to_idx.items():
            d_i = raw.filter(lambda x, y, idx=i: tf.equal(y, idx))
            d_i = d_i.repeat()
            per_class.append(d_i)
            # use previously computed class_weight (higher for minority classes)
            weights_raw.append(float(class_weight[i]))

        # Avoid passing float weights into sample_from_datasets (can trigger GPU JIT errors).
        # Create integer multipliers proportional to class_weight to duplicate minority class
        # datasets so they are sampled more often.
        min_w = min(weights_raw)
        multipliers = [min(20, max(1, int(round(w / min_w)))) for w in weights_raw]

        per_class_expanded = []
        for d_i, m in zip(per_class, multipliers):
            for _ in range(m):
                per_class_expanded.append(d_i)

        balanced = tf.data.experimental.sample_from_datasets(per_class_expanded)

        # Apply augmentation (still conditional) then preprocess and batch
        augmented = balanced.map(augment_if_minority, num_parallel_calls=AUTOTUNE)
        ds = augmented.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
    else:
        # validation: no augmentation, just preprocess
        ds = ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)

train_ds = make_dataset("training",   shuffle=True)
val_ds   = make_dataset("validation", shuffle=False)

# ==============================================================================
# MODEL
# FIX: Lambda eliminada (no hay mixed_float16 activo)
# ==============================================================================
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

os.makedirs(os.path.join(ruta_output, "Modelo"), exist_ok=True)
with open(os.path.join(ruta_output, "Modelo", nombre_archivo + "_summary.txt"), 'w') as f:
    model.summary(print_fn=lambda line: f.write(line + '\n'))

# ==============================================================================
# CALLBACKS — reutilizados entre fases con rutas distintas
# ==============================================================================
def make_callbacks(suffix="", patience_es=5, patience_lr=3):
    ruta = os.path.join(ruta_output, "Modelo", f"mejor_{nombre_archivo}{suffix}.keras")
    return [
        ModelCheckpoint(ruta, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=patience_es, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience_lr, min_lr=1e-7, verbose=1),
    ]

# ==============================================================================
# FASE 1 — solo cabeza
# ==============================================================================
print("\n" + "="*60)
print("FASE 1: cabeza de clasificación")
print("="*60)

t0 = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=make_callbacks(),
    class_weight=class_weight,   # FIX: pesos sobre datos reales
    verbose=2
)
t1 = time.time()
print(f"Fase 1: {(t1-t0)/60:.2f} min")

# ==============================================================================
# FASE 2 — fine-tuning (últimas 50 capas)
# ==============================================================================
print("\n" + "="*60)
print("FASE 2: fine-tuning (últimas 50 capas)")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

t2 = time.time()
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=make_callbacks("_ft", patience_es=4, patience_lr=2),
    class_weight=class_weight,   # FIX: consistente con fase 1
    verbose=2
)
t3 = time.time()
print(f"Fase 2: {(t3-t2)/60:.2f} min")

# ==============================================================================
# GUARDAR MÉTRICAS
# ==============================================================================
os.makedirs(os.path.join(ruta_output, "Entrenamiento"), exist_ok=True)
with open(os.path.join(ruta_output, "Entrenamiento", f_name), "w", encoding='utf-8') as f:
    f.write(f"Modelo: EfficientNetB3\n")
    f.write(f"Image size = {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write(f"Clases: {class_names}\n")
    f.write(f"Índices minoritarios: {minority_indices.numpy().tolist()}\n")
    f.write(f"Conteo real por clase: {counts_real}\n")
    f.write(f"Class weights: {class_weight}\n\n")
    f.write(f"Epochs fase 1 = {len(history.history['accuracy'])}\n")
    f.write(f"Epochs fase 2 = {len(history_ft.history['accuracy'])}\n")
    f.write(f"Tiempo fase 1 = {(t1-t0)/60:.2f} min\n")
    f.write(f"Tiempo fase 2 = {(t3-t2)/60:.2f} min\n")
    f.write(f"Mejor val_accuracy fase 1: {max(history.history['val_accuracy']):.4f}\n")
    f.write(f"Mejor val_accuracy fase 2: {max(history_ft.history['val_accuracy']):.4f}\n")

print("\nDONE")