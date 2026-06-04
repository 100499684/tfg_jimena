# ==============================================================================
# IMPORTS
# ==============================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import numpy as np
from collections import Counter
import datetime
import time

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

tf.keras.backend.clear_session()

# ==============================================================================
# GPU CONFIG
# ==============================================================================
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.optimizer.set_jit(True)

# ==============================================================================
# PATHS
# ==============================================================================
ruta_train = "./Training"
ruta_output = "./Estudios"

nombre_archivo = "modelo_efficientnetb3_augm_eliminados_9clases_train"
f_name = f"{nombre_archivo}_{datetime.date.today()}.txt"

IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 20

AUTOTUNE = tf.data.AUTOTUNE

# ==============================================================================
# DATASET
# ==============================================================================
def make_dataset(subset, shuffle=False):

    ds = tf.keras.utils.image_dataset_from_directory(
        ruta_train,
        validation_split=0.1,
        subset=subset,
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = ds.class_names

    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 127.5 - 1.0, y),
        num_parallel_calls=AUTOTUNE
    )

    if shuffle:
        ds = ds.shuffle(2000, reshuffle_each_iteration=True)

    return ds.prefetch(AUTOTUNE), class_names


train_ds, class_names = make_dataset("training", shuffle=True)
val_ds, _ = make_dataset("validation", shuffle=False)

NUM_CLASSES = len(class_names)
print("Clases:", class_names)

# ==============================================================================
# CLASS WEIGHT (IMPORTANTE)
# ==============================================================================
y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)

counts = Counter(y_train)
total = sum(counts.values())

class_weight = {
    cls: total / (len(counts) * count)
    for cls, count in counts.items()
}

print("\nClass weights:", class_weight)

# ==============================================================================
# AUGMENTATION (GLOBAL)
# ==============================================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# ==============================================================================
# MODEL
# ==============================================================================
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================================================================
# CALLBACKS
# ==============================================================================
callbacks = [
    ModelCheckpoint(
        os.path.join(ruta_output, "Modelo", f"{nombre_archivo}.keras"),
        monitor='val_accuracy',
        save_best_only=True
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3
    )
]

# ==============================================================================
# FASE 1
# ==============================================================================
print("\nFASE 1")

start = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight
)

end = time.time()
print("Fase 1 tiempo:", (end - start)/60, "min")

# ==============================================================================
# FINE TUNING
# ==============================================================================
print("\nFASE 2")

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weight
)

# ==============================================================================
# SAVE
# ==============================================================================
os.makedirs(os.path.join(ruta_output, "Entrenamiento"), exist_ok=True)

with open(os.path.join(ruta_output, "Entrenamiento", f_name), "w") as f:
    f.write(f"Clases: {class_names}\n")
    f.write(f"Epochs F1: {len(history.history['accuracy'])}\n")
    f.write(f"Epochs F2: {len(history_ft.history['accuracy'])}\n")
    f.write(f"Best val acc F1: {max(history.history['val_accuracy'])}\n")
    f.write(f"Best val acc F2: {max(history_ft.history['val_accuracy'])}\n")

print("\nDONE")