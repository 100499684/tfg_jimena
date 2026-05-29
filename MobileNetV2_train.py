# Imports
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import os
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# CONFIGURACIÓN
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

ruta_train  = "./Training"
ruta_test   = "./Testing"
ruta_output = "./Estudios"
nombre_archivo = "modelo_mobilenetv2_combinacion2"
f_name = f"MobileNetV2_history_combinacion2.txt"

# MobileNetV2 usa 224x224 normalmente (más pequeño que EfficientNet)
IMG_SIZE    = 224        # MobileNetV2 está optimizado para 224x224
BATCH_SIZE  = 128        # Puede ser más grande porque MobileNet es más ligero
NUM_CLASSES = 11
EPOCHS      = 20

print("Configuración lista")

# DATASETS
AUTOTUNE = tf.data.AUTOTUNE

train_datagen = tf.keras.utils.image_dataset_from_directory(
    ruta_train,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_datagen = tf.keras.utils.image_dataset_from_directory(
    ruta_train,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Preprocesado específico de MobileNetV2
train_datagen = (
    train_datagen
    .shuffle(1000)
    .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_datagen = (
    val_datagen
    .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

print("\nDatasets listos 🚀")

# MODELO — MobileNetV2 + cabeza personalizada
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False    # Fase 1: solo entrenar la cabeza

# Capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nResumen del modelo:")
model.summary()

# Guardar el modelo
path_modelo = os.path.join(ruta_output, "Modelo", nombre_archivo)
os.makedirs(os.path.dirname(path_modelo), exist_ok=True)
with open(path_modelo, 'w', encoding='utf-8') as f:
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    f.write('\n'.join(summary_lines))

# CALLBACKS
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

# ENTRENAMIENTO — Fase 1 (solo cabeza)
print("\n" + "="*60)
print("FASE 1: ENTRENANDO CABEZA DE CLASIFICACIÓN")
print("="*60)

start_time = time.time()

history = model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

tiempo_total = time.time() - start_time
print(f"\nFase 1 completada en {tiempo_total/60:.2f} minutos")

# ENTRENAMIENTO — Fase 2: Fine-tuning (últimas capas)
print("\n" + "="*60)
print("🎯 FASE 2: FINE-TUNING (últimas 50 capas de MobileNetV2)")
print("="*60)

base_model.trainable = True

# MobileNetV2 tiene ~155 capas, descongelamos las últimas 50
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR muy bajo
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

n = "mejor_" + nombre_archivo + "_ft.keras"
ruta_mejor_modelo_ft = os.path.join(ruta_output, "Modelo", n)

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
    epochs=10,  # MobileNetV2 necesita menos fine-tuning
    callbacks=callbacks_ft,
    verbose=2
)

tiempo_ft = time.time() - start_ft
print(f"\n✅ Fase 2 completada en {tiempo_ft/60:.2f} minutos")

# GUARDAR MÉTRICAS DE ENTRENAMIENTO
os.makedirs(os.path.join(ruta_output, "Entrenamiento"), exist_ok=True)
ruta_archivo = os.path.join(ruta_output, "Entrenamiento", f_name)

with open(ruta_archivo, "w", encoding='utf-8') as f:
    f.write(f"Modelo: MobileNetV2\n")
    f.write(f"Image size = {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write(f"Epochs fase 1 = {len(history.history['accuracy'])}\n")
    f.write(f"Epochs fase 2 = {len(history_ft.history['accuracy'])}\n")
    f.write(f"Tiempo fase 1 = {tiempo_total/60:.2f} min\n")
    f.write(f"Tiempo fase 2 = {tiempo_ft/60:.2f} min\n\n")
    
    best_acc_f1 = max(history.history['val_accuracy'])
    best_acc_f2 = max(history_ft.history['val_accuracy'])
    f.write(f"Mejor val_accuracy fase 1: {best_acc_f1:.4f}\n")
    f.write(f"Mejor val_accuracy fase 2: {best_acc_f2:.4f}\n")
    
    # Guardar historial completo
    f.write("\n📊 HISTORIAL FASE 1:\n")
    for epoch in range(len(history.history['accuracy'])):
        f.write(f"Epoch {epoch+1}: acc={history.history['accuracy'][epoch]:.4f}, "
                f"val_acc={history.history['val_accuracy'][epoch]:.4f}, "
                f"loss={history.history['loss'][epoch]:.4f}, "
                f"val_loss={history.history['val_loss'][epoch]:.4f}\n")
    
    f.write("\n📊 HISTORIAL FASE 2:\n")
    for epoch in range(len(history_ft.history['accuracy'])):
        f.write(f"Epoch {epoch+1}: acc={history_ft.history['accuracy'][epoch]:.4f}, "
                f"val_acc={history_ft.history['val_accuracy'][epoch]:.4f}, "
                f"loss={history_ft.history['loss'][epoch]:.4f}, "
                f"val_loss={history_ft.history['val_loss'][epoch]:.4f}\n")

print(f"Métricas guardadas en: {ruta_archivo}")

print("\n" + "="*60)
print("✨ ENTRENAMIENTO COMPLETADO EXITOSAMENTE ✨")
print("="*60)