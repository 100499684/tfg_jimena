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

# Mixed precision: float16 en capas internas → más rápido en GPU
#mixed_precision.set_global_policy('mixed_float16')

# XLA JIT compilation
tf.config.optimizer.set_jit(True)

ruta_train  = "/remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Training"
ruta_test   = "/remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Testing"
ruta_output = "./Estudios"
nombre_archivo = "modelo_efficientnetb3_eliminacion2"
f_name = f"EfficientNetB3_{datetime.date.today()}_eliminacion2.txt"

IMG_SIZE    = 300
BATCH_SIZE  = 32
NUM_CLASSES = 9
EPOCHS      = 20

print("Configuración lista")


# ==============================================================================
# DATASETS OPTIMIZADOS
# ==============================================================================
AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(ruta, subset, shuffle=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        ruta,
        validation_split=0.1,
        subset=subset,
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='rgb'
    )
    # Preprocesado específico de EfficientNet (NO dividir por 255)
    ds = ds.map(
        lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
        num_parallel_calls=AUTOTUNE
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=2000, reshuffle_each_iteration=True)

    #cache_path = f"/tmp/ds_cache_{subset}"
    #ds = ds.cache(cache_path)       # Cachea en RAM tras la primera época → épocas siguientes mucho más rápidas
    ds = ds.prefetch(AUTOTUNE)
    return ds

# NOTA: si el dataset es muy grande y se queda sin RAM, cambia .cache() por
# .cache("/tmp/ds_cache_train") y .cache("/tmp/ds_cache_val") para cachear en disco

train_datagen = make_dataset(ruta_train, "training",   shuffle=True)
val_datagen   = make_dataset(ruta_train, "validation", shuffle=False)

print("\nDatasets listos 🚀")


# ==============================================================================
# MODELO — EfficientNetB3 + cabeza personalizada
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

# Cast explícito a float32 ANTES de la capa final para evitar conflictos mixed_float16
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

# Recompilar con LR muy bajo para fine-tuning
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
    f.write(f"Epochs fase 1 = {len(history.history['accuracy'])}\n")
    f.write(f"Epochs fase 2 = {len(history_ft.history['accuracy'])}\n")
    f.write(f"Tiempo fase 1 = {tiempo_total/60:.2f} min\n")
    f.write(f"Tiempo fase 2 = {tiempo_ft/60:.2f} min\n\n")

    best_acc_f1 = max(history.history['val_accuracy'])
    best_acc_f2 = max(history_ft.history['val_accuracy'])
    f.write(f"Mejor val_accuracy fase 1: {best_acc_f1:.4f}\n")
    f.write(f"Mejor val_accuracy fase 2: {best_acc_f2:.4f}\n")

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





