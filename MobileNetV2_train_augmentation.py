# Imports
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight
import os
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from Codigo_graficas.guardar_history import guardar_historial, augment_data
from Codigo_graficas.graficas import graficar_accuracy, graficar_loss
from matplotlib.backends.backend_pdf import PdfPages

# CONFIGURACIÓN
mixed_precision.set_global_policy('mixed_float16')
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

ruta_train  = "./Training"
ruta_test   = "./Testing"
ruta_output = "./Estudios"
nombre_archivo = "MobileNetV2_combinacion2_augmentation"
f_name = f"MobileNetV2_combinacion2_augmentation.txt"  # Para metricas de entrenamiento


# MobileNetV2 usa 224x224 normalmente (más pequeño que EfficientNet)
IMG_SIZE    = 224        # MobileNetV2 está optimizado para 224x224
BATCH_SIZE  = 128        # Puede ser más grande porque MobileNet es más ligero
NUM_CLASSES = 11
EPOCHS_FASE1 = 20  
EPOCHS_FASE2 = 15

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

# Obtener nombres de clases
class_names = train_datagen.class_names
print(f"\n📋 Clases: {class_names}")
print(f"Total clases: {len(class_names)}")









# ============================================
# CALCULAR PESOS DE CLASE (para balancear)
# ============================================
print("\n📊 Calculando pesos de clase...")

# Contar muestras por clase temporalmente
temp_dataset = train_datagen.map(lambda x, y: y)
y_train_temp = np.concatenate(list(temp_dataset.as_numpy_iterator()))
class_counts = np.bincount(y_train_temp)

print("Distribución original:")
for i, cls in enumerate(class_names):
    print(f"  {cls:25s}: {class_counts[i]:6d} muestras")

# Calcular pesos balanceados
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_temp),
    y=y_train_temp
)
class_weight_dict = dict(enumerate(class_weights))

print("\nPesos asignados (mayor peso = clase más importante):")
for i, cls in enumerate(class_names):
    print(f"  {cls:25s}: {class_weight_dict[i]:.4f}")


# ============================================
# DATA AUGMENTATION ESPECÍFICO
# ============================================
print("\n🔧 Configurando Data Augmentation...")

# Identificar clases minoritarias (accuracy < 70% según análisis previo)
minority_classes = ['texting', 'talking_to_passenger', 'radio', 'reach_backseat']
minority_indices = [class_names.index(cls) for cls in minority_classes if cls in class_names]
print(f"Clases con augmentation especial: {minority_classes}")
print(f"Índices: {minority_indices}")

# Augmentation suave (para todas las clases)
basic_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05),
])

# Augmentation agresivo (solo para clases minoritarias)
aggressive_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])


def augment_data(x, y):
    """Aplica augmentation según la clase"""
    # Augmentation básico para todas
    x = basic_augmentation(x, training=True)
    
    # Augmentation extra para clases minoritarias
    is_minority = tf.reduce_any(tf.equal(y, minority_indices))
    x = tf.cond(is_minority, 
                lambda: aggressive_augmentation(x, training=True), 
                lambda: x)
    return x, y


# Preprocesado específico de MobileNetV2
train_datagen = (
    train_datagen
    .shuffle(100)
    .map(lambda x, y: (tf.cast(x, tf.float32), y), 
         num_parallel_calls=AUTOTUNE)
    .map(augment_data, num_parallel_calls=AUTOTUNE)
    .map(lambda x, y: (preprocess_input(x), y), 
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
ruta_mejor_modelo_f1 = os.path.join(ruta_output, "Modelo", f"mejor_{nombre_archivo}_fase1.keras")
ruta_mejor_modelo_f2 = os.path.join(ruta_output, "Modelo", f"mejor_{nombre_archivo}_fase2.keras")

callbacks_fase1 = [
    ModelCheckpoint(ruta_mejor_modelo_f1, monitor='val_accuracy', 
                    save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
]

callbacks_fase2 = [
    ModelCheckpoint(ruta_mejor_modelo_f2, monitor='val_accuracy', 
                    save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-8, verbose=1)
]

print("✅ Callbacks configurados")







# ENTRENAMIENTO — Fase 1 (solo cabeza)
print("\n" + "="*60)
print("\n" + "="*60)
print("🎯 FASE 1: ENTRENANDO CABEZA DE CLASIFICACIÓN")
print("="*60)

start_time = time.time()

history_fase1 = model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=EPOCHS_FASE1,
    class_weight=class_weight_dict,
    callbacks=callbacks_fase1,
    verbose=2
)

tiempo_fase1 = time.time() - start_time
print(f"\n✅ Fase 1 completada en {tiempo_fase1/60:.2f} minutos")




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

start_ft = time.time()

history_fase2 = model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=EPOCHS_FASE2,
    class_weight=class_weight_dict,
    callbacks=callbacks_fase2,
    verbose=2
)

tiempo_fase2 = time.time() - start_ft
print(f"\n✅ Fase 2 completada en {tiempo_fase2/60:.2f} minutos")



# Guardar historiales
guardar_historial(history_fase1, f"{nombre_archivo}_history_fase1")
guardar_historial(history_fase2, f"{nombre_archivo}_history_fase2")






# GUARDAR MÉTRICAS DE ENTRENAMIENTO
os.makedirs(os.path.join(ruta_output, "Entrenamiento"), exist_ok=True)
ruta_archivo = os.path.join(ruta_output, "Entrenamiento", f_name)

with open(ruta_archivo, "w", encoding='utf-8') as f:
    f.write(f"Modelo: MobileNetV2 con Data Augmentation\n")
    f.write(f"Image size = {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write(f"Epochs fase 1 = {len(history_fase1.history['accuracy'])}\n")
    f.write(f"Epochs fase 2 = {len(history_fase2.history['accuracy'])}\n")
    f.write(f"Tiempo fase 1 = {tiempo_fase1/60:.2f} min\n")
    f.write(f"Tiempo fase 2 = {tiempo_fase2/60:.2f} min\n\n")
    
    best_acc_f1 = max(history_fase1.history['val_accuracy'])
    best_acc_f2 = max(history_fase2.history['val_accuracy'])
    f.write(f"🏆 Mejor val_accuracy fase 1: {best_acc_f1:.4f}\n")
    f.write(f"🏆 Mejor val_accuracy fase 2: {best_acc_f2:.4f}\n\n")
    
    f.write("📊 Pesos de clase aplicados:\n")
    for i, cls in enumerate(class_names):
        f.write(f"  {cls}: {class_weight_dict[i]:.4f}\n")
    
    # Guardar historial completo
    f.write("\n📊 HISTORIAL FASE 1:\n")
    for epoch in range(len(history_fase1.history['accuracy'])):
        f.write(f"Epoch {epoch+1}: acc={history_fase1.history['accuracy'][epoch]:.4f}, "
                f"val_acc={history_fase1.history['val_accuracy'][epoch]:.4f}, "
                f"loss={history_fase1.history['loss'][epoch]:.4f}, "
                f"val_loss={history_fase1.history['val_loss'][epoch]:.4f}\n")
    
    f.write("\n📊 HISTORIAL FASE 2:\n")
    for epoch in range(len(history_fase2.history['accuracy'])):
        f.write(f"Epoch {epoch+1}: acc={history_fase2.history['accuracy'][epoch]:.4f}, "
                f"val_acc={history_fase2.history['val_accuracy'][epoch]:.4f}, "
                f"loss={history_fase2.history['loss'][epoch]:.4f}, "
                f"val_loss={history_fase2.history['val_loss'][epoch]:.4f}\n")

print(f"✅ Métricas guardadas en: {ruta_archivo}")





# =====================================
# GRAFICA ACCURACY
# =====================================

# Fase 1
n = nombre_archivo + "accuracy1.png"
fig1 = graficar_accuracy(history_fase1, "Fase 1 Train MobileNetV2 - Accuracy", True, os.path.join(ruta_output, "Graficas", n))

# Fase 2
n = nombre_archivo + "accuracy2.png"
fig2 = graficar_accuracy(history_fase2, "Fase 2 Train MobileNetV2 - Accuracy", True, os.path.join(ruta_output, "Graficas", n))

# guardamos las figuras en un pdf
n = nombre_archivo + "accuracy.pdf"
with PdfPages(os.path.join(ruta_output, "Graficas", n)) as pdf:
    pdf.savefig(fig1, bbox_inches='tight')
    pdf.savefig(fig2, bbox_inches='tight')

# Cerrar figuras para liberar memoria
plt.close(fig1)
plt.close(fig2)    


# =====================================
# GRAFICA LOSS
# =====================================

# Fase 1
n = nombre_archivo + "loss1.png"
fig1 = graficar_loss(history_fase1, "Fase 1 Train MobileNetV2 - Loss", True, os.path.join(ruta_output, "Graficas", n))

# Fase 2
n = nombre_archivo + "loss2.png"
fig2 = graficar_loss(history_fase2, "Fase 2 Train MobileNetV2 - Loss", True, os.path.join(ruta_output, "Graficas", n))

# guardamos las figuras en un pdf
n = nombre_archivo + "loss.pdf"
with PdfPages(os.path.join(ruta_output, "Graficas", n)) as pdf:
    pdf.savefig(fig1, bbox_inches='tight')
    pdf.savefig(fig2, bbox_inches='tight')
    
# Cerrar figuras para liberar memoria
plt.close(fig1)
plt.close(fig2) 



print("\n" + "="*60)
print("✨ ENTRENAMIENTO COMPLETADO EXITOSAMENTE ✨")
print("="*60)