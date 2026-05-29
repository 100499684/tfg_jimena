#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
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


# In[2]:


# CONFIGURACIÓN
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
 
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)
 
ruta_train  = "./Training"
ruta_test   = "./Testing"
ruta_output = "./Estudios"
 
IMG_SIZE    = 300       
BATCH_SIZE  = 64
NUM_CLASSES = 11
EPOCHS      = 20
 
 
print("Configuración lista")


# In[3]:


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
 
test_datagen = tf.keras.utils.image_dataset_from_directory(
    ruta_test,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False       
)
 
# Preprocesado específico de EfficientNet (NO dividir por 255)
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
 
test_datagen = (
    test_datagen
    .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
 
print("\nDatasets listos 🚀")


# In[5]:


# MODELO — EfficientNetB3 + cabeza personalizada
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
path_modelo = os.path.join(ruta_output, "Modelo", "modelo_efficientnetb3_combinacion1")
with open(path_modelo, 'w', encoding='utf-8') as f:
    # Pasamos el archivo como parámetro 'print_fn'
    model.summary(print_fn=lambda x: f.write(x + '\n'))


# In[6]:


# CALLBACKS
ruta_mejor_modelo = os.path.join(ruta_output, "mejor_modelo_efficientnetb3_combinacion1.keras")
 
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


# In[7]:


# ENTRENAMIENTO — Fase 1 (solo cabeza)
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
 


# In[8]:


# ENTRENAMIENTO — Fase 2: Fine-tuning (opcional)
print("\n" + "="*60)
print("🎯 FASE 2: FINE-TUNING (últimas 30 capas)")
print("="*60)
 
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR muy bajo
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
 
ruta_mejor_modelo_ft = os.path.join(ruta_output, "Modelo", "mejor_modelo_efficientnetb3_combinacion1_ft.keras")
 
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
 


# In[9]:


# GUARDAR MÉTRICAS DE ENTRENAMIENTO
f_name = f"EfficientNetB3_{datetime.date.today()}_combinacion1.txt"
ruta_archivo = os.path.join(ruta_output, "Entrenamiento", f_name)
 
with open(ruta_archivo, "w", encoding='utf-8') as f:
    f.write(f"Modelo: EfficientNetB3\n")
    f.write(f"Image size = {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write(f"Epochs fase 1 = {len(history.history['accuracy'])}\n")
    f.write(f"Epochs fase 2 = {len(history_ft.history['accuracy'])}\n")
    f.write(f"Tiempo fase 1 = {tiempo_total/60:.2f} min\n")
    f.write(f"Tiempo fase 2 = {tiempo_ft/60:.2f} min\n\n")
 
    best_acc_f1  = max(history.history['val_accuracy'])
    best_acc_f2  = max(history_ft.history['val_accuracy'])
    f.write(f"Mejor val_accuracy fase 1: {best_acc_f1:.4f}\n")
    f.write(f"Mejor val_accuracy fase 2: {best_acc_f2:.4f}\n")
 
print(f"Métricas guardadas en: {ruta_archivo}")


# In[10]:


# EVALUACIÓN CON EL MEJOR MODELO
print("\n" + "="*60)
print("📊 EVALUACIÓN FINAL")
print("="*60)
 
# Cargar el mejor modelo del fine-tuning
print(f"Cargando mejor modelo desde: {ruta_mejor_modelo_ft}")
best_model = load_model(ruta_mejor_modelo_ft)
 
loss_best, acc_best = best_model.evaluate(test_datagen, verbose=1)
print(f"\nAccuracy en test: {acc_best:.4f} ({acc_best*100:.2f}%)")
 


# In[11]:


# MATRIZ DE CONFUSIÓN — predicción por batches
print("\n" + "="*60)
print("📊 GENERANDO MATRIZ DE CONFUSIÓN")
print("="*60)
 
# Dataset de test fresco con shuffle=False garantizado
test_fresh = tf.keras.utils.image_dataset_from_directory(
    ruta_test,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)
 
class_names = test_fresh.class_names
print(f"📌 Clases: {class_names}")
 
test_fresh = (
    test_fresh
    .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
 
# Predicción batch a batch (no carga todo en RAM)
print("🔮 Prediciendo por batches...")
y_true_list = []
y_pred_list = []
 
for i, (x_batch, y_batch) in enumerate(test_fresh):
    preds = best_model(x_batch, training=False)
    y_pred_list.append(np.argmax(preds.numpy(), axis=1))
    y_true_list.append(y_batch.numpy())
 
    if i % 500 == 0:
        print(f"  → Batch {i+1} | ~{(i+1)*BATCH_SIZE:,} imágenes procesadas")
 
y_true = np.concatenate(y_true_list, axis=0)
y_pred = np.concatenate(y_pred_list, axis=0)

# Verificación de coherencia
acc_manual = np.mean(y_true == y_pred)
print(f"\n✅ Total muestras:          {len(y_true):,}")
print(f"📊 Accuracy manual:         {acc_manual:.4f}")
print(f"📊 Accuracy del evaluate(): {acc_best:.4f}")
print(f"{'✅ Coinciden' if abs(acc_manual - acc_best) < 0.01 else '❌ No coinciden — revisar pipeline'}")
 
# Classification report
print("\n📋 CLASSIFICATION REPORT:")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names))
 
# Guardar classification report
matriz_dir = os.path.join(ruta_output, "Matriz confusion")
with open(os.path.join(matriz_dir, f"classification_report_{datetime.date.today()}_combinacion1.txt"), 'w') as f:
    f.write(classification_report(y_true, y_pred, target_names=class_names))

# Calcular y guardar matriz
cm = confusion_matrix(y_true, y_pred)
 
plt.figure(figsize=(14, 11))
sns.heatmap(
    cm,
    annot=True,             # ✅ Mostrar números
    fmt='d',
    cmap='Blues',
    cbar=True,
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title(
    f"Matriz de Confusión — EfficientNetB3\nAccuracy: {acc_manual:.4f} ({acc_manual*100:.2f}%)",
    fontsize=14, fontweight='bold'
)
plt.xlabel("Predicción", fontsize=12)
plt.ylabel("Valor Real", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()



# PNG
ruta_matriz = os.path.join(matriz_dir, f"matriz_confusion_{datetime.date.today()}_combinacion1.png")
plt.savefig(ruta_matriz, dpi=300, bbox_inches='tight')
print(f"PNG guardado en: {ruta_matriz}")
 
# PDF
ruta_matriz_pdf = os.path.join(matriz_dir, f"matriz_confusion_{datetime.date.today()}_combinacion1.pdf")


# In[12]:


# GRÁFICAS DE EVOLUCIÓN
print("\n" + "="*60)
print("📈 GENERANDO GRÁFICAS DE EVOLUCIÓN")
print("="*60)
 
graficas_dir = os.path.join(ruta_output, "Graficas")
os.makedirs(graficas_dir, exist_ok=True)
 
# Combinar historial fase 1 + fase 2
train_acc  = history.history['accuracy']  + history_ft.history['accuracy']
val_acc    = history.history['val_accuracy'] + history_ft.history['val_accuracy']
train_loss = history.history['loss']      + history_ft.history['loss']
val_loss   = history.history['val_loss']  + history_ft.history['val_loss']
 
epochs_range     = range(1, len(train_acc) + 1)
num_epochs       = len(train_acc)
annotation_step  = max(1, num_epochs // 5)
 
best_epoch_acc   = int(np.argmax(val_acc)) + 1
best_acc_val     = max(val_acc)
best_epoch_loss  = int(np.argmin(val_loss)) + 1
best_loss_val    = min(val_loss)


# Línea divisoria entre fase 1 y fase 2
fase1_epochs = len(history.history['accuracy'])
 
def add_phase_line(ax, fase1_epochs, num_epochs):
    """Añade línea vertical indicando donde empieza el fine-tuning."""
    if fase1_epochs < num_epochs:
        ax.axvline(x=fase1_epochs + 0.5, color='purple', linestyle=':', alpha=0.6, linewidth=1.5)
        ax.text(fase1_epochs + 0.7, ax.get_ylim()[0] + 0.01,
                'Fine-tuning →', color='purple', fontsize=9, alpha=0.8)
 
 


# In[13]:


# --- GRÁFICA ACCURACY ---

print("📊 Generando gráfica de Accuracy...")
fig, ax = plt.subplots(figsize=(12, 8))
 
ax.plot(list(epochs_range), train_acc, 'b-o', label='Entrenamiento', linewidth=2, markersize=5)
ax.plot(list(epochs_range), val_acc,   'r-o', label='Validación',    linewidth=2, markersize=5)
ax.scatter(best_epoch_acc, best_acc_val, color='gold', s=200, zorder=5,
           marker='*', edgecolors='black', linewidth=2,
           label=f'✨ Mejor (época {best_epoch_acc}, acc={best_acc_val:.4f})')
ax.axvline(x=best_epoch_acc, color='gold', linestyle='--', alpha=0.7, linewidth=1.5)
 
if num_epochs >= 5:
    z = np.polyfit(list(epochs_range), val_acc, 2)
    ax.plot(list(epochs_range), np.poly1d(z)(list(epochs_range)),
            "g--", alpha=0.5, linewidth=1, label='Tendencia')
 
all_acc = train_acc + val_acc
ax.set_ylim([max(0, min(all_acc) - 0.05), min(1.0, max(all_acc) + 0.05)])
ax.set_title('Evolución del Accuracy — EfficientNetB3', fontsize=16, fontweight='bold')
ax.set_xlabel('Época', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(list(epochs_range))

add_phase_line(ax, fase1_epochs, num_epochs)
 
for i, (_, acc_v) in enumerate(zip(train_acc, val_acc)):
    ep = i + 1
    if ep % annotation_step == 0 or ep == best_epoch_acc or ep == num_epochs:
        ax.annotate(f'{acc_v:.3f}', xy=(ep, acc_v), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
 
plt.tight_layout()
plt.savefig(os.path.join(graficas_dir, f"accuracy_evolution_{datetime.date.today()}_combinacion1.png"),
            dpi=300, bbox_inches='tight')
print("Gráfica de Accuracy guardada")
plt.close()


# In[14]:


# --- GRÁFICA LOSS ---
print("📊 Generando gráfica de Loss...")
fig, ax = plt.subplots(figsize=(12, 8))
 
ax.plot(list(epochs_range), train_loss, 'b-o', label='Entrenamiento', linewidth=2, markersize=5)
ax.plot(list(epochs_range), val_loss,   'r-o', label='Validación',    linewidth=2, markersize=5)
ax.scatter(best_epoch_loss, best_loss_val, color='gold', s=200, zorder=5,
           marker='*', edgecolors='black', linewidth=2,
           label=f'✨ Mejor pérdida (época {best_epoch_loss}, loss={best_loss_val:.4f})')
ax.axvline(x=best_epoch_loss, color='gold', linestyle='--', alpha=0.7, linewidth=1.5)
 
if num_epochs >= 5:
    z = np.polyfit(list(epochs_range), val_loss, 2)
    ax.plot(list(epochs_range), np.poly1d(z)(list(epochs_range)),
            "g--", alpha=0.5, linewidth=1, label='Tendencia')



all_loss = train_loss + val_loss
ax.set_ylim([max(0, min(all_loss) - 0.05), max(all_loss) + 0.05])
ax.set_title('Evolución de la Pérdida (Loss) — EfficientNetB3', fontsize=16, fontweight='bold')
ax.set_xlabel('Época', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(list(epochs_range))
 
add_phase_line(ax, fase1_epochs, num_epochs)
 
for i, (_, loss_v) in enumerate(zip(train_loss, val_loss)):
    ep = i + 1
    if ep % annotation_step == 0 or ep == best_epoch_loss or ep == num_epochs:
        ax.annotate(f'{loss_v:.3f}', xy=(ep, loss_v), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
 
plt.tight_layout()
plt.savefig(os.path.join(graficas_dir, f"loss_evolution_{datetime.date.today()}_combinacion1.png"),
            dpi=300, bbox_inches='tight')
print("✅ Gráfica de Loss guardada")
plt.close()

