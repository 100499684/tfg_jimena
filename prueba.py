"""
Script de diagnóstico para encontrar el problema en el entrenamiento
Ejecutar ANTES de entrenar para verificar data generator, labels y modelo
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
import os

# ==============================================================================
# CONFIG
# ==============================================================================
TRAIN_PATH = "../../../remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Training"
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
CLASSES = ['safe_drive', 'phonecall', 'hair_and_makeup', 'texting',
           'drinking', 'reach_side', 'radio', 'talking_to_passenger',
           'reach_backseat']

# ==============================================================================
# TEST 1: Verificar imágenes
# ==============================================================================
print("\n" + "="*60)
print("TEST 1: Verificar integridad de imágenes")
print("="*60)

for cls in CLASSES[:3]:  # Solo primeras 3 clases
    cls_path = os.path.join(TRAIN_PATH, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    
    print(f"\n{cls}: Verificando {len(imgs)} imágenes...")
    for img_name in imgs:
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  ❌ ROTA: {img_name}")
        else:
            h, w = img.shape[:2]
            print(f"  ✅ OK: {img_name} ({h}x{w})")

# ==============================================================================
# TEST 2: Verificar data generator
# ==============================================================================
print("\n" + "="*60)
print("TEST 2: Verificar data generator")
print("="*60)

class TestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=8, target_size=(256, 256), is_training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.is_training = is_training
        self.classes = CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        
        for cls in self.classes:
            cls_path = os.path.join(data_dir, cls)
            if not os.path.exists(cls_path):
                continue
            imgs = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in imgs:
                self.samples.append((os.path.join(cls_path, img), cls))
        
        self.on_epoch_end()
    
    def __len__(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.samples))
        batch_samples = self.samples[start_idx:end_idx]
        
        X = np.empty((len(batch_samples), *self.target_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_samples), len(self.classes)), dtype=np.float32)
        
        for i, (img_path, cls) in enumerate(batch_samples):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"❌ No se pudo leer: {img_path}")
                    X[i] = np.ones((*self.target_size, 3), dtype=np.float32) * 0.5
                    y[i, self.class_to_idx[cls]] = 1
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
                
                X[i] = img.astype(np.float32) / 255.0
                y[i, self.class_to_idx[cls]] = 1
                
            except Exception as e:
                print(f"❌ Error: {img_path} - {e}")
                X[i] = np.ones((*self.target_size, 3), dtype=np.float32) * 0.5
                y[i, self.class_to_idx[cls]] = 1
        
        return X, y
    
    def on_epoch_end(self):
        np.random.shuffle(self.samples)

print("Creando data generator...")
train_gen = TestDataGenerator(TRAIN_PATH, is_training=True)
print(f"✅ Dataset cargado: {len(train_gen.samples)} muestras")
print(f"✅ Batches por época: {len(train_gen)}")

# ==============================================================================
# TEST 3: Verificar batch
# ==============================================================================
print("\n" + "="*60)
print("TEST 3: Verificar contenido de un batch")
print("="*60)

X_batch, y_batch = train_gen[0]

print(f"\n✅ X_batch.shape: {X_batch.shape}")
print(f"✅ y_batch.shape: {y_batch.shape}")
print(f"\nValores X:")
print(f"  Min: {X_batch.min():.4f}")
print(f"  Max: {X_batch.max():.4f}")
print(f"  Mean: {X_batch.mean():.4f}")

if X_batch.min() < 0 or X_batch.max() > 1:
    print(f"❌ PROBLEMA: Imágenes no están en [0, 1]")
else:
    print(f"✅ Imágenes están correctamente normalizadas [0, 1]")

print(f"\nValores y (one-hot):")
print(f"  Shape: {y_batch.shape}")
print(f"  Sum por sample: {y_batch.sum(axis=1)}")  # Debe ser [1, 1, 1, ...]

if not np.allclose(y_batch.sum(axis=1), 1):
    print(f"❌ PROBLEMA: Labels no son one-hot válidos")
else:
    print(f"✅ Labels son one-hot válidos")

print(f"\nDistribución de clases en batch:")
for i, cls in enumerate(CLASSES):
    count = (y_batch.argmax(axis=1) == i).sum()
    print(f"  {cls:25s}: {count}")

# ==============================================================================
# TEST 4: Verificar modelo
# ==============================================================================
print("\n" + "="*60)
print("TEST 4: Verificar modelo")
print("="*60)

print("Creando modelo...")
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(len(CLASSES), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo compilado")
print(f"✅ Loss: categorical_crossentropy")
print(f"✅ Metrics: accuracy")

# ==============================================================================
# TEST 5: Predicción de prueba
# ==============================================================================
print("\n" + "="*60)
print("TEST 5: Predicción de prueba (1 batch)")
print("="*60)

preds = model.predict(X_batch, verbose=0)

print(f"\n✅ Predicciones shape: {preds.shape}")
print(f"✅ Suma de predicciones por sample: {preds.sum(axis=1)}")

if not np.allclose(preds.sum(axis=1), 1):
    print(f"❌ PROBLEMA: Predicciones no suman a 1")
else:
    print(f"✅ Predicciones válidas (suman a 1)")

print(f"\nConfianza media: {preds.max(axis=1).mean():.4f}")

# Comparar predicción vs label
pred_labels = preds.argmax(axis=1)
true_labels = y_batch.argmax(axis=1)
accuracy = (pred_labels == true_labels).mean()

print(f"\n✅ Accuracy en batch (antes de entrenar): {accuracy:.4f}")
print(f"   Esperado (random): ~{1/len(CLASSES):.4f} ({len(CLASSES)} clases)")

if accuracy < 0.2:
    print(f"✅ OK: Accuracy baja es normal antes de entrenar")

# ==============================================================================
# TEST 6: Un paso de entrenamiento
# ==============================================================================
print("\n" + "="*60)
print("TEST 6: Un paso de entrenamiento (1 batch)")
print("="*60)

history = model.fit(X_batch, y_batch, batch_size=8, epochs=1, verbose=0)

print(f"\n✅ Loss después de 1 batch: {history.history['loss'][0]:.4f}")
print(f"✅ Accuracy después de 1 batch: {history.history['accuracy'][0]:.4f}")

if history.history['loss'][0] < 2.5:
    print(f"✅ Loss está bajando correctamente")
else:
    print(f"⚠️ Loss muy alto")

# ==============================================================================
# RESUMEN
# ==============================================================================
print("\n" + "="*60)
print("✅ RESUMEN")
print("="*60)
print("""
Si todo está en verde (✅), el problema NO está en:
  - Data generator
  - Labels
  - Modelo
  - Pérdida
  
Entonces el problema probablemente está en:
  - Class weights muy desbalanceados
  - Learning rate muy bajo
  - Hyperparameters del modelo
  - Orden de las augmentations

Si hay algún ❌, revisa ese test específico.
""")