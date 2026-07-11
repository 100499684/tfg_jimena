import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import json
import random
import warnings

from Codigo_graficas.graficas import (
    graficar_accuracy,
    graficar_loss,
    graficar_accuracy_por_clase
)

warnings.filterwarnings('ignore')

# ==============================================================================
# GPU SETUP
# ==============================================================================
physical_devices = tf.config.list_physical_devices('GPU')
AUTOTUNE = tf.data.AUTOTUNE

if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        print(f"GPU disponible: {len(physical_devices)}")

    except:
        pass

# ==============================================================================
# MIXED PRECISION
# ==============================================================================
tf.keras.mixed_precision.set_global_policy('float32')

# ==============================================================================
# CONFIG
# ==============================================================================
class Config:

    TRAIN_PATH = "./Training"
    TEST_PATH = "./Testing"

    OUTPUT_PATH = "./Estudios"

    IMG_SIZE = (300, 300)
    BATCH_SIZE = 32

    ENABLE_UNDERSAMPLING = True
    SAFE_DRIVE_TARGET = 15000
    PHONECALL_TARGET = 15000

    EPOCHS_PHASE1 = 20
    EPOCHS_PHASE2 = 10

    LEARNING_RATE_PHASE1 = 1e-4
    LEARNING_RATE_PHASE2 = 1e-6

    CLASSES = [
        'safe_drive',
        'phonecall',
        'hair_and_makeup',
        'texting',
        'drinking',
        'reach_side',
        'radio',
        'talking_to_passenger',
        'reach_backseat'
    ]

nombre_archivo = "modelo_aug_final"
cfg = Config()

# ==============================================================================
# AUGMENTATIONS
# ==============================================================================
class DriverAugmentations:

    @staticmethod
    def get_medium_augmentation():
        return A.Compose([
            #A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.4),
            A.ShiftScaleRotate(0.05, 0.1, 10, p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.RandomGamma((80, 120), p=0.3),
        ])

augmentations = DriverAugmentations()

augmentation_dict = {
    'medium': augmentations.get_medium_augmentation()
}

# ==============================================================================
# DATA PIPELINE
# ==============================================================================
class TFDataPipeline:

    def __init__(self, data_dir, is_training=True):
        self.data_dir = data_dir
        self.is_training = is_training
        self.classes = cfg.CLASSES
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        for cls in self.classes:
            cls_path = os.path.join(self.data_dir, cls)
            if not os.path.exists(cls_path):
                continue

            imgs = [
                os.path.join(cls_path, f)
                for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            for img in imgs:
                samples.append((img, cfg.CLASSES.index(cls)))

        random.shuffle(samples)
        return samples

    def get_dataset(self):
        paths = [s[0] for s in self.samples]
        labels = [s[1] for s in self.samples]

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        if self.is_training:
            ds = ds.shuffle(10000)

        ds = ds.map(self._load_image, num_parallel_calls=AUTOTUNE)

        if self.is_training:
            ds = ds.map(self._augment, num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda img, label: (preprocess_input(img), label),
                        num_parallel_calls=AUTOTUNE)

        ds = ds.batch(cfg.BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    def _load_image(self, path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])

        img = tf.image.resize(img, cfg.IMG_SIZE)
        img = tf.cast(img, tf.float32)

        label = tf.one_hot(label, len(cfg.CLASSES))
        return img, label

    def _albumentations_aug(self, img):
        img = img.astype(np.uint8)

        aug = augmentation_dict['medium']
        img = aug(image=img)['image']

        img = img.astype(np.float32)
        return img

    def _augment(self, img, label):
        img = tf.numpy_function(
            self._albumentations_aug,
            [img],
            tf.float32
        )

        img.set_shape((cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))

        # Si usas EfficientNet de tf.keras, normalmente puedes incluso omitir esto,
        # porque EfficientNet ya incluye preprocessing interno.
        img = preprocess_input(img)

        return img, label

# ==============================================================================
# MODEL
# ==============================================================================
def create_efficientnet_model(num_classes):

    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3)
    )

    # Fase 1: congelar EfficientNet completo
    base_model.trainable = False

    inputs = layers.Input(shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs)

    return model, base_model

# ==============================================================================
# TRAIN
# ==============================================================================
def train_model(train_gen, val_gen, class_weights):

    model, base_model = create_efficientnet_model(len(cfg.CLASSES))

    model.compile(
        optimizer=keras.optimizers.Adam(cfg.LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.EPOCHS_PHASE1,
        class_weight=class_weights
    )

    # =========================
    # FINE TUNING CORRECTO
    # =========================
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(cfg.LEARNING_RATE_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.EPOCHS_PHASE2,
        class_weight=class_weights
    )

    return model, history1, history2

# ==============================================================================
# MAIN
# ==============================================================================
def main():

    train_pipe = TFDataPipeline(cfg.TRAIN_PATH, True)
    val_pipe = TFDataPipeline(cfg.TEST_PATH, False)

    train_gen = train_pipe.get_dataset()
    val_gen = val_pipe.get_dataset()

    labels = [s[1] for s in train_pipe.samples]

    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(cfg.CLASSES)),
        y=labels
    )

    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    model, h1, h2 = train_model(train_gen, val_gen, class_weight_dict)
    
    model.save(f"./Estudios/Modelos/{nombre_archivo}.h5")

    # ==============================================================================
    # GRÁFICAS DE ENTRENAMIENTO
    # ==============================================================================
    print("\n" + "="*60)
    print("📊 GENERANDO GRÁFICAS DE ENTRENAMIENTO")
    print("="*60)

    os.makedirs(os.path.join(cfg.OUTPUT_PATH, "Graficas"), exist_ok=True)

    # FASE 1
    graficar_accuracy(
        h1,
        titulo="Accuracy - Fase 1",
        guardar=True,
        ruta_guardado=os.path.join(
            cfg.OUTPUT_PATH,
            "Graficas",
            f"acc_fase1_{nombre_archivo}.png"
        )
    )

    graficar_loss(
        h1,
        titulo="Loss - Fase 1",
        guardar=True,
        ruta_guardado=os.path.join(
            cfg.OUTPUT_PATH,
            "Graficas",
            f"loss_fase1_{nombre_archivo}.png"
        )
    )

    # FASE 2
    graficar_accuracy(
        h2,
        titulo="Accuracy - Fine-Tuning",
        guardar=True,
        ruta_guardado=os.path.join(
            cfg.OUTPUT_PATH,
            "Graficas",
            f"acc_fase2_{nombre_archivo}.png"
        )
    )

    graficar_loss(
        h2,
        titulo="Loss - Fine-Tuning",
        guardar=True,
        ruta_guardado=os.path.join(
            cfg.OUTPUT_PATH,
            "Graficas",
            f"loss_fase2_{nombre_archivo}.png"
        )
    )

    print("\n✅ Todo completado.")

# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    main()