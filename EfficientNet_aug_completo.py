import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB3
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
    graficar_accuracy_por_clase,
    graficar_distribucion_clases
)

warnings.filterwarnings('ignore')

# ==============================================================================
# GPU SETUP
# ==============================================================================
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        print(f"PU disponible: {len(physical_devices)}")

    except:
        pass

# ==============================================================================
# MIXED PRECISION
# ==============================================================================
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

print(f"Compute dtype: {policy.compute_dtype}")
print(f"Variable dtype: {policy.variable_dtype}")

# ==============================================================================
# CONFIG
# ==============================================================================
class Config:

    TRAIN_PATH = "/remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Training"
    TEST_PATH = "/remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Testing"
    
    print(f"TRAIN_PATH: {TRAIN_PATH}")
    print(f"TEST_PATH: {TEST_PATH}")

    OUTPUT_PATH = "./Estudios"

    # Imagen
    IMG_SIZE = (300, 300)

    # Batch
    BATCH_SIZE = 32

    # Undersampling SOLO EN MEMORIA
    ENABLE_UNDERSAMPLING = True
    SAFE_DRIVE_TARGET = 15000
    PHONECALL_TARGET = 15000

    # Entrenamiento
    EPOCHS_PHASE1 = 20
    EPOCHS_PHASE2 = 10

    LEARNING_RATE_PHASE1 = 1e-4
    LEARNING_RATE_PHASE2 = 1e-5

    # Clases
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

cfg = Config()

# ==============================================================================
# OUTPUT DIRS
# ==============================================================================
for folder in [
    "Entrenamiento",
    "Graficas",
    "Matriz confusion",
    "Evaluacion",
    "Modelo"
]:
    os.makedirs(os.path.join(cfg.OUTPUT_PATH, folder), exist_ok=True)

fecha_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_archivo = f"EfficientNetB3_augm_new_{fecha_actual}"

# ==============================================================================
# DISTRIBUCIÓN
# ==============================================================================
def get_class_distribution(train_path, class_names):

    distribution = {}

    for cls in class_names:

        cls_path = os.path.join(train_path, cls)

        if os.path.exists(cls_path):

            distribution[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        else:
            distribution[cls] = 0

    return distribution

# ==============================================================================
# AUGMENTATIONS
# ==============================================================================
class DriverAugmentations:

    @staticmethod
    def get_majority_augmentation():

        return A.Compose([
            A.HorizontalFlip(p=0.3),

            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),

            A.RandomGamma(
                gamma_limit=(90, 110),
                p=0.2
            ),
        ])

    @staticmethod
    def get_medium_augmentation():

        return A.Compose([

            A.HorizontalFlip(p=0.5),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),

            A.GaussNoise(
                var_limit=(10.0, 30.0),
                p=0.3
            ),

            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),
        ])

    @staticmethod
    def get_minority_augmentation():

        return A.Compose([

            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.7
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6
            ),

            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.4
            ),

            A.CoarseDropout(
                max_holes=8,
                max_height=50,
                max_width=50,
                p=0.3
            ),

            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),

            A.Perspective(
                scale=(0.05, 0.1),
                p=0.3
            ),
        ])

    @staticmethod
    def get_test_augmentation():

        return A.Compose([
            A.Resize(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]),
        ])

    @staticmethod
    def get_strategy_for_class(class_name):

        if class_name in ['safe_drive', 'phonecall']:
            return 'majority'

        elif class_name in [
            'hair_and_makeup',
            'texting',
            'drinking',
            'reach_side'
        ]:
            return 'medium'

        else:
            return 'minority'

augmentations = DriverAugmentations()

augmentation_dict = {

    'majority': augmentations.get_majority_augmentation(),
    'medium': augmentations.get_medium_augmentation(),
    'minority': augmentations.get_minority_augmentation(),
    'test': augmentations.get_test_augmentation()
}

# ==============================================================================
# DATA GENERATOR
# ==============================================================================
class BalancedDataGenerator(keras.utils.Sequence):

    def __init__(
        self,
        data_dir,
        batch_size=cfg.BATCH_SIZE,
        target_size=cfg.IMG_SIZE,
        is_training=True
    ):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.is_training = is_training

        self.classes = cfg.CLASSES

        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.classes)
        }

        self.samples = []
        self.class_counts = {}

        print(f"\nCargando dataset: {data_dir}")

        for cls in self.classes:

            cls_path = os.path.join(data_dir, cls)

            if not os.path.exists(cls_path):

                self.class_counts[cls] = 0
                continue

            imgs = [
                f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            # ==========================================================
            # UNDERSAMPLING SOLO EN MEMORIA
            # ==========================================================
            if cfg.ENABLE_UNDERSAMPLING:

                if cls == 'safe_drive' and len(imgs) > cfg.SAFE_DRIVE_TARGET:

                    random.seed(42)
                    imgs = random.sample(imgs, cfg.SAFE_DRIVE_TARGET)

                if cls == 'phonecall' and len(imgs) > cfg.PHONECALL_TARGET:

                    random.seed(42)
                    imgs = random.sample(imgs, cfg.PHONECALL_TARGET)

            self.class_counts[cls] = len(imgs)

            for img in imgs:

                self.samples.append(
                    (os.path.join(cls_path, img), cls)
                )

        print("\nDistribución:")

        for cls in self.classes:

            print(f"  {cls:25s}: {self.class_counts[cls]:6d}")

        self.balanced_samples = self.samples

        print(f"\nDataset listo: {len(self.balanced_samples):,}")

        self.on_epoch_end()

    def __len__(self):

        return max(
            1,
            int(np.ceil(len(self.balanced_samples) / self.batch_size))
        )

    def __getitem__(self, idx):

        start_idx = idx * self.batch_size
        end_idx = min(
            start_idx + self.batch_size,
            len(self.balanced_samples)
        )

        batch_samples = self.balanced_samples[start_idx:end_idx]

        if len(batch_samples) == 0:

            batch_samples = self.balanced_samples[:self.batch_size]

        X = np.empty(
            (len(batch_samples), *self.target_size, 3),
            dtype=np.float32
        )

        y = np.zeros(
            (len(batch_samples), len(self.classes)),
            dtype=np.float32
        )

        for i, (img_path, cls) in enumerate(batch_samples):

            try:

                img = cv2.imread(img_path)

                if img is None:
                    raise ValueError(f"No se pudo leer: {img_path}")

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = cv2.resize(
                    img,
                    self.target_size,
                    interpolation=cv2.INTER_LANCZOS4
                )

                if self.is_training:

                    strategy = augmentations.get_strategy_for_class(cls)

                    augmented = augmentation_dict[strategy](image=img)

                    img = augmented['image']

                else:

                    augmented = augmentation_dict['test'](image=img)

                    img = augmented['image']

                X[i] = img.astype(np.float32) / 255.0

                y[i, self.class_to_idx[cls]] = 1

            except Exception as e:

                X[i] = np.ones(
                    (*self.target_size, 3),
                    dtype=np.float32
                ) * 0.5

                y[i, self.class_to_idx[cls]] = 1

        return X, y

    def on_epoch_end(self):

        if self.is_training:
            np.random.shuffle(self.balanced_samples)

# ==============================================================================
# MODELO
# ==============================================================================
def create_efficientnet_model(num_classes):

    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(*cfg.IMG_SIZE, 3)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(*cfg.IMG_SIZE, 3))

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu')(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        dtype='float32'
    )(x)

    model = models.Model(inputs, outputs)

    return model, base_model

# ==============================================================================
# TRAIN
# ==============================================================================
def train_model(train_gen, val_gen, class_weights):

    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("="*60)

    num_classes = len(cfg.CLASSES)

    model, base_model = create_efficientnet_model(num_classes)

    # ==========================================================
    # FASE 1
    # ==========================================================
    print("\nFASE 1")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=cfg.LEARNING_RATE_PHASE1
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    t0 = time.time()

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.EPOCHS_PHASE1,
        class_weight=class_weights,
        callbacks=[

            callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),

            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                verbose=1
            )
        ],
        verbose=1
    )

    tiempo_fase1 = time.time() - t0

    # ==========================================================
    # FASE 2
    # ==========================================================
    print("\nFASE 2 - Fine Tuning")

    base_model.trainable = True

    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=cfg.LEARNING_RATE_PHASE2
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    t1 = time.time()

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.EPOCHS_PHASE2,
        class_weight=class_weights,
        callbacks=[

            callbacks.EarlyStopping(
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),

            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=2,
                verbose=1
            )
        ],
        verbose=1
    )

    tiempo_fase2 = time.time() - t1

    return model, history1, history2, tiempo_fase1, tiempo_fase2

# ==============================================================================
# MAIN
# ==============================================================================
def main():

    print("\n" + "="*60)
    print("EfficientNetB3 DRIVER ACTIONS")
    print("="*60)

    before_distribution = get_class_distribution(
        cfg.TRAIN_PATH,
        cfg.CLASSES
    )

    print("\nDISTRIBUCIÓN ORIGINAL")

    for cls, count in before_distribution.items():

        print(f"{cls:25s}: {count}")

    # ==========================================================
    # GENERATORS
    # ==========================================================
    print("\nCreando generators...")

    train_gen = BalancedDataGenerator(
        cfg.TRAIN_PATH,
        is_training=True
    )

    if not os.path.exists(cfg.TEST_PATH):

        raise Exception(
            f"\nERROR: No existe TEST_PATH\n{cfg.TEST_PATH}"
        )

    val_gen = BalancedDataGenerator(
        cfg.TEST_PATH,
        is_training=False
    )

    # ==========================================================
    # CLASS WEIGHTS
    # ==========================================================
    labels = [
        cfg.CLASSES.index(cls)
        for _, cls in train_gen.balanced_samples
    ]

    class_weights = compute_class_weight(
        'balanced',
        classes=np.array(range(len(cfg.CLASSES))),
        y=labels
    )

    class_weight_dict = {
        i: float(w)
        for i, w in enumerate(class_weights)
    }

    print("\nCLASS WEIGHTS")

    for i, cls in enumerate(cfg.CLASSES):

        count = sum(1 for label in labels if label == i)

        print(
            f"{cls:25s} "
            f"weight={class_weight_dict[i]:.4f} "
            f"samples={count}"
        )

    # ==========================================================
    # TRAIN
    # ==========================================================
    model, history1, history2, t1, t2 = train_model(
        train_gen,
        val_gen,
        class_weight_dict
    )

    # ==========================================================
    # SAVE MODEL
    # ==========================================================
    model_path = os.path.join(
        cfg.OUTPUT_PATH,
        "Modelo",
        f"modelo_{nombre_archivo}.h5"
    )

    model.save(model_path)

    print(f"\nModelo guardado en:")
    print(model_path)

# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    main()