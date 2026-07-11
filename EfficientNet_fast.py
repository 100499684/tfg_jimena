import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from datetime import datetime
import json

# Config (ajusta rutas si es necesario)
class Config:
    TRAIN_PATH = "./Training"
    VAL_PATH = "./Testing"
    OUTPUT_PATH = "./Estudios/Modelo"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    CLASSES = ['safe_drive', 'phonecall', 'hair_and_makeup', 'texting',
               'drinking', 'reach_side', 'radio', 'talking_to_passenger',
               'reach_backseat']

cfg = Config()

# Mixed precision if available
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

def list_images_and_labels(base_path, classes):
    filepaths = []
    labels = []
    for i, cls in enumerate(classes):
        cls_path = os.path.join(base_path, cls)
        if not os.path.exists(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(cls_path, fname))
                labels.append(i)
    return np.array(filepaths), np.array(labels)

def decode_and_preprocess(filename, label, img_size, is_training):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)

    if is_training:
        # lightweight augmentations that run on GPU
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        # small random rotations (if tfa available)
        angles = tf.random.uniform([], -0.05, 0.05)
        img = tfa.image.rotate(img, angles) if 'tfa' in globals() else img

    return img, label

def make_dataset(filepaths, labels, batch_size, img_size, is_training):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=len(filepaths))
    ds = ds.map(lambda f, l: decode_and_preprocess(f, l, img_size, is_training), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def create_model(num_classes, input_shape):
    try:
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input
    except Exception:
        # fallback to MobileNetV2 if EfficientNet not available
        from tensorflow.keras.applications import MobileNetV2 as EfficientNetB0
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    inputs = layers.Input(shape=(*input_shape, 3))
    x = preprocess_input(inputs)
    base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = models.Model(inputs, outputs)
    return model

def compute_class_weights(labels, num_classes):
    classes = np.arange(num_classes)
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return {i: float(w) for i, w in enumerate(weights)}

def main():
    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)
    print("Listing training files...")
    train_files, train_labels = list_images_and_labels(cfg.TRAIN_PATH, cfg.CLASSES)
    val_files, val_labels = list_images_and_labels(cfg.VAL_PATH, cfg.CLASSES)

    if len(train_files) == 0:
        print(f"No training images found under {cfg.TRAIN_PATH}")
        return

    print(f"Train samples: {len(train_files)}  Val samples: {len(val_files)}")

    # build datasets
    train_ds = make_dataset(train_files, train_labels, cfg.BATCH_SIZE, cfg.IMG_SIZE, is_training=True)
    val_ds = make_dataset(val_files, val_labels, cfg.BATCH_SIZE, cfg.IMG_SIZE, is_training=False)

    class_weights = compute_class_weights(train_labels, len(cfg.CLASSES))
    print("Class weights:")
    for i, cls in enumerate(cfg.CLASSES):
        print(f"  {cls:25s}: {class_weights.get(i,1.0):.3f}")

    model = create_model(len(cfg.CLASSES), cfg.IMG_SIZE)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.LR),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = callbacks.ModelCheckpoint(os.path.join(cfg.OUTPUT_PATH, 'best_model.h5'), save_best_only=True, monitor='val_accuracy')
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.EPOCHS,
                        class_weight=class_weights,
                        callbacks=[checkpoint, rlrop, es], verbose=1)

    # save final model and history
    model.save(os.path.join(cfg.OUTPUT_PATH, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'))

    with open(os.path.join(cfg.OUTPUT_PATH, 'history.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    print('Training complete. Models and history saved in', cfg.OUTPUT_PATH)

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    try:
        import tensorflow_addons as tfa
        globals()['tfa'] = tfa
    except Exception:
        pass
    main()
