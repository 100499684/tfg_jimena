import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout
)
from tensorflow.keras.models import Model

NUM_CLASSES = 12
IMG_SIZE = 300

print("Construyendo modelo...")

base_model = EfficientNetB3(
    weights=None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax",
    dtype="float32"
)(x)

model = Model(base_model.input, outputs)

print("Cargando pesos...")

model.load_weights(
    "./Estudios/Modelo/mejor_modelo_efficientnetb3_principal_train_ft.keras"
)

print("✅ CARGA CORRECTA")