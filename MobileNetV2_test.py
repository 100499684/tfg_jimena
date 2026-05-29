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
import json
from Codigo_graficas.graficas import graficar_confianza_vs_acierto, graficar_accuracy_por_clase
from Codigo_graficas.m_confusion import generar_matriz_confusion

# CONFIGURACIÓN
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

ruta_train  = "./Training"
ruta_test   = "./Testing"
ruta_output = "./Estudios"
nombre_archivo = "modelo_mobilenetv2_combinacion2"
f_name = f"MobileNetV2_{datetime.date.today()}_combinacion2.txt"

# MobileNetV2 usa 224x224 normalmente (más pequeño que EfficientNet)
IMG_SIZE    = 224        # MobileNetV2 está optimizado para 224x224
BATCH_SIZE  = 128        # Puede ser más grande porque MobileNet es más ligero
NUM_CLASSES = 11
EPOCHS      = 20

print("Configuración lista")







# 1. CARGAR DATOS DE TEST
AUTOTUNE = tf.data.AUTOTUNE
test_datagen = tf.keras.utils.image_dataset_from_directory(
    ruta_test,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    labels='inferred'
)

class_names = test_datagen.class_names
print(f"\n📋 Clases encontradas: {class_names}")

test_datagen = (
    test_datagen
    .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)










# 2. CARGAR MODELO
ruta_mejor_modelo = os.path.join(ruta_output, "Modelo", "mejor_modelo_mobilenetv2_combinacion2_ft.keras")
model = load_model(ruta_mejor_modelo)
print(f"\n✅ Modelo cargado desde: {ruta_mejor_modelo}")







# 3. EVALUACIÓN GLOBAL
loss, acc = model.evaluate(test_datagen, verbose=1)
print(f"\n🎯 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"📉 Test Loss: {loss:.4f}")








# 4. MATRIZ DE CONFUSIÓN
test_datagen_fresh = tf.keras.utils.image_dataset_from_directory(
    ruta_test,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    labels='inferred'
)
test_datagen_fresh = (
    test_datagen_fresh
    .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

n = nombre_archivo + "_confusion_matrix_test.png"
resultados_matriz = generar_matriz_confusion(
    model=model,
    test_datagen=test_datagen_fresh,
    class_names=class_names,
    guardar=True,
    ruta_guardado=os.path.join(ruta_output, "Matriz confusion", n),
    mostrar_reporte=True
)

y_true = resultados_matriz['y_true']
y_pred = resultados_matriz['y_pred']
cm = resultados_matriz['confusion_matrix']
acc = resultados_matriz['accuracy']








# 5. Métricas por clase
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, cls in enumerate(class_names):
    print(f"{cls:20s}: {class_accuracy[i]:.4f} ({class_accuracy[i]*100:.2f}%)")



# 6. Confianza de predicciones
y_pred_probs = model.predict(test_datagen_fresh, verbose=0)
confianzas = np.max(y_pred_probs, axis=1)

print(f"\n📊 Estadísticas de confianza:")
print(f"  - Media: {np.mean(confianzas):.4f}")
print(f"  - Mediana: {np.median(confianzas):.4f}")
print(f"  - Mínima: {np.min(confianzas):.4f}")
print(f"  - Máxima: {np.max(confianzas):.4f}")
print(f"  - Desviación: {np.std(confianzas):.4f}")







# 7. Guardar resultados completos
results_df = pd.DataFrame({
    'true_label': y_true,
    'true_class': [class_names[i] for i in y_true],
    'pred_label': y_pred,
    'pred_class': [class_names[i] for i in y_pred],
    'confidence': confianzas,
    'correct': y_true == y_pred
})
n = nombre_archivo + "_test_predictions.csv"
results_df.to_csv(os.path.join(ruta_output, "Entrenamiento", n), index=False)

metrics = {
    'test_accuracy': float(acc),
    'test_loss': float(loss),
    'total_samples': len(y_true),
    'correct_predictions': int(np.sum(y_true == y_pred)),
    'mean_confidence': float(np.mean(confianzas)),
    'num_classes': NUM_CLASSES
}
n = nombre_archivo + "_test_metrics.json"
with open(os.path.join(ruta_output, "Metricas", n), 'w') as f:
    json.dump(metrics, f, indent=4)







# 8. Gráficas Test
n = nombre_archivo + "_accuracy_por_clase_test.png"
graficar_accuracy_por_clase(
    cm=cm,
    class_names=class_names,
    titulo="Accuracy por clase - Test",
    guardar=True,
    ruta_guardado=os.path.join(ruta_output, "Graficas", n)
)

n = nombre_archivo + "_confianza_vs_acierto_test.png"
graficar_confianza_vs_acierto(
    confianzas=confianzas,
    y_true=y_true,
    y_pred=y_pred,
    titulo="Confianza vs acierto - Test",
    guardar=True,
    ruta_guardado=os.path.join(ruta_output, "Graficas", n)
)




# 9. Resumen final
print("\n" + "="*60)
print("RESUMEN FINAL DEL MODELO")
print("="*60)
print(f"🎯 Accuracy en test: {acc:.4f} ({acc*100:.2f}%)")
print(f"✅ Predicciones correctas: {np.sum(y_true == y_pred)}/{len(y_true)}")
print(f"❌ Predicciones incorrectas: {np.sum(y_true != y_pred)}/{len(y_true)}")
print(f"🎲 Confianza promedio: {np.mean(confianzas):.4f}")
print("="*60)