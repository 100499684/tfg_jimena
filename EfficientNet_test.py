#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación del mejor modelo EfficientNetB3.
Script robusto que carga el modelo (con fallback para capas Lambda / quantization)
y evalúa el dataset de `./Testing`.
"""
import os
# Respect a restart marker so we can restart CPU-only without being overridden below
if os.environ.get("DISABLE_GPU_ON_RESTART") == "1":
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Usamos keras.layers.Dense para definir PatchedDense compatible
from keras.layers import Dense as KerasDense

# Módulos propios
from Codigo_graficas.graficas import (
	graficar_accuracy_por_clase,
	graficar_confianza_vs_acierto
)
from Codigo_graficas.m_confusion import generar_matriz_confusion

tf.keras.backend.clear_session()

def restart_without_gpu():
	print('\n⚠️  Reiniciando proceso en CPU-only para evitar OOM...')
	# set marker so the restarted process disables GPU
	os.environ["DISABLE_GPU_ON_RESTART"] = "1"
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	# ensure TF uses CPU-only
	os.execv(sys.executable, [sys.executable] + sys.argv)


# -----------------------------
# Configuración (ajusta si hace falta)
# -----------------------------
ruta_test = '/remote-repositorio/afrodita/repo-fast/tfg_jcabrera/Testing'
ruta_output = './Estudios'
nombre_modelo_ft = 'mejor_modelo_efficientnetb3_principal_train_ft.keras'
ruta_modelo_ft = os.path.join(ruta_output, 'Modelo', nombre_modelo_ft)

# Alias utilizado en fragmentos de evaluación/otros notebooks
ruta_mejor_modelo_ft = ruta_modelo_ft

IMG_SIZE = 300
# Empieza con un batch razonable; si hay OOM se reducirá automáticamente
BATCH_SIZE = 32

fecha_hoy = datetime.date.today()
f_eval = f"Evaluacion_EfficientNetB3_Principal_FT_{fecha_hoy}.txt"


# -----------------------------
# GPU
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
print('GPUs disponibles:', gpus)
for gpu in gpus:
	try:
		tf.config.experimental.set_memory_growth(gpu, True)
	except Exception:
		pass


# -----------------------------
# Dataset de test
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
	ruta_test,
	image_size=(IMG_SIZE, IMG_SIZE),
	batch_size=BATCH_SIZE,
	shuffle=False,
	color_mode='rgb'
)

class_names = test_ds_raw.class_names
print(f"\nClases detectadas ({len(class_names)}): {class_names}")

test_ds = test_ds_raw.map(
	lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
	num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)


# -----------------------------
# Carga segura del modelo
# -----------------------------
class PatchedDense(KerasDense):
	def __init__(self, *args, **kwargs):
		kwargs.pop('quantization_config', None)
		super().__init__(*args, **kwargs)

def load_model_safe(path):
	print(f"\nCargando modelo: {path}")
	if not os.path.exists(path):
		raise FileNotFoundError(f"Modelo no encontrado: {path}")

	try:
		return tf.keras.models.load_model(path)
	except Exception as e:
		print('⚠️  Error cargando modelo (intento fallback):', e)
		# Intentar cargar forzando asignación en CPU (evita OOM en GPU)
		try:
			print('Intentando cargar en CPU (/CPU:0) como fallback...')
			with tf.device('/CPU:0'):
				return tf.keras.models.load_model(path)
		except Exception as e_cpu:
			print('⚠️  Error al cargar en CPU:', e_cpu)
		# Intentar con custom_objects + safe_mode=False
		try:
			return tf.keras.models.load_model(
				path,
				custom_objects={'Dense': PatchedDense},
				safe_mode=False
			)
		except Exception as e2:
			print('⚠️  Error cargando con safe_mode=False:', e2)
			# Habilitar deserialización insegura globalmente y reintentar
			try:
				from keras import config as keras_config
				keras_config.enable_unsafe_deserialization()
				return tf.keras.models.load_model(
					path,
					custom_objects={'Dense': PatchedDense}
				)
			except Exception as e3:
				print('❌ Fallo definitivo al cargar modelo:', e3)
				# Último intento: usar la API de Keras independiente (si está instalada)
				try:
					import keras as standalone_keras
					print('Intentando cargar con keras.models.load_model (compile=False)...')
					return standalone_keras.models.load_model(
						path,
						custom_objects={'Dense': PatchedDense},
						compile=False
					)
				except Exception as e4:
					print('Fallo también con keras standalone:', e4)
					# Intentar reconstruir la arquitectura conocida (EfficientNetB3 + cabeza)
					try:
						print('Intentando reconstruir arquitectura EfficientNetB3 y cargar solo pesos...')
						num_classes_local = len(class_names) if 'class_names' in globals() else None
						if num_classes_local is None:
							raise RuntimeError('No se puede inferir num_classes para reconstruir la arquitectura')
						# Construir modelo igual que en entrenamiento
						base = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
						base.trainable = False
						from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
						x = base.output
						x = GlobalAveragePooling2D()(x)
						x = Dropout(0.3)(x)
						x = Dense(128, activation='relu')(x)
						x = Dropout(0.2)(x)
						pred = Dense(num_classes_local, activation='softmax', dtype='float32')(x)
						model_recon = tf.keras.models.Model(inputs=base.input, outputs=pred)
						# Cargar pesos desde el archivo (intentar en CPU)
						with tf.device('/CPU:0'):
							model_recon.load_weights(path)
						print('Pesos cargados en el modelo reconstruido (load_weights).')
						# Compilar antes de usar evaluate/predict
						model_recon.compile(
							optimizer=tf.keras.optimizers.Adam(),
							loss='sparse_categorical_crossentropy',
							metrics=['accuracy']
						)
						return model_recon
					except Exception as e_weights:
						print('Fallo al cargar pesos en modelo reconstruido:', e_weights)
						raise


try:
	model = load_model_safe(ruta_modelo_ft)
	print('✅ Modelo cargado correctamente')
except tf.errors.ResourceExhaustedError as e:
	print('❌ ResourceExhausted al cargar el modelo:', e)
	restart_without_gpu()
except RuntimeError as e:
	# Algunos OOM se propagan como RuntimeError con texto OOM/ResourceExhausted
	msg = str(e).lower()
	if 'resourceexhausted' in msg or 'oom' in msg or 'out of memory' in msg:
		print('❌ RuntimeError indicando OOM al cargar el modelo:', e)
		restart_without_gpu()
	else:
		raise


# -----------------------------
# Evaluación
# -----------------------------
print('\n' + '='*60)
print('📊 EVALUACIÓN GLOBAL')
print('='*60)

def evaluate_with_oom_retry(model, test_ds_raw, preprocess_fn, initial_batch):
	batch = initial_batch
	while batch >= 1:
		print(f"Intentando evaluar con batch_size={batch}...")
		try:
			test_ds_try = test_ds_raw.unbatch().batch(batch)
			test_ds_try = test_ds_try.map(lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
			loss, accuracy = model.evaluate(test_ds_try, verbose=1)
			return loss, accuracy, test_ds_try
		except tf.errors.ResourceExhaustedError as e:
			print(f"⚠️  ResourceExhausted with batch {batch}: {e}")
		except RuntimeError as e:
			if 'ResourceExhausted' in str(e) or 'OOM' in str(e) or 'out of memory' in str(e).lower():
				print(f"⚠️  Runtime OOM with batch {batch}: {e}")
			else:
				raise
		batch = max(1, batch // 2)
		print(f"Reduciendo batch a {batch} y reintentando...")
	raise RuntimeError("No se pudo evaluar: OOM incluso con batch=1")


loss, accuracy, test_ds = evaluate_with_oom_retry(model, test_ds_raw, preprocess_input, BATCH_SIZE)
print(f"\n  Loss     : {loss:.4f}")
print(f"  Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")


# -----------------------------
# Matriz de confusión y métricas
# -----------------------------
os.makedirs(os.path.join(ruta_output, 'Evaluacion'), exist_ok=True)
ruta_eval_graficas = os.path.join(ruta_output, 'Graficas')
os.makedirs(ruta_eval_graficas, exist_ok=True)
# Directorio para matriz de confusión y reportes
matriz_dir = os.path.join(ruta_output, 'Matriz confusion')
os.makedirs(matriz_dir, exist_ok=True)

resultados = generar_matriz_confusion(
	model=model,
	test_datagen=test_ds,
	class_names=class_names,
	guardar=True,
	ruta_guardado=os.path.join(ruta_output, 'Evaluacion'),
	mostrar_reporte=True
)

y_true = resultados['y_true']
y_pred = resultados['y_pred']
cm = resultados['confusion_matrix']
acc_eval = resultados['accuracy']
report = resultados['classification_report']


# -----------------------------
# Análisis simple de confianza
# -----------------------------
print('\n' + '='*60)
print('🎯 ANÁLISIS DE CONFIANZA DE PREDICCIONES')
print('='*60)

y_probs = model.predict(test_ds, verbose=1)
confianzas = np.max(y_probs, axis=1)

print('\n📊 Estadísticas de confianza:')
print(f"  - Media     : {confianzas.mean():.4f}")
print(f"  - Mediana   : {np.median(confianzas):.4f}")
print(f"  - Mínima    : {confianzas.min():.4f}")
print(f"  - Máxima    : {confianzas.max():.4f}")
print(f"  - Desviación: {confianzas.std():.4f}")


# -----------------------------
# Guardar resumen en TXT
# -----------------------------
ruta_txt = os.path.join(ruta_output, 'Evaluacion', f_eval)
with open(ruta_txt, 'w', encoding='utf-8') as f:
	f.write('=' * 60 + '\n')
	f.write('EVALUACIÓN — EfficientNetB3\n')
	f.write('=' * 60 + '\n\n')
	f.write(f'Fecha         : {fecha_hoy}\n')
	f.write(f'Modelo        : {nombre_modelo_ft}\n')
	f.write(f'Dataset test  : {ruta_test}\n')
	f.write(f'Img size      : {IMG_SIZE}x{IMG_SIZE}\n')
	f.write(f'Batch size    : {BATCH_SIZE}\n\n')
	f.write(f'Loss (Keras)  : {loss:.4f}\n')
	f.write(f'Accuracy      : {acc_eval:.4f} ({acc_eval*100:.2f}%)\n\n')
	f.write('CLASSIFICATION REPORT:\n')
	f.write('-' * 60 + '\n')
	f.write(report + '\n')

print('Resultados guardados en:', ruta_txt)

print('\n✅ Evaluación completada')
