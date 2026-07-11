# TFG Jimena

Este repositorio contiene un proyecto de visión por computadora para la clasificación de acciones del conductor a partir de imágenes. El objetivo principal es entrenar y evaluar modelos de redes neuronales convolucionales, especialmente EfficientNet y MobileNetV2, para detectar situaciones como conducción segura, uso del móvil, texting, consumo de bebidas o alcanzar elementos en el asiento delantero o trasero.

## Qué incluye este proyecto

- Scripts de entrenamiento para [EfficientNet_aug_completo.py](EfficientNet_aug_completo.py) y [EfficientNet_train2.py](EfficientNet_train2.py)
- Scripts de evaluación para [EfficientNet_test2.py](EfficientNet_test2.py) y [MobileNetV2_test.py](MobileNetV2_test.py)
- Versiones con aumento de datos en [EfficientNet_train_aug.py](EfficientNet_train_aug.py) y [MobileNetV2_train_augmentation.py](MobileNetV2_train_augmentation.py)
- Carpeta de experimentos y métricas en [Estudios](Estudios)
- Herramientas auxiliares para gráficas y análisis en [Codigo_graficas](Codigo_graficas)

## Requisitos

Se recomienda usar Python 3.10+ y un entorno virtual.

Instala las dependencias principales con:

```bash
pip install numpy pandas opencv-python tensorflow matplotlib seaborn scikit-learn albumentations
```

Si vas a trabajar con GPU, asegúrate de tener TensorFlow compatible con tu instalación de CUDA/cuDNN.

## Estructura del repositorio

- [Training](Training): imágenes usadas para entrenamiento
- [Testing](Testing): imágenes usadas para evaluación
- [Estudios](Estudios): resultados, métricas, modelos y gráficas
- [Extras](Extras): utilidades auxiliares para preparación y análisis de datos
- [Codigo_graficas](Codigo_graficas): scripts para generar gráficas de accuracy, loss y matriz de confusión

## Inicio rápido

### 1. Preparar datos

Asegúrate de que las carpetas [Training](Training) y [Testing](Testing) contengan subcarpetas por clase, por ejemplo:

- safe_drive
- phonecall
- texting
- drinking
- reach_side
- reach_backseat

### 2. Entrenar un modelo

Para entrenar el modelo principal con EfficientNet:

```bash
python EfficientNet_aug_completo.py
```

También puedes probar otras variantes con:

```bash
python EfficientNet_train2.py
python MobileNetV2_train.py
```

### 3. Evaluar un modelo

```bash
python EfficientNet_test2.py
```

O para MobileNetV2:

```bash
python MobileNetV2_test.py
```

### 4. Ver resultados

Los resultados, gráficas y métricas se guardan en [Estudios](Estudios).

## Notas importantes

- Los archivos grandes de logs y salidas se han archivado en [outputs.zip](outputs.zip) y los ficheros `.out` se ignoran para evitar problemas con GitHub.
- Los pesos de modelos pesados y otros artefactos grandes pueden ocupar bastante espacio en disco; conviene mantenerlos en carpetas externas o en un almacenamiento dedicado si no se van a usar en cada ejecución.

## Scripts útiles

- [estudio.sh](estudio.sh): comprobación básica del entorno y del acceso a GPU
- [ejecutar.sh](ejecutar.sh): script de ejecución rápida para entornos de cluster o SLURM

## Estado del proyecto

Proyecto de fin de grado en desarrollo y experimentación. Está orientado a la investigación y comparación de arquitecturas CNN para clasificación visual de conductas de conducción.
