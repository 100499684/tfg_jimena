import numpy as np                      # Para operaciones con arrays
import matplotlib.pyplot as plt         # Para crear gráficas
import seaborn as sns                   # Para el heatmap de la matriz
import pandas as pd                     # Para guardar CSV
import os                               # Para manejar rutas de archivos
from datetime import datetime           # Para nombres de archivo con fecha
from sklearn.metrics import classification_report, confusion_matrix




def generar_matriz_confusion(model, test_datagen, 
                            class_names=None, 
                            guardar=False, 
                            ruta_guardado=None,
                            mostrar_reporte=True):
    """
    Genera la matriz de confusión y el reporte de clasificación a partir de un modelo y datos de prueba.
    
    Parámetros:
    -----------
    model : keras.Model
        Modelo entrenado (preferiblemente el mejor modelo cargado)
    test_datagen : tf.data.Dataset
        Generador de datos de prueba (debe estar fresco, sin consumir)
    class_names : list, optional
        Lista con los nombres de las clases. Si es None, usa números
    guardar : bool, default=False
        Si es True, guarda los resultados en archivos
    ruta_guardado : str, optional
        Ruta donde guardar los archivos. Si es None, usa directorio por defecto
    mostrar_reporte : bool, default=True
        Si es True, imprime el classification_report en consola
    
    Retorna:
    --------
    dict : Diccionario con los siguientes elementos:
        - 'y_true': etiquetas reales
        - 'y_pred': etiquetas predichas
        - 'confusion_matrix': matriz de confusión
        - 'classification_report': reporte de clasificación
        - 'accuracy': precisión del modelo
    """
    
    print("📊 Generando matriz de confusión...")
    print("="*60)
    
    # Verificar que el test_datagen está fresco
    print("🔄 Preparando generador de prueba...")
    
    # Obtener etiquetas reales
    print("📥 Obteniendo etiquetas reales...")
    y_true = np.concatenate([y for x, y in test_datagen], axis=0)
    
    # Obtener predicciones
    print("🔮 Generando predicciones...")
    y_pred = model.predict(test_datagen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calcular precisión
    accuracy = np.mean(y_true == y_pred_classes)
    print(f"\n✅ Precisión del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generar reporte de clasificación
    if class_names:
        report = classification_report(y_true, y_pred_classes, target_names=class_names)
    else:
        report = classification_report(y_true, y_pred_classes)
    
    if mostrar_reporte:
        print("\n📋 CLASSIFICATION REPORT:")
        print("="*60)
        print(report)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Definir tamaño de anotaciones según el tamaño de la matriz
    if len(cm) <= 10:
        annot = True
        fmt = 'd'
        annot_kws = {'size': 10}
    else:
        annot = False
        fmt = 'd'
        annot_kws = None
    
    # Crear heatmap
    if class_names:
        sns.heatmap(cm, annot=annot, fmt=fmt, cmap="Blues", cbar=True,
                   xticklabels=class_names, yticklabels=class_names,
                   annot_kws=annot_kws, ax=ax)
    else:
        sns.heatmap(cm, annot=annot, fmt=fmt, cmap="Blues", cbar=True,
                   annot_kws=annot_kws, ax=ax)
    
    # Configurar título y etiquetas
    ax.set_title(f'Matriz de Confusión\nPrecisión: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicción', fontsize=14)
    ax.set_ylabel('Valor Real', fontsize=14)
    
    # Rotar etiquetas si son muchas
    if class_names and len(class_names) > 8:
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
    else:
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
    
    plt.tight_layout()
    
    # Guardar si es necesario
    if guardar:
        # Determinar ruta de guardado
        if ruta_guardado is None:
            # Crear directorio por defecto
            os.makedirs("./Matriz_Confusion", exist_ok=True)
            base_path = "./Matriz_Confusion"
        else:
            base_path = ruta_guardado
            os.makedirs(base_path, exist_ok=True)
        
        # Guardar PNG
        fecha = datetime.today().strftime('%Y%m%d_%H%M%S')
        ruta_png = os.path.join(base_path, f"matriz_confusion_{fecha}.png")
        plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
        print(f"\n✅ Matriz de confusión guardada como PNG: {ruta_png}")
        
        # Guardar PDF
        ruta_pdf = os.path.join(base_path, f"matriz_confusion_{fecha}.pdf")
        plt.savefig(ruta_pdf, bbox_inches='tight')
        print(f"✅ Matriz de confusión guardada como PDF: {ruta_pdf}")
        
        # Guardar matriz como CSV
        df_cm = pd.DataFrame(cm)
        if class_names:
            df_cm.index = class_names
            df_cm.columns = class_names
        ruta_csv = os.path.join(base_path, f"matriz_confusion_{fecha}.csv")
        df_cm.to_csv(ruta_csv)
        print(f"✅ Matriz de confusión guardada como CSV: {ruta_csv}")
        
        # Guardar classification report
        ruta_report = os.path.join(base_path, f"classification_report_{fecha}.txt")
        with open(ruta_report, 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n")
            f.write(report)
            f.write("\n\n" + "="*60 + "\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        print(f"✅ Classification report guardado: {ruta_report}")
    
    # Mostrar la figura
    plt.show()
    
    # Retornar resultados
    resultados = {
        'y_true': y_true,
        'y_pred': y_pred_classes,
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy,
        'figura': fig
    }
    
    return resultados