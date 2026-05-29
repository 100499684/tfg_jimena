import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


def graficar_accuracy(history, titulo=None, guardar=False, ruta_guardado=None):
    """
    Genera una gráfica de la evolución del Accuracy durante el entrenamiento.
    
    Parámetros:
    -----------
    history : History object
        Objeto devuelto por model.fit() que contiene el historial de entrenamiento
    titulo : str, optional
        Título personalizado para la gráfica. Si es None, usa título por defecto
    guardar : bool, default=False
        Si es True, guarda la gráfica como archivo
    ruta_guardado : str, optional
        Ruta donde guardar la gráfica. Si es None, usa directorio por defecto
    
    Retorna:
    --------
    fig : matplotlib.figure.Figure
        La figura generada
    """
    # Obtener datos del historial
    epochs = range(1, len(history.history['accuracy']) + 1)
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    # Encontrar la mejor época
    best_epoch = np.argmax(val_acc) + 1
    best_acc = max(val_acc)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Graficar líneas principales
    ax.plot(epochs, train_acc, 'b-o', label='Entrenamiento', linewidth=2, markersize=6, markevery=1)
    ax.plot(epochs, val_acc, 'r-o', label='Validación', linewidth=2, markersize=6, markevery=1)
    
    # Resaltar el mejor punto
    ax.scatter(best_epoch, best_acc, color='gold', s=200, zorder=5, 
               marker='*', edgecolors='black', linewidth=2,
               label=f'✨ Mejor modelo (época {best_epoch}, acc={best_acc:.4f})')
    ax.axvline(x=best_epoch, color='gold', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Línea de tendencia (polinomio de grado 2)
    if len(epochs) > 2:
        z = np.polyfit(epochs, val_acc, 2)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "g--", alpha=0.5, linewidth=1, label='Tendencia')
    
    # Configurar título y etiquetas
    if titulo:
        ax.set_title(titulo, fontsize=16, fontweight='bold')
    else:
        ax.set_title('Evolución del Accuracy durante el Entrenamiento', fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Época', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.05])
    ax.set_xlim([0, len(epochs) + 1])
    
    # Añadir valores en puntos clave
    for i, (acc_t, acc_v) in enumerate(zip(train_acc, val_acc)):
        if (i+1) % 5 == 0 or i+1 == best_epoch or i+1 == 1 or i+1 == len(epochs):
            ax.annotate(f'{acc_v:.3f}', xy=(i+1, acc_v), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Guardar si es necesario
    if guardar:
        if ruta_guardado is None:
            # Crear directorio por defecto
            os.makedirs("./Graficas", exist_ok=True)
            ruta_guardado = os.path.join("./Graficas", f"accuracy_{datetime.today().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfica de Accuracy guardada en: {ruta_guardado}")
    
    return fig





def graficar_loss(history, titulo=None, guardar=False, ruta_guardado=None):
    """
    Genera una gráfica de la evolución de la Pérdida (Loss) durante el entrenamiento.
    
    Parámetros:
    -----------
    history : History object
        Objeto devuelto por model.fit() que contiene el historial de entrenamiento
    titulo : str, optional
        Título personalizado para la gráfica. Si es None, usa título por defecto
    guardar : bool, default=False
        Si es True, guarda la gráfica como archivo
    ruta_guardado : str, optional
        Ruta donde guardar la gráfica. Si es None, usa directorio por defecto
    
    Retorna:
    --------
    fig : matplotlib.figure.Figure
        La figura generada
    """
    # Obtener datos del historial
    epochs = range(1, len(history.history['loss']) + 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Encontrar la mejor época (menor pérdida)
    best_epoch = np.argmin(val_loss) + 1
    best_loss = min(val_loss)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Graficar líneas principales
    ax.plot(epochs, train_loss, 'b-o', label='Entrenamiento', linewidth=2, markersize=6, markevery=1)
    ax.plot(epochs, val_loss, 'r-o', label='Validación', linewidth=2, markersize=6, markevery=1)
    
    # Resaltar el mejor punto
    ax.scatter(best_epoch, best_loss, color='gold', s=200, zorder=5,
               marker='*', edgecolors='black', linewidth=2,
               label=f'✨ Mejor pérdida (época {best_epoch}, loss={best_loss:.4f})')
    ax.axvline(x=best_epoch, color='gold', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Línea de tendencia (polinomio de grado 2)
    if len(epochs) > 2:
        z = np.polyfit(epochs, val_loss, 2)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "g--", alpha=0.5, linewidth=1, label='Tendencia')
    
    # Configurar título y etiquetas
    if titulo:
        ax.set_title(titulo, fontsize=16, fontweight='bold')
    else:
        ax.set_title('Evolución de la Pérdida (Loss) durante el Entrenamiento', fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Época', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir valores en puntos clave
    for i, (loss_t, loss_v) in enumerate(zip(train_loss, val_loss)):
        if (i+1) % 5 == 0 or i+1 == best_epoch or i+1 == 1 or i+1 == len(epochs):
            ax.annotate(f'{loss_v:.3f}', xy=(i+1, loss_v), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Guardar si es necesario
    if guardar:
        if ruta_guardado is None:
            # Crear directorio por defecto
            os.makedirs("./Graficas", exist_ok=True)
            ruta_guardado = os.path.join("./Graficas", f"loss_{datetime.today().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfica de Loss guardada en: {ruta_guardado}")
    
    return fig




def graficar_accuracy_por_clase(cm, class_names, titulo="Accuracy por clase", guardar=False, ruta_guardado=None):
    """
    Grafica el accuracy por clase a partir de una matriz de confusión.

    Parámetros:
    -----------
    cm : np.ndarray
        Matriz de confusión (num_classes x num_classes)
    class_names : list
        Lista de nombres de clases
    titulo : str
        Título de la gráfica
    guardar : bool
        Si True, guarda la figura
    ruta_guardado : str
        Ruta para guardar la figura
    """
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(12,6))
    plt.bar(class_names, class_accuracy, color='steelblue')
    plt.title(titulo)
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.xticks(rotation=45, ha='right')

    for i, v in enumerate(class_accuracy):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=8)

    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if guardar:
        if ruta_guardado is None:
            os.makedirs("./Graficas", exist_ok=True)
            ruta_guardado = os.path.join("./Graficas", "accuracy_por_clase.png")
        plt.savefig(ruta_guardado, dpi=300)
        print(f"✅ Gráfica guardada en: {ruta_guardado}")

    plt.show()



def graficar_confianza_vs_acierto(confianzas, y_true, y_pred, titulo="Confianza vs Acierto", guardar=False, ruta_guardado=None):
    """
    Grafica la relación entre la confianza de predicciones y si fueron correctas.

    Parámetros:
    -----------
    confianzas : np.ndarray
        Array con la probabilidad máxima de cada predicción
    y_true : np.ndarray
        Labels verdaderos
    y_pred : np.ndarray
        Labels predichos
    titulo : str
        Título de la gráfica
    guardar : bool
        Si True, guarda la figura
    ruta_guardado : str
        Ruta para guardar la figura
    """
    correct = (y_true == y_pred).astype(int)

    plt.figure(figsize=(8,5))
    plt.scatter(confianzas, correct, alpha=0.3)
    plt.title(titulo)
    plt.xlabel("Confianza")
    plt.ylabel("Correcto (1) / Incorrecto (0)")
    plt.grid(True, alpha=0.3)

    mean_conf = np.mean(confianzas)
    plt.axvline(mean_conf, color='red', linestyle='--', label=f"Media: {mean_conf:.3f}")
    plt.legend()

    if guardar:
        if ruta_guardado is None:
            os.makedirs("./Graficas", exist_ok=True)
            ruta_guardado = os.path.join("./Graficas", "confianza_vs_acierto.png")
        plt.savefig(ruta_guardado, dpi=300)
        print(f"✅ Gráfica guardada en: {ruta_guardado}")

    plt.show()
    