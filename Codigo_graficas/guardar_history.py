import os
import json



def guardar_historial(history, nombre):
    history_dict = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'lr': history.history.get('lr', [])
    }
    ruta_json = os.path.join(ruta_output, "Entrenamiento", f"{nombre}.json")
    with open(ruta_json, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"✅ Guardado: {ruta_json}")



def augment_data(x, y):
    """Aplica augmentation según la clase"""
    # Augmentation básico para todas
    x = basic_augmentation(x, training=True)
    
    # Augmentation extra para clases minoritarias
    is_minority = tf.reduce_any(tf.equal(y, minority_indices))
    x = tf.cond(is_minority, 
                lambda: aggressive_augmentation(x, training=True), 
                lambda: x)
    return x, y