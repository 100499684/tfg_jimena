import os
import time
import pandas as pd
from PIL import Image
from tqdm import tqdm
from google import genai

'''

import google.generativeai as genai

genai.configure(api_key="AQ.Ab8RN6LBi4D9ldxUvk1aVcMIfmWTh6CKOyclhYWJpt43yeyBMQ") # Pon tu clave aquí

print("Modelos disponibles en tu cuenta para generación:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)





'''
# ==========================================
# CONFIGURACIÓN
# ==========================================
API_KEY = "AQ.Ab8RN6LBi4D9ldxUvk1aVcMIfmWTh6CKOyclhYWJpt43yeyBMQ" # Sustituye por tu clave
CARPETA_BASE_TEST = "../../../../remote-repositorio/afrodita/repo-ultra/tfg_jcabrera/Testing" 
ARCHIVO_SALIDA = "./Estudios/resultados_gemini.csv"

# Iniciamos el cliente con la nueva sintaxis
client = genai.Client(api_key=API_KEY)

# Elegimos uno de los modelos ultra-rápidos que tienes disponibles
MODELO_ELEGIDO = 'gemini-2.5-flash' 

PROMPT_SISTEMA = """
Eres un sistema experto de seguridad vial. Tu tarea es analizar la imagen de la cámara interior de un coche y clasificar la acción que está realizando el conductor.
Debes responder ÚNICA Y EXCLUSIVAMENTE con UNA de las siguientes etiquetas exactas (sin puntos, sin comillas, sin explicaciones adicionales):

drinking
hair_and_makeup
phonecall
radio
reach_backseat
reach_side
safe_drive
talking_to_passenger
texting

Si hay dudas, elige la etiqueta que mejor describa la acción principal.
"""

def analizar_imagenes_por_carpetas(carpeta_base, archivo_salida):
    resultados = []
    
    clases_reales = [d for d in os.listdir(carpeta_base) if os.path.isdir(os.path.join(carpeta_base, d))]
    print(f"Se han detectado {len(clases_reales)} clases: {clases_reales}")
    
    for clase in clases_reales:
        ruta_clase = os.path.join(carpeta_base, clase)
        imagenes = [f for f in os.listdir(ruta_clase) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nProcesando la clase: '{clase}' ({len(imagenes)} imágenes)")
        
        for nombre_archivo in tqdm(imagenes):
            ruta_completa = os.path.join(ruta_clase, nombre_archivo)
            
            try:
                img = Image.open(ruta_completa)
                
                # NUEVA sintaxis de llamada a la API
                response = client.models.generate_content(
                    model=MODELO_ELEGIDO,
                    contents=[img, PROMPT_SISTEMA]
                )
                
                prediccion = response.text.strip().lower()
                
                resultados.append({
                    "imagen": nombre_archivo,
                    "valor_real": clase,
                    "prediccion_llm": prediccion
                })
                
                time.sleep(2) # Pausa por si la cuota gratuita tiene límite por minuto
                
            except Exception as e:
                print(f"\nError en {nombre_archivo}: {e}")
                resultados.append({
                    "imagen": nombre_archivo,
                    "valor_real": clase,
                    "prediccion_llm": "ERROR"
                })
                time.sleep(5)
                
    df = pd.DataFrame(resultados)
    df.to_csv(archivo_salida, index=False)
    print(f"\n¡Proceso completado! CSV guardado con {len(df)} registros.")

if __name__ == "__main__":
    analizar_imagenes_por_carpetas(CARPETA_BASE_TEST, ARCHIVO_SALIDA)