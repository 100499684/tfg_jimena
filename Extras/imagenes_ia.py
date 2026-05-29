"""
Generación de imágenes sintéticas con InstructPix2Pix
Toma fotos reales y las modifica según instrucciones de texto.
Sin problemas de dtype, más simple y directo.

Requisitos:
    pip install diffusers transformers accelerate torch torchvision Pillow tqdm
"""

import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "dir_dataset":            "Training",
    "dir_salida":             "Training_sinteticas",
    "clases_objetivo":        ["reach_backseat", "talking_to"],
    "variaciones_por_imagen": 5,
    "img_size":               512,
    "num_inference_steps":    50,
    "guidance_scale":         9.0,   # fidelidad al texto (3-15)
    "image_guidance_scale":   1.2,   # fidelidad a la imagen original (1-3)
}

# Instrucciones de modificación por clase
# Describe QUÉ CAMBIAR, no qué generar desde cero
INSTRUCCIONES = {
    "reach_backseat": [
        "make it daytime with bright sunlight through the car windows",
        "make the driver a young man with short dark hair",
        "make the driver an older woman with gray hair",
        "make it raining outside the car windows",
        "make the driver wearing a red jacket instead",
    ],
    "talking_to": [
        "change the lighting to morning sunlight inside the car",
        "make the driver wear a hat",
        "change the background outside the car window to city street",
        "make it look like a cloudy day outside",
        "change the hair color and style of the driver",
    ],
}


# ─────────────────────────────────────────────────────────────
# GENERADOR
# ─────────────────────────────────────────────────────────────

class GeneradorPix2Pix:

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        print(f"\n🖥️  Dispositivo: {self.device.upper()}")

    def cargar_modelos(self):
        print("\n📥 Cargando InstructPix2Pix (~3GB primera vez)...")

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

        print("✅ Modelo cargado\n")

    def generar_variaciones(
        self,
        imagen_referencia: Image.Image,
        clase: str,
        n_variaciones: int,
        idx_base: int,
    ) -> list:

        instrucciones_clase = INSTRUCCIONES.get(
            clase,
            [f"change the lighting of the scene with driver {clase.replace('_', ' ')}"]
        )

        size = self.config["img_size"]
        imagen_ref = imagen_referencia.resize((size, size)).convert("RGB")

        imagenes_generadas = []
        for i in range(n_variaciones):
            instruccion = instrucciones_clase[i % len(instrucciones_clase)]
            generator = torch.Generator(device=self.device).manual_seed(idx_base * 100 + i)

            resultado = self.pipe(
                prompt=instruccion,
                image=imagen_ref,
                num_inference_steps=self.config["num_inference_steps"],
                guidance_scale=self.config["guidance_scale"],
                image_guidance_scale=self.config["image_guidance_scale"],
                generator=generator,
            ).images[0]

            imagenes_generadas.append(resultado)

        return imagenes_generadas

    def procesar_clase(self, clase: str) -> int:
        dir_clase = Path(self.config["dir_dataset"]) / clase
        dir_salida = Path(self.config["dir_salida"]) / clase
        dir_salida.mkdir(parents=True, exist_ok=True)

        extensiones = {".jpg", ".jpeg", ".png", ".bmp"}
        imagenes_ref = [
            p for p in dir_clase.iterdir()
            if p.suffix.lower() in extensiones
        ]

        if not imagenes_ref:
            print(f"  ⚠️  No se encontraron imágenes en {dir_clase}")
            return 0

        print(f"\n📁 Clase: {clase}")
        print(f"   Imágenes de referencia : {len(imagenes_ref)}")
        print(f"   Variaciones por imagen : {self.config['variaciones_por_imagen']}")
        print(f"   Total a generar        : {len(imagenes_ref) * self.config['variaciones_por_imagen']}")

        total_generadas = 0
        n_var = self.config["variaciones_por_imagen"]

        for idx, ruta_ref in enumerate(tqdm(imagenes_ref, desc=f"  {clase}")):
            try:
                imagen_ref = Image.open(ruta_ref).convert("RGB")
                variaciones = self.generar_variaciones(imagen_ref, clase, n_var, idx)

                for j, img_gen in enumerate(variaciones):
                    nombre = f"synth_{ruta_ref.stem}_v{j:02d}.jpg"
                    img_gen.save(dir_salida / nombre, quality=95)
                    total_generadas += 1

            except Exception as e:
                print(f"\n  ⚠️  Error con {ruta_ref.name}: {e}")
                continue

        print(f"  ✅ {total_generadas} imágenes guardadas en {dir_salida}")
        return total_generadas

    def ejecutar(self):
        self.cargar_modelos()

        resumen = {}
        for clase in self.config["clases_objetivo"]:
            n = self.procesar_clase(clase)
            resumen[clase] = n

        print("\n" + "=" * 50)
        print("  RESUMEN FINAL")
        print("=" * 50)
        for clase, n in resumen.items():
            print(f"  {clase:<25} → {n:>6} imágenes generadas")
        print(f"\n  Guardadas en: {self.config['dir_salida']}/")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generador = GeneradorPix2Pix(CONFIG)
    generador.ejecutar()