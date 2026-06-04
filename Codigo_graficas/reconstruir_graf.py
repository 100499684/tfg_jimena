import re
import os
import matplotlib.pyplot as plt

OUT_FILE = "./Estudios/terminal/process.trainEffectiveNet2_augmentation.1717.out"
OUTPUT_DIR = "./Estudios/Graficas/Effectivenetb3_combinacion2_augmentation_train"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# MÉTRICAS
# ==========================================================

acc, loss, val_acc, val_loss = [], [], [], []
acc_f2, loss_f2, val_acc_f2, val_loss_f2 = [], [], [], []

current_phase = 1
current_epoch = None

print("Leyendo:", OUT_FILE)

# ==========================================================
# LECTURA
# ==========================================================

with open(OUT_FILE, "r", encoding="utf-8", errors="ignore") as f:

    for line in f:

        line_upper = line.upper()

        # --------------------------------------------------
        # DETECTAR FASE
        # --------------------------------------------------
        if "FASE 1" in line_upper:
            current_phase = 1
            continue

        if "FASE 2" in line_upper:
            current_phase = 2
            continue

        # --------------------------------------------------
        # DETECTAR EPOCH
        # --------------------------------------------------
        epoch_match = re.search(r"Epoch\s+(\d+)/\d+", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        # --------------------------------------------------
        # DETECTAR MÉTRICAS
        # --------------------------------------------------
        match = re.search(
            r"accuracy:\s*([0-9.]+)\s*-\s*loss:\s*([0-9.]+)\s*-\s*val_accuracy:\s*([0-9.]+)\s*-\s*val_loss:\s*([0-9.]+)",
            line
        )

        if match:

            values = (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4))
            )

            # opcional: debug coherencia
            # print(current_phase, current_epoch)

            if current_phase == 1:
                acc.append(values[0])
                loss.append(values[1])
                val_acc.append(values[2])
                val_loss.append(values[3])

            else:
                acc_f2.append(values[0])
                loss_f2.append(values[1])
                val_acc_f2.append(values[2])
                val_loss_f2.append(values[3])

# ==========================================================
# COMPROBACIONES
# ==========================================================

print(f"\nEpochs Fase 1: {len(acc)}")
print(f"Epochs Fase 2: {len(acc_f2)}")

if len(acc) == 0:
    raise ValueError("No se han encontrado métricas de Fase 1")

# ==========================================================
# FUNCIÓN PARA GRAFICAR
# ==========================================================

def guardar_grafica(train_values,
                    val_values,
                    titulo,
                    ylabel,
                    ruta):

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(train_values) + 1),
        train_values,
        marker='o',
        label='Train'
    )

    plt.plot(
        range(1, len(val_values) + 1),
        val_values,
        marker='s',
        label='Validation'
    )

    plt.title(titulo)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(ruta, dpi=300)
    plt.close()

    print("Guardada:", ruta)

# ==========================================================
# FASE 1
# ==========================================================

guardar_grafica(
    acc,
    val_acc,
    "Accuracy - Fase 1",
    "Accuracy",
    os.path.join(OUTPUT_DIR, "acc_fase1.png")
)

guardar_grafica(
    loss,
    val_loss,
    "Loss - Fase 1",
    "Loss",
    os.path.join(OUTPUT_DIR, "loss_fase1.png")
)

# ==========================================================
# FASE 2 (si existe)
# ==========================================================

if len(acc_f2) > 0:

    guardar_grafica(
        acc_f2,
        val_acc_f2,
        "Accuracy - Fine Tuning (Fase 2)",
        "Accuracy",
        os.path.join(OUTPUT_DIR, "acc_fase2.png")
    )

    guardar_grafica(
        loss_f2,
        val_loss_f2,
        "Loss - Fine Tuning (Fase 2)",
        "Loss",
        os.path.join(OUTPUT_DIR, "loss_fase2.png")
    )

# ==========================================================
# RESUMEN
# ==========================================================

print("\n====================================")
print("GRÁFICAS RECUPERADAS CORRECTAMENTE")
print("====================================")

print("\nMejor val_accuracy Fase 1:", max(val_acc))

if len(val_acc_f2) > 0:
    print("Mejor val_accuracy Fase 2:", max(val_acc_f2))