# config.py

import torch
import os

# ======================================================
#   RUTA DONDE SE GUARDARAN RESULTADOS
# ======================================================
RESULTS_ROOT = r"C:\Users\amesa\OneDrive\GITA\Trabajo de grado\Resultados FCN"

# ======================================================
#   LISTA DE REPRESENTACIONES A PROCESAR
# ======================================================
REPRESENTATIONS = [
    r"C:\Users\amesa\OneDrive\GITA\Trabajo de grado\representaciones\Habla\Wav2Vec_XLSR_Emociones"
]

#r"C:\Users\amesa\OneDrive\GITA\Trabajo de grado\representaciones\Texto\Word2Vec"

# ======================================================
#       CLASES
# ======================================================
LABELS = {
    "Aliviado": 0,
    "Depresivo": 1
}

# ======================================================
#       HIPERPARÁMETROS
# ======================================================
INPUT_SIZE = 1024        # AJUSTAR Para cada tipo, falta 300
HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 64
NUM_CLASSES = 2

LR = 0.0001
BATCH_SIZE = 32
EPOCHS = 150
DROPOUT = 0.3
PATIENCE = 20

# ======================================================
#   DEVICE
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
