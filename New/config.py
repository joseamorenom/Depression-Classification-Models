# config.py
import torch
import os

# ======================================================
#   RUTAS
# ======================================================
RESULTS_ROOT = "/home/gitalab2/Documents/Jose/Individual Results"
SAVE_MODEL   = "/home/gitalab2/Documents/Jose/Individual Results"

REPRESENTATIONS = [
    "/home/gitalab2/Documents/Jose/Representaciones/Habla/Wav2Vec_XLSR_Base"
]

# ======================================================
#   CLASES
# ======================================================
LABELS = {
    "Aliviado": 0,
    "Depresivo": 1
}

# ======================================================
#   MODELO (ARQUITECTURA)
# ======================================================
# ¡ATENCIÓN! Ajustar INPUT_SIZE según la representación:
# - Wav2Vec2 Base: usualmente 768
# - Wav2Vec2 Large / XLSR: usualmente 1024
# - BERT Base: 768
# - Word2Vec: 300
INPUT_SIZE = 1024  
NUM_CLASSES = 2

# Tamaños de capas ocultas (por si quieres cambiarlos fácil)
HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 64

# ======================================================
#   ENTRENAMIENTO E HIPERPARÁMETROS
# ======================================================
BATCH_SIZE = 32     # Mínimo 32 para que BatchNorm funcione bien
EPOCHS = 150
PATIENCE = 25

# --- Control de Overfitting ---
LR = 1e-4           # Subido de 1e-5 a 1e-4 para aprendizaje más efectivo
DROPOUT = 0.5       # Aumentado a 0.5 (agresivo) para combatir overfitting severo
WEIGHT_DECAY = 1e-4 # Penalización L2 para el optimizador (NUEVO)
FOCAL_GAMMA = 2.0   # Enfoque en clases difíciles (NUEVO)

# ======================================================
#   DEVICE
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG] Usando dispositivo: {DEVICE}")