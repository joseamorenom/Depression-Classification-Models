# model.py
import torch
import torch.nn as nn
from config import INPUT_SIZE, NUM_CLASSES, DROPOUT

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        # Definimos la red con BatchNorm para mayor estabilidad
        self.net = nn.Sequential(
            # Capa 1: Entrada -> 256
            nn.Linear(INPUT_SIZE, 256),
            nn.BatchNorm1d(256),        # <--- MEJORA: Normalización por lote
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            # Capa 2: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),        # <--- MEJORA
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            # Capa 3: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),         # <--- MEJORA
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            # Salida: 64 -> Clases (Logits, SIN Softmax)
            nn.Linear(64, NUM_CLASSES)
        )
        
        # Inicialización de pesos inteligente (He / Kaiming Initialization)
        self._init_weights()

    def _init_weights(self):
        """Inicializa los pesos para evitar gradientes que explotan o desaparecen."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization optimizada para ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)