# model.py

import torch
import torch.nn as nn
from config import INPUT_SIZE, NUM_CLASSES, DROPOUT

class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)
