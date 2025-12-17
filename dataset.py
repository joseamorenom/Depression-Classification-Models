# dataset.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from config import LABELS

# ----------------------------------------
#   Función: cargar un embedding desde CSV
# ----------------------------------------
def load_embedding(csv_path):
    df = pd.read_csv(csv_path, header=None)
    values = df.iloc[1].values.astype("float32")  # segunda fila
    return torch.tensor(values)


# ----------------------------------------
#   Dataset por segmento
# ----------------------------------------
class SegmentDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            label = LABELS[class_name]

            for patient_id in os.listdir(class_path):
                pat_path = os.path.join(class_path, patient_id)
                if not os.path.isdir(pat_path):
                    continue

                for file in os.listdir(pat_path):
                    if file.endswith(".csv"):
                        csv_path = os.path.join(pat_path, file)
                        self.samples.append(
                            (csv_path, label, patient_id)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, label, patient_id = self.samples[idx]

        embedding = load_embedding(csv_path)

        return embedding, label, patient_id
