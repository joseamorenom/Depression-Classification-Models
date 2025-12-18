# dataset.py

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from config import LABELS

# ----------------------------------------
#   Función: cargar un embedding desde CSV
# ----------------------------------------
def load_embedding(csv_path):
    """
    Lee el archivo CSV asegurando obtener los datos numéricos de la SEGUNDA FILA.
    Maneja dos casos:
    1. El archivo se lee con múltiples columnas (estándar).
    2. El archivo se lee como una sola columna de texto con comas.
    """
    try:
        # header=None: Leemos todo "crudo", la fila 1 de títulos será la fila 0 del DF
        df = pd.read_csv(csv_path, header=None)
        
        # Seleccionamos la SEGUNDA FILA (índice 1) como indicaste
        row_data = df.iloc[1]

        # CASO A: Pandas detectó múltiples columnas (lo normal si hay comas)
        if len(row_data) > 1:
            values = row_data.values.astype("float32")

        # CASO B: Pandas leyó todo como una sola celda (string con comas)
        # Esto cumple con tu descripción literal: "datos en la primera celda... separados por comas"
        else:
            # Tomamos el valor de la única celda
            cell_value = row_data.iloc[0]
            if isinstance(cell_value, str):
                # Separamos por comas y convertimos a float
                values = np.array([float(x) for x in cell_value.split(',') if x.strip()], dtype="float32")
            else:
                # Si ya es un número (caso raro de 1 sola feature)
                values = np.array([cell_value], dtype="float32")

        return torch.tensor(values)

    except Exception as e:
        print(f"[ERROR CRÍTICO] Fallo al leer {csv_path}. Verifica el formato. Error: {e}")
        # Retornamos None o lanzamos error. Mejor lanzar error para no entrenar con basura.
        raise e


# ----------------------------------------
#   Dataset por segmento
# ----------------------------------------
class SegmentDataset(Dataset):
    def __init__(self, root_dir):
        """
        Recorre las carpetas: Clase -> Paciente -> Archivos CSV
        """
        self.samples = []

        if not os.path.exists(root_dir):
            print(f"[ADVERTENCIA] La ruta {root_dir} no existe.")
            return

        # Recorrer carpetas de clases (Aliviado / Depresivo)
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue

            # Verificar etiqueta válida
            if class_name not in LABELS:
                continue
            label = LABELS[class_name]

            # Recorrer pacientes dentro de la clase
            for patient_id in os.listdir(class_path):
                pat_path = os.path.join(class_path, patient_id)
                
                if not os.path.isdir(pat_path):
                    continue

                # Recorrer archivos CSV del paciente
                for file in os.listdir(pat_path):
                    if file.endswith(".csv"):
                        csv_path = os.path.join(pat_path, file)
                        
                        # Guardamos la tupla (Ruta, Label, ID Paciente)
                        # El ID Paciente es VITAL para el subsampling correcto en utils.py
                        self.samples.append(
                            (csv_path, label, patient_id)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, label, patient_id = self.samples[idx]

        # Cargar los datos numéricos de la fila 1
        embedding = load_embedding(csv_path)

        return embedding, label, patient_id