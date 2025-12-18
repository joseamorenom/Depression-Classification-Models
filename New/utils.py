# utils.py
import random
import numpy as np
import torch
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score

# -----------------------------------------------------------
# 1. Subsampling por PACIENTES COMPLETOS (Correcto para Tesis)
# -----------------------------------------------------------
def patient_subsampling(dataset):
    # Agrupar índices por paciente y detectar su etiqueta
    patient_map = defaultdict(list)
    patient_labels = {}

    for idx in range(len(dataset)):
        _, label, pid = dataset[idx]
        patient_map[pid].append(idx)
        # Asumimos que todos los segmentos de un paciente tienen la misma etiqueta
        if pid not in patient_labels:
            patient_labels[pid] = label

    # Separar pacientes por clase
    pats_0 = [p for p, l in patient_labels.items() if l == 0]
    pats_1 = [p for p, l in patient_labels.items() if l == 1]

    # Encontrar el mínimo numero de pacientes
    n_min = min(len(pats_0), len(pats_1))
    
    print(f"[SUBSAMPLING] Original: {len(pats_0)} Sanos vs {len(pats_1)} Depresivos.")

    # Aleatorizar y recortar la clase mayoritaria
    random.shuffle(pats_0)
    random.shuffle(pats_1)
    
    keep_p0 = pats_0[:n_min]
    keep_p1 = pats_1[:n_min] # En teoría tomamos todos, pero por seguridad hacemos slice

    selected_patients = set(keep_p0 + keep_p1)
    
    # Recuperar todos los índices de los segmentos de esos pacientes
    final_indices = []
    for pid in selected_patients:
        final_indices.extend(patient_map[pid])

    print(f"[SUBSAMPLING] Final: {len(keep_p0)} Sanos vs {len(keep_p1)} Depresivos (se mantienen sus segmentos completos).")
    
    return final_indices

# -----------------------------------------------------------
# 2. Predicción por paciente (Con desempate inteligente)
# -----------------------------------------------------------
def aggregate_predictions(patient_ids, preds):
    per_patient = defaultdict(list)
    for pid, p in zip(patient_ids, preds):
        per_patient[pid].append(p)

    final = {}
    for pid, votes in per_patient.items():
        # Promedio de votos (probabilidad empírica)
        mean_vote = np.mean(votes)
        # Si >= 0.5 es Depresivo (1). Esto maneja el empate 0.5 clasificándolo como 1 (prioriza sensibilidad)
        final[pid] = 1 if mean_vote >= 0.5 else 0
            
    return final

# -----------------------------------------------------------
# 3. Métricas Completas (Incluye UAR)
# -----------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except:
        tn=fp=fn=tp=0

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    
    # Sensibilidad (Recall clase 1)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    # Especificidad (Recall clase 0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # UAR (Unweighted Average Recall) = Promedio macro de recalls
    uar = recall_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        "acc": acc,
        "f1": f1,
        "uar": uar,  # <--- AQUÍ ESTÁ LO QUE PIDIÓ TU ASESOR
        "sensitivity": sens,
        "specificity": spec,
        "confusion": (tn, fp, fn, tp)
    }

def metrics_segment(y_true, y_pred):
    return calculate_metrics(y_true, y_pred)

def metrics_patient(true_dict, pred_dict):
    sorted_pids = sorted(true_dict.keys())
    y_true = [true_dict[p] for p in sorted_pids]
    y_pred = [pred_dict[p] for p in sorted_pids]
    return calculate_metrics(y_true, y_pred)