# utils.py

import numpy as np
import random
import torch
from collections import defaultdict, Counter
from sklearn.metrics import (
    f1_score, 
    accuracy_score,
    confusion_matrix,
    recall_score
)

# -----------------------------------------------------------
# Subsampling por paciente (para train o val)
# Mantiene todos los depresivos y selecciona 50% de aliviados
# -----------------------------------------------------------
def patient_subsampling(dataset):
    by_patient = defaultdict(list)

    for idx in range(len(dataset)):
        _, label, patient = dataset[idx]
        by_patient[patient].append(idx)

    new_indices = []

    for patient, idx_list in by_patient.items():
        # Label del paciente
        _, label, _ = dataset[idx_list[0]]

        if label == 0:     # Aliviado
            keep = random.sample(idx_list, max(1, len(idx_list) // 2))
            new_indices.extend(keep)
        else:              # Depresivo
            new_indices.extend(idx_list)

    return new_indices


# -----------------------------------------------------------
# Predicción por paciente (voto por mayoría)
# -----------------------------------------------------------
def aggregate_predictions(patient_ids, preds):
    per_patient = defaultdict(list)

    for pid, p in zip(patient_ids, preds):
        per_patient[pid].append(p)

    final = {}
    for pid, votes in per_patient.items():
        final[pid] = Counter(votes).most_common(1)[0][0]

    return final


# -----------------------------------------------------------
# Metricas por segmento
# -----------------------------------------------------------
def metrics_segment(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # sensitivity = recall de clase depresiva (1)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)

    # specificity = recall de clase aliviada (0)
    specificity = recall_score(y_true, y_pred, pos_label=0)

    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "conf_matrix": cm
    }


# -----------------------------------------------------------
# Métricas por paciente
# -----------------------------------------------------------
def metrics_patient(true_dict, pred_dict):
    y_true = []
    y_pred = []

    for pid in true_dict.keys():
        y_true.append(true_dict[pid])
        y_pred.append(pred_dict[pid])

    cm = confusion_matrix(y_true, y_pred)

    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)

    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "conf_matrix": cm
    }


# -----------------------------------------------------------
# Guardar métricas a archivo TXT
# -----------------------------------------------------------
def save_metrics_to_txt(path, seg_results, pat_results):
    with open(path, "w", encoding="utf-8") as f:

        f.write("=== RESULTADOS POR SEGMENTO ===\n")
        f.write(f"Accuracy:      {seg_results['acc']:.4f}\n")
        f.write(f"F1-score:      {seg_results['f1']:.4f}\n")
        f.write(f"Sensitivity:   {seg_results['sensitivity']:.4f}\n")
        f.write(f"Specificity:   {seg_results['specificity']:.4f}\n")
        f.write("Matriz de confusión:\n")
        f.write(str(seg_results["conf_matrix"]))
        f.write("\n\n")

        f.write("=== RESULTADOS POR PACIENTE ===\n")
        f.write(f"Accuracy:      {pat_results['acc']:.4f}\n")
        f.write(f"F1-score:      {pat_results['f1']:.4f}\n")
        f.write(f"Sensitivity:   {pat_results['sensitivity']:.4f}\n")
        f.write(f"Specificity:   {pat_results['specificity']:.4f}\n")
        f.write("Matriz de confusión:\n")
        f.write(str(pat_results["conf_matrix"]))
