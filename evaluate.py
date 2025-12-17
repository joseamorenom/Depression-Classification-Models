# evaluate.py

import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)
from config import DEVICE, BATCH_SIZE
import numpy as np


def compute_sensitivity_specificity(cm):
    """
    Calcula sensibilidad y especificidad desde la matriz de confusión.
    cm: confusion_matrix (2x2)
    """
    if cm.shape != (2, 2):
        return None, None

    TN, FP = cm[0]
    FN, TP = cm[1]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return sensitivity, specificity


def evaluate_model(
    model,
    dataset,
    mode_name="test",
    result_path=""
):
    print(f"\n>>> Evaluando: {mode_name}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_preds = []
    all_trues = []
    all_patients = []

    model.eval()
    with torch.no_grad():
        for xb, yb, patient_ids in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(yb.cpu().numpy())
            all_patients.extend(list(patient_ids))

    # ==========================================================================
    # MÉTRICAS POR SEGMENTO
    # ==========================================================================
    seg_acc = accuracy_score(all_trues, all_preds)
    seg_f1 = f1_score(all_trues, all_preds, average="macro")
    seg_recall = recall_score(all_trues, all_preds, average="macro")
    seg_cm = confusion_matrix(all_trues, all_preds)

    seg_sens, seg_spec = compute_sensitivity_specificity(seg_cm)

    # ==========================================================================
    # MÉTRICAS POR PACIENTE (voto mayoritario)
    # ==========================================================================
    patient_predictions = {}
    for p, pred in zip(all_patients, all_preds):
        patient_predictions.setdefault(p, []).append(pred)

    patient_trues = []
    patient_preds = []

    for p, preds in patient_predictions.items():
        # mayoría de votos
        p_pred = max(set(preds), key=preds.count)

        # obtenemos la etiqueta real mirando primera ocurrencia
        idx = all_patients.index(p)
        p_true = all_trues[idx]

        patient_preds.append(p_pred)
        patient_trues.append(p_true)

    pat_acc = accuracy_score(patient_trues, patient_preds)
    pat_f1 = f1_score(patient_trues, patient_preds, average="macro")
    pat_recall = recall_score(patient_trues, patient_preds, average="macro")
    pat_cm = confusion_matrix(patient_trues, patient_preds)

    pat_sens, pat_spec = compute_sensitivity_specificity(pat_cm)

    # ==========================================================================
    #  IMPRIMIR EN CONSOLA
    # ==========================================================================
    print("\n======= RESULTADOS POR SEGMENTO =======")
    print(f"Accuracy:      {seg_acc:.4f}")
    print(f"F1:            {seg_f1:.4f}")
    print(f"Sensibilidad:  {seg_sens:.4f}")
    print(f"Especificidad: {seg_spec:.4f}")
    print("Matriz de confusión:")
    print(seg_cm)

    print("\n======= RESULTADOS POR PACIENTE =======")
    print(f"Accuracy:      {pat_acc:.4f}")
    print(f"F1:            {pat_f1:.4f}")
    print(f"Sensibilidad:  {pat_sens:.4f}")
    print(f"Especificidad: {pat_spec:.4f}")
    print("Matriz de confusión:")
    print(pat_cm)

    # ==========================================================================
    # GUARDAR A TXT (OPCIONAL)
    # ==========================================================================
    if result_path != "":
        os.makedirs(result_path, exist_ok=True)
        filepath = os.path.join(result_path, f"{mode_name}_metrics.txt")

        with open(filepath, "w", encoding="utf-8") as f:

            f.write("===== RESULTADOS POR SEGMENTO =====\n")
            f.write(f"Accuracy:      {seg_acc:.4f}\n")
            f.write(f"F1:            {seg_f1:.4f}\n")
            f.write(f"Sensibilidad:  {seg_sens:.4f}\n")
            f.write(f"Especificidad: {seg_spec:.4f}\n")
            f.write("Matriz de confusión:\n")
            f.write(str(seg_cm))
            f.write("\n\n")

            f.write("===== RESULTADOS POR PACIENTE =====\n")
            f.write(f"Accuracy:      {pat_acc:.4f}\n")
            f.write(f"F1:            {pat_f1:.4f}\n")
            f.write(f"Sensibilidad:  {pat_sens:.4f}\n")
            f.write(f"Especificidad: {pat_spec:.4f}\n")
            f.write("Matriz de confusión:\n")
            f.write(str(pat_cm))
            f.write("\n")

        print(f"\nResultados guardados en: {filepath}")

    # ==========================================================================
    # RETORNO PARA USOS POSTERIORES
    # ==========================================================================
    return {
        "segment": {
            "accuracy": seg_acc,
            "f1": seg_f1,
            "sensitivity": seg_sens,
            "specificity": seg_spec,
            "cm": seg_cm,
        },
        "patient": {
            "accuracy": pat_acc,
            "f1": pat_f1,
            "sensitivity": pat_sens,
            "specificity": pat_spec,
            "cm": pat_cm,
        },
    }
