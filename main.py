# main.py
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset

from dataset import SegmentDataset
from model import FCN
from utils import (
    patient_subsampling,
    aggregate_predictions,
    metrics_segment,
    metrics_patient
)
from config import (
    REPRESENTATIONS,
    RESULTS_ROOT,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LR,
    PATIENCE
)

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, recall_score, f1_score


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =======================================================================
#  MÉTODO PARA GUARDAR MÉTRICAS DE SEGMENTO Y PACIENTE
# =======================================================================
def evaluate_and_save(model, dataset, out_prefix):

    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds, trues, pids = [], [], []

    with torch.no_grad():
        for xb, yb, pid in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            pred = torch.argmax(out, dim=1)

            preds.extend(pred.cpu().numpy())
            trues.extend(yb.numpy())
            pids.extend(pid)

    # ------------------ MÉTRICAS SEGMENTO ----------------------
    seg_m = metrics_segment(trues, preds)

    try:
        tn, fp, fn, tp = confusion_matrix(trues, preds, labels=[0,1]).ravel()
    except:
        tn = fp = fn = tp = 0

    seg_sens = recall_score(trues, preds, pos_label=1, zero_division=0)
    seg_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ------------------ MÉTRICAS PACIENTE ----------------------
    real_per_patient = {}
    for i, pid in enumerate(pids):
        if pid not in real_per_patient:
            real_per_patient[pid] = int(trues[i])

    pred_per_patient = aggregate_predictions(pids, preds)

    y_true_pat = [real_per_patient[p] for p in sorted(real_per_patient.keys())]
    y_pred_pat = [pred_per_patient[p] for p in sorted(real_per_patient.keys())]

    try:
        tn2, fp2, fn2, tp2 = confusion_matrix(y_true_pat, y_pred_pat, labels=[0,1]).ravel()
    except:
        tn2 = fp2 = fn2 = tp2 = 0

    pat_sens = recall_score(y_true_pat, y_pred_pat, pos_label=1, zero_division=0)
    pat_spec = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0

    pat_m = metrics_patient(real_per_patient, pred_per_patient)

    # ------------------ GUARDAR REPORTE ----------------------
    report_lines = [
        "=== METRICAS - SEGMENTO ===",
        f"Accuracy: {seg_m.get('acc', np.nan):.4f}",
        f"F1 macro: {seg_m.get('f1', np.nan):.4f}",
        f"Sensibilidad: {seg_sens:.4f}",
        f"Especificidad: {seg_spec:.4f}",
        f"Confusion: {tn}, {fp}, {fn}, {tp}",
        "",
        "=== METRICAS - PACIENTE ===",
        f"Accuracy: {pat_m.get('acc', np.nan):.4f}",
        f"F1 macro: {pat_m.get('f1', np.nan):.4f}",
        f"Sensibilidad: {pat_sens:.4f}",
        f"Especificidad: {pat_spec:.4f}",
        f"Confusion: {tn2}, {fp2}, {fn2}, {tp2}",
        ""
    ]

    with open(out_prefix + "_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "segment": seg_m,
        "patient": pat_m
    }


# =======================================================================
#  PINTAR CURVAS
# =======================================================================
def plot_curves(history, out_path, title_suffix=""):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.title(f"Loss {title_suffix}")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.title(f"Accuracy {title_suffix}")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =======================================================================
#  ENTRENAMIENTO
# =======================================================================
def train_loop(model, train_dataset, val_dataset, experiment_name, out_dir,
               subs_train=False, subs_val=False):

    print(f"\n=== Entrenando experimento: {experiment_name} ===")

    # --- subsampling train ---
    if subs_train:
        idx = patient_subsampling(train_dataset)
        train_ds = Subset(train_dataset, idx)
    else:
        train_ds = train_dataset

    # --- subsampling val ---
    if subs_val:
        idx = patient_subsampling(val_dataset)
        val_ds = Subset(val_dataset, idx)
    else:
        val_ds = val_dataset

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = -1
    patience_count = 0
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(1, EPOCHS + 1):

        model.train()
        tot_loss = 0
        correct = 0
        total = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * xb.size(0)
            pred = torch.argmax(out, 1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

        train_loss = tot_loss / total
        train_acc = correct / total

        # ---------- VALIDACIÓN ----------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                out = model(xb)
                loss = criterion(out, yb)

                val_loss += loss.item() * xb.size(0)
                pred = torch.argmax(out, 1)

                val_correct += (pred == yb).sum().item()
                val_total += xb.size(0)

                y_true.extend(yb.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(y_true, y_pred, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:03d} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | F1={val_f1:.4f}")

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping activado.")
            break

    model.load_state_dict(best_state)

    plot_curves(history, os.path.join(out_dir, f"{experiment_name}_curves.png"), experiment_name)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{experiment_name}_model.pt"))

    return model


# =======================================================================
#  MAIN
# =======================================================================
def main():

    print("\n=========== INICIO EXPERIMENTO COMPLETO ===========")

    for rep in REPRESENTATIONS:

        rep_name = os.path.basename(rep)

        train_dir = os.path.join(rep, "train")
        val_dir   = os.path.join(rep, "val")
        test_dir  = os.path.join(rep, "test")

        print(f"\n\n=== Cargando datos de representación: {rep_name} ===")

        train_ds = SegmentDataset(train_dir)
        val_ds   = SegmentDataset(val_dir)
        test_ds  = SegmentDataset(test_dir)

        out_rep_dir = os.path.join(RESULTS_ROOT, rep_name)
        ensure_dir(out_rep_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        experiments = [
            ("sub_train_val", True, True)
        ]

        #("no_subsampling", False, False),
        #("sub_train", True, False),
        for tag, st, sv in experiments:

            exp_name = f"{timestamp}_{rep_name}_{tag}"
            exp_dir  = os.path.join(out_rep_dir, exp_name)
            ensure_dir(exp_dir)

            model = FCN()

            model = train_loop(
                model,
                train_ds,
                val_ds,
                exp_name,
                exp_dir,
                subs_train=st,
                subs_val=sv
            )

            results = evaluate_and_save(model, test_ds, os.path.join(exp_dir, exp_name))

            print(f"\n--- RESULTADOS ({tag}) ---")
            print("Seg:", results["segment"])
            print("Pat:", results["patient"])
            print("------------------------------------")

    print("\n=========== EXPERIMENTO TERMINADO ===========\n")


if __name__ == "__main__":
    main()
