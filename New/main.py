# main.py
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# --- TUS IMPORTS ---
from dataset import SegmentDataset
from model import FCN
from loss import FocalLoss  # <--- NUEVO: Tu archivo loss.py

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
    PATIENCE,
    SAVE_MODEL,
    WEIGHT_DECAY,
    FOCAL_GAMMA
)

from sklearn.metrics import f1_score

# =======================================================================
# 1. SEMILLA PARA REPRODUCIBILIDAD
# =======================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Semilla fijada en: {seed}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# =======================================================================
# 2. EVALUACIÓN Y GUARDADO (Con UAR incluido)
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

    # --- MÉTRICAS SEGMENTO ---
    # Nota: metrics_segment (en utils corregido) ya devuelve UAR, Sens, Spec, etc.
    seg_m = metrics_segment(trues, preds)

    # --- MÉTRICAS PACIENTE ---
    real_per_patient = {}
    for i, pid in enumerate(pids):
        if pid not in real_per_patient:
            real_per_patient[pid] = int(trues[i])

    pred_per_patient = aggregate_predictions(pids, preds)
    pat_m = metrics_patient(real_per_patient, pred_per_patient)

    # --- GENERAR REPORTE TXT ---
    # Extraemos los valores con seguridad (.get) por si algo falla
    report_lines = [
        "=== METRICAS - SEGMENTO ===",
        f"Accuracy:      {seg_m.get('acc', 0):.4f}",
        f"F1 macro:      {seg_m.get('f1', 0):.4f}",
        f"UAR (Avg Rec): {seg_m.get('uar', 0):.4f}",  # <--- RECOMENDADO ASESOR
        f"Sensibilidad:  {seg_m.get('sensitivity', 0):.4f}",
        f"Especificidad: {seg_m.get('specificity', 0):.4f}",
        f"Confusion:     {seg_m.get('confusion', 'N/A')}",
        "",
        "=== METRICAS - PACIENTE ===",
        f"Accuracy:      {pat_m.get('acc', 0):.4f}",
        f"F1 macro:      {pat_m.get('f1', 0):.4f}",
        f"UAR (Avg Rec): {pat_m.get('uar', 0):.4f}",  # <--- RECOMENDADO ASESOR
        f"Sensibilidad:  {pat_m.get('sensitivity', 0):.4f}",
        f"Especificidad: {pat_m.get('specificity', 0):.4f}",
        f"Confusion:     {pat_m.get('confusion', 'N/A')}",
        ""
    ]

    with open(out_prefix + "_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {"segment": seg_m, "patient": pat_m}

# =======================================================================
# 3. PINTAR CURVAS
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
# 4. ENTRENAMIENTO (Training Loop)
# =======================================================================
def train_loop(model, train_dataset, val_dataset, experiment_name, out_dir,
               subs_train=False, subs_val=False):

    print(f"\n=== Entrenando experimento: {experiment_name} ===")

    # --- Subsampling Train ---
    if subs_train:
        idx = patient_subsampling(train_dataset)
        train_ds = Subset(train_dataset, idx)
        print(f"[INFO] Train con subsampling activado. Muestras: {len(train_ds)}")
    else:
        train_ds = train_dataset

    # --- Subsampling Val (Generalmente debe ser False) ---
    if subs_val:
        idx = patient_subsampling(val_dataset)
        val_ds = Subset(val_dataset, idx)
        print(f"[INFO] Val con subsampling activado. Muestras: {len(val_ds)}")
    else:
        val_ds = val_dataset

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.to(DEVICE)

    # --- OPTIMIZADOR Y PÉRDIDA ---
    # Weight decay ayuda a evitar overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)

    best_metric = -1
    patience_count = 0
    best_state = None

    history = {
        "train_loss": [], "val_loss": [], 
        "train_acc": [], "val_acc": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0; correct = 0; total = 0

        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
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

        # --- VALIDACIÓN ---
        model.eval()
        val_loss = 0; val_correct = 0; val_total = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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
        
        # Usamos F1 Macro como métrica principal para Early Stopping
        # (Podrías cambiarlo a UAR si prefieres priorizar recall balanceado)
        val_metric = f1_score(y_true, y_pred, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:03d} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Val F1={val_metric:.4f}")

        # --- EARLY STOPPING ---
        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping activado.")
            break

    # Cargar mejor modelo
    if best_state:
        model.load_state_dict(best_state)

    # Guardar resultados
    plot_curves(history, os.path.join(out_dir, f"{experiment_name}_curves.png"), experiment_name)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{experiment_name}_model.pt"))
    
    # Copia de seguridad si se configuró SAVE_MODEL
    if 'SAVE_MODEL' in globals() and SAVE_MODEL and isinstance(SAVE_MODEL, str):
        try:
            os.makedirs(SAVE_MODEL, exist_ok=True)
            save_path2 = os.path.join(SAVE_MODEL, f"{experiment_name}_model.pt")
            torch.save(model.state_dict(), save_path2)
            print(f"Modelo COPIADO en ruta global: {save_path2}")
        except Exception as e:
            print(f"Error guardando copia en SAVE_MODEL: {e}")

    return model

# =======================================================================
# 5. MAIN
# =======================================================================
def main():
    # Fijar semilla al inicio de todo
    set_seed(42)

    print("\n=========== INICIO EXPERIMENTO COMPLETO ===========")

    for rep in REPRESENTATIONS:
        rep_name = os.path.basename(rep)
        train_dir = os.path.join(rep, "train")
        val_dir   = os.path.join(rep, "val")
        test_dir  = os.path.join(rep, "test")

        print(f"\n\n=== Cargando datos de representación: {rep_name} ===")

        # Asumimos que dataset.py ya está corregido (pd.read_csv)
        train_ds = SegmentDataset(train_dir)
        val_ds   = SegmentDataset(val_dir)
        test_ds  = SegmentDataset(test_dir)

        out_rep_dir = os.path.join(RESULTS_ROOT, rep_name)
        ensure_dir(out_rep_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CONFIGURACIÓN DEL EXPERIMENTO
        # ("Nombre", Subsampling_Train, Subsampling_Val)
        experiments = [
            ("sub_train_only", True, False) # <--- LA CONFIGURACIÓN CORRECTA
        ]

        for tag, st, sv in experiments:
            exp_name = f"{timestamp}_{rep_name}_{tag}"
            exp_dir  = os.path.join(out_rep_dir, exp_name)
            ensure_dir(exp_dir)

            model = FCN()
            
            # Entrenar
            model = train_loop(
                model, train_ds, val_ds, exp_name, exp_dir,
                subs_train=st, subs_val=sv
            )

            # Evaluar en TEST (Nunca subsampling aquí)
            results = evaluate_and_save(model, test_ds, os.path.join(exp_dir, exp_name))

            print(f"\n--- RESULTADOS TEST ({tag}) ---")
            print("Seg:", results["segment"])
            print("Pat:", results["patient"])
            print("------------------------------------")

    print("\n=========== EXPERIMENTO TERMINADO ===========\n")

if __name__ == "__main__":
    main()