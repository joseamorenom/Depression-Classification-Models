# train.py

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score
from config import DEVICE, LR, EPOCHS, BATCH_SIZE, PATIENCE
from subsampling import patient_subsampling


def train_model(
    model,
    train_set,
    val_set,
    use_sub_train=False,
    use_sub_val=False
):
    print("\nIniciando entrenamiento...")

    # ============================================================
    #  SUBSAMPLING TRAIN
    # ============================================================
    if use_sub_train:
        print(">>> Aplicando subsampling en TRAIN")
        train_idx = patient_subsampling(train_set)
        train_set = Subset(train_set, train_idx)
        print(f"Segmentos restantes en train: {len(train_set)}")

    # ============================================================
    #  SUBSAMPLING VAL
    # ============================================================
    if use_sub_val:
        print(">>> Aplicando subsampling en VALIDATION")
        val_idx = patient_subsampling(val_set)
        val_set = Subset(val_set, val_idx)
        print(f"Segmentos restantes en val: {len(val_set)}")

    # ============================================================
    #  DATA LOADERS
    # ============================================================
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0
    patience_counter = 0
    best_state = None

    # ============================================================
    #  PARA GUARDAR HISTORIA DE ENTRENAMIENTO
    # ============================================================
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    # ============================================================
    #  LOOP DE ENTRENAMIENTO
    # ============================================================
    for epoch in range(1, EPOCHS + 1):

        # --------------------------------------------------------
        #  TRAIN
        # --------------------------------------------------------
        model.train()
        train_preds = []
        train_trues = []
        running_loss = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

            pred = torch.argmax(out, dim=1)
            train_preds.extend(pred.cpu().numpy())
            train_trues.extend(yb.cpu().numpy())

        train_loss = running_loss / len(train_set)
        train_acc = accuracy_score(train_trues, train_preds)

        # --------------------------------------------------------
        #  VALIDATION
        # --------------------------------------------------------
        model.eval()
        val_preds = []
        val_trues = []
        val_running_loss = 0

        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                out = model(xb)
                loss = criterion(out, yb)
                val_running_loss += loss.item() * xb.size(0)

                pred = torch.argmax(out, dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(yb.cpu().numpy())

        val_loss = val_running_loss / len(val_set)
        val_acc = accuracy_score(val_trues, val_preds)
        val_f1 = f1_score(val_trues, val_preds, average="macro")

        # Guardar historia
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(">>> EARLY STOPPING ACTIVADO")
            break

    model.load_state_dict(best_state)
    return model, history
