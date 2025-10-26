# search_params.py
import os, json, time
import torch, optuna
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from common_meda import (
    DATASETS_INFO, DATA_ROOTS_3D,
    CombinedDataset, collate_mixed, make_subset,
    MultiTaskNet, build_labels_for_dataset, compute_loss_dataset, get_transforms
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
HPO_MAX_SAMPLES = 50_000
OPTUNA_TRIALS = 100
EPOCHS_PER_TRIAL = 1
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Datasets (sin DDP)
    train_dataset = CombinedDataset(
        DATASETS_INFO.keys(), 'train.pt',
        data_roots=DATA_ROOTS_3D, use_25d=True, mode_25d="triplet",
        stride_25d=1, out_size=(64,64), transform=get_transforms(train=True)
    )
    val_dataset = CombinedDataset(
        DATASETS_INFO.keys(), 'val.pt',
        data_roots=DATA_ROOTS_3D, use_25d=True, mode_25d="triplet",
        stride_25d=1, out_size=(64,64), transform=get_transforms(train=False)
    )

    def objective(trial):
        train_ds = make_subset(train_dataset, HPO_MAX_SAMPLES)
        val_ds   = make_subset(val_dataset,   HPO_MAX_SAMPLES // 5)

        model = MultiTaskNet(trial).to(DEVICE)

        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

        # grupos de LR (backbone menor)
        bb_params = list(model.backbone.parameters())
        head_params = [p for n,p in model.named_parameters() if not n.startswith('backbone.')]
        optimizer = getattr(optim, optimizer_name)([
            {'params': bb_params,   'lr': lr * 0.2},
            {'params': head_params, 'lr': lr * 1.0},
        ])

        # CE ponderada para classifier de dataset (frecuencia)
        counts = []
        for name in DATASETS_INFO.keys():
            _, labels = train_dataset.data_cache[name]
            counts.append(len(labels))
        counts = torch.tensor(counts, dtype=torch.float32, device=DEVICE)
        ds_weights = (counts.sum() / counts).clamp(min=1.0)
        ds_weights = ds_weights / ds_weights.mean()
        ce_dataset = nn.CrossEntropyLoss(weight=ds_weights)

        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=False, collate_fn=collate_mixed)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=False, collate_fn=collate_mixed)

        # warmup corto
        model.train()
        for k,(image, dataset_label, task_label) in enumerate(train_loader):
            image = image.to(DEVICE, non_blocking=True)
            dataset_label = dataset_label.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=('cuda' if torch.cuda.is_available() else 'cpu'),
                          dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
                dataset_pred, task_preds = model(image, dataset_label)
                total_loss = ce_dataset(dataset_pred, dataset_label)

                # tocar siempre todos los heads
                for idx, name in enumerate(DATASETS_INFO.keys()):
                    logits_all = task_preds[name]
                    total_loss = total_loss + 0.0 * logits_all.sum()
                    mask = (dataset_label == idx)
                    y, idxs = build_labels_for_dataset(task_label, mask, name, DEVICE)
                    if idxs.numel() > 0:
                        head_out = logits_all[mask]
                        total_loss = total_loss + compute_loss_dataset(name, head_out, y, ce_loss, bce_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer); scaler.update()
            if k >= 1: break

        # entrenamiento corto
        for epoch in range(EPOCHS_PER_TRIAL):
            model.train()
            for image, dataset_label, task_label in tqdm(train_loader, desc=f"[Trial {trial.number}] Epoch {epoch+1}/{EPOCHS_PER_TRIAL}", disable=False):
                image = image.to(DEVICE, non_blocking=True)
                dataset_label = dataset_label.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=('cuda' if torch.cuda.is_available() else 'cpu'),
                              dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
                    dataset_pred, task_preds = model(image, dataset_label)
                    total_loss = ce_dataset(dataset_pred, dataset_label)
                    for idx, name in enumerate(DATASETS_INFO.keys()):
                        logits_all = task_preds[name]
                        total_loss = total_loss + 0.0 * logits_all.sum()
                        mask = (dataset_label == idx)
                        y, idxs = build_labels_for_dataset(task_label, mask, name, DEVICE)
                        if idxs.numel() > 0:
                            head_out = logits_all[mask]
                            total_loss = total_loss + compute_loss_dataset(name, head_out, y, ce_loss, bce_loss)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer); scaler.update()

        # validación
        model.eval()
        total_correct, total_elems = 0, 0
        with torch.no_grad():
            for image, dataset_label, task_label in val_loader:
                image = image.to(DEVICE, non_blocking=True)
                dataset_label = dataset_label.to(DEVICE, non_blocking=True)
                dataset_pred, task_preds = model(image, dataset_label)
                for i, name in enumerate(DATASETS_INFO.keys()):
                    mask = (dataset_label == i)
                    y, idxs = build_labels_for_dataset(task_label, mask, name, DEVICE)
                    if idxs.numel() == 0: continue
                    logits = task_preds[name][mask]
                    # pred simple por dataset
                    info = DATASETS_INFO[name]
                    if info.get("type") == "multilabel":
                        y_pred = (torch.sigmoid(logits) >= 0.5).to(torch.long)
                        total_correct += (y_pred == y).sum().item()
                        total_elems += y.numel()
                    elif info.get("type") == "ordinal":
                        probs = torch.sigmoid(logits)
                        y_pred = (probs >= 0.5).sum(dim=1)
                        y = y.view(-1).long()
                        total_correct += (y_pred == y).sum().item()
                        total_elems += y.numel()
                    else:
                        y_pred = torch.argmax(logits, dim=1)
                        y = y.view(-1).long()
                        total_correct += (y_pred == y).sum().item()
                        total_elems += y.numel()

        acc = 100.0 * total_correct / max(total_elems, 1)
        trial.report(acc, step=EPOCHS_PER_TRIAL)
        print(f"[Trial {trial.number}] ValAcc={acc:.2f}")
        return acc

    print("INICIANDO BÚSQUEDA DE HIPERPARÁMETROS")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    best_params = study.best_params
    with open(os.path.join(ART_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f)
    print("[OK] Guardado artifacts/best_params.json")

    # Warm-start opcional (1-2 epochs muy cortas)
    model = MultiTaskNet(optuna.trial.FixedTrial(best_params)).to(DEVICE)
    torch.save(model.state_dict(), os.path.join(ART_DIR, "warmstart.pt"))
    print("[OK] Guardado artifacts/warmstart.pt")

if __name__ == "__main__":
    main()
