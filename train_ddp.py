# train_ddp.py
import os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from datetime import timedelta
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ===== extra para exportar métricas/CM =====
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from common_meda import (
    DATASETS_INFO, DATA_ROOTS_3D,
    CombinedDataset, BalancedByDatasetBatchSampler, collate_mixed,
    MultiTaskNet, build_labels_for_dataset, compute_loss_dataset, get_transforms
)

ART_DIR = "artifacts/artifacts50de3"
CKPT_DIR = "checkpoints250es224"
CKPT_PATH = os.path.join(CKPT_DIR, "final_model.pt")
CM_DIR = "matricesDG250es224"

BATCH_SIZE = 32
FINAL_TRAIN_EPOCHS = 250

def get_dist_env():
    env = os.environ
    if "LOCAL_RANK" in env:
        return int(env["LOCAL_RANK"]), int(env.get("RANK", 0)), int(env.get("WORLD_SIZE", 1)), True
    elif "SLURM_LOCALID" in env:
        return int(env["SLURM_LOCALID"]), int(env["SLURM_PROCID"]), int(env["SLURM_NTASKS"]), True
    else:
        return 0, 0, 1, False

local_rank, global_rank, world_size, is_distributed = get_dist_env()

if is_distributed:
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=local_rank,
            timeout=timedelta(minutes=30),
        )
    except TypeError:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=30),
        )
    DEVICE = torch.device(f"cuda:{local_rank}")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')

# ---------- helpers de métrica/CM ----------
def ordinal_predict(logits):
    probs = torch.sigmoid(logits)
    return (probs >= 0.5).sum(dim=1)

def predict_for_dataset(name, logits):
    info = DATASETS_INFO[name]
    ttype = info.get("type", "categorical")
    if ttype == "multilabel":
        probs = torch.sigmoid(logits)
        return (probs >= 0.3).to(torch.long)
    elif ttype == "ordinal":
        return ordinal_predict(logits)
    else:
        return torch.argmax(logits, dim=1)

def multilabel_f1(y_true, y_pred, eps=1e-9):
    tp = (y_true * y_pred).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)
    f1_per_class = (2 * tp) / (2 * tp + fp + fn + eps)
    return f1_per_class.mean().item()

def eval_f1_micro_universal(model, val_loader, device, six_map):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for image, dataset_label, task_label in val_loader:
            image = image.to(device, non_blocking=True)
            dataset_label = dataset_label.to(device, non_blocking=True)
            _, _, uni_logits = model(image, dataset_label)

            for idx, name in enumerate(DATASETS_INFO.keys()):
                if DATASETS_INFO[name].get("type") == "multilabel":
                    continue
                mask = (dataset_label == idx)
                if mask.sum() == 0:
                    continue
                y, idxs = build_labels_for_dataset(task_label, mask, name, device)
                y = y.view(-1).long()
                y6 = six_map[name].to(device)[y]
                pred6 = uni_logits[mask].argmax(dim=1)
                y_true_all.append(y6.detach().cpu().numpy())
                y_pred_all.append(pred6.detach().cpu().numpy())
    if len(y_true_all) == 0:
        return 0.0
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return f1_score(y_true, y_pred, average='micro', zero_division=0) * 100.0

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.ylabel('Etiqueta Verdadera'); plt.xlabel('Etiqueta Predicha')
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(filename); plt.close()
    print(f"[CM] guardada: {filename}")
# -------------------------------------------

def save_checkpoint(path, model, optimizer, scaler, epoch, best_params):
    if (not is_distributed) or (global_rank == 0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_to_save = model.module if is_distributed else model
        ckpt = {
            "epoch": epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_params": best_params,
            "datasets_info": DATASETS_INFO,
        }
        torch.save(ckpt, path)
        print(f"[CKPT] guardado en {path}")

def main():
    # Lee best params (provenientes de search_params.py)
    with open(os.path.join(ART_DIR, "best_params.json"), "r") as f:
        best_params = json.load(f)

    if (not is_distributed) or (global_rank == 0):
        print("\nENTRENANDO EL MODELO FINAL")

    # Datasets
    train_dataset = CombinedDataset(
        DATASETS_INFO.keys(), 'train.pt',
        data_roots=DATA_ROOTS_3D, use_25d=True, mode_25d="triplet",
        stride_25d=1, out_size=(224,224), transform=get_transforms(train=True)
    )
    val_dataset = CombinedDataset(
        DATASETS_INFO.keys(), 'val.pt',
        data_roots=DATA_ROOTS_3D, use_25d=True, mode_25d="triplet",
        stride_25d=1, out_size=(224,224), transform=get_transforms(train=False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_mixed,
        persistent_workers=False
    )

    full_train_dataset = ConcatDataset([train_dataset, val_dataset])
    balanced_bs = BalancedByDatasetBatchSampler(full_train_dataset, batch_size=BATCH_SIZE, drop_last=True)
    final_train_loader = DataLoader(
        full_train_dataset,
        batch_sampler=balanced_bs,
        num_workers=0, pin_memory=False, collate_fn=collate_mixed, persistent_workers=False
    )

    # ===== Modelo =====
    model = MultiTaskNet(params=best_params).to(DEVICE)

    # Warmstart: carga sólo claves compatibles (evita size mismatch)
    ws_path = os.path.join(ART_DIR, "warmstart.pt")
    if os.path.exists(ws_path):
        sd = torch.load(ws_path, map_location="cpu")
        model_named = model
        cur = model_named.state_dict()
        filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        missing, unexpected = model_named.load_state_dict(filtered, strict=False)
        if (not is_distributed) or (global_rank == 0):
            print(f"[Warmstart] loaded filtered. loaded={len(filtered)} missing={len(missing)}")

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    # ===== Optimizer (grupos) =====
    model_named = model.module if is_distributed else model
    bb_params   = list(model_named.backbone.parameters())
    head_params = [p for n,p in model_named.named_parameters() if not n.startswith('backbone.')]
    base_lr = float(best_params.get('lr', 1e-3))
    optimizer = getattr(optim, best_params.get('optimizer','Adam'))([
        {'params': bb_params,   'lr': base_lr * 0.2},
        {'params': head_params, 'lr': base_lr * 1.0},
    ])

    # CE ponderada para dataset classifier (según train real)
    counts = []
    for name in DATASETS_INFO.keys():
        _, labels = train_dataset.data_cache[name]
        counts.append(len(labels))
    counts = torch.tensor(counts, dtype=torch.float32, device=DEVICE)
    ds_weights = (counts.sum() / counts).clamp(min=1.0)
    ds_weights = ds_weights / ds_weights.mean()
    ce_dataset = nn.CrossEntropyLoss(weight=ds_weights)

    ce_loss  = nn.CrossEntropyLoss(label_smoothing=0.05)
    bce_loss = nn.BCEWithLogitsLoss()

    # ===== Entrenamiento =====
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)

    patience = 30
    patience_left = patience
    best_f1 = -1.0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_f1.pt")

    start_time = time.time()
    stop_all = False

    for epoch in range(FINAL_TRAIN_EPOCHS):
        if is_distributed:
            balanced_bs.set_epoch(epoch)
        model.train()

        for step, (image, dataset_label, task_label) in enumerate(tqdm(
            final_train_loader,
            desc=f"Epoch {epoch+1}/{FINAL_TRAIN_EPOCHS}",
            disable=(is_distributed and global_rank != 0)
        )):
            image = image.to(DEVICE, non_blocking=True)
            dataset_label = dataset_label.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=('cuda' if torch.cuda.is_available() else 'cpu'),
                        dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
                dataset_pred, task_preds, uni_logits = model(image, dataset_label)
                total_loss = ce_dataset(dataset_pred, dataset_label)

                for idx, name in enumerate(DATASETS_INFO.keys()):
                    logits_all = task_preds[name]  # [B, C]
                    total_loss = total_loss + 0.0 * logits_all.sum()  # tocar buffers
                    mask = (dataset_label == idx)
                    y, idxs = build_labels_for_dataset(task_label, mask, name, DEVICE)
                    if idxs.numel() > 0:
                        head_out = logits_all[mask]
                        total_loss = total_loss + compute_loss_dataset(name, head_out, y, ce_loss, bce_loss)

                        # universal 6-way (saltamos multilabel)
                        if DATASETS_INFO[name].get("type") != "multilabel":
                            y6 = (model.module if is_distributed else model).six_map[name].to(DEVICE)[y.view(-1).long()]
                            uni_out = uni_logits[mask]
                            total_loss = total_loss + ce_loss(uni_out, y6)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer); scaler.update()

        scheduler.step()
        save_checkpoint(CKPT_PATH, model, optimizer, scaler, epoch+1, best_params)

        # ====== VALIDACIÓN + EARLY STOP (solo rank0) ======
        if (not is_distributed) or (global_rank == 0):
            model_to_val = model.module if is_distributed else model
            f1_val = eval_f1_micro_universal(model_to_val, val_loader, DEVICE,
                                             (model.module if is_distributed else model).six_map)
            print(f"[Val-Uni] Epoch {epoch+1}: F1(micro)={f1_val:.2f}% (best={best_f1:.2f}%)")

            if f1_val > best_f1 + 1e-6:
                best_f1 = f1_val
                patience_left = patience
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save(model_to_val.state_dict(), best_ckpt_path)
                print(f"[CKPT] Nuevo mejor F1 -> guardado {best_ckpt_path}")
                stop_flag = 0
            else:
                patience_left -= 1
                stop_flag = 1 if patience_left == 0 else 0
                if stop_flag:
                    print("[EarlyStop] paciencia agotada — deteniendo entrenamiento.")

            stop_tensor = torch.tensor([stop_flag], device=DEVICE)
        else:
            stop_tensor = torch.tensor([0], device=DEVICE)

        if is_distributed:
            dist.broadcast(stop_tensor, src=0)

        if int(stop_tensor.item()) == 1:
            stop_all = True

        if stop_all:
            break

    if (not is_distributed) or (global_rank == 0):
        print(f"\nentrenamiento completado en {time.time() - start_time:.2f} s")

    # =================== EVALUACIÓN + EXPORTS ===================
    model_to_eval = model.module if is_distributed else model
    model_to_eval.eval()

    if (not is_distributed) or (global_rank == 0):
        # loaders de test
        test_dataset = CombinedDataset(
            DATASETS_INFO.keys(), 'test.pt',
            data_roots=DATA_ROOTS_3D, use_25d=True, mode_25d="triplet",
            stride_25d=1, out_size=(224,224), transform=get_transforms(train=False)
        )

        # cargar el mejor por F1 universal si existe (antes de evaluar)
        if os.path.exists(best_ckpt_path):
            (model.module if is_distributed else model).load_state_dict(
                torch.load(best_ckpt_path, map_location="cpu")
            )
            print(f"[LOAD] Evaluando con best F1: {best_ckpt_path}")
        else:
            print("[WARN] No se encontró best_f1.pt; se evaluará el último estado.")

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_mixed,
            persistent_workers=False
        )

        print("\nEVALUACIÓN SOBRE EL CONJUNTO DE PRUEBA (per-dataset)")
        all_dataset_true, all_dataset_pred = [], []
        all_task_true = {name: [] for name in DATASETS_INFO}
        all_task_pred = {name: [] for name in DATASETS_INFO}

        with torch.no_grad():
            for image, dataset_label, task_label in tqdm(test_loader, desc="Evaluando en Test (rank0)"):
                image = image.to(DEVICE, non_blocking=True)
                dataset_label = dataset_label.to(DEVICE, non_blocking=True)
                dataset_logits, task_logits, uni_logits = model_to_eval(image, dataset_label)

                # pred del clasificador de dataset (para CM de dominios)
                dataset_pred_labels = torch.argmax(dataset_logits, dim=1)
                all_dataset_true.extend(dataset_label.cpu().numpy().tolist())
                all_dataset_pred.extend(dataset_pred_labels.cpu().numpy().tolist())

                # preds por dataset
                for idx, name in enumerate(DATASETS_INFO.keys()):
                    mask = (dataset_label == idx)
                    y, idxs = build_labels_for_dataset(task_label, mask, name, DEVICE)
                    if idxs.numel() == 0:
                        continue
                    logits = task_logits[name][mask]
                    y_pred = predict_for_dataset(name, logits)

                    ttype = DATASETS_INFO[name].get("type", "categorical")
                    if ttype == "multilabel":
                        all_task_true[name].append(y.cpu())
                        all_task_pred[name].append(y_pred.cpu())
                    else:
                        y = y.view(-1).long()
                        all_task_true[name].append(y.cpu())
                        all_task_pred[name].append(y_pred.cpu())

        # ---- imprimir métricas por dataset ----
        print("\nResultados por dataset (Test):")
        for name in DATASETS_INFO.keys():
            t_list = all_task_true[name]
            p_list = all_task_pred[name]
            if len(t_list) == 0:
                print(f"  {name}: (sin muestras)")
                continue

            y_true = torch.cat(t_list, dim=0)
            y_pred = torch.cat(p_list, dim=0)
            ttype = DATASETS_INFO[name].get("type", "categorical")

            if ttype == "multilabel":
                f1 = multilabel_f1(y_true, y_pred)
                acc = (y_true.eq(y_pred).all(dim=1).float().mean().item()) * 100
                print(f"  {name}: Accuracy = {acc:.2f}%,  F1 (micro-promedio clases) = {100*f1:.2f}%")
            else:
                acc = (y_true.numpy() == y_pred.numpy()).mean() * 100
                f1 = f1_score(y_true.numpy(), y_pred.numpy(), average='micro', zero_division=0) * 100
                print(f"  {name}: Accuracy = {acc:.2f}%,  F1 (micro) = {f1:.2f}%")

        # ---- EVAL UNIVERSAL 6-way (aprox. protocolo del reto) ----
        print("\nEVAL UNIVERSAL 6-way (aprox. protocolo del reto)")
        y_true_u, y_pred_u = [], []
        with torch.no_grad():
            for image, dataset_label, task_label in tqdm(test_loader, desc="Eval Universal"):
                image = image.to(DEVICE, non_blocking=True)
                dataset_label = dataset_label.to(DEVICE, non_blocking=True)
                _, _, uni_logits = model_to_eval(image, dataset_label)
                for idx, name in enumerate(DATASETS_INFO.keys()):
                    if DATASETS_INFO[name].get("type") == "multilabel":
                        continue
                    mask = (dataset_label == idx)
                    if mask.sum() == 0: 
                        continue
                    y, _ = build_labels_for_dataset(task_label, mask, name, DEVICE)
                    y6 = (model.module if is_distributed else model).six_map[name].to(DEVICE)[y.view(-1).long()]
                    pred6 = uni_logits[mask].argmax(dim=1)
                    y_true_u.append(y6.cpu().numpy())
                    y_pred_u.append(pred6.cpu().numpy())
        if len(y_true_u):
            y_true_u = np.concatenate(y_true_u); y_pred_u = np.concatenate(y_pred_u)
            acc_u = (y_true_u == y_pred_u).mean() * 100
            f1_u  = f1_score(y_true_u, y_pred_u, average='micro', zero_division=0) * 100
            print(f"[Universal6] Accuracy={acc_u:.2f}%  F1(micro)={f1_u:.2f}%")

        # ---- matrices de confusión (opcionales) ----
        os.makedirs(CM_DIR, exist_ok=True)
        plot_confusion_matrix(
            all_dataset_true, all_dataset_pred,
            class_names=list(DATASETS_INFO.keys()),
            title='CM: Identificación de Dataset',
            filename=os.path.join(CM_DIR, 'cm_datasets.png')
        )
        for name in DATASETS_INFO.keys():
            if DATASETS_INFO[name].get("type") == "multilabel":
                continue
            t_list = all_task_true[name]
            p_list = all_task_pred[name]
            if len(t_list) == 0:
                continue
            y_true = torch.cat(t_list, dim=0).numpy()
            y_pred = torch.cat(p_list, dim=0).numpy()
            class_names = [f"Clase {i}" for i in range(DATASETS_INFO[name]["num_classes"])]
            plot_confusion_matrix(
                y_true, y_pred,
                class_names=class_names,
                title=f"CM: {name}",
                filename=os.path.join(CM_DIR, f"cm_{name}.png")
            )

        # ---- exportar state_dict “limpio” ----
        model_sd_path = os.path.join(CKPT_DIR, "final_model_state_dict.pt")
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(model_to_eval.state_dict(), model_sd_path)
        print(f"[MODEL] state_dict exportado en {model_sd_path}")

        # ===================================================================
        # =============== ONNX: SOLO LA CABEZA UNIVERSAL (6) ================
        # ===================================================================
        # Obtiene referencia a la cabeza universal (acepta universal_head_6 o universal_head)
        head6 = getattr(model_to_eval, "universal_head_6", None)
        if head6 is None:
            head6 = getattr(model_to_eval, "universal_head", None)
        if head6 is None:
            raise AttributeError("No se encontró la cabeza universal de 6 clases (universal_head_6 o universal_head).")

        # Wrapper que expone SOLO la cabeza (entrada: features del backbone)
        class Universal6End2End(nn.Module):
            def _init_(self, model_backbone, head6, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
                super()._init_()
                self.backbone = model_backbone
                self.head6 = head6
                self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
                self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))
            def forward(self, x):
                x = (x - self.mean) / self.std
                feats = self.backbone(x, domain_idx=None)
                logits = self.head6(feats)
                return logits
        
        head6 = getattr(model_to_eval, "universal_head_6", None) or getattr(model_to_eval, "universal_head", None)
                if head6 is None: raise AttributeError("No se encontró la cabeza universal de 6 clases.")
                e2e = Universal6End2End(model_to_eval.backbone, head6).to(DEVICE).eval()
                dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
                onnx_path = os.path.join(CKPT_DIR, "universal6_end2end.onnx")
                torch.onnx.export(
                    e2e, dummy, onnx_path,
                    input_names=["input"], output_names=["logits6"],
                    opset_version=13,
                    dynamic_axes={"input": {0: "batch"}, "logits6": {0: "batch"}},
                    do_constant_folding=True
                )
                print(f"[ONNX] {onnx_path}")

    if is_distributed:
        dist.barrier()

if __name__ == "__main__":
    try:
        main()
    finally:
        if is_distributed:
            try:
                torch.cuda.synchronize()
                dist.barrier()
            finally:
                dist.destroy_process_group()
