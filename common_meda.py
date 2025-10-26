# common_meda.py
import os, math, json, random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, Sampler
from torchvision import models, transforms as T
from sklearn.metrics import confusion_matrix
import numpy as np

# ===================== Configs compartidas =====================
DATASETS_INFO = {
    "PathMNIST": {"num_classes": 9, "type": "categorical"},
    "DermaMNIST": {"num_classes": 7, "type": "categorical"},
    "OCTMNIST": {"num_classes": 4, "type": "categorical"},
    "PneumoniaMNIST": {"num_classes": 2, "type": "categorical"},
    "BreastMNIST": {"num_classes": 2, "type": "categorical"},
    "BloodMNIST": {"num_classes": 8, "type": "categorical"},
    "TissueMNIST": {"num_classes": 8, "type": "categorical"},
    "OrganAMNIST": {"num_classes": 11, "type": "categorical"},
    "OrganCMNIST": {"num_classes": 11, "type": "categorical"},
    "OrganSMNIST": {"num_classes": 11, "type": "categorical"},
    "ChestMNIST": {"num_classes": 14, "type": "multilabel"},
    "RetinaMNIST": {"num_classes": 5, "type": "ordinal"},

    "AdrenalMNIST3D":  {"num_classes": 2,  "type": "categorical"},
    "FractureMNIST3D": {"num_classes": 3,  "type": "categorical"},
    "NoduleMNIST3D":   {"num_classes": 2,  "type": "categorical"},
    "OrganMNIST3D":    {"num_classes": 11, "type": "categorical"},
    "SynapseMNIST3D":  {"num_classes": 2,  "type": "categorical"},
    "VesselMNIST3D":   {"num_classes": 2,  "type": "categorical"}
}

DATA_ROOTS_3D = {
    "AdrenalMNIST3D":  "../medmnist/data3d",
    "FractureMNIST3D": "../medmnist/data3d",
    "NoduleMNIST3D":   "../medmnist/data3d",
    "OrganMNIST3D":    "../medmnist/data3d",
    "SynapseMNIST3D":  "../medmnist/data3d",
    "VesselMNIST3D":   "../medmnist/data3d",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ===================== Utils Img =====================
def _resize2d(x, size=(64, 64)):
    if x.dim() == 2: x = x.unsqueeze(0)
    x = x.unsqueeze(0).float()
    y = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    return y.squeeze(0)

def _minmax_norm(x, eps=1e-6):
    x = x.float()
    mn = x.amin(dim=(-2,-1), keepdim=True)
    mx = x.amax(dim=(-2,-1), keepdim=True)
    return (x - mn) / (mx - mn + eps)

def to_25d_from_volume(vol, mode="triplet", stride=1, out_size=(64,64)):
    if vol.dim() == 4: vol = vol[0]
    D, H, W = vol.shape
    if mode == "triplet":
        c = D // 2
        i0 = max(c - stride, 0); i2 = min(c + stride, D - 1)
        ch0, ch1, ch2 = vol[i0], vol[c], vol[i2]
        out = torch.stack([ch0, ch1, ch2], dim=0)
        out = _minmax_norm(out)
        out = _resize2d(out, out_size)
        return out
    elif mode == "mip":
        axial    = vol.max(dim=0).values
        sagittal = vol.max(dim=2).values
        coronal  = vol.max(dim=1).values
        sagittal = F.interpolate(sagittal[None,None].float(), size=(H,W), mode='bilinear', align_corners=False).squeeze()
        coronal  = F.interpolate(coronal [None,None].float(), size=(H,W), mode='bilinear', align_corners=False).squeeze()
        out = torch.stack([axial, sagittal, coronal], dim=0)
        out = _minmax_norm(out)
        out = _resize2d(out, out_size)
        return out
    else:
        raise ValueError(f"Modo 2.5D no soportado: {mode}")

# ===================== Datasets =====================
class CombinedDataset(Dataset):
    def __init__(self, dataset_names, filename, transform=None, data_roots=None,
                 use_25d=True, mode_25d="triplet", stride_25d=1, out_size=(64,64)):
        self.dataset_names = list(dataset_names)
        self.samples = []
        self.data_cache = {}
        self.transform = transform
        self.data_roots = data_roots or {}
        self.use_25d = use_25d
        self.mode_25d = mode_25d
        self.stride_25d = stride_25d
        self.out_size = out_size

        for dataset in self.dataset_names:
            base = self.data_roots.get(dataset, "../medmnist/data")
            path_pt = os.path.join(base, dataset, filename)
            path_tp = path_pt[:-2] + "tp" if path_pt.endswith(".pt") else path_pt
            if os.path.exists(path_pt):
                imgs, labels = torch.load(path_pt, map_location="cpu")
            elif os.path.exists(path_tp):
                imgs, labels = torch.load(path_tp, map_location="cpu")
            else:
                raise FileNotFoundError(f"No se encontró {path_pt} ni {path_tp}")
            self.data_cache[dataset] = (imgs, labels)
        for idx, dataset_name in enumerate(self.dataset_names):
            num_samples = len(self.data_cache[dataset_name][1])
            for i in range(num_samples):
                self.samples.append((dataset_name, idx, i))

    def __getitem__(self, idx):
        dataset, dataset_label, i = self.samples[idx]
        imgs, labels = self.data_cache[dataset]
        x, y = imgs[i], labels[i]
        if self.use_25d and (x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1)):
            x = to_25d_from_volume(x, mode=self.mode_25d, stride=self.stride_25d, out_size=self.out_size)
        else:
            if x.dim() == 2: x = x.unsqueeze(0)
            if x.shape[0] == 1: x = x.repeat(3, 1, 1)
            x = _resize2d(x, self.out_size)
        x = x.contiguous().clone()
        if torch.is_tensor(y): y = y.contiguous().clone()
        if self.transform: x = self.transform(x)
        return x, dataset_label, y

    def __len__(self): return len(self.samples)

# ===================== Sampler balanceado (DDP-safe) =====================
class BalancedByDatasetBatchSampler(Sampler):
    """
    Balancea por dataset y evita deadlocks DDP (len == batches emitidos en el rank).
    """
    def __init__(self, dataset, batch_size, drop_last=True):
        self.ds = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

        buckets = defaultdict(list)

        def add_from_combined(ds_obj, base_offset: int):
            for local_idx, (_name, dataset_label, _i) in enumerate(ds_obj.samples):
                global_idx = base_offset + local_idx
                buckets[int(dataset_label)].append(global_idx)

        if isinstance(self.ds, ConcatDataset):
            lengths = [len(d) for d in self.ds.datasets]
            offsets = np.cumsum([0] + lengths[:-1]).tolist()
            for off, child in zip(offsets, self.ds.datasets):
                if hasattr(child, "samples"):
                    add_from_combined(child, off)
                else:
                    for li in range(len(child)):
                        _, dataset_label, _ = child[li]
                        buckets[int(dataset_label)].append(off + li)
        else:
            if not hasattr(self.ds, "samples"):
                raise ValueError("Requiere CombinedDataset o ConcatDataset de CombinedDataset.")
            add_from_combined(self.ds, 0)

        self.buckets = {k: torch.tensor(v, dtype=torch.long) for k, v in buckets.items()}
        self.domains = sorted(self.buckets.keys())
        if len(self.domains) == 0:
            raise ValueError("No hay dominios/datasets para balancear.")

        self.per_dom = max(1, self.batch_size // max(1, len(self.domains)))
        self.batch_size_eff = self.per_dom * len(self.domains)
        min_per_dom = min(len(v) for v in self.buckets.values())
        self.num_batches_total = math.floor(min_per_dom / self.per_dom)
        if not drop_last and self.num_batches_total == 0:
            self.num_batches_total = 1

        self._epoch = 0
        self._local_batches_count = None

    def set_epoch(self, epoch:int):
        self._epoch = int(epoch)
        self._local_batches_count = None

    def __len__(self):
        if self._local_batches_count is not None:
            return self._local_batches_count
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
            q, r = divmod(self.num_batches_total, world)
            self._local_batches_count = q + (1 if rank < r else 0)
            return self._local_batches_count
        self._local_batches_count = int(self.num_batches_total)
        return self._local_batches_count

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._epoch + 12345)
        perms = {k: v[torch.randperm(len(v), generator=g)] for k, v in self.buckets.items()}
        ptrs  = {k: 0 for k in self.buckets}
        all_batches = []
        for _ in range(self.num_batches_total):
            batch = []
            for k in self.domains:
                start = ptrs[k]; end = start + self.per_dom
                if end > len(perms[k]):
                    perms[k] = perms[k][torch.randperm(len(perms[k]), generator=g)]
                    start, end = 0, self.per_dom
                batch.extend(perms[k][start:end].tolist())
                ptrs[k] = end
            if len(batch) < self.batch_size_eff and self.drop_last:
                break
            all_batches.append(batch)

        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            rank  = dist.get_rank()
            local_batches = all_batches[rank::world]
        else:
            local_batches = all_batches

        self._local_batches_count = len(local_batches)
        for b in local_batches:
            yield b

# ===================== Modelo =====================
try:
    from torchvision.models import ResNet18_Weights
    _HAS_WEIGHTS_ENUM = True
except Exception:
    _HAS_WEIGHTS_ENUM = False

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps
    def forward(self, x, domain_idx=None):
        if (not self.training) or (torch.rand(1, device=x.device) > self.p):
            return x
        B, C, H, W = x.size()
        x_ = x.view(B, C, -1)
        mu  = x_.mean(-1, keepdim=True)
        var = x_.var(-1, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        x_norm = (x_ - mu) / sigma
        perm = torch.randperm(B, device=x.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha).sample((B,1,1)).to(x.device)
        mu2, sigma2 = mu[perm], sigma[perm]
        x_mix = x_norm * (beta*sigma + (1-beta)*sigma2) + (beta*mu + (1-beta)*mu2)
        return x_mix.view(B, C, H, W)

class ResNetBackboneWithMixStyle(nn.Module):
    def __init__(self, mixstyle: MixStyle, freeze_stem=True):
        super().__init__()
        if _HAS_WEIGHTS_ENUM:
            resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18(pretrained=True)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool= resnet.avgpool
        self.mixstyle = mixstyle
        if freeze_stem:
            for p in self.stem.parameters(): p.requires_grad = False

    def forward(self, x, domain_idx=None):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.mixstyle(x, domain_idx)
        x = self.layer2(x)
        x = self.mixstyle(x, domain_idx)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class MultiTaskNet(nn.Module):
    def __init__(self, trial=None, params: dict=None):
        super().__init__()

        if params is not None:
            fc_neurons  = int(params.get('fc_neurons', 256))
            freeze_stem = bool(params.get('freeze_stem', True))
        elif trial is not None:
            fc_neurons  = trial.suggest_categorical('fc_neurons', [128, 256, 512])
            freeze_stem = trial.suggest_categorical('freeze_stem', [True, False])
        else:
            fc_neurons, freeze_stem = 256, True

        self.mixstyle = MixStyle(p=0.5, alpha=0.3)
        self.backbone = ResNetBackboneWithMixStyle(self.mixstyle, freeze_stem=freeze_stem)
        trunk_output_size = 512  # ResNet18

        self.dataset_classifier = nn.Sequential(
            nn.Linear(trunk_output_size, fc_neurons), nn.ReLU(),
            nn.Linear(fc_neurons, len(DATASETS_INFO))
        )
        self.task_classifiers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(trunk_output_size, fc_neurons),
                nn.ReLU(),
                nn.Linear(fc_neurons, (info["num_classes"] - 1) if info.get("type") == "ordinal" else info["num_classes"])
            )
            for name, info in DATASETS_INFO.items()
        })

        # ---- HEAD UNIVERSAL 6-CLASES
        self.universal_head_6 = nn.Sequential(
            nn.Linear(trunk_output_size, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )

        # ---- TABLA DE MAPEOS A 6 BINS (balanceado por clase)
        self.six_map = {}
        for name, info in DATASETS_INFO.items():
            K = info["num_classes"]
            self.six_map[name] = torch.tensor([(c * 6) // K for c in range(K)], dtype=torch.long)

    def forward(self, x, dataset_label=None):
        features = self.backbone(x, domain_idx=dataset_label)
        dataset_logits = self.dataset_classifier(features)
        tasks = {name: head(features) for name, head in self.task_classifiers.items()}
        universal_logits = self.universal_head_6(features)
        return dataset_logits, tasks, universal_logits

# ===================== Helpers training =====================
def collate_mixed(batch):
    xs, dls, ys = zip(*batch)
    images = torch.stack(xs, dim=0)
    dataset_labels = torch.as_tensor(dls, dtype=torch.long)
    ys_t = []
    for y in ys:
        if not torch.is_tensor(y): y = torch.as_tensor(y)
        ys_t.append(y)
    return images, dataset_labels, ys_t

def build_labels_for_dataset(task_labels_list, mask, dataset_name, device):
    idxs = mask.nonzero(as_tuple=True)[0]
    if idxs.numel() == 0: return None, idxs
    ys = [task_labels_list[k] for k in idxs.tolist()]
    task_type = DATASETS_INFO[dataset_name].get("type", "categorical")
    if task_type == "multilabel":
        ys = [y if y.ndim > 0 else y.unsqueeze(0) for y in ys]
        y = torch.stack(ys, dim=0).to(device)
        return y, idxs
    else:
        ys = [(y.view(-1)[0] if y.ndim > 0 else y) for y in ys]
        y = torch.stack(ys, dim=0).to(device)
        return y, idxs

def ordinal_targets(y, K):
    y = y.view(-1).long()
    t = (torch.arange(K - 1, device=y.device).unsqueeze(0) < y.unsqueeze(1))
    return t.long()

def compute_loss_dataset(name, head_out, y, ce_loss, bce_loss):
    info = DATASETS_INFO[name]
    ttype = info.get("type", "categorical")
    K = info["num_classes"]
    if ttype == "multilabel":
        return bce_loss(head_out, y.float())
    elif ttype == "ordinal":
        y_ord = ordinal_targets(y, K)
        return bce_loss(head_out, y_ord.float())
    else:
        return ce_loss(head_out, y.long())

def make_subset(ds, max_samples):
    n = len(ds)
    if n <= max_samples: return ds
    idx = random.sample(range(n), max_samples)
    return Subset(ds, idx)

def get_transforms(train=True):
    if train:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=12),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
