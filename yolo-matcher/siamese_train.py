#!/usr/bin/env python3
"""
Siamese/triplet training with DINO-style ViT embeddings (NO YOLO at train time)

Update summary
- Uses **pre‑cropped images only**; no detector calls.
- Positives: Mint vs Mint of the **same `product_name`**
- Negatives: prefer **Counterfeit of the same `product_name`**; fallback to any global counterfeit; if none exist, fallback to a different product's mint.
- Optional `--allowed-faces` (e.g., Face_Up Face_Down) to drop 'na' or other poses.

CLI Examples
------------
# Cropped images are in ./cropped_root/images/scans/...; products.json in ./dataset
python siamese_train.py \
  --root dataset \
  --images-root cropped_root \
  --products-json dataset/images/scans/products.json \
  --classes counterfeit mint \
  --allowed-faces Face_Up Face_Down \
  --epochs 10 --batch-size 32 --lr 3e-4 --num-workers 8 --device cuda:0

# If cropped images live under the same root as products.json
python siamese_train.py \
  --root dataset \
  --products-json dataset/images/scans/products.json \
  --classes counterfeit mint

"""
from __future__ import annotations
import json, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import timm
except Exception:
    timm = None

# ------------------------- Utilities -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl_or_json(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        return json.loads(path.read_text())


# -------------------- Dataset: Triplets with product_name semantics --------------------
@dataclass
class Item:
    img_path: Path
    condition: str  # "Mint" or "Counterfeit"
    product_name: str
    face: str


class ProductTripletDataset(Dataset):
    def __init__(self,
                 meta_root: Path,
                 images_root: Path,
                 products_json: Path,
                 classes: Tuple[str, str] = ("mint", "counterfeit"),
                 image_size: int = 336,
                 augment: bool = True,
                 allowed_faces: Optional[List[str]] = None):
        """
        meta_root: root containing products.json (usually original dataset root)
        images_root: root containing **pre-cropped** images mirroring the same structure
                     images_root/images/scans/<condition>/<filename>.jpg
        classes: keep only these conditions from products.json
        allowed_faces: optional whitelist, e.g. ["Face_Up", "Face_Down"] to drop 'na'
        """
        self.meta_root = Path(meta_root)
        self.images_root = Path(images_root)
        self.items: List[Item] = []
        self.image_size = image_size
        self.augment = augment

        data = read_jsonl_or_json(products_json)
        keep_conditions = {c.lower() for c in classes}
        face_whitelist = None if not allowed_faces else {f.lower() for f in allowed_faces}

        for rec in data:
            cond = str(rec.get("condition", "")).strip()
            cond_lower = cond.lower()
            if cond_lower not in keep_conditions:
                continue
            face = str(rec.get("face", "na"))
            if face_whitelist and face.lower() not in face_whitelist:
                continue
            # prefer explicit relative path from products.json if given
            path_rel = rec.get("path") or rec.get("jpg_filename")
            if not path_rel:
                continue
            img_path = self.images_root / "images" / "scans" / path_rel
            if not img_path.exists():
                alt = self.images_root / "images" / "scans" / cond_lower / rec.get("jpg_filename", "")
                if alt.exists():
                    img_path = alt
                else:
                    continue
            self.items.append(Item(img_path=img_path,
                                   condition=cond.capitalize(),
                                   product_name=str(rec.get("product_name", "")).strip(),
                                   face=face))

        # Build indices per product_name and condition
        self.by_product: Dict[str, Dict[str, List[int]]] = {}
        for idx, it in enumerate(self.items):
            d = self.by_product.setdefault(it.product_name, {})
            d.setdefault(it.condition, []).append(idx)

        # Precompute global counterfeit list for fallback
        self.global_counterfeits = [i for i, it in enumerate(self.items) if it.condition.lower() == "counterfeit"]

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.transpose(2, 0, 1)).float() / 255.0

    def __len__(self):
        return len(self.items)

    def _read_image(self, img_path: Path) -> np.ndarray:
        """Read full image and letterbox-pad to a square (keep aspect ratio).
        No random cropping. Output is image_size x image_size.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        H, W = img.shape[:2]
        # letterbox resize while preserving aspect ratio
        target = int(self.image_size)
        r = min(target / H, target / W)
        newH, newW = max(1, int(round(H * r))), max(1, int(round(W * r)))
        interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, (newW, newH), interpolation=interp)
        # pad to target size (centered)
        pad_top = (target - newH) // 2
        pad_bottom = target - newH - pad_top
        pad_left = (target - newW) // 2
        pad_right = target - newW - pad_left
        # neutral gray padding works well for X-ray style images
        pad_color = (114, 114, 114)
        img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=pad_color)
        if self.augment:
            if random.random() < 0.5:
                img_padded = cv2.flip(img_padded, 1)
            if random.random() < 0.15:
                hsv = cv2.cvtColor(img_padded, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] *= random.uniform(0.9, 1.1)
                hsv[..., 2] *= random.uniform(0.95, 1.05)
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                img_padded = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        return img_padded

    def _sample_positive(self, prod: str, exclude_idx: int) -> Optional[int]:
        mint_ids = self.by_product.get(prod, {}).get("Mint", [])
        mint_ids = [i for i in mint_ids if i != exclude_idx]
        if not mint_ids:
            return None
        return random.choice(mint_ids)

    def _sample_negative(self, prod: str) -> Optional[int]:
        # 1) Prefer counterfeit of the SAME product
        ct_ids = self.by_product.get(prod, {}).get("Counterfeit", [])
        if ct_ids:
            return random.choice(ct_ids)
        # 2) Fallback: any global counterfeit
        if self.global_counterfeits:
            return random.choice(self.global_counterfeits)
        # 3) Last resort: any other product mint
        return None

    def __getitem__(self, idx: int):
        anchor_it = self.items[idx]
        # Ensure anchor is Mint when possible
        if anchor_it.condition != "Mint":
            mint_ids = self.by_product.get(anchor_it.product_name, {}).get("Mint", [])
            if mint_ids:
                idx = random.choice(mint_ids)
                anchor_it = self.items[idx]
        pos_idx = self._sample_positive(anchor_it.product_name, exclude_idx=idx)
        neg_idx = self._sample_negative(anchor_it.product_name)
        # If a product has no other mint image, duplicate anchor for positive (still useful with augmentation)
        if pos_idx is None:
            pos_idx = idx
        # If truly no counterfeit exists anywhere, use a random other product mint as a weak negative
        if neg_idx is None:
            candidates = [i for i in range(len(self.items)) if i not in {idx, pos_idx}]
            neg_idx = random.choice(candidates) if candidates else idx

        a = self._read_image(anchor_it.img_path)
        p = self._read_image(self.items[pos_idx].img_path)
        n = self._read_image(self.items[neg_idx].img_path)
        return self._to_tensor(a), self._to_tensor(p), self._to_tensor(n)


# -------------------- Backbones (DINO family) --------------------
class DinoEncoder(nn.Module):
    def __init__(self, name: str = "dinov3_vitb14", pretrained: bool = True, out_dim: int = 768, trainable: bool = False, image_size: int = 336):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required to load a DINO-style ViT. Please `pip install timm`.")
        model_name = {
            "dinov3_vitb14": "vit_base_patch14_dinov2.lvd142m",
            "dinov3_vitl14": "vit_large_patch14_dinov2.lvd142m",
        }.get(name, "vit_base_patch14_dinov2.lvd142m")
        # NOTE: DINOv2 weights expect token pooling; using global_pool="avg" adds fc_norm
        # which mismatches checkpoint keys (fc_norm vs norm). So we avoid passing global_pool here.
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        self.out_dim = self.encoder.num_features
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)


class SiameseModel(nn.Module):
    def __init__(self, backbone_name: str = "dinov3_vitb14", proj_dim: int = 256, train_backbone: bool = False, image_size: int = 336):
        super().__init__()
        self.encoder = DinoEncoder(backbone_name, pretrained=True, trainable=train_backbone, image_size=image_size)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim), nn.GELU(),
            nn.Linear(self.encoder.out_dim, proj_dim)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.training or any(p.requires_grad for p in self.encoder.parameters())):
            z = self.encoder(x)
        z = self.proj(z)
        return F.normalize(z, dim=-1)

    def forward(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.embed(a), self.embed(p), self.embed(n)


# -------------------- Args & Training --------------------
@dataclass
class Args:
    root: Path
    images_root: Path
    products_json: Path
    classes: Tuple[str, str]
    image_size: int
    epochs: int
    batch_size: int
    lr: float
    backbone: str
    proj_dim: int
    train_backbone: bool
    num_workers: int
    device: str
    save_dir: Path
    allowed_faces: Optional[List[str]]
    viz_every: int
    viz_samples: int
    pca_subset: int
    viz_dir: Path


def parse_args() -> Args:
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', type=Path, required=True, help='meta root (where products.json schema originates)')
    p.add_argument('--images-root', type=Path, default=None, help='root with pre-cropped images mirroring images/scans/...; defaults to --root')
    p.add_argument('--products-json', type=Path, required=True)
    p.add_argument('--classes', nargs='+', default=['mint','counterfeit'])
    p.add_argument('--allowed-faces', nargs='+', default=None, help='whitelist faces e.g. Face_Up Face_Down (omit to use all, including na)')
    p.add_argument('--image-size', type=int, default=336)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--backbone', type=str, default='dinov3_vitb14')
    p.add_argument('--proj-dim', type=int, default=256)
    p.add_argument('--train-backbone', action='store_true')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save-dir', type=Path, default=Path('runs/siamese'))
    # viz args
    p.add_argument('--viz-every', type=int, default=1, help='save visualizations every N epochs')
    p.add_argument('--viz-samples', type=int, default=8, help='number of triplet panels to save per viz epoch')
    p.add_argument('--pca-subset', type=int, default=400, help='number of samples for PCA scatter (approx)')
    p.add_argument('--viz-dir', type=Path, default=None, help='override visualization dir (default: save_dir/viz)')

    a = p.parse_args()
    images_root = a.images_root if a.images_root is not None else a.root
    viz_dir = a.viz_dir if a.viz_dir is not None else (a.save_dir / 'viz')
    return Args(
        root=a.root, images_root=images_root, products_json=a.products_json, classes=tuple(a.classes),
        image_size=a.image_size, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        backbone=a.backbone, proj_dim=a.proj_dim, train_backbone=a.train_backbone,
        num_workers=a.num_workers, device=a.device, save_dir=a.save_dir, allowed_faces=a.allowed_faces,
        viz_every=a.viz_every, viz_samples=a.viz_samples, pca_subset=a.pca_subset, viz_dir=viz_dir
    )


# -------------------- Visualization helpers --------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _put_caption(img: np.ndarray, text: str, y: int = 24) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _hstack_images(imgs: List[np.ndarray], pad: int = 8, pad_color=(32, 32, 32)) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    w = sum(im.shape[1] for im in imgs) + pad * (len(imgs) - 1)
    out = np.full((h, w, 3), pad_color, dtype=np.uint8)
    x = 0
    for i, im in enumerate(imgs):
        out[0:im.shape[0], x:x+im.shape[1]] = im
        x += im.shape[1] + pad
    return out


def _sample_triplet_arrays(ds: ProductTripletDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    # mirror __getitem__ but return arrays + meta
    idx = random.randrange(len(ds))
    anchor_it = ds.items[idx]
    if anchor_it.condition != "Mint":
        mint_ids = ds.by_product.get(anchor_it.product_name, {}).get("Mint", [])
        if mint_ids:
            idx = random.choice(mint_ids)
            anchor_it = ds.items[idx]
    pos_idx = ds._sample_positive(anchor_it.product_name, exclude_idx=idx)
    neg_idx = ds._sample_negative(anchor_it.product_name)
    if pos_idx is None:
        pos_idx = idx
    if neg_idx is None:
        candidates = [i for i in range(len(ds.items)) if i not in {idx, pos_idx}]
        neg_idx = random.choice(candidates) if candidates else idx
    a = ds._read_image(anchor_it.img_path)
    p = ds._read_image(ds.items[pos_idx].img_path)
    n = ds._read_image(ds.items[neg_idx].img_path)
    meta = {
        'product_name': anchor_it.product_name,
        'anchor_path': str(anchor_it.img_path.name),
        'pos_path': str(ds.items[pos_idx].img_path.name),
        'neg_path': str(ds.items[neg_idx].img_path.name),
        'has_same_product_ct': neg_idx in ds.by_product.get(anchor_it.product_name, {}).get("Counterfeit", []),
    }
    return a, p, n, meta


def save_epoch_visuals(model: SiameseModel, ds: ProductTripletDataset, device: torch.device, out_root: Path, epoch: int, viz_samples: int = 8, pca_subset: int = 400):
    model.eval()
    out_dir = out_root / f"epoch_{epoch:03d}"
    _ensure_dir(out_dir)
    # 1) Triplet panels + distances
    rows = []
    for i in range(viz_samples):
        a_img, p_img, n_img, meta = _sample_triplet_arrays(ds)
        # embeddings
        with torch.no_grad():
            ta = ds._to_tensor(a_img).unsqueeze(0).to(device)
            tp = ds._to_tensor(p_img).unsqueeze(0).to(device)
            tn = ds._to_tensor(n_img).unsqueeze(0).to(device)
            za = model.embed(ta)
            zp = model.embed(tp)
            zn = model.embed(tn)
            d_ap = torch.cdist(za, zp, p=2).item()
            d_an = torch.cdist(za, zn, p=2).item()
        # panel
        cap_a = f"A | {meta['product_name']}"
        cap_p = f"P | d_ap={d_ap:.3f}"
        cap_n = f"N | d_an={d_an:.3f}"
        panel = _hstack_images([
            _put_caption(a_img, cap_a),
            _put_caption(p_img, cap_p),
            _put_caption(n_img, cap_n)
        ])
        cv2.imwrite(str(out_dir / f"triplet_{i:02d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        rows.append((i, meta['product_name'], d_ap, d_an, meta['anchor_path'], meta['pos_path'], meta['neg_path']))
    # write distances csv
    import csv
    with open(out_dir / 'triplet_distances.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx','product_name','d_ap','d_an','anchor','positive','negative'])
        w.writerows(rows)

    # 2) PCA scatter of a subset
    try:
        import matplotlib.pyplot as plt
        # sample subset indices
        n = min(pca_subset, len(ds.items))
        ids = random.sample(range(len(ds.items)), n)
        X = []
        conds = []
        for j in ids:
            img = ds._read_image(ds.items[j].img_path)
            x = ds._to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z = model.embed(x).cpu().numpy()[0]
            X.append(z)
            conds.append(ds.items[j].condition)
        X = np.stack(X, axis=0)
        # PCA via SVD
        Xc = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:2].T
        plt.figure(figsize=(6,6))
        for label in ['Mint','Counterfeit']:
            idxs = [i for i,c in enumerate(conds) if c==label]
            if idxs:
                plt.scatter(Z[idxs,0], Z[idxs,1], label=label, s=12)
        plt.legend()
        plt.title(f'PCA (epoch {epoch})')
        plt.tight_layout()
        plt.savefig(out_dir / 'pca_scatter.png', dpi=160)
        plt.close()
    except Exception as e:
        (out_dir / 'WARN_no_matplotlib.txt').write_text(f'Could not create PCA scatter: {e}').write_text(f'Could not create PCA scatter: {e}')


def train():
    cfg = parse_args()
    set_seed(1337)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dir(cfg.viz_dir)

    ds = ProductTripletDataset(meta_root=cfg.root, images_root=cfg.images_root, products_json=cfg.products_json,
                               classes=cfg.classes, image_size=cfg.image_size, augment=True,
                               allowed_faces=cfg.allowed_faces)

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=True, drop_last=True)

    model = SiameseModel(backbone_name=cfg.backbone, proj_dim=cfg.proj_dim, train_backbone=cfg.train_backbone, image_size=cfg.image_size)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != 'cpu' else 'cpu')
    model.to(device)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best = float('inf')

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, (a, p, n) in enumerate(loader, 1):
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                za, zp, zn = model(a, p, n)
                loss = F.triplet_margin_loss(za, zp, zn, margin=0.2, p=2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()
            if step % 20 == 0:
                print(f"Epoch {epoch} | step {step} | loss {epoch_loss/step:.4f}")
        dt = time.time() - t0
        avg = epoch_loss / max(1, step)
        print(f"Epoch {epoch} done in {dt:.1f}s | avg loss {avg:.4f}")
        ckpt_path = cfg.save_dir / f"epoch_{epoch:03d}_loss_{avg:.4f}.pt"
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'cfg': vars(cfg), 'avg_loss': avg}, ckpt_path)
        # Visualize this epoch
        if (epoch % cfg.viz_every) == 0:
            save_epoch_visuals(model, ds, device, cfg.viz_dir, epoch, viz_samples=cfg.viz_samples, pca_subset=cfg.pca_subset)
        if avg < best:
            best = avg
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'cfg': vars(cfg), 'avg_loss': avg}, cfg.save_dir / 'best.pt')
            print(f"✓ Saved new best to {cfg.save_dir / 'best.pt'}")


@torch.no_grad()
def export_embeddings(ckpt_path: Path, images_root: Path, products_json: Path, out_npz: Path,
                      image_size: int = 336, device: str = 'cuda'):
    """Export embeddings for the (cropped) dataset to a compressed NPZ."""
    payload = torch.load(ckpt_path, map_location='cpu')
    cfg_dict = payload['cfg']
    model = SiameseModel(backbone_name=cfg_dict['backbone'], proj_dim=cfg_dict['proj_dim'], train_backbone=False)
    model.load_state_dict(payload['model'])
    model.eval()
    model.to(device if torch.cuda.is_available() else 'cpu')

    data = read_jsonl_or_json(products_json)
    paths = []
    for rec in data:
        path_rel = rec.get('path') or rec.get('jpg_filename')
        if not path_rel:
            continue
        paths.append(str(Path(images_root) / 'images' / 'scans' / path_rel))

    embs = []
    out_paths = []
    for pth in paths:
        img = cv2.imread(pth)
        if img is None:
            continue
        H, W = img.shape[:2]
        if H != W:
            m = min(H, W)
            y0 = (H - m)//2; x0 = (W - m)//2
            img = img[y0:y0+m, x0:x0+m]
        if img.shape[0] != image_size:
            img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0) / 255.0
        x = x.to(next(model.parameters()).device)
        z = model.embed(x).cpu().numpy()
        embs.append(z)
        out_paths.append(pth)
    if embs:
        embs = np.concatenate(embs, axis=0)
        np.savez_compressed(out_npz, embeddings=embs, paths=np.array(out_paths))


if __name__ == "__main__":
    # Use spawn by default for safety with CUDA
    import torch.multiprocessing as mp
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    train()
