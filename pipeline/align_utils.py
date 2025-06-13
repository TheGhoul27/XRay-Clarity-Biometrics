# """
# align_utils.py – helpers for geometric / embedding-based alignment.

# Public API
# ----------
# rotate_align(...)          – brute-force rotation against ONE template
# rotate_align_two_refs(...) – rotation + orientation (front / back) using TWO
#                              reference templates
# """

# from pathlib import Path
# from typing import Optional, Tuple, List

# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T

# __all__ = [
#     "rotate_align",
#     "rotate_align_two_refs",
# ]

# # ────────────────────────────── ResNet-50 setup ──────────────────────────
# _IMAGENET_MEAN = (0.485, 0.456, 0.406)
# _IMAGENET_STD  = (0.229, 0.224, 0.225)
# _pre = T.Compose([
#     T.ToTensor(),
#     T.Resize((224, 224), antialias=True),
#     T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
# ])

# def _load_resnet(device: str) -> torch.nn.Module:
#     resnet = torch.hub.load('pytorch/vision', 'resnet50',
#                             weights='IMAGENET1K_V2')
#     resnet.fc = torch.nn.Identity()
#     return resnet.to(device).eval()

# def _embed(bgr: np.ndarray, model: torch.nn.Module,
#            device: str) -> torch.Tensor:
#     """Return ℓ2-normalised 2048-D embedding for a BGR image."""
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     t   = _pre(rgb).unsqueeze(0).to(device)
#     with torch.no_grad():
#         feat = model(t)
#     return F.normalize(feat.squeeze(0), dim=0)

# # ───────────────────── geometry helpers (internal) ───────────────────────
# def _rotate_expand(img: np.ndarray, angle_deg: float) -> np.ndarray:
#     """Rotate around centre with canvas expansion so nothing clips."""
#     h, w = img.shape[:2]
#     centre = (w / 2, h / 2)
#     M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
#     cos, sin = abs(M[0, 0]), abs(M[0, 1])
#     nW = int(h * sin + w * cos)
#     nH = int(h * cos + w * sin)
#     M[0, 2] += (nW / 2) - centre[0]
#     M[1, 2] += (nH / 2) - centre[1]
#     return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,
#                           borderMode=cv2.BORDER_REPLICATE)

# # ─────────────────────────── rotate_align (ONE ref) ──────────────────────
# def rotate_align(
#     crop_moving: np.ndarray,
#     reference:   np.ndarray,
#     save_dir:    Path,
#     *,
#     resnet:      Optional[torch.nn.Module] = None,
#     device:      str   = "cuda",
#     angle_step:  int   = 5,
#     min_improve: float = 0.005,
# ) -> Tuple[np.ndarray, int, float]:
#     """
#     Rotate `crop_moving` so its embedding best matches `reference`.
#     Returns (best_rotated, best_angle_deg, best_similarity).
#     """
#     if crop_moving.size == 0:
#         raise ValueError("Empty crop passed to rotate_align()")

#     save_dir.mkdir(parents=True, exist_ok=True)
#     device = device if (device != "cuda" or torch.cuda.is_available()) \
#                 else "cpu"
#     resnet = resnet or _load_resnet(device)

#     # embeddings for reference and un-rotated crop
#     ref_emb  = _embed(reference,   resnet, device)
#     crop0_emb = _embed(crop_moving, resnet, device)

#     sim0 = F.cosine_similarity(crop0_emb, ref_emb, dim=0).item()
#     best_sim   = sim0
#     best_angle = 0

#     for ang in range(angle_step, 360, angle_step):
#         rot = _rotate_expand(crop_moving, ang)
#         sim = F.cosine_similarity(_embed(rot, resnet, device),
#                                   ref_emb, dim=0).item()
#         if sim > best_sim:
#             best_sim, best_angle = sim, ang

#     if best_sim - sim0 < min_improve:
#         best_angle, best_sim = 0, sim0
#         best_rot = crop_moving
#     else:
#         best_rot = _rotate_expand(crop_moving, best_angle)

#     _dump_pair(save_dir / "best_rotated.jpg",
#                save_dir / "side_by_side.jpg",
#                best_rot, reference)

#     return best_rot, best_angle, best_sim

# # ─────────────── rotate_align_two_refs (FRONT/BACK refs) ────────────────
# def rotate_align_two_refs(
#     crop_moving: np.ndarray,
#     front_ref:   np.ndarray,
#     back_ref:    np.ndarray,
#     save_dir:    Path,
#     *,
#     resnet:     Optional[torch.nn.Module] = None,
#     device:     str  = "cuda",
#     angle_step: int  = 5,
#     dump_debug: bool = False,
# ) -> Tuple[np.ndarray, int, str, float, List[float], List[float]]:
#     """
#     Rotate `crop_moving` and decide **which side it is** by comparing
#     embeddings against *both* `front_ref` and `back_ref`.

#     Returns
#     -------
#     best_rotated : np.ndarray
#     best_angle   : int    (0–359 ° CCW)
#     best_side    : str    "front" | "back"
#     best_sim     : float  Cos-similarity of the chosen match
#     sims_front   : List[float] – similarity curve vs. angle for `front_ref`
#     sims_back    : List[float] – same for `back_ref`
#     """
#     if crop_moving.size == 0:
#         raise ValueError("Empty crop passed to rotate_align_two_refs()")

#     save_dir.mkdir(parents=True, exist_ok=True)
#     device = device if (device != "cuda" or torch.cuda.is_available()) \
#                 else "cpu"
#     resnet = resnet or _load_resnet(device)

#     refF = _embed(front_ref, resnet, device)
#     refB = _embed(back_ref,  resnet, device)

#     sims_front, sims_back = [], []
#     best_sim   = -1.0
#     best_angle = 0
#     best_side  = "front"

#     for ang in range(0, 360, angle_step):
#         rot = _rotate_expand(crop_moving, ang)
#         emb = _embed(rot, resnet, device)
#         simF = F.cosine_similarity(emb, refF, dim=0).item()
#         simB = F.cosine_similarity(emb, refB, dim=0).item()
#         sims_front.append(simF)
#         sims_back.append(simB)

#         if simF > best_sim:
#             best_sim, best_angle, best_side = simF, ang, "front"
#         if simB > best_sim:
#             best_sim, best_angle, best_side = simB, ang, "back"

#     best_rot = _rotate_expand(crop_moving, best_angle)

#     # ─── diagnostics ────────────────────────────────────────────────
#     _dump_pair(save_dir / "best_rotated.jpg",
#                save_dir / f"side_by_side_{best_side}.jpg",
#                best_rot,
#                front_ref if best_side == "front" else back_ref)

#     # save similarity curve only when debugging
#     if dump_debug:
#         with open(save_dir / "sims.csv", "w", newline="") as f:
#             import csv
#             writer = csv.writer(f)
#             writer.writerow(["angle_deg", "sim_front", "sim_back"])
#             for a, sF, sB in zip(range(0, 360, angle_step),
#                                  sims_front, sims_back):
#                 writer.writerow([a, f"{sF:.6f}", f"{sB:.6f}"])

#     return best_rot, best_angle, best_side, best_sim, sims_front, sims_back

# # ────────────────────────── I/O helpers ─────────────────────────────────
# def _dump_pair(best_path: Path, side_path: Path,
#                best_rot: np.ndarray, reference: np.ndarray):
#     cv2.imwrite(str(best_path), best_rot)
#     h, w = reference.shape[:2]
#     side = cv2.hconcat([reference, cv2.resize(best_rot, (w, h))])
#     cv2.imwrite(str(side_path), side)


"""
align_utils.py – helpers for geometric / embedding-based alignment.

Public API
----------
rotate_align(...)          – rotation search against ONE reference
rotate_align_two_refs(...) – rotation + orientation (front / back)
                             using TWO reference templates

Both take an optional `model_name` kw-arg:

    model_name="resnet50"                       # (default, ImageNet-V2 weights)
    model_name="vit_large_patch14_dinov2.lvd"   # any timm model id

Install timm once:
    pip install -U timm
"""

from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm

__all__ = [
    "rotate_align",
    "rotate_align_two_refs",
]

# ────────────────────────────── defaults ───────────────────────────────
_DEFAULT_MODEL = "resnet50"          # keep old behaviour if caller omits arg

# for ResNet and for timm models that don’t ship preprocessing values
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# ────────────────────────── backbone loader ────────────────────────────
_CACHE = {}  # (model_name, device) → (nn.Module, transforms)

def _load_backbone(model_name: str, device: str):
    """
    Returns (model, transform) and caches them by (name, device).
    • "resnet50" uses torchvision weights (IMAGENET1K_V2).
    • Any other string is passed to timm.create_model(..., pretrained=True).
    """
    key = (model_name, device)
    if key in _CACHE:
        return _CACHE[key]

    if model_name == "resnet50":
        model = torch.hub.load("pytorch/vision", "resnet50",
                               weights="IMAGENET1K_V2")
        model.fc = torch.nn.Identity()
        cfg = dict(mean=_IMAGENET_MEAN, std=_IMAGENET_STD, size=(224, 224))
    else:
        model = timm.create_model(model_name, pretrained=True)
        # remove classification layer if present
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(0)
        cfg_timm = model.default_cfg
        H, W = cfg_timm.get("input_size", (3, 224, 224))[1:]
        cfg = dict(
            mean=cfg_timm.get("mean", _IMAGENET_MEAN),
            std=cfg_timm.get("std",  _IMAGENET_STD),
            size=(H, W),
        )

    transform = T.Compose([
        T.ToTensor(),
        T.Resize(cfg["size"], antialias=True,
                 interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(cfg["mean"], cfg["std"]),
    ])

    model = model.to(device).eval()
    _CACHE[key] = (model, transform)
    return model, transform

# ──────────────────────── embed util ───────────────────────────────────
@torch.inference_mode()
def _embed(bgr: np.ndarray,
           model: torch.nn.Module,
           transform: T.Compose,
           device: str) -> torch.Tensor:
    """Return ℓ2-normalised embedding for a single BGR image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = transform(rgb).unsqueeze(0).to(device)
    feat= model(t)
    return F.normalize(feat.squeeze(0), dim=0)

# ───────────────────── geometry helper ─────────────────────────────────
def _rotate_expand(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around centre and expand canvas so nothing clips."""
    h, w = img.shape[:2]
    centre = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2) - centre[0]
    M[1, 2] += (nH / 2) - centre[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

# ───────────────────── rotate_align (ONE ref) ──────────────────────────
def rotate_align(
    crop_moving: np.ndarray,
    reference:   np.ndarray,
    save_dir:    Path,
    *,
    model_name:  str   = _DEFAULT_MODEL,
    device:      str   = "cuda",
    angle_step:  int   = 5,
    min_improve: float = 0.005,
) -> Tuple[np.ndarray, int, float]:
    """
    Rotate `crop_moving` so its embedding best matches `reference`.
    Returns (best_rotated, best_angle_deg, best_similarity).
    """
    if crop_moving.size == 0:
        raise ValueError("Empty crop passed to rotate_align()")

    save_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, tfm = _load_backbone(model_name, device)

    ref_emb   = _embed(reference,   model, tfm, device)
    crop0_emb = _embed(crop_moving, model, tfm, device)

    sim0 = F.cosine_similarity(crop0_emb, ref_emb, dim=0).item()
    best_sim, best_angle = sim0, 0

    for ang in range(angle_step, 360, angle_step):
        rot = _rotate_expand(crop_moving, ang)
        sim = F.cosine_similarity(
            _embed(rot, model, tfm, device), ref_emb, dim=0
        ).item()
        if sim > best_sim:
            best_sim, best_angle = sim, ang

    best_rot = crop_moving if best_angle == 0 else _rotate_expand(crop_moving, best_angle)

    _dump_pair(save_dir / "best_rotated.jpg",
               save_dir / "side_by_side.jpg",
               best_rot, reference)

    return best_rot, best_angle, best_sim

# ───────── rotate_align_two_refs (FRONT / BACK) ────────────────────────
def rotate_align_two_refs(
    crop_moving: np.ndarray,
    front_ref:   np.ndarray,
    back_ref:    np.ndarray,
    save_dir:    Path,
    *,
    model_name:  str  = _DEFAULT_MODEL,
    device:      str  = "cuda",
    angle_step:  int  = 5,
    dump_debug:  bool = False,
) -> Tuple[np.ndarray, int, str, float, List[float], List[float]]:
    """
    Rotate `crop_moving` and decide **which side it is** by comparing
    embeddings against BOTH references.

    Returns
    -------
    best_rotated : np.ndarray
    best_angle   : int    (0–359 ° CCW)
    best_side    : str    "front" | "back"
    best_sim     : float  cosine similarity of chosen match
    sims_front   : List[float] – similarity curve vs angle for front_ref
    sims_back    : List[float] – same for back_ref
    """
    if crop_moving.size == 0:
        raise ValueError("Empty crop passed to rotate_align_two_refs()")

    save_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, tfm = _load_backbone(model_name, device)

    refF = _embed(front_ref, model, tfm, device)
    refB = _embed(back_ref,  model, tfm, device)

    sims_front, sims_back = [], []
    best_sim, best_angle, best_side = -1.0, 0, "front"

    for ang in range(0, 360, angle_step):
        rot = _rotate_expand(crop_moving, ang)
        emb = _embed(rot, model, tfm, device)
        simF = F.cosine_similarity(emb, refF, dim=0).item()
        simB = F.cosine_similarity(emb, refB, dim=0).item()
        sims_front.append(simF)
        sims_back.append(simB)
        if simF > best_sim:
            best_sim, best_angle, best_side = simF, ang, "front"
        if simB > best_sim:
            best_sim, best_angle, best_side = simB, ang, "back"

    best_rot = _rotate_expand(crop_moving, best_angle)

    _dump_pair(save_dir / "best_rotated.jpg",
               save_dir / f"side_by_side_{best_side}.jpg",
               best_rot,
               front_ref if best_side == "front" else back_ref)

    if dump_debug:
        with open(save_dir / "sims.csv", "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["angle_deg", "sim_front", "sim_back"])
            for a, sF, sB in zip(range(0, 360, angle_step),
                                 sims_front, sims_back):
                writer.writerow([a, f"{sF:.6f}", f"{sB:.6f}"])

    return best_rot, best_angle, best_side, best_sim, sims_front, sims_back

# ────────────────────────── I/O helper ────────────────────────────────
def _dump_pair(best_path: Path, side_path: Path,
               best_rot: np.ndarray, reference: np.ndarray):
    cv2.imwrite(str(best_path), best_rot)
    h, w = reference.shape[:2]
    combo = cv2.hconcat([reference, cv2.resize(best_rot, (w, h))])
    cv2.imwrite(str(side_path), combo)