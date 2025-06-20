"""
align_utils.py – orientation / alignment helpers
================================================

Public API
----------
rotate_align(...)          – rotation search against ONE reference
rotate_align_two_refs(...) – rotation + side decision (front / back)

Keyword parameters
------------------
model_name : str   (default "resnet50")
    Any model id recognised by timm; common examples:
        "resnet50", 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
score_mode : {"hybrid", "cos", "orb", "rmse"}
    Metric used to rate each candidate angle (see table above).
alpha      : float (0-1)  – weight of ORB score in "hybrid" mode.
angle_step : int   – degrees between tested angles (divides 360).
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ───────────────────────── backbone loader ────────────────────────────
_DEFAULT_MODEL = "resnet50"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, T.Compose]] = {}

_DINO_IDS = {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"}

def _load_backbone(name: str, device: str) -> tuple[torch.nn.Module, T.Compose]:
    key = (name, device)
    if key in _CACHE:
        return _CACHE[key]

    if name == "resnet50":
        model = torch.hub.load("pytorch/vision", "resnet50",
                               weights="IMAGENET1K_V2")
        model.fc = torch.nn.Identity()
        size = 224
        mean, std = _IMAGENET_MEAN, _IMAGENET_STD

    elif name in _DINO_IDS:
        model = torch.hub.load("facebookresearch/dinov2", name)
        size = 224                                # all three hub variants
        mean, std = _IMAGENET_MEAN, _IMAGENET_STD

    else:
        raise ValueError(f"Unsupported model_name '{name}'. "
                         "Use 'resnet50' or one of "
                         "{'dinov2_vits14','dinov2_vitb14','dinov2_vitl14'}.")

    tfm = T.Compose([
        T.ToTensor(),
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(size),
        T.Normalize(mean, std),
    ])

    model = model.to(device).eval()
    _CACHE[key] = (model, tfm)
    return model, tfm

@torch.inference_mode()
def _embed(bgr: np.ndarray, model: torch.nn.Module,
           tfm: T.Compose, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = tfm(rgb).unsqueeze(0).to(device)
    return F.normalize(model(t).squeeze(0), dim=0)

# ───────────────────────── ORB & RMSE helpers ─────────────────────────
_ORB   = cv2.ORB_create(nfeatures=4000)
_MATCH = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def _orb_ratio(gA: np.ndarray, gB: np.ndarray) -> float:
    kpA, desA = _ORB.detectAndCompute(gA, None)
    kpB, desB = _ORB.detectAndCompute(gB, None)
    if desA is None or desB is None:
        return 0.0
    matches = _MATCH.match(desA, desB)
    if len(matches) < 4:
        return 0.0
    src = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inl = mask.ravel().sum() if mask is not None else 0
    return inl / len(matches)

def _rmse_sim(gA: np.ndarray, gB: np.ndarray) -> float:
    if gA.shape != gB.shape:
        gA = cv2.resize(gA, (gB.shape[1], gB.shape[0]),
                        interpolation=cv2.INTER_AREA)
    diff = gA.astype(np.float32) - gB.astype(np.float32)
    return 1.0 - (np.sqrt(np.mean(diff**2)) / 255.0)

# ───────────────────────── similarity dispatcher ──────────────────────
def _score_pair(
    rot_bgr: np.ndarray, ref_bgr: np.ndarray,
    gRot: np.ndarray,   gRef: np.ndarray,
    *, mode: Literal["hybrid", "cos", "orb", "rmse"] = "hybrid",
    alpha: float = 0.5,
    model: torch.nn.Module | None = None,
    tfm: T.Compose | None = None,
    device: str = "cpu",
    ref_emb: torch.Tensor | None = None,
) -> float:
    if mode == "rmse":
        return _rmse_sim(gRot, gRef)
    if mode == "orb":
        return _orb_ratio(gRot, gRef)
    if mode == "cos":
        emb = _embed(rot_bgr, model, tfm, device)
        return float(F.cosine_similarity(emb, ref_emb, dim=0).cpu())

    # hybrid
    emb = _embed(rot_bgr, model, tfm, device)
    cos = float(F.cosine_similarity(emb, ref_emb, dim=0).cpu())
    orb = _orb_ratio(gRot, gRef)
    return (1 - alpha) * cos + alpha * orb

# ───────────────────────── rotation util ──────────────────────────────
def _rotate_expand(img: np.ndarray, ang: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW, nH = int(h*sin + w*cos), int(h*cos + w*sin)
    M[0,2] += nW/2 - w/2
    M[1,2] += nH/2 - h/2
    return cv2.warpAffine(img, M, (nW, nH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

# ───────────────────────── rotate_align (one ref) ─────────────────────
def rotate_align(
    crop_moving: np.ndarray, reference: np.ndarray, save_dir: Path, *,
    model_name: str = _DEFAULT_MODEL,
    score_mode: Literal["hybrid", "cos", "orb", "rmse"] = "hybrid",
    alpha: float = 0.5, device: str = "cuda", angle_step: int = 5,
) -> Tuple[np.ndarray, int, float]:
    if crop_moving.size == 0:
        raise ValueError("Empty crop supplied")

    save_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, tfm = _load_backbone(model_name, device)
    ref_emb = _embed(reference, model, tfm, device)
    gRef    = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    best_rot, best_angle, best_score = crop_moving, 0, -1.0
    for ang in range(0, 360, angle_step):
        rot  = _rotate_expand(crop_moving, ang)
        gRot = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        s    = _score_pair(rot, reference, gRot, gRef,
                           mode=score_mode, alpha=alpha,
                           model=model, tfm=tfm, device=device, ref_emb=ref_emb)
        if s > best_score:
            best_rot, best_angle, best_score = rot, ang, s

    _dump_pair(save_dir/"best_rotated.jpg",
               save_dir/"side_by_side.jpg",
               best_rot, reference)
    return best_rot, best_angle, best_score

# ───────────────── rotate_align_two_refs (front/back) ─────────────────
def rotate_align_two_refs(
    crop_moving: np.ndarray,
    front_ref:   np.ndarray,
    back_ref:    np.ndarray,
    save_dir:    Path,
    *,
    model_name: str = _DEFAULT_MODEL,
    score_mode: Literal["hybrid", "cos", "orb", "rmse"] = "hybrid",
    alpha: float = 0.5,
    device: str = "cuda",
    angle_step: int = 5,
    dump_debug: bool = False,
) -> Tuple[np.ndarray, int, str, float, List[float], List[float]]:

    if crop_moving.size == 0:
        raise ValueError("Empty crop supplied")

    save_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, tfm = _load_backbone(model_name, device)
    refF_emb = _embed(front_ref, model, tfm, device)
    refB_emb = _embed(back_ref,  model, tfm, device)
    gF, gB = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in (front_ref, back_ref))

    simsF, simsB = [], []
    best_rot, best_angle, best_side, best_score = crop_moving, 0, "front", -1.0

    for ang in range(0, 360, angle_step):
        rot  = _rotate_expand(crop_moving, ang)
        gRot = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)

        sF = _score_pair(rot, front_ref, gRot, gF, mode=score_mode,
                         alpha=alpha, model=model, tfm=tfm,
                         device=device, ref_emb=refF_emb)
        sB = _score_pair(rot, back_ref,  gRot, gB, mode=score_mode,
                         alpha=alpha, model=model, tfm=tfm,
                         device=device, ref_emb=refB_emb)

        simsF.append(sF); simsB.append(sB)

        if sF > best_score:
            best_rot, best_angle, best_side, best_score = rot, ang, "front", sF
        if sB > best_score:
            best_rot, best_angle, best_side, best_score = rot, ang, "back",  sB

    _dump_pair(save_dir/"best_rotated.jpg",
               save_dir/f"side_by_side_{best_side}.jpg",
               best_rot,
               front_ref if best_side == "front" else back_ref)

    if dump_debug:
        import csv
        with open(save_dir/"sims.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["angle_deg", "score_front", "score_back"])
            for a, sF, sB in zip(range(0, 360, angle_step), simsF, simsB):
                w.writerow([a, f"{sF:.6f}", f"{sB:.6f}"])

    return best_rot, best_angle, best_side, best_score, simsF, simsB

# ─────────────────────────── I/O helper ────────────────────────────────
def _dump_pair(best_path: Path, side_path: Path,
               best_rot: np.ndarray, reference: np.ndarray):
    cv2.imwrite(str(best_path), best_rot)
    h, w = reference.shape[:2]
    combo = cv2.hconcat([reference, cv2.resize(best_rot, (w, h))])
    cv2.imwrite(str(side_path), combo)