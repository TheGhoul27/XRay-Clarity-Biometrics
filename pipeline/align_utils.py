# """
# align_utils.py  ·  orientation search helpers
# ------------------------------------------------
# New in this version
# • For every tested angle we now compute:
#       score = (1-alpha) · cosine_sim  +  alpha · ORB_inlier_ratio
#   – `alpha` defaults to 0.5 and may be passed to both public functions.
# """

# from pathlib import Path
# from typing import Optional, Tuple, List
# import cv2, numpy as np, torch, timm
# import torch.nn.functional as F
# import torchvision.transforms as T

# __all__ = ["rotate_align", "rotate_align_two_refs"]

# # ─────── backbone loader (unchanged) ───────────────────────────────────
# _DEFAULT_MODEL = "resnet50"
# _IMAGENET_MEAN = (0.485, 0.456, 0.406)
# _IMAGENET_STD  = (0.229, 0.224, 0.225)
# _CACHE = {}
# def _load_backbone(name: str, device: str):
#     key = (name, device)
#     if key in _CACHE: return _CACHE[key]
#     if name == "resnet50":
#         m = torch.hub.load("pytorch/vision", "resnet50",
#                            weights="IMAGENET1K_V2"); m.fc = torch.nn.Identity()
#         cfg = dict(mean=_IMAGENET_MEAN, std=_IMAGENET_STD, size=(224,224))
#     else:
#         m = timm.create_model(name, pretrained=True)
#         if hasattr(m, "reset_classifier"): m.reset_classifier(0)
#         d = m.default_cfg
#         H,W = d.get("input_size", (3,224,224))[1:]
#         cfg = dict(mean=d.get("mean", _IMAGENET_MEAN),
#                    std =d.get("std",  _IMAGENET_STD),
#                    size=(H,W))
#     tfm = T.Compose([T.ToTensor(),
#                      T.Resize(cfg["size"], antialias=True,
#                               interpolation=T.InterpolationMode.BICUBIC),
#                      T.Normalize(cfg["mean"], cfg["std"])])
#     m = m.to(device).eval(); _CACHE[key]=(m,tfm); return m,tfm

# @torch.inference_mode()
# def _embed(bgr: np.ndarray, model, tfm, device):
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     t   = tfm(rgb).unsqueeze(0).to(device)
#     return F.normalize(model(t).squeeze(0), dim=0)

# # ─────── ORB helper ────────────────────────────────────────────────────
# _ORB = cv2.ORB_create(nfeatures=4000)
# _MATCH = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# def _orb_ratio(imgA_gray, imgB_gray) -> float:
#     kpA, desA = _ORB.detectAndCompute(imgA_gray, None)
#     kpB, desB = _ORB.detectAndCompute(imgB_gray, None)
#     if desA is None or desB is None: return 0.0
#     m = _MATCH.match(desA, desB)
#     if len(m) < 4: return 0.0
#     src = np.float32([kpA[i.queryIdx].pt for i in m]).reshape(-1,1,2)
#     dst = np.float32([kpB[i.trainIdx].pt for i in m]).reshape(-1,1,2)
#     _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#     inl = mask.ravel().sum() if mask is not None else 0
#     return inl / len(m)

# # ─────── geometry util ─────────────────────────────────────────────────
# def _rotate_expand(img: np.ndarray, ang: float):
#     h,w = img.shape[:2]; c=(w/2,h/2)
#     M=cv2.getRotationMatrix2D(c,ang,1); cos,sin=abs(M[0,0]),abs(M[0,1])
#     nW,nH=int(h*sin+w*cos),int(h*cos+w*sin)
#     M[0,2]+=nW/2-c[0]; M[1,2]+=nH/2-c[1]
#     return cv2.warpAffine(img,M,(nW,nH),flags=cv2.INTER_LINEAR,
#                           borderMode=cv2.BORDER_REPLICATE)

# # ───────────────── rotate_align (ONE ref) ──────────────────────────────
# def rotate_align(
#     crop_moving: np.ndarray, reference: np.ndarray, save_dir: Path, *,
#     model_name=_DEFAULT_MODEL, device="cuda", angle_step=5,
#     alpha: float=0.5,   # ← weight for ORB ratio
#     min_improve=0.005
# ) -> Tuple[np.ndarray,int,float]:

#     if crop_moving.size==0: raise ValueError("empty crop")
#     save_dir.mkdir(parents=True,exist_ok=True)
#     if device=="cuda" and not torch.cuda.is_available(): device="cpu"
#     model,tfm=_load_backbone(model_name,device)

#     ref_emb=_embed(reference,model,tfm,device)
#     gRef=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)

#     def score(rot):
#         cos=F.cosine_similarity(_embed(rot,model,tfm,device),ref_emb,dim=0).item()
#         ratio=_orb_ratio(cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY),gRef)
#         return (1-alpha)*cos + alpha*ratio

#     best_rot=crop_moving; best_angle=0; best_score=score(crop_moving)
#     for ang in range(angle_step,360,angle_step):
#         rot=_rotate_expand(crop_moving,ang); sc=score(rot)
#         if sc>best_score: best_score,best_angle,best_rot=sc,ang,rot

#     _dump_pair(save_dir/"best_rotated.jpg",save_dir/"side_by_side.jpg",
#                best_rot,reference)
#     return best_rot,best_angle,best_score

# # ─────────── rotate_align_two_refs (FRONT/BACK) ────────────────────────
# def rotate_align_two_refs(
#     crop_moving: np.ndarray, front_ref: np.ndarray, back_ref: np.ndarray,
#     save_dir: Path, *, model_name=_DEFAULT_MODEL, device="cuda",
#     angle_step=5, alpha: float=0.5, dump_debug=False
# ) -> Tuple[np.ndarray,int,str,float,List[float],List[float]]:

#     if crop_moving.size==0: raise ValueError("empty crop")
#     save_dir.mkdir(parents=True,exist_ok=True)
#     if device=="cuda" and not torch.cuda.is_available(): device="cpu"
#     model,tfm=_load_backbone(model_name,device)

#     refF,_gF = _embed(front_ref,model,tfm,device), cv2.cvtColor(front_ref,cv2.COLOR_BGR2GRAY)
#     refB,_gB = _embed(back_ref ,model,tfm,device), cv2.cvtColor(back_ref ,cv2.COLOR_BGR2GRAY)

#     simsF,simsB=[],[]
#     best_score,best_angle,best_side=-1.0,0,"front"

#     for ang in range(0,360,angle_step):
#         rot=_rotate_expand(crop_moving,ang)
#         emb=_embed(rot,model,tfm,device)
#         gRot=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY)

#         cosF=F.cosine_similarity(emb,refF,dim=0).item()
#         cosB=F.cosine_similarity(emb,refB,dim=0).item()
#         rF = _orb_ratio(gRot,_gF);  rB = _orb_ratio(gRot,_gB)

#         scoreF=(1-alpha)*cosF + alpha*rF
#         scoreB=(1-alpha)*cosB + alpha*rB
#         simsF.append(scoreF); simsB.append(scoreB)

#         if scoreF>best_score: best_score,best_angle,best_side=scoreF,ang,"front"
#         if scoreB>best_score: best_score,best_angle,best_side=scoreB,ang,"back"

#     best_rot=_rotate_expand(crop_moving,best_angle)
#     _dump_pair(save_dir/"best_rotated.jpg",
#                save_dir/f"side_by_side_{best_side}.jpg",
#                best_rot, front_ref if best_side=="front" else back_ref)

#     if dump_debug:
#         with open(save_dir/"sims.csv","w",newline="") as f:
#             import csv; w=csv.writer(f)
#             w.writerow(["angle_deg","score_front","score_back"])
#             for a,sf,sb in zip(range(0,360,angle_step),simsF,simsB):
#                 w.writerow([a,f"{sf:.6f}",f"{sb:.6f}"])

#     return best_rot,best_angle,best_side,best_score,simsF,simsB

# # ─────────── misc I/O ──────────────────────────────────────────────────
# def _dump_pair(best_path: Path, side_path: Path,
#                best_rot: np.ndarray, reference: np.ndarray):
#     cv2.imwrite(str(best_path),best_rot)
#     h,w=reference.shape[:2]
#     combo=cv2.hconcat([reference,cv2.resize(best_rot,(w,h))])
#     cv2.imwrite(str(side_path),combo)


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
        "resnet50", "vit_large_patch14_dinov2.lvd142m", ...
score_mode : {"hybrid", "cos", "orb", "rmse"}
    Metric used to rate each candidate angle (see table above).
alpha      : float (0-1)  – weight of ORB score in "hybrid" mode.
angle_step : int   – degrees between tested angles (divides 360).
"""

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm

# ───────────────────────── default backbone ────────────────────────────
_DEFAULT_MODEL = "resnet50"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_CACHE = {}                         # (model_name, device) → (model, transform)

def _load_backbone(name: str, device: str):
    key = (name, device)
    if key in _CACHE:
        return _CACHE[key]

    if name == "resnet50":
        m = torch.hub.load("pytorch/vision", "resnet50",
                           weights="IMAGENET1K_V2")
        m.fc = torch.nn.Identity()
        cfg = dict(mean=_IMAGENET_MEAN, std=_IMAGENET_STD, size=(224, 224))
    else:
        m = timm.create_model(name, pretrained=True)
        if hasattr(m, "reset_classifier"):
            m.reset_classifier(0)
        d = m.default_cfg
        H, W = d.get("input_size", (3, 224, 224))[1:]
        cfg = dict(
            mean=d.get("mean", _IMAGENET_MEAN),
            std=d.get("std",  _IMAGENET_STD),
            size=(H, W),
        )

    tfm = T.Compose([
        T.ToTensor(),
        T.Resize(cfg["size"], antialias=True,
                 interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(cfg["mean"], cfg["std"]),
    ])

    m = m.to(device).eval()
    _CACHE[key] = (m, tfm)
    return m, tfm

@torch.inference_mode()
def _embed(bgr: np.ndarray, model, tfm, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = tfm(rgb).unsqueeze(0).to(device)
    return F.normalize(model(t).squeeze(0), dim=0)

# ───────────────────────── ORB helpers ─────────────────────────────────
_ORB   = cv2.ORB_create(nfeatures=4000)
_MATCH = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def _orb_ratio(gA, gB) -> float:
    kpA, desA = _ORB.detectAndCompute(gA, None)
    kpB, desB = _ORB.detectAndCompute(gB, None)
    if desA is None or desB is None:
        return 0.0
    matches = _MATCH.match(desA, desB)
    if len(matches) < 4:
        return 0.0
    src = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inl = mask.ravel().sum() if mask is not None else 0
    return inl / len(matches)                     # 0-1 similarity

# ───────────────────────── RMSE helper ────────────────────────────────
def _rmse_sim(gA, gB) -> float:
    if gA.shape != gB.shape:                       # ← NEW
        gA = cv2.resize(gA, (gB.shape[1], gB.shape[0]),
                        interpolation=cv2.INTER_AREA)
    diff = gA.astype(np.float32) - gB.astype(np.float32)
    rmse = np.sqrt(np.mean(diff**2)) / 255.0       # normalise
    return 1.0 - rmse

# ───────────────────────── pixel → tensor helper ──────────────────────
def _to_torch(img_bgr, device):
    return torch.tensor(img_bgr / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device)

# ───────────────────────── similarity dispatcher ──────────────────────

def _score_pair(rot_bgr, ref_bgr, gRot, gRef,
                *, mode="hybrid", alpha=0.5,
                model=None, tfm=None, device="cpu",
                ref_emb=None):
    if mode == "rmse":
        return _rmse_sim(gRot, gRef)

    if mode == "orb":
        return _orb_ratio(gRot, gRef)

    if mode == "cos":
        emb = _embed(rot_bgr, model, tfm, device)
        return float(F.cosine_similarity(emb, ref_emb, dim=0).cpu())

    # hybrid: weighted cosine + ORB
    emb = _embed(rot_bgr, model, tfm, device)
    cos = float(F.cosine_similarity(emb, ref_emb, dim=0).cpu())
    orb = _orb_ratio(gRot, gRef)
    return (1 - alpha) * cos + alpha * orb

# ───────────────────────── geometry util ──────────────────────────────
def _rotate_expand(img: np.ndarray, ang: float):
    h, w = img.shape[:2]; cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nW / 2 - cx
    M[1, 2] += nH / 2 - cy
    return cv2.warpAffine(img, M, (nW, nH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

# ───────────────────────── rotate_align (1 ref) ───────────────────────
def rotate_align(
    crop_moving: np.ndarray,
    reference:   np.ndarray,
    save_dir:    Path,
    *,
    model_name=_DEFAULT_MODEL,
    score_mode="hybrid",
    alpha: float = 0.5,
    device="cuda",
    angle_step=5,
) -> Tuple[np.ndarray, int, float]:

    if crop_moving.size == 0:
        raise ValueError("Empty crop passed to rotate_align")

    save_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, tfm = _load_backbone(model_name, device)
    ref_emb = _embed(reference, model, tfm, device)
    gRef = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    best_rot = crop_moving
    best_angle = 0
    best_score = -1.0

    for ang in range(0, 360, angle_step):
        rot = _rotate_expand(crop_moving, ang)
        gRot = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        score = _score_pair(rot, reference, gRot, gRef,
                            mode=score_mode, alpha=alpha,
                            model=model, tfm=tfm, device=device,
                            ref_emb=ref_emb)
        if score > best_score:
            best_rot, best_angle, best_score = rot, ang, score

    _dump_pair(save_dir / "best_rotated.jpg",
               save_dir / "side_by_side.jpg",
               best_rot, reference)
    return best_rot, best_angle, best_score

# ───────────────── rotate_align_two_refs (front/back) ─────────────────
def rotate_align_two_refs(
    crop_moving: np.ndarray,
    front_ref:   np.ndarray,
    back_ref:    np.ndarray,
    save_dir:    Path,
    *,
    model_name=_DEFAULT_MODEL,
    score_mode="hybrid",
    alpha: float = 0.5,
    device="cuda",
    angle_step=5,
    dump_debug=False,
) -> Tuple[np.ndarray, int, str, float, List[float], List[float]]:

    if crop_moving.size == 0:
        raise ValueError("Empty crop passed to rotate_align_two_refs")

    save_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, tfm = _load_backbone(model_name, device)
    refF_emb = _embed(front_ref, model, tfm, device)
    refB_emb = _embed(back_ref , model, tfm, device)
    gF = cv2.cvtColor(front_ref, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(back_ref , cv2.COLOR_BGR2GRAY)

    simsF, simsB = [], []
    best_rot = crop_moving
    best_angle, best_side, best_score = 0, "front", -1.0

    for ang in range(0, 360, angle_step):
        rot = _rotate_expand(crop_moving, ang)
        gRot = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)

        sF = _score_pair(rot, front_ref, gRot, gF,
                         mode=score_mode, alpha=alpha,
                         model=model, tfm=tfm, device=device,
                         ref_emb=refF_emb)
        sB = _score_pair(rot, back_ref, gRot, gB,
                         mode=score_mode, alpha=alpha,
                         model=model, tfm=tfm, device=device,
                         ref_emb=refB_emb)

        simsF.append(sF); simsB.append(sB)

        if sF > best_score:
            best_rot, best_angle, best_side, best_score = rot, ang, "front", sF
        if sB > best_score:
            best_rot, best_angle, best_side, best_score = rot, ang, "back",  sB

    _dump_pair(save_dir / "best_rotated.jpg",
               save_dir / f"side_by_side_{best_side}.jpg",
               best_rot,
               front_ref if best_side == "front" else back_ref)

    if dump_debug:
        import csv
        with open(save_dir / "sims.csv", "w", newline="") as f:
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
