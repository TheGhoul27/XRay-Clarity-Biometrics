import os; os.environ["WANDB_DISABLED"] = "true"
"""
segment_obb_rotation_match.py
=============================
Segment a moving X‑ray image with YOLO‑Seg, convert the mask polygon to an
**oriented bounding‑box (OBB)**, perspective‑crop that patch, then **estimate its
rotation** by comparing ResNet‑50 embeddings of the reference template and all
rotations of the crop.  If the embedding‑based search fails for any reason, the
script gracefully falls back to the rectangle’s geometric angle so you always
get the *closest* rotated crop instead of an error.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from ultralytics import YOLO

# ───────────────────────────── helpers ──────────────────────────────────

def crop_from_obb(img: np.ndarray, box: np.ndarray, scale: float = 1.05) -> np.ndarray:
    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])
    W, H = int(w * scale), int(h * scale)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    centre = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - centre[0]
    M[1, 2] += (nH / 2) - centre[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ─────────── ResNet‑50 embedding utilities ────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def resnet_embedding(bgr: np.ndarray, model: torch.nn.Module, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = _preprocess(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return F.normalize(emb.squeeze(0), dim=0)


def find_best_rotation(crop: np.ndarray, ref: np.ndarray, model: torch.nn.Module,
                       device: str, angle_step: int = 5):
    """Return (best_angle, similarity). May raise if embedding fails."""
    ref_emb = resnet_embedding(ref, model, device)
    best_sim, best_angle = -1.0, 0
    for angle in range(0, 360, angle_step):
        rot = rotate_image(crop, angle)
        emb = resnet_embedding(rot, model, device)
        sim = F.cosine_similarity(emb, ref_emb, dim=0).item()
        if sim > best_sim:
            best_sim, best_angle = sim, angle
    return best_angle, best_sim

# ─────────── CLI ────────────

def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt', required=True, type=Path)
    p.add_argument('--out_dir', required=True, type=Path)
    # batch vs single
    p.add_argument('--img_dir', type=Path)
    p.add_argument('--ref_dir', type=Path)
    p.add_argument('--moving', type=Path)
    p.add_argument('--reference', type=Path)
    # misc
    p.add_argument('--classes', nargs='*', type=int)
    p.add_argument('--imgsz', default=640, type=int)
    p.add_argument('--conf', default=0.1, type=float)
    p.add_argument('--device', default='cuda')
    p.add_argument('--angle_step', default=5, type=int)
    return p

# ─────────── main ────────────

def main():
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    yolo = YOLO(str(args.ckpt))
    resnet50 = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
    resnet50.fc = torch.nn.Identity()
    resnet50 = resnet50.to(device).eval()

    keep_cls = set(args.classes) if args.classes else None
    single = bool(args.moving and args.reference)
    if single:
        srcs = [str(args.moving)]
    else:
        if not (args.img_dir and args.ref_dir):
            sys.exit('Need (--img_dir & --ref_dir) OR (--moving & --reference).')
        srcs = str(args.img_dir)

    for r in yolo.predict(srcs, imgsz=args.imgsz, conf=args.conf, device=args.device, stream=True, verbose=False):
        if r.masks is None:
            continue
        mov_path = Path(r.path)
        ref_path = args.reference if single else (args.ref_dir)
        if not ref_path.exists():
            print(f'⚠ reference missing for {mov_path.name}, skipping'); continue

        img_mov = cv2.imread(str(mov_path))
        img_ref = cv2.imread(str(ref_path))

        polys = r.masks.xy
        cids = r.boxes.cls.cpu().numpy().astype(int)

        for inst, (poly, cid) in enumerate(zip(polys, cids)):
            if keep_cls and cid not in keep_cls:
                continue
            sub = args.out_dir / f"{mov_path.stem}_cls{cid}_{inst}"
            sub.mkdir(parents=True, exist_ok=True)

            # OBB
            pts = np.asarray(poly, np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(int)
            over = img_mov.copy(); cv2.polylines(over, [box], True, (0,255,0), 2)
            cv2.imwrite(str(sub/'obb_overlay.jpg'), over)

            # crop
            crop = crop_from_obb(img_mov, box)
            if crop.size == 0:
                print(f'⚠ empty crop {mov_path.name} inst{inst}'); continue
            cv2.imwrite(str(sub/'crop_moving.jpg'), crop)
            cv2.imwrite(str(sub/'reference.jpg'), img_ref)

            # rotation search with graceful fallback
            try:
                best_angle, best_sim = find_best_rotation(crop, img_ref, resnet50, device, args.angle_step)
                reason = 'search'
            except Exception as e:
                # fallback to geometric angle from minAreaRect (rect[2])
                base_angle = -rect[2]  # rect angle is [-90,0)
                best_angle = base_angle % 360
                best_sim = None
                reason = f'fallback ({e})'

            best_rot = rotate_image(crop, best_angle)
            cv2.imwrite(str(sub/'best_rotated.jpg'), best_rot)

            # side by side
            h_ref, w_ref = img_ref.shape[:2]
            resized = cv2.resize(best_rot, (w_ref, h_ref))
            cv2.imwrite(str(sub/'side_by_side.jpg'), cv2.hconcat([img_ref, resized]))

            msg_sim = f'sim={best_sim:.3f}' if best_sim is not None else 'sim=N/A'
            print(f'✓ {mov_path.name} cls{cid} inst{inst} | {reason} angle={best_angle:.1f}° {msg_sim}')

    print(f"\nAll artefacts → {args.out_dir.resolve()}")

if __name__ == '__main__':
    main()
