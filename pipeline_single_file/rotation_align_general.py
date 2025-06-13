import os; os.environ["WANDB_DISABLED"] = "true"
"""rotation_align_general.py
================================
Segment a moving X‑ray image, crop each oriented bounding box (OBB), then decide
**which side (front / back)** it belongs to and its **rotation angle** by
comparing ResNet‑50 embeddings of *both* reference templates across a rotation
sweep.

Output per detected instance
---------------------------
* **obb_overlay.jpg**       – original image with green OBB
* **crop_moving.jpg**       – perspective crop (un‑rotated)
* **best_rotated.jpg**      – crop rotated to best angle & reference
* **side_by_side_front.jpg** / **side_by_side_back.jpg** – diagnostic overlay
  depending on the chosen orientation
* **sims.csv**              – CSV table: `angle,sim_front,sim_back`

Single‑image example
```bash
python rotation_align_general.py \
  --ckpt runs/segment/best.pt \
  --moving data/135656-V0.bmp \
  --front_ref refs/front.bmp \
  --back_ref  refs/back.bmp \
  --out_dir outputs \
  --classes 0 1
```
"""

from pathlib import Path
import argparse
import sys
import csv

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from ultralytics import YOLO

# ────────────────────────────── helpers ─────────────────────────────────

def crop_from_obb(img: np.ndarray, box: np.ndarray, scale: float = 1.05) -> np.ndarray:
    """Perspective‑crop an oriented bounding box region."""
    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])
    W, H = int(w * scale), int(h * scale)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


def rotate_expand(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    centre = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2) - centre[0]
    M[1, 2] += (nH / 2) - centre[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ───────────── ResNet‑50 embedding utils ─────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def _embed(bgr: np.ndarray, model: torch.nn.Module, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = _preprocess(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(t)
    return F.normalize(feat.squeeze(0), dim=0)


# ────────────────────── rotation + orientation ─────────────────────────

def find_best_rotation_two_refs(
    crop: np.ndarray,
    front_ref: np.ndarray,
    back_ref:  np.ndarray,
    model: torch.nn.Module,
    device: str,
    angle_step: int = 5,
):
    """Return (best_angle, best_side:str, best_sim, sims_front, sims_back)."""
    refF = _embed(front_ref, model, device)
    refB = _embed(back_ref,  model, device)

    sims_front, sims_back = [], []
    best_sim = -1.0
    best_angle = 0
    best_side = "front"

    for ang in range(0, 360, angle_step):
        rot = rotate_expand(crop, ang)
        emb = _embed(rot, model, device)
        simF = F.cosine_similarity(emb, refF, dim=0).item()
        simB = F.cosine_similarity(emb, refB, dim=0).item()
        sims_front.append(simF)
        sims_back.append(simB)
        if simF > best_sim:
            best_sim, best_angle, best_side = simF, ang, "front"
        if simB > best_sim:
            best_sim, best_angle, best_side = simB, ang, "back"

    sims_front = np.array(sims_front, np.float32)
    sims_back  = np.array(sims_back,  np.float32)
    return best_angle, best_side, best_sim, sims_front, sims_back


# ───────────────────────────── CLI ─────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Segment → rotation‑orient X‑ray images using two references.")
    p.add_argument('--ckpt', required=True, type=Path, help='YOLO segmentation checkpoint')
    p.add_argument('--out_dir', required=True, type=Path, help='Folder to save outputs')

    # batch vs single
    p.add_argument('--img_dir',   type=Path, help='Folder of moving images')
    p.add_argument('--front_dir', type=Path, help='Folder of FRONT reference templates')
    p.add_argument('--back_dir',  type=Path, help='Folder of BACK reference templates')

    p.add_argument('--moving',     type=Path, help='Single moving image')
    p.add_argument('--front_ref',  type=Path, help='Single front reference')
    p.add_argument('--back_ref',   type=Path, help='Single back reference')

    # misc
    p.add_argument('--classes', nargs='*', type=int, help='Filter by YOLO class IDs')
    p.add_argument('--imgsz', default=640, type=int, help='YOLO inference size')
    p.add_argument('--conf',  default=0.1, type=float, help='YOLO confidence threshold')
    p.add_argument('--device', default='cuda', help='cuda | cpu')
    p.add_argument('--angle_step', default=5, type=int, help='Rotation step (°)')
    return p


# ─────────────────────────── main ───────────────────────────────────────

def main():
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    yolo = YOLO(str(args.ckpt))
    resnet = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()

    keep_cls = set(args.classes) if args.classes else None

    single = bool(args.moving and args.front_ref and args.back_ref)
    if single:
        mov_paths = [args.moving]
    else:
        if not (args.img_dir and args.front_dir and args.back_dir):
            sys.exit('Need (--img_dir & --front_dir & --back_dir) OR (--moving & --front_ref & --back_ref)')
        mov_paths = sorted(args.img_dir.glob('*'))

    for mov_p in mov_paths:
        if single:
            front_p, back_p = args.front_ref, args.back_ref
        else:
            front_p = args.front_dir / mov_p.name
            back_p  = args.back_dir  / mov_p.name
        if not (front_p.exists() and back_p.exists()):
            print(f'⚠ reference(s) missing for {mov_p.name}, skipping'); continue

        img_front = cv2.imread(str(front_p))
        img_back  = cv2.imread(str(back_p))
        if img_front is None or img_back is None:
            print(f'⚠ cannot read references for {mov_p.name}'); continue

        # segment instances
        inst_iter = yolo.predict(str(mov_p), imgsz=args.imgsz, conf=args.conf, device=device, stream=False)[0]
        if inst_iter.masks is None:
            continue

        polys = inst_iter.masks.xy
        cids  = inst_iter.boxes.cls.cpu().numpy().astype(int)
        scores= inst_iter.boxes.conf.cpu().numpy().astype(float)
        img_mov_full = cv2.imread(str(mov_p))
        if img_mov_full is None:
            print(f'⚠ cannot read {mov_p}'); continue

        for inst_id, (poly, cid, score) in enumerate(zip(polys, cids, scores)):
            if keep_cls and cid not in keep_cls:
                continue
            sub = args.out_dir / f"{mov_p.stem}_cls{cid}_{inst_id}"
            sub.mkdir(parents=True, exist_ok=True)

            # OBB & overlay
            pts  = np.asarray(poly, np.float32)
            rect = cv2.minAreaRect(pts)
            box  = cv2.boxPoints(rect).astype(int)
            over = img_mov_full.copy(); cv2.polylines(over, [box], True, (0,255,0), 2)
            cv2.imwrite(str(sub / 'obb_overlay.jpg'), over)

            # crop
            crop = crop_from_obb(img_mov_full, box)
            if crop.size == 0:
                print(f'⚠ empty crop {mov_p.name} inst{inst_id}'); continue
            cv2.imwrite(str(sub / 'crop_moving.jpg'), crop)
            cv2.imwrite(str(sub / 'front_reference.jpg'), img_front)
            cv2.imwrite(str(sub / 'back_reference.jpg' ), img_back)

            # rotation + orientation search
            try:
                ang, side, sim, simsF, simsB = find_best_rotation_two_refs(
                    crop, img_front, img_back, resnet, device, angle_step=args.angle_step)
            except Exception as e:
                print(f'x embedding failed {mov_p.name} inst{inst_id}: {e}')
                continue

            best_rot = rotate_expand(crop, ang)
            cv2.imwrite(str(sub / 'best_rotated.jpg'), best_rot)

            # save side‑by‑side diagnostic according to chosen side
            hR, wR = (img_front if side == 'front' else img_back).shape[:2]
            combo = cv2.hconcat([
                img_front if side == 'front' else img_back,
                cv2.resize(best_rot, (wR, hR))
            ])
            cv2.imwrite(str(sub / f'side_by_side_{side}.jpg'), combo)

            # save similarity matrix
            angles = list(range(0, 360, args.angle_step))
            with open(sub / 'sims.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['angle_deg', 'sim_front', 'sim_back'])
                for a, sF, sB in zip(angles, simsF, simsB):
                    writer.writerow([a, f'{sF:.6f}', f'{sB:.6f}'])

            print(f'✓ {mov_p.name} inst{inst_id}  → {side}  angle={ang:3d}°  sim={sim:.3f}')

    print(f"All artefacts → {args.out_dir.resolve()}")

if __name__ == '__main__':
    main()

