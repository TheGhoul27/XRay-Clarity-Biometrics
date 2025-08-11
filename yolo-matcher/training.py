#!/usr/bin/env python3
"""
train_and_visualize_yolo_seg_scans.py  (updated)

Usage examples
--------------
# 1) Predict-only with existing weights and save oriented boxes
python train_and_visualize_yolo_seg_scans.py \
  --root root_folder --names object --weights runs/segment/exp/weights/best.pt \
  --imgsz 640 --predict-only --save-obb

# 2) Train then predict (unchanged behavior)
python train_and_visualize_yolo_seg_scans.py --root root_folder --names object
"""
import os
os.environ["WANDB_DISABLED"] = "true"
import argparse, yaml, shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ───────────────────────── helper: YAML ─────────────────────────
# def write_data_yaml(root: Path, class_names, scans_subdir="scans") -> Path:
#     yaml_path = root / 'data.yaml'
#     data = dict(
#         path=str(root),
#         train=f'images/{scans_subdir}',
#         val=f'images/{scans_subdir}',
#         names=class_names
#     )
#     with open(yaml_path, 'w') as f:
#         yaml.safe_dump(data, f)
#     return yaml_path

def write_data_yaml(root: Path, class_names, scans_subdir="scans",
                    train_list: Path | None = None, val_list: Path | None = None) -> Path:
    yaml_path = root / 'data.yaml'
    data = dict(
        path=str(root),
        train=str(train_list) if train_list else f'images/{scans_subdir}',
        val=str(val_list)     if val_list   else f'images/{scans_subdir}',
        names=class_names
    )
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f)
    return yaml_path

def check_labels_exist(root: Path, scans_subdir="scans", exts=(".jpg", ".jpeg", ".png", ".bmp")):
    img_root = root / "images" / scans_subdir
    lbl_root = root / "labels" / scans_subdir
    missing = []
    for img in img_root.rglob("*"):
        if img.is_file() and img.suffix.lower() in exts:
            rel = img.relative_to(img_root).with_suffix(".txt")
            if not (lbl_root / rel).exists():
                missing.append((img, lbl_root / rel))
    if missing:
        print(f"[WARN] {len(missing)} images have no matching label .txt (showing up to 20):")
        for i, (img, lbl) in enumerate(missing[:20], 1):
            print(f"  {i:>3}. {img}  -> expected {lbl}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("✓ Label sanity check passed (1:1 images↔labels).")

# ─────────────────────────── args ───────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root',  type=Path, required=True, help='dataset root folder')
    p.add_argument('--names', nargs='+', required=True,
                   help='space-separated class names, e.g. --names crack rust')
    p.add_argument('--model', default='yolov8n-seg.pt', help='base checkpoint')
    p.add_argument('--train-list', type=Path, help='txt with training image paths')
    p.add_argument('--val-list',   type=Path, help='txt with validation image paths')
    p.add_argument('--weights', default='', help='path to trained weights (best.pt). If set with --predict-only, training is skipped.')
    p.add_argument('--epochs', default=100, type=int)
    p.add_argument('--imgsz',  default=640, type=int)
    p.add_argument('--batch',  default=4,  type=int)
    p.add_argument('--device', default='0', help='"cpu" or cuda index (e.g. 0)')
    p.add_argument('--scans-subdir', default='scans', help='subdir under images/ and labels/')
    p.add_argument('--predict-only', action='store_true', help='skip training and only run prediction')
    p.add_argument('--save-obb', action='store_true', help='also write oriented boxes as txt (class x1 y1 x2 y2 x3 y3 x4 y4, normalized)')
    return p.parse_args()

# ──────────────── oriented box helpers (from segmentation masks) ─────────────
def polygon_to_minrect(poly_xy: np.ndarray):
    """poly_xy: (N,2) float32 array in absolute image coords"""
    if poly_xy.dtype != np.float32:
        poly_xy = poly_xy.astype(np.float32)
    rect = cv2.minAreaRect(poly_xy)                  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)                        # 4x2 float32, ordered
    return box                                       # [[x,y],...]

def draw_obb(image: np.ndarray, box: np.ndarray, color=(255, 0, 0), thickness=2):
    pts = box.reshape(-1, 1, 2).astype(int)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

def save_obb_txt(txt_path: Path, cls_id: int, box: np.ndarray, W: int, H: int):
    # normalize x,y to [0,1]
    norm = []
    for x, y in box:
        norm.extend([x / W, y / H])
    line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm) + "\n"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "a") as f:
        f.write(line)

# ─────────────────────────── predict (OBB) ───────────────────────────────
def predict_with_oriented_boxes(weights_path: Path, args):
    model = YOLO(str(weights_path))
    vis_dir = Path('runs/segment/train_vis_obb')
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # TXT export dir (mirrors dataset structure)
    obb_txt_root = Path('runs/segment/train_vis_obb_txt')

    # Absolute path to images/scans
    root_images = (args.root / 'images' / args.scans_subdir).resolve()

    sources = str(root_images / '**' / '*')
    results = model.predict(
        source=sources,
        imgsz=args.imgsz,
        device=args.device,
        conf=0.1,
        max_det=300,
        stream=True,
        verbose=False
    )

    for r in results:
        im_path = Path(r.path).resolve()
        img = cv2.imread(str(im_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        if r.masks is None:
            # fallback: draw axis-aligned boxes
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = b.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imwrite(str(vis_dir / im_path.name), img)
            continue

        polys = r.masks.xy
        clses = r.boxes.cls.cpu().numpy().astype(int)

        for poly, cls_id in zip(polys, clses):
            box = polygon_to_minrect(poly)
            draw_obb(img, box, color=(0, 0, 255), thickness=2)

            if args.save_obb:
                try:
                    rel = im_path.relative_to(root_images)
                except ValueError:
                    # If not under root_images, just use filename
                    rel = im_path.name
                txt_path = obb_txt_root / rel
                txt_path = txt_path.with_suffix('.txt')
                save_obb_txt(txt_path, int(cls_id), box, W, H)

        cv2.imwrite(str(vis_dir / im_path.name), img)

    print(f"\n✓ Oriented visualizations saved to {vis_dir.resolve()}")
    if args.save_obb:
        print(f"✓ OBB txt files saved to {obb_txt_root.resolve()} (class x1 y1 x2 y2 x3 y3 x4 y4, normalized)")


# ─────────────────────────── main ───────────────────────────────
def main():
    args = parse_args()
    check_labels_exist(args.root, scans_subdir=args.scans_subdir)
    # data_yaml = write_data_yaml(args.root, args.names, scans_subdir=args.scans_subdir)
    args.root = args.root.resolve()
    data_yaml = write_data_yaml(
        args.root, args.names, scans_subdir=args.scans_subdir,
        train_list=args.train_list, val_list=args.val_list
)

    # Train unless predict-only is requested with weights
    if args.predict_only and args.weights:
        best_ckpt = Path(args.weights)
    else:
        print('\n=== Training YOLO-v8 segmentation model ===')
        model = YOLO(args.model)
        train_res = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            close_mosaic=True, mosaic=0,
        )
        best_ckpt = Path(train_res.save_dir) / 'weights/best.pt'
        print(f'\n✓ Training finished. Best weights: {best_ckpt}')

    # Predict with oriented boxes (from masks)
    print('\n=== Predicting and saving ORIENTED boxes (from seg masks) ===')
    predict_with_oriented_boxes(best_ckpt, args)

if __name__ == '__main__':
    main()
