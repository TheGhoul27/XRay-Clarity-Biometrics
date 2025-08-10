#!/usr/bin/env python3
"""
train_and_visualize_yolo_seg_scans.py

Dataset layout (example)
root_folder/
 ├─ images/
 │   └─ scans/
 │       ├─ counterfeit/  *.jpg|png|bmp
 │       └─ mint/        *.jpg|png|bmp
 └─ labels/
     └─ scans/
         ├─ counterfeit/  *.txt  (seg polygons)
         └─ mint/         *.txt

Usage
-----
python train_and_visualize_yolo_seg_scans.py --root root_folder \
       --names object --model yolov8n-seg.pt --epochs 100 --imgsz 640 --batch 4
"""
import os
os.environ["WANDB_DISABLED"] = "true"
import argparse, yaml, shutil
from pathlib import Path
from ultralytics import YOLO

# ───────────────────────── helper: YAML ─────────────────────────
def write_data_yaml(root: Path, class_names, scans_subdir="scans") -> Path:
    """
    Create data.yaml for Ultralytics-YOLO with images under images/scans
    and labels auto-resolved under labels/scans.
    """
    yaml_path = root / 'data.yaml'
    data = dict(
        path=str(root),
        train=f'images/{scans_subdir}',
        val=f'images/{scans_subdir}',     # use the same set by default
        names=class_names
    )
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f)
    return yaml_path

def check_labels_exist(root: Path, scans_subdir="scans", exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Quick sanity check: for each image under images/scans/**,
    verify a matching labels/scans/**.txt exists.
    """
    img_root = root / "images" / scans_subdir
    lbl_root = root / "labels" / scans_subdir
    missing = []

    for img in img_root.rglob("*"):
        if img.is_file() and img.suffix.lower() in exts:
            rel = img.relative_to(img_root).with_suffix(".txt")
            lbl = lbl_root / rel
            if not lbl.exists():
                missing.append((img, lbl))

    if missing:
        print(f"[WARN] {len(missing)} images have no matching label .txt:")
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
    p.add_argument('--epochs', default=100, type=int)
    p.add_argument('--imgsz',  default=640, type=int)
    p.add_argument('--batch',  default=4,  type=int)
    p.add_argument('--device', default='0', help='"cpu" or cuda index (e.g. 0)')
    p.add_argument('--scans-subdir', default='scans',
                   help='subdirectory under images/ and labels/ that holds data')
    return p.parse_args()

# ─────────────────────────── main ───────────────────────────────
def main():
    args = parse_args()

    # 0. quick sanity check that mirrors exist
    check_labels_exist(args.root, scans_subdir=args.scans_subdir)

    # 1. create data.yaml
    data_yaml = write_data_yaml(args.root, args.names, scans_subdir=args.scans_subdir)

    # 2. TRAIN
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

    # 3. VISUALISE on the whole scans set
    print('\n=== Predicting on images/scans and saving overlays ===')
    vis_dir = Path('runs/segment/train_vis')
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(best_ckpt))  # reload best weights
    model.predict(
        source=str(args.root / 'images' / args.scans_subdir / '**' / '*'),
        imgsz=args.imgsz,
        project='runs/segment',
        name='train_vis',
        exist_ok=True,
        save=True,       # writes annotated images (PNG)
        conf=0.1,
        max_det=300,
        stream=False
    )

    print(f'\n✓ Visualisations saved to {vis_dir.resolve()}')
    print('Open any PNG inside that folder to inspect the masks.')

if __name__ == '__main__':
    main()