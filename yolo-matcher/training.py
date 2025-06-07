#!/usr/bin/env python3
"""
train_and_visualize_yolo_seg.py
================================
1. Creates a minimal data.yaml that points to your images/labels-seg structure
2. Trains a YOLO-v8 segmentation model
3. Runs inference on *all* training images and saves visualisations

Requirements
------------
pip install ultralytics opencv-python           # full OpenCV build (GUI optional)

Directory expected
------------------
dataset/
 ├─ images/
 │   └─ train/  *.jpg|png|bmp
 └─ labels-seg/          ←  bbox+polygon files from the previous converter
     └─ train/  *.txt

Usage
-----
python train_and_visualize_yolo_seg.py --root dataset \
       --names object                               \
       --model yolov8n-seg.pt                       \
       --epochs 100 --imgsz 640 --batch 4
"""
import os                    
os.environ["WANDB_DISABLED"] = "true"
import argparse, yaml, shutil, sys
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────── helper: YAML ────────────────────────────────
def write_data_yaml(root: Path, class_names, split='train') -> Path:
    """Create a temporary data.yaml for Ultralytics-YOLO."""
    yaml_path = root / 'data.yaml'
    data = dict(
        path=str(root),
        train=f'images/{split}',
        val=f'images/{split}',           # over-fit on train for a quick check
        names=class_names
    )
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f)
    return yaml_path


# ─────────────────────────── main flow ───────────────────────────────────
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
    return p.parse_args()


def main():
    args = parse_args()

    # 1. create data.yaml
    data_yaml = write_data_yaml(args.root, args.names)

    # 2. TRAIN
    print('\n=== Training YOLO-v8 segmentation model ===')
    model = YOLO(args.model)
    train_res = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        close_mosaic=True, mosaic=0,         # helpful when over-fitting
    )

    best_ckpt = Path(train_res.save_dir) / 'weights/best.pt'
    print(f'\n✓ Training finished. Best weights: {best_ckpt}')

    # 3. VISUALISE on *all* training images
    print('\n=== Predicting on training set and saving overlays ===')
    vis_dir = Path('runs/segment/train_vis')
    if vis_dir.exists():
        shutil.rmtree(vis_dir)              # clean previous run
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(best_ckpt))            # reload best weights
    model.predict(
        source=str(args.root / 'images/train'),
        imgsz=args.imgsz,
        project='runs/segment',
        name='train_vis',
        exist_ok=True,
        save=True,       # writes annotated images (PNG) into the folder above
        conf=0.1,        # low threshold to see every mask
        max_det=300,
        stream=False
    )

    print(f'\n✓ Visualisations saved to {vis_dir.resolve()}')
    print('Open any PNG inside that folder to inspect the masks.')


if __name__ == '__main__':
    main()
