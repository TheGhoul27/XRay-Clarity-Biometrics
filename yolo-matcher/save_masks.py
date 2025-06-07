# # save_masks.py  (or paste into your training script)
# import cv2, numpy as np
# from pathlib import Path
# from ultralytics import YOLO


# def dump_yolo_masks(
#         ckpt: str | Path,
#         img_dir: str | Path,
#         out_dir: str | Path,
#         imgsz: int = 640,
#         conf: float = 0.1,
#         device: str = "0",
# ):
#     """
#     Run YOLO-v8 segmentation inference and save *raw* binary masks.

#     Args
#     ----
#     ckpt    : path to trained .pt checkpoint
#     img_dir : folder with images to segment
#     out_dir : where to write mask PNGs
#     imgsz   : inference image size
#     conf    : confidence threshold
#     device  : "cpu" or cuda index, e.g. "0"
#     """
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     model = YOLO(str(ckpt))

#     for res in model.predict(
#         source=str(img_dir),
#         imgsz=imgsz,
#         conf=conf,
#         device=device,
#         stream=True,        # yield results one by one
#         verbose=False
#     ):
#         if res.masks is None:
#             continue                      # no detections in this image

#         im_name = Path(res.path).stem     # e.g. IMG_123
#         masks   = res.masks.data.cpu().numpy()      # (N, H, W)  bool/0-1
#         classes = res.boxes.cls.cpu().numpy().astype(int)  # (N,)

#         # save each instance mask separately
#         for i, (m, cls) in enumerate(zip(masks, classes)):
#             mask_uint8 = (m * 255).astype(np.uint8)
#             mask_path  = out_dir / f"{im_name}_cls{cls}_{i}.png"
#             cv2.imwrite(str(mask_path), mask_uint8)

#     print(f"✓ Masks saved to {out_dir.resolve()}")

"""
save_masks.py  –  Dump per-instance binary masks from a trained YOLO-v8
-----------------------------------------------------------------------

Example
-------
python save_masks.py \
       --ckpt runs/segment/train/weights/best.pt \
       --img_dir dataset/images/train \
       --out_dir runs/segment/train_masks \
       --imgsz 640 --conf 0.1 --device 0
"""
import argparse, cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys


def dump_yolo_masks(ckpt, img_dir, out_dir, imgsz, conf, device):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'✓ Loading model  {ckpt}')
    model = YOLO(str(ckpt))

    # stream=True yields one result at a time (low RAM, good progress prints)
    for res in model.predict(
        source=str(img_dir),
        imgsz=imgsz,
        conf=conf,
        device=device,
        stream=True,
        verbose=False
    ):
        if res.masks is None:          # no detections in that image
            continue

        im_name = Path(res.path).stem
        masks   = res.masks.data.cpu().numpy()        # (N,H,W) 0/1
        classes = res.boxes.cls.cpu().numpy().astype(int)

        for i, (m, cls) in enumerate(zip(masks, classes)):
            mask = (m * 255).astype(np.uint8)         # 0/255 binary PNG
            fout = out_dir / f'{im_name}_cls{cls}_{i}.png'
            cv2.imwrite(str(fout), mask)

    print(f'✓ Masks saved to  {out_dir.resolve()}')


# ----------------- CLI wrapper -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Dump YOLO-v8 segmentation masks")
    p.add_argument('--ckpt',    required=True, type=Path, help='best.pt checkpoint')
    p.add_argument('--img_dir', required=True, type=Path, help='folder with images')
    p.add_argument('--out_dir', required=True, type=Path, help='where to save PNGs')
    p.add_argument('--imgsz',   default=640, type=int)
    p.add_argument('--conf',    default=0.1, type=float)
    p.add_argument('--device',  default='0',
                   help='"cpu" or CUDA index, e.g. 0 or 0,1')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        dump_yolo_masks(
            ckpt=args.ckpt,
            img_dir=args.img_dir,
            out_dir=args.out_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device
        )
    except Exception as e:
        # print full traceback but keep window open on double-click
        import traceback, sys
        traceback.print_exc()
        input("\n[ERROR]  Press <Enter> to exit…")
        sys.exit(1)

    # keep window open if user double-clicked the file
    if sys.stdout.isatty() is False:     # likely double-clicked
        input("\nDone. Press <Enter> to close.")
