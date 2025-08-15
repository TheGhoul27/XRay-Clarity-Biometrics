#!/usr/bin/env python3
"""
Crop images with YOLOv8-seg oriented boxes (masks.xy -> minAreaRect -> rotate+crop).

Inputs:
  dataset/
    images/
      scans/
        mint/
        counterfeit/
        ... (optionally nested)

Outputs:
  cropped/
    images/
      scans/
        mint/
        counterfeit/
        (mirrors input structure)

Usage:
  python crop_with_yolo_obb.py \
    --root dataset \
    --yolo-weights runs/segment/exp/weights/best.pt \
    --imgsz 640 \
    --yolo-conf 0.1

Optional:
  --max-side 1600       # run YOLO on resized copy, polygons rescaled back to original
  --exts .jpg .jpeg .png .tif .tiff .bmp .webp
  --overwrite           # allow overwriting existing crops
  --visualize           # also save a vis image with the OBB drawn (next to crop)
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# -------------------------- OBB helpers --------------------------

def polygon_to_minrect(poly_xy: np.ndarray) -> np.ndarray:
    """poly_xy: (N,2) float32 array in absolute image coords -> returns 4x2 box points."""
    if poly_xy.dtype != np.float32:
        poly_xy = poly_xy.astype(np.float32)
    rect = cv2.minAreaRect(poly_xy)     # ((cx,cy),(w,h),angle)
    box  = cv2.boxPoints(rect)          # 4x2 float32
    return box

def draw_obb(image: np.ndarray, box: np.ndarray, color=(0, 0, 255), thickness=2) -> np.ndarray:
    out = image.copy()
    pts = box.reshape(-1, 1, 2).astype(int)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out

# -------------------------- YOLO cropper --------------------------

class YOLOSegCropper:
    """
    YOLOv8-seg -> choose best mask (highest conf) -> masks.xy polygon -> minAreaRect -> rotate+crop.

    If --max-side is set, runs inference on a resized copy and rescales polygons back to original
    before cropping, so the final crop is in the original resolution.
    """
    def __init__(self, weights_path: str, conf_thres: float = 0.1, imgsz: int = 640, max_side: Optional[int] = None):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "Ultralytics not installed. `pip install ultralytics`.\n"
                f"Original import error: {e}"
            )
        self.model = YOLO(weights_path)
        self.conf_thres = float(conf_thres)
        self.imgsz = int(imgsz)
        self.max_side = max_side if (max_side is None or int(max_side) > 0) else None

    def _resize_long_side(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.max_side:
            return img, 1.0
        h, w = img.shape[:2]
        long = max(h, w)
        if long <= self.max_side:
            return img, 1.0
        scale = self.max_side / long
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale

    def _crop_rotated_rect(self, img: np.ndarray, rect) -> np.ndarray:
        (cx, cy), (rw, rh), angle = rect
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        x1 = int(cx - rw / 2); y1 = int(cy - rh / 2)
        x2 = int(cx + rw / 2); y2 = int(cy + rh / 2)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(rotated.shape[1], x2); y2 = min(rotated.shape[0], y2)
        return rotated[y1:y2, x1:x2].copy()

    def _center_crop(self, img: np.ndarray, frac: float = 0.9) -> np.ndarray:
        h, w = img.shape[:2]
        th, tw = int(h * frac), int(w * frac)
        y0 = (h - th) // 2
        x0 = (w - tw) // 2
        return img[y0:y0 + th, x0:x0 + tw].copy()

    def crop_with_vis(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns (crop, vis). 'vis' is None if no polygon is found.
        """
        orig = img_bgr
        resized, scale = self._resize_long_side(img_bgr)

        # Run prediction (imgsz must be an int to avoid TypeError)
        results = self.model.predict(
            resized,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            max_det=300,
            device=None,
            verbose=False
        )
        if not results:
            return self._center_crop(orig, 0.9), None

        r = results[0]

        # If no masks, try best box; else center-crop fallback
        if r.masks is None or len(r.masks) == 0:
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                idx = int(np.argmax(confs))
                x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[idx].astype(int)
                # rescale to original
                x1 = int(x1 / scale); y1 = int(y1 / scale)
                x2 = int(x2 / scale); y2 = int(y2 / scale)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(orig.shape[1], x2); y2 = min(orig.shape[0], y2)
                crop = orig[y1:y2, x1:x2].copy()
                vis  = orig.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                return (crop if crop.size else self._center_crop(orig, 0.9), vis)
            return self._center_crop(orig, 0.9), None

        # Choose mask with highest confidence
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([1.0])
        best_idx = int(np.argmax(confs)) if confs.size > 0 else 0

        polys = r.masks.xy
        if best_idx >= len(polys):
            best_idx = 0
        poly = np.array(polys[best_idx], dtype=np.float32)

        # Rescale polygon back to original coords if we resized
        if scale != 1.0:
            poly = poly / scale

        rect = cv2.minAreaRect(poly)
        crop = self._crop_rotated_rect(orig, rect)
        vis  = draw_obb(orig, polygon_to_minrect(poly))
        return (crop if crop.size else self._center_crop(orig, 0.9), vis)

# -------------------------- IO helpers --------------------------

def iter_images(root: Path, subdir: str, exts: Iterable[str]) -> Iterable[Path]:
    img_root = root / "images" / "scans" / subdir
    if not img_root.exists():
        return []
    exts_l = {e.lower() for e in exts}
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_l:
            yield p

def make_out_path(root: Path, p: Path) -> Path:
    """
    Input: .../dataset/images/scans/{mint|counterfeit}/.../file.ext
    Output: .../cropped/images/scans/{mint|counterfeit}/.../file.ext
    """
    parts = list(p.parts)
    try:
        idx = parts.index("images")
    except ValueError:
        # fallback: just mirror under cropped/ keeping relative name
        return root / "cropped" / p.name
    rel_from_images = Path(*parts[idx + 1:])  # scans/.../file.ext
    return root / "cropped" / "images" / rel_from_images

# -------------------------- main --------------------------

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Dataset root (contains images/ and labels/)")
    ap.add_argument("--yolo-weights", required=True, help="Path to YOLOv8-seg weights (.pt)")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--yolo-conf", type=float, default=0.1, help="YOLO confidence threshold")
    ap.add_argument("--max-side", type=int, default=None, help="Optional: resize long side for speed (polygons rescaled back)")
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"], help="Image extensions")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing crops")
    ap.add_argument("--visualize", action="store_true", help="Also save *_vis.jpg with the OBB drawn")
    return ap.parse_args()

def main():
    args = parse_args()
    root = args.root.resolve()

    cropper = YOLOSegCropper(
        weights_path=args.yolo_weights,
        conf_thres=args.yolo_conf,
        imgsz=args.imgsz,
        max_side=args.max_side
    )

    # Process both classes (folders)
    classes = ["mint", "counterfeit"]
    files = []
    for cls in classes:
        files.extend(list(iter_images(root, cls, args.exts)))

    if not files:
        print("No images found. Check --root and folder structure.")
        return

    for src_path in tqdm(files, desc="Cropping", unit="img"):
        img = cv2.imread(str(src_path))
        if img is None:
            continue

        crop, vis = cropper.crop_with_vis(img)

        out_path = make_out_path(root, src_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.overwrite and out_path.exists():
            continue

        # Save crop
        cv2.imwrite(str(out_path), crop)

        # Optional visualization with OBB
        if args.visualize:
            vis_path = out_path.with_name(out_path.stem + "_vis.jpg")
            if vis is None:
                # create a simple vis by drawing a yellow frame around the crop
                vis = crop.copy()
                cv2.rectangle(vis, (2, 2), (max(0, vis.shape[1]-3), max(0, vis.shape[0]-3)), (0, 255, 255), 2)
            cv2.imwrite(str(vis_path), vis)

    print("Done. Crops saved under:", root / "cropped" / "images" / "scans")

if __name__ == "__main__":
    main()