# seg_utils.py
"""
YOLO-v8 segmentation helpers:
    • Segmenter   – wraps a checkpoint and returns polygons
    • polygon_to_obb
    • crop_from_obb
    • save_mask_or_box (optional debugging helper)
"""
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class Segmenter:
    def __init__(self, ckpt: str,
                 imgsz: int = 640,
                 conf: float = 0.1,
                 device: str = "cuda"):
        self.model  = YOLO(str(ckpt))
        self.imgsz  = imgsz
        self.conf   = conf
        self.device = device

    def segment(self, img_path: str,
                keep_cls: Optional[set[int]] = None
                ) -> List[Tuple[np.ndarray, int]]:
        """Return list of (polygon Nx2 float32, class-id)."""
        res = self.model.predict(
            source=str(img_path),
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            stream=False)[0]

        if res.masks is None:
            return []

        polys = res.masks.xy                      # list of arrays
        cls   = res.boxes.cls.cpu().numpy().astype(int)
        out   = []
        for p,c in zip(polys, cls):
            if keep_cls and c not in keep_cls:
                continue
            out.append((np.asarray(p, np.float32), c))
        return out


# ───────── geometry helpers ────────────────────────────────────────────
def polygon_to_obb(poly: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(poly)
    return cv2.boxPoints(rect).astype(int)       # 4×2


def crop_from_obb(img: np.ndarray, box: np.ndarray, scale: float = 1.05):
    w = np.linalg.norm(box[0]-box[1])
    h = np.linalg.norm(box[1]-box[2])
    W,H = int(w*scale), int(h*scale)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    M   = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W,H))


def save_mask_or_box(img_shape, poly, box, save_mask=None, save_box=None):
    if save_mask:
        mask = np.zeros(img_shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly.astype(int)], 255)
        cv2.imwrite(str(save_mask), mask)
    if save_box:
        np.savetxt(str(save_box), box, fmt='%d')
