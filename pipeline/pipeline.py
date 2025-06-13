"""
pipeline.py – high-level convenience wrappers.

Functions
---------
process_instance_dual(...)  – front/back orientation + rotation
batch_dir_dual(...)         – iterate three folders (images / front / back)
single_pair_dual(...)       – single image with two templates

Legacy one-reference helpers (process_instance / batch_dir / single_pair)
are retained for compatibility.
"""

from pathlib import Path
import cv2

from .seg_utils import (Segmenter, polygon_to_obb,
                        crop_from_obb, save_mask_or_box)
from .align_utils import rotate_align, rotate_align_two_refs

def process_instance_dual(img_mov, front_ref, back_ref, out_dir: Path,
                          seg: Segmenter, *,
                          class_filter=None, dump_debug=False,
                          angle_step: int = 5):
    img_m = cv2.imread(str(img_mov))
    img_f = cv2.imread(str(front_ref))
    img_b = cv2.imread(str(back_ref))
    if img_m is None or img_f is None or img_b is None:
        raise FileNotFoundError("Could not read moving or reference images")

    for i, (poly, cid) in enumerate(
            seg.segment(img_mov, keep_cls=class_filter)):
        box   = polygon_to_obb(poly)
        cropM = crop_from_obb(img_m, box)

        inst_dir = out_dir / f"{Path(img_mov).stem}_cls{cid}_{i}"
        inst_dir.mkdir(parents=True, exist_ok=True)

        # ─── save extra diagnostics only when dump_debug ────────────
        if dump_debug:
            cv2.imwrite(str(inst_dir / "obb_overlay.jpg"),
                        cv2.polylines(img_m.copy(), [box], True, (0, 255, 0), 2))
            cv2.imwrite(str(inst_dir / "crop_moving.jpg"), cropM)
            save_mask_or_box(img_m.shape, poly, box,
                             inst_dir / "mask.png", inst_dir / "obb.txt")

        # ─── orientation + rotation search ─────────────────────────
        best_rot, ang, side, sim, *_ = rotate_align_two_refs(
            cropM, img_f, img_b, inst_dir,
            device=seg.device, angle_step=angle_step,
            dump_debug=dump_debug, model_name="vit_base_patch14_dinov2.lvd142m")             # <── pass the flag

        # save ONLY the winning reference
        best_ref = img_f if side == "front" else img_b
        cv2.imwrite(str(inst_dir / "best_reference.jpg"), best_ref)

        print(f"✓ {img_mov.name} inst{i} → {side:<5} "
              f"angle={ang:3d}°  sim={sim:.3f}")


def batch_dir_dual(img_dir: Path, front_dir: Path, back_dir: Path,
                   out_root: Path, ckpt: Path, *,
                   class_filter=None, dump_debug=False,
                   imgsz=640, conf=0.1, device="cuda", angle_step=5):
    seg = Segmenter(ckpt, imgsz=imgsz, conf=conf, device=device)

    for img in sorted(img_dir.glob("*")):
        front = front_dir / img.name
        back  = back_dir  / img.name
        if not front.exists() or not back.exists():
            print(f"⚠ references missing for {img.name}")
            continue
        process_instance_dual(img, front, back, out_root, seg,
                              class_filter=class_filter,
                              dump_debug=dump_debug,
                              angle_step=angle_step)


def single_pair_dual(moving: Path, front_ref: Path, back_ref: Path,
                     out_dir: Path, ckpt: Path, *,
                     class_filter=None, dump_debug=False,
                     imgsz=640, conf=0.1, device="cuda", angle_step=5):
    seg = Segmenter(ckpt, imgsz=imgsz, conf=conf, device=device)
    process_instance_dual(moving, front_ref, back_ref, out_dir, seg,
                          class_filter=class_filter,
                          dump_debug=dump_debug,
                          angle_step=angle_step)


# ─────────────────────── legacy single-reference helpers ────────────────
def process_instance(img_mov, img_ref, out_dir: Path, seg: Segmenter, *,
                     class_filter=None, dump_debug=False, angle_step=5):
    """(Older) helper that aligns against a single template."""
    img_m = cv2.imread(str(img_mov))
    img_r = cv2.imread(str(img_ref))
    if img_m is None or img_r is None:
        raise FileNotFoundError("Failed to read images")

    for i, (poly, cid) in enumerate(
            seg.segment(img_mov, keep_cls=class_filter)):
        box   = polygon_to_obb(poly)
        cropM = crop_from_obb(img_m, box)

        inst_dir = out_dir / f"{Path(img_mov).stem}_cls{cid}_{i}"
        inst_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(inst_dir / "obb_overlay.jpg"),
                    cv2.polylines(img_m.copy(), [box], True, (0, 255, 0), 2))
        cv2.imwrite(str(inst_dir / "crop_moving.jpg"), cropM)
        cv2.imwrite(str(inst_dir / "reference.jpg"),   img_r)

        if dump_debug:
            save_mask_or_box(img_m.shape, poly, box,
                             inst_dir / "mask.png", inst_dir / "obb.txt")

        try:
            best_rot, ang, sim = rotate_align(
                cropM, img_r, inst_dir,
                device=seg.device, angle_step=angle_step, model_name="vit_base_patch14_dinov2.lvd142m")
        except Exception as e:
            print(f"× align fail {img_mov} inst{i}: {e}")
        else:
            status = "search" if ang != 0 else "fallback"
            print(f"✓ {img_mov.name} inst{i}  angle={ang:3d}°  {status}")


def batch_dir(img_dir: Path, ref_dir: Path, out_root: Path, ckpt: Path, *,
              class_filter=None, dump_debug=False,
              imgsz=640, conf=0.1, device="cuda", angle_step=5):
    seg = Segmenter(ckpt, imgsz=imgsz, conf=conf, device=device)
    for img in sorted(img_dir.glob("*")):
        ref = ref_dir / img.name
        if not ref.exists():
            print(f"⚠ {img.name} missing template")
            continue
        process_instance(img, ref, out_root, seg,
                         class_filter=class_filter,
                         dump_debug=dump_debug,
                         angle_step=angle_step)


def single_pair(moving: Path, reference: Path, out_dir: Path, ckpt: Path, *,
                class_filter=None, dump_debug=False,
                imgsz=640, conf=0.1, device="cuda", angle_step=5):
    seg = Segmenter(ckpt, imgsz=imgsz, conf=conf, device=device)
    process_instance(moving, reference, out_dir, seg,
                     class_filter=class_filter,
                     dump_debug=dump_debug,
                     angle_step=angle_step)
