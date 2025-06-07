# # pipeline.py
# """
# High-level convenience:  batch_dir() and single_pair()
# """
# from pathlib import Path
# import cv2
# from .seg_utils import Segmenter, polygon_to_obb, crop_from_obb, save_mask_or_box
# from .align_utils import orb_align


# def process_instance(
#     img_mov:  str,
#     img_ref:  str,
#     out_dir:  Path,
#     seg: Segmenter,
#     class_filter=None,
#     dump_debug=False
# ):
#     img_m = cv2.imread(str(img_mov))
#     img_r = cv2.imread(str(img_ref))
#     if img_m is None or img_r is None:
#         raise FileNotFoundError("Could not read images")

#     for i,(poly,cid) in enumerate(seg.segment(img_mov, keep_cls=class_filter)):
#         box   = polygon_to_obb(poly)
#         cropM = crop_from_obb(img_m, box)

#         inst_dir = out_dir / f"{Path(img_mov).stem}_cls{cid}_{i}"
#         inst_dir.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(inst_dir/'obb_overlay.jpg'),
#                     cv2.polylines(img_m.copy(),[box],True,(0,255,0),2))
#         cv2.imwrite(str(inst_dir/'crop_moving.jpg'), cropM)
#         cv2.imwrite(str(inst_dir/'reference.jpg'),   img_r)

#         if dump_debug:
#             save_mask_or_box(img_m.shape, poly, box,
#                              inst_dir/'mask.png', inst_dir/'obb.txt')

#         try:
#             orb_align(cropM, img_r, inst_dir)
#         except Exception as e:
#             print(f"× align fail {img_mov} inst{i}: {e}")
#         else:
#             print(f"✓ {img_mov} cls{cid} inst{i}")


# def batch_dir(img_dir: Path, ref_dir: Path, out_root: Path,
#               ckpt: Path, *, class_filter=None, **seg_kwargs):
#     seg = Segmenter(ckpt, **seg_kwargs)          # ← no class_filter here
#     for img in sorted(img_dir.glob('*')):
#         ref = ref_dir / img.name
#         if not ref.exists():
#             print(f"⚠ {img.name} missing template"); continue
#         process_instance(img, ref, out_root, seg,
#                          class_filter=class_filter)

# def single_pair(moving: Path, reference: Path, out_dir: Path,
#                 ckpt: Path, *, class_filter=None, **seg_kwargs):
#     seg = Segmenter(ckpt, **seg_kwargs)          # ← same fix
#     process_instance(moving, reference, out_dir, seg,
#                      class_filter=class_filter)


# pipeline.py
"""
High-level helpers:
    batch_dir(...)   – iterate two folders with matching filenames
    single_pair(...) – one moving+reference pair
Both call process_instance().
"""
from pathlib import Path
import cv2
from .seg_utils import (Segmenter, polygon_to_obb,
                       crop_from_obb, save_mask_or_box)
from .align_utils import orb_align


def process_instance(img_mov, img_ref, out_dir: Path,
                     seg: Segmenter,
                     class_filter=None,
                     dump_debug=False):
    img_m = cv2.imread(str(img_mov))
    img_r = cv2.imread(str(img_ref))
    if img_m is None or img_r is None:
        raise FileNotFoundError("Failed to read images")

    for i,(poly,cid) in enumerate(seg.segment(img_mov, keep_cls=class_filter)):
        box   = polygon_to_obb(poly)
        cropM = crop_from_obb(img_m, box)

        inst = out_dir / f"{Path(img_mov).stem}_cls{cid}_{i}"
        inst.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(inst/'obb_overlay.jpg'),
                    cv2.polylines(img_m.copy(), [box], True, (0,255,0),2))
        cv2.imwrite(str(inst/'crop_moving.jpg'), cropM)
        cv2.imwrite(str(inst/'reference.jpg'),   img_r)

        if dump_debug:
            save_mask_or_box(img_m.shape, poly, box,
                             inst/'mask.png', inst/'obb.txt')

        try:
            orb_align(cropM, img_r, inst)
        except Exception as e:
            print(f"× align fail {img_mov} inst{i}: {e}")
        else:
            print(f"✓ {img_mov} cls{cid} inst{i}")


def batch_dir(img_dir: Path, ref_dir: Path, out_root: Path, ckpt: Path,
              *, class_filter=None, dump_debug=False,
              imgsz=640, conf=0.1, device="cuda"):
    seg = Segmenter(ckpt, imgsz=imgsz, conf=conf, device=device)
    for img in sorted(img_dir.glob('*')):
        ref = ref_dir / img.name
        if not ref.exists():
            print(f"⚠ {img.name} missing template"); continue
        process_instance(img, ref, out_root, seg,
                         class_filter=class_filter,
                         dump_debug=dump_debug)


def single_pair(moving: Path, reference: Path, out_dir: Path, ckpt: Path,
                *, class_filter=None, dump_debug=False,
                imgsz=640, conf=0.1, device="cuda"):
    seg = Segmenter(ckpt, imgsz=imgsz, conf=conf, device=device)
    process_instance(moving, reference, out_dir, seg,
                     class_filter=class_filter,
                     dump_debug=dump_debug)
