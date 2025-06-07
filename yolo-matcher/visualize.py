# """
# visualize_yolo_obb.py
# ---------------------
# Render every labelled oriented-bounding-box (OBB) in a YOLO dataset.

# Usage
# -----
# python visualize_yolo_obb.py --root dataset --split train --save

# Requirements
# ------------
# pip install ultralytics opencv-python
# """
# import argparse, math, cv2, sys
# from pathlib import Path
# from ultralytics.utils.plotting import colors          # optional: nice class colours

# # ------------------------- helpers ------------------------------------------------
# def _five_to_poly(xc, yc, w, h, theta, W, H):
#     """(cx,cy,w,h,θ) → 4×2 polygon in pixel coords."""
#     xc, yc, w, h = xc*W, yc*H, w*W, h*H
#     cos_t, sin_t = math.cos(theta), math.sin(theta)
#     dx, dy = w/2, h/2
#     base = [(-dx,-dy), ( dx,-dy), ( dx, dy), (-dx, dy)]
#     return [(int(x*cos_t - y*sin_t + xc),
#              int(x*sin_t + y*cos_t + yc)) for x, y in base]

# def _draw_poly(img, poly, col, thick=2):
#     p = poly + [poly[0]]
#     for i in range(4):
#         cv2.line(img, p[i], p[i+1], col, thick, cv2.LINE_AA)

# # ------------------------- main ---------------------------------------------------
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--root',  type=Path, required=True, help='dataset root folder')
#     p.add_argument('--split', default='train',          help='sub-folder to visualise')
#     p.add_argument('--save',  action='store_true',      help='write composites to <root>/vis')
#     return p.parse_args()

# def main():
#     args  = parse_args()
#     img_d = args.root / 'images' / args.split
#     lbl_d = args.root / 'labels' / args.split
#     out_d = args.root / 'vis'    / args.split
#     if args.save:
#         out_d.mkdir(parents=True, exist_ok=True)

#     images = sorted(img_d.glob('*'))
#     if not images:
#         sys.exit(f'✗ No images found under {img_d}')

#     for im_p in images:
#         lbl_p = lbl_d / (im_p.stem + '.txt')
#         if not lbl_p.exists():
#             print(f'⚠  missing label for {im_p.name}', file=sys.stderr);  continue

#         img = cv2.imread(str(im_p));  H, W = img.shape[:2]

#         with open(lbl_p) as f:
#             for line in f:
#                 nums = [float(x) for x in line.strip().split()]
#                 cls, vals = int(nums[0]), nums[1:]

#                 # --- handle 5-tuple or 8-tuple -----------------------------------
#                 if len(vals) == 5:            # cx cy w h θ  (Ultralytics OBB)  :contentReference[oaicite:0]{index=0}
#                     poly = _five_to_poly(*vals, W, H)
#                 elif len(vals) == 8:          # x1 y1 … x4 y4
#                     poly = [(int(vals[i]*W), int(vals[i+1]*H)) for i in range(0,8,2)]
#                 else:
#                     print(f'⚠  bad line in {lbl_p}: {line}', file=sys.stderr);  continue

#                 clr = colors(cls, True)       # distinct colour per class
#                 _draw_poly(img, poly, clr)
#                 cv2.putText(img, str(cls), poly[0], cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5, clr, 1, cv2.LINE_AA)

#         # -------------------- show or save ----------------------------------------
#         if args.save:
#             cv2.imwrite(str(out_d / im_p.name), img)
#         else:
#             cv2.imshow('YOLO-OBB viewer', img)
#             if cv2.waitKey(0) == 27:   # Esc to quit early
#                 break
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()


"""
visualize_yolo_labels.py
========================
Render every YOLO-style label (OBB or polygon) on top of its image.

Directory layout expected
-------------------------
<root>/
 ├─ images/<split>/  img1.jpg|png|bmp …
 └─ labels/<split>/  img1.txt …

Label line formats supported
----------------------------
①  5-tuple  OBB …  cls cx cy w h θ
②  8-tuple  quad … cls x1 y1 x2 y2 x3 y3 x4 y4
③  Polygon ≥ 6 …   cls x1 y1 x2 y2 … xn yn   (any even length ≥ 6)

All coordinates are **normalised** to [0,1] (YOLO convention).

Usage examples
--------------
# pop-up windows (needs GUI build of OpenCV)
python visualize_yolo_labels.py --root dataset --split train

# just save composites (always works)
python visualize_yolo_labels.py --root dataset --split train --save

Dependencies
------------
pip install ultralytics opencv-python
(
    or  opencv-python-headless  if you never want windows
)
"""
from pathlib import Path
import argparse, math, cv2, sys
from ultralytics.utils.plotting import colors


# ───────────────────────── helpers ────────────────────────────────────────
def five_to_poly(cx, cy, w, h, theta, W, H):
    """(cx,cy,w,h,θ)  →  list[(x,y)] (4 corners, int px)."""
    cx, cy, w, h = cx * W, cy * H, w * W, h * H
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = w / 2, h / 2
    base = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return [(int(x * c - y * s + cx), int(x * s + y * c + cy)) for x, y in base]


def draw_poly(img, pts, col, thick=2):
    """Draw closed polyline."""
    n = len(pts)
    for i in range(n):
        cv2.line(img, pts[i], pts[(i + 1) % n], col, thick, cv2.LINE_AA)


# ───────────────────────── main ───────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="dataset root folder")
    p.add_argument("--split", default="train", help="sub-folder (train / val / test)")
    p.add_argument("--save", action="store_true", help="save composites instead of/showing them")
    p.add_argument("--exts", nargs="+", default=["jpg", "png", "bmp"],
                   help="image file extensions to consider")
    return p.parse_args()


def main():
    args = parse_args()
    img_dir = args.root / "images" / args.split
    lbl_dir = args.root / "labels" / args.split
    out_dir = args.root / "vis" / args.split
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # gather images
    imgs = []
    for ext in args.exts:
        imgs.extend(img_dir.glob(f"*.{ext}"))
    imgs.sort()

    if not imgs:
        sys.exit(f"✗ No images found in {img_dir}")

    # decide whether we will try to show windows
    want_show = not args.save
    gui_available = True

    for im_path in imgs:
        lbl_path = lbl_dir / (im_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"⚠  missing label for {im_path.name}", file=sys.stderr)
            continue

        img = cv2.imread(str(im_path))
        if img is None:
            print(f"⚠  cannot read {im_path}", file=sys.stderr)
            continue
        H, W = img.shape[:2]

        with open(lbl_path) as f:
            for line in f:
                nums = [float(x) for x in line.strip().split()]
                if len(nums) < 3 or (len(nums) - 1) % 2:
                    print(f"⚠  bad line in {lbl_path}: {line.strip()}", file=sys.stderr)
                    continue

                cls, vals = int(nums[0]), nums[1:]
                col = colors(cls, True)

                # ① OBB 5-tuple
                if len(vals) == 5:
                    poly = five_to_poly(*vals, W, H)

                # ② Quad 8-tuple
                elif len(vals) == 8:
                    poly = [(int(vals[i] * W), int(vals[i + 1] * H))
                            for i in range(0, 8, 2)]

                # ③ Generic polygon ≥ 6
                else:
                    poly = [(int(vals[i] * W), int(vals[i + 1] * H))
                            for i in range(0, len(vals), 2)]

                draw_poly(img, poly, col)
                cv2.putText(img, str(cls), poly[0], cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 1, cv2.LINE_AA)

        # ─────── write vs. show ───────────────────────────────────────────
        if args.save:
            cv2.imwrite(str(out_dir / im_path.name), img)
        elif want_show and gui_available:
            try:
                cv2.imshow("YOLO label viewer", img)
                if cv2.waitKey(0) == 27:  # Esc to quit early
                    want_show = False
            except cv2.error:
                # Headless build – fall back to 'save' mode for rest of run
                gui_available = False
                print("⚠  OpenCV installed without GUI support. "
                      "Continuing in --save-only mode.", file=sys.stderr)
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / im_path.name), img)

    # close window only if one was opened successfully
    if want_show and gui_available:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()