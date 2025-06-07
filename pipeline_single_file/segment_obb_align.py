# python segment_obb_align.py --ckpt E:\Masters_College_Work\RA_CyLab\X-Ray\yolo_type\runs\segment\train\weights\best.pt --moving E:\Masters_College_Work\RA_CyLab\X-Ray\data\images\135656-V0.bmp --reference E:\Masters_College_Work\RA_CyLab\X-Ray\code\135644-V0.bmp --out_dir E:\Masters_College_Work\RA_CyLab\X-Ray\pipeline\sample --classes 0

"""
segment_obb_align_ref_is_template.py
====================================
Segment a moving image, convert mask polygon → oriented bounding-box,
crop that region, and ORB-align *the crop* to an already-cropped
reference template.

Artefacts per instance
----------------------
obb_overlay.jpg        moving image with green OBB drawn
crop_moving.jpg        perspective crop from moving image
reference.jpg          the (unchanged) template you supplied
side_by_side.jpg       reference | warped-moving
aligned_keypoints.jpg  ORB inliers

Run either:

• batch  : --img_dir folder_moving  --ref_dir folder_reference  (same filenames)
• single : --moving  moving.jpg      --reference reference.jpg
"""
import os; os.environ["WANDB_DISABLED"] = "true"
import argparse, cv2, numpy as np, sys
from pathlib import Path
from ultralytics import YOLO


# ───────────────────── helpers ───────────────────────────────────────────
def crop_from_obb(img: np.ndarray, box: np.ndarray, scale: float = 1.05):
    """Perspective-crop the OBB region."""
    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])
    W, H = int(w * scale), int(h * scale)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                   np.float32)
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


def orb_align(mov: np.ndarray, ref: np.ndarray, outdir: Path,
              n_feat=5000, ratio=.75, ransac=6.0):
    """Align moving patch to reference template, dump JPEGs."""
    orb = cv2.ORB_create(nfeatures=n_feat)
    k1, d1 = orb.detectAndCompute(mov, None)
    k2, d2 = orb.detectAndCompute(ref, None)
    if d1 is None or d2 is None:
        raise RuntimeError("No ORB keypoints.")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if len(good) < 4:
        raise RuntimeError("Not enough good matches.")
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
    if H is None:
        raise RuntimeError("Homography failed.")
    h, w = ref.shape[:2]
    warped = cv2.warpPerspective(mov, H, (w, h))

    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "side_by_side.jpg"),
                cv2.hconcat([ref, warped]))

    vis, off = cv2.hconcat([ref, warped]), np.array([w, 0], int)
    for s, t, keep in zip(src, dst, inl.ravel()):
        if not keep:
            continue
        s_w = cv2.perspectiveTransform(s.reshape(1, 1, 2), H).reshape(2)
        cv2.circle(vis, tuple(t.astype(int)),      4, (0, 255, 0), -1)
        cv2.circle(vis, tuple((s_w + off).astype(int)), 4, (0, 0, 255), -1)
        cv2.line(vis, tuple(t.astype(int)),
                 tuple((s_w + off).astype(int)), (255, 200, 200), 1)
    cv2.imwrite(str(outdir / "aligned_keypoints.jpg"), vis)


# ───────────────────── CLI ───────────────────────────────────────────────
def args_():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt', required=True, type=Path)
    p.add_argument('--out_dir', required=True, type=Path)

    # batch
    p.add_argument('--img_dir', type=Path)
    p.add_argument('--ref_dir', type=Path)
    # single
    p.add_argument('--moving', type=Path)
    p.add_argument('--reference', type=Path)

    p.add_argument('--classes', nargs='*', type=int)
    p.add_argument('--imgsz', default=640, type=int)
    p.add_argument('--conf',  default=.1,  type=float)
    p.add_argument('--device', default='0')
    return p.parse_args()


# ───────────────────── main ──────────────────────────────────────────────
def main():
    a = args_()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(a.classes) if a.classes else None

    single = bool(a.moving and a.reference)
    if single:
        srcs = [str(a.moving)]
    else:
        if not (a.img_dir and a.ref_dir):
            sys.exit("Need (--img_dir & --ref_dir) OR (--moving & --reference).")
        srcs = str(a.img_dir)

    model = YOLO(str(a.ckpt))

    for r in model.predict(srcs, imgsz=a.imgsz, conf=a.conf,
                           device=a.device, stream=True, verbose=False):

        if r.masks is None:
            continue

        mov_path = Path(r.path)
        ref_path = a.reference if single else (a.ref_dir / mov_path.name)
        if not ref_path.exists():
            print(f"⚠ reference missing for {mov_path.name}, skipping"); continue

        img_mov = cv2.imread(str(mov_path))
        img_ref = cv2.imread(str(ref_path))

        polys = r.masks.xy                          # list of arrays Nx2
        cids  = r.boxes.cls.cpu().numpy().astype(int)

        for inst, (poly, cid) in enumerate(zip(polys, cids)):
            if keep and cid not in keep:
                continue

            sub = a.out_dir / f"{mov_path.stem}_cls{cid}_{inst}"
            sub.mkdir(parents=True, exist_ok=True)

            # 1️⃣  full-res polygon → OBB
            pts  = np.asarray(poly, np.float32)
            rect = cv2.minAreaRect(pts)
            box  = cv2.boxPoints(rect).astype(int)  # 4×2

            # visual overlay on moving
            over = img_mov.copy()
            cv2.polylines(over, [box], True, (0, 255, 0), 2)
            cv2.imwrite(str(sub/'obb_overlay.jpg'), over)

            # 2️⃣  crop only moving image
            crop_mov = crop_from_obb(img_mov, box)
            cv2.imwrite(str(sub/'crop_moving.jpg'), crop_mov)
            cv2.imwrite(str(sub/'reference.jpg'),   img_ref)

            # 3️⃣  align crop to provided reference
            try:
                orb_align(crop_mov, img_ref, sub)
            except Exception as e:
                print(f"x align fail {mov_path.name} inst{inst}: {e}")
            else:
                print(f"✓ {mov_path.name} cls{cid} inst{inst}")

    print(f"\nAll artefacts →  {a.out_dir.resolve()}")


if __name__ == '__main__':
    main()