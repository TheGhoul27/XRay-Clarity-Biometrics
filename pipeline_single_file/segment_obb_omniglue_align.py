import os; os.environ["WANDB_DISABLED"] = "true"
import argparse, cv2, numpy as np, sys
from pathlib import Path
from ultralytics import YOLO
import omniglue  # NEW – deep matcher


# ───────────────────── helpers ───────────────────────────────────────────

def crop_from_obb(img: np.ndarray, box: np.ndarray, scale: float = 1.05):
    """Perspective‑crop the OBB region."""
    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])
    W, H = int(w * scale), int(h * scale)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


def orb_align(mov: np.ndarray, ref: np.ndarray, outdir: Path,
              n_feat: int = 5_000, ratio: float = .75, ransac: float = 6.0):
    """Align moving patch to reference with ORB; dump diagnostics (side_by_side.jpg, aligned_keypoints.jpg)."""
    orb = cv2.ORB_create(nfeatures=n_feat)
    k1, d1 = orb.detectAndCompute(mov, None)
    k2, d2 = orb.detectAndCompute(ref, None)
    if d1 is None or d2 is None:
        raise RuntimeError("No ORB keypoints.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if len(good) < 4:
        raise RuntimeError("Not enough good ORB matches.")

    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
    if H is None:
        raise RuntimeError("Homography failed (ORB).")

    h, w = ref.shape[:2]
    warped = cv2.warpPerspective(mov, H, (w, h))

    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "side_by_side.jpg"), cv2.hconcat([ref, warped]))

    vis, off = cv2.hconcat([ref, warped]), np.array([w, 0], int)
    for s, t, keep in zip(src, dst, inl.ravel()):
        if not keep:
            continue
        s_w = cv2.perspectiveTransform(s.reshape(1, 1, 2), H).reshape(2)
        cv2.circle(vis, tuple(t.astype(int)), 4, (0, 255, 0), -1)
        cv2.circle(vis, tuple((s_w + off).astype(int)), 4, (0, 0, 255), -1)
        cv2.line(vis, tuple(t.astype(int)), tuple((s_w + off).astype(int)), (255, 200, 200), 1)
    cv2.imwrite(str(outdir / "aligned_keypoints.jpg"), vis)


def omniglue_align(mov: np.ndarray, ref: np.ndarray, outdir: Path,
                   matcher: "omniglue.OmniGlue", conf_thresh: float = 0.5,
                   ransac_thresh: float = 6.0):
    """Align via OmniGlue; writes *_omniglue.jpg diagnostics."""
    # OmniGlue expects RGB float images in np.uint8 ‑ we convert to RGB but keep uint8
    mov_rgb = cv2.cvtColor(mov, cv2.COLOR_BGR2RGB)
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    kp0, kp1, conf = matcher.FindMatches(mov_rgb, ref_rgb)  # (N,2), (N,2), (N,)
    good = conf > conf_thresh
    if good.sum() < 4:
        raise RuntimeError("Not enough confident OmniGlue matches.")

    src = kp0[good].astype(np.float32)
    dst = kp1[good].astype(np.float32)

    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("Homography failed (OmniGlue).")

    h, w = ref.shape[:2]
    warped = cv2.warpPerspective(mov, H, (w, h))

    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "side_by_side_omniglue.jpg"), cv2.hconcat([ref, warped]))

    vis, off = cv2.hconcat([ref, warped]), np.array([w, 0], int)
    for p_ref, p_src, keep in zip(dst, src, inl.ravel()):
        if not keep:
            continue
        p_src_w = cv2.perspectiveTransform(p_src.reshape(1, 1, 2), H).reshape(2)
        cv2.circle(vis, tuple(p_ref.astype(int)), 4, (0, 255, 0), -1)
        cv2.circle(vis, tuple((p_src_w + off).astype(int)), 4, (0, 0, 255), -1)
        cv2.line(vis, tuple(p_ref.astype(int)), tuple((p_src_w + off).astype(int)), (200, 200, 255), 1)
    cv2.imwrite(str(outdir / "aligned_keypoints_omniglue.jpg"), vis)

    return warped, H


# ───────────────────── CLI ───────────────────────────────────────────────

def args_():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt', required=True, type=Path, help='YOLO segmentation checkpoint')
    p.add_argument('--out_dir', required=True, type=Path, help='Destination folder for artefacts')

    # batch‑mode paths
    p.add_argument('--img_dir', type=Path, help='Folder of moving images (batch mode)')
    p.add_argument('--ref_dir', type=Path, help='Folder of reference templates (batch mode)')
    # single image
    p.add_argument('--moving', type=Path, help='Single moving image path')
    p.add_argument('--reference', type=Path, help='Single reference template path')

    p.add_argument('--classes', nargs='*', type=int, help='Keep only these class IDs')
    p.add_argument('--imgsz', default=640, type=int, help='YOLO inference resolution')
    p.add_argument('--conf', default=.1, type=float, help='YOLO confidence threshold')
    p.add_argument('--device', default='0', help='CUDA device for YOLO')

    # OmniGlue specific
    p.add_argument('--og_export', default='E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\models\\og_export', type=Path, help='OmniGlue export folder')
    p.add_argument('--sp_export', default='E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\models\\sp_v6', type=Path, help='SuperPoint weights folder')
    p.add_argument('--dino_ckpt', default='E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\models\\dinov2_vitb14_pretrain.pth', type=Path, help='DINOv2 checkpoint path')
    p.add_argument('--no_omni', action='store_true', help='Disable OmniGlue alignment (keep ORB only)')

    return p.parse_args()


# ───────────────────── main ──────────────────────────────────────────────

def main():
    a = args_()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(a.classes) if a.classes else None

    # Decide on batch vs single
    single = bool(a.moving and a.reference)
    if single:
        srcs = [str(a.moving)]
    else:
        if not (a.img_dir and a.ref_dir):
            sys.exit('Need (--img_dir & --ref_dir) OR (--moving & --reference).')
        srcs = str(a.img_dir)

    # YOLO model
    yolo_model = YOLO(str(a.ckpt))

    # OmniGlue matcher (optional)
    og_matcher = None
    if not a.no_omni:
        og_matcher = omniglue.OmniGlue(
            og_export=str(a.og_export),
            sp_export=str(a.sp_export),
            dino_export=str(a.dino_ckpt),
        )
        print('✔ OmniGlue loaded.')

    # ───────────────── inference loop ──────────────────────────────────
    for r in yolo_model.predict(srcs, imgsz=a.imgsz, conf=a.conf, device=a.device, stream=True, verbose=False):
        if r.masks is None:
            continue  # no detections

        mov_path = Path(r.path)
        ref_path = a.reference if single else (a.ref_dir)
        if not ref_path.exists():
            print(f'⚠ reference missing for {mov_path.name}, skipping'); continue

        img_mov = cv2.imread(str(mov_path))
        img_ref = cv2.imread(str(ref_path))

        polys = r.masks.xy                     # list of arrays Nx2
        cids  = r.boxes.cls.cpu().numpy().astype(int)

        for inst, (poly, cid) in enumerate(zip(polys, cids)):
            if keep and cid not in keep:
                continue

            sub = a.out_dir / f"{mov_path.stem}_cls{cid}_{inst}"
            sub.mkdir(parents=True, exist_ok=True)

            # 1️⃣  polygon → oriented bounding box
            pts  = np.asarray(poly, np.float32)
            rect = cv2.minAreaRect(pts)
            box  = cv2.boxPoints(rect).astype(int)  # 4×2

            # overlay OBB on moving
            over = img_mov.copy()
            cv2.polylines(over, [box], True, (0, 255, 0), 2)
            cv2.imwrite(str(sub / 'obb_overlay.jpg'), over)

            # 2️⃣  perspective crop of moving
            crop_mov = crop_from_obb(img_mov, box)
            cv2.imwrite(str(sub / 'crop_moving.jpg'), crop_mov)
            cv2.imwrite(str(sub / 'reference.jpg'),   img_ref)

            # 3️⃣-a ORB alignment
            try:
                orb_align(crop_mov, img_ref, sub)
            except Exception as e:
                print(f'x ORB align fail {mov_path.name} inst{inst}: {e}')
            else:
                print(f'✓ ORB {mov_path.name} cls{cid} inst{inst}')

            # 3️⃣-b OmniGlue alignment (if enabled)
            if og_matcher is not None:
                try:
                    omniglue_align(crop_mov, img_ref, sub, og_matcher)
                except Exception as e:
                    print(f'x OmniGlue align fail {mov_path.name} inst{inst}: {e}')
                else:
                    print(f'✓ OmniGlue {mov_path.name} cls{cid} inst{inst}')

    print(f"\nAll artefacts →  {a.out_dir.resolve()}")


if __name__ == '__main__':
    main()
