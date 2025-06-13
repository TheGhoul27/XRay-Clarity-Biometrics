# orb_ransac_ablation.py
# ---------------------------------------------------------------------
# Example
#   python orb_ransac_ablation.py ^
#          --csv  "Clarity-test-images-data-Sheet1.csv" ^
#          --root "E:/Masters_College_Work/RA_CyLab/X-Ray/results/Matches" ^
#          --thr  0.10 0.15 0.20 0.25 0.30 ^
#          --device cpu
# ---------------------------------------------------------------------
import argparse, cv2
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ───────────────────────────── indexing ───────────────────────────────
def index_instance_dirs(root: Path):
    """Map image‐stem → list of *_cls* folders."""
    mapping = {}
    for d in root.glob('*_cls*'):
        if d.is_dir():
            stem = d.name.split('_cls')[0]
            mapping.setdefault(stem, []).append(d)
    return mapping

# ─────────────────────── ORB + RANSAC similarity ──────────────────────
orb = cv2.ORB_create(nfeatures=5000)

def orb_inlier_ratio(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """Return (#RANSAC inliers) / (#raw matches)."""
    kpA, desA = orb.detectAndCompute(imgA, None)
    kpB, desB = orb.detectAndCompute(imgB, None)
    if desA is None or desB is None:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desA, desB)
    if len(matches) < 4:                              # not enough for homography
        return 0.0

    src_pts = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    inliers = mask.ravel().sum()
    return inliers / len(matches)

# ───────────────────────── evaluation loop ────────────────────────────
def evaluate_threshold(stem_to_dirs, df, ratio_thr):
    y_true, y_pred = [], []

    FS = 'Frontside ("FS") Scan Filename'
    BS = 'Backside ("BS") Scan Filename'

    for _, row in df.iterrows():
        for col in (FS, BS):
            fname = row.get(col)
            if pd.isna(fname):
                continue
            stem = Path(fname).stem
            dirs = stem_to_dirs.get(stem, [])
            if not dirs:
                continue

            worst_ratio = 1.0
            for d in dirs:
                br  = cv2.imread(str(d/'best_rotated.jpg'), cv2.IMREAD_GRAYSCALE)
                ref = cv2.imread(str(d/'best_reference.jpg'), cv2.IMREAD_GRAYSCALE)
                if br is None or ref is None:
                    continue
                # resize rotated to reference size for fairness
                h,w = ref.shape[:2]
                br_r = cv2.resize(br, (w,h))

                ratio = orb_inlier_ratio(br_r, ref)
                worst_ratio = min(worst_ratio, ratio)

            is_anom = worst_ratio < ratio_thr
            y_true.append(bool(row['Anomaly Status']))
            y_pred.append(is_anom)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        thr      = ratio_thr,
        accuracy = accuracy_score(y_true, y_pred),
        precision= precision_score(y_true, y_pred),
        recall   = recall_score(y_true, y_pred),
        f1       = f1_score(y_true, y_pred),
        TP=tp, FP=fp, FN=fn, TN=tn
    )

# ────────────────────────────── main ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--csv',   required=True, type=Path, help='Clarity CSV')
    ap.add_argument('--root',  required=True, type=Path, help='Matches root')
    ap.add_argument('--thr',   nargs='+', type=float, required=True,
                    help='space-separated list of inlier-ratio thresholds')
    ap.add_argument('--out', default='orb_ablation')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    stem_to_dirs = index_instance_dirs(args.root)

    rows = []
    for t in sorted(args.thr):
        m = evaluate_threshold(stem_to_dirs, df, t)
        rows.append(m)
        print(f"thr={t:.2f}  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}")

    results = pd.DataFrame(rows)
    csv_path = Path(f"{args.out}.csv")
    md_path  = Path(f"{args.out}.md")
    results.to_csv(csv_path, index=False)
    md_path.write_text(
        "# ORB + RANSAC inlier-ratio ablation\n\n" +
        results.to_markdown(index=False, floatfmt=".3f")
    )
    print(f"\nSaved:\n• {csv_path}\n• {md_path}")

if __name__ == '__main__':
    main()
