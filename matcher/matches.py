"""
batch_match_ratio_eval.py
-------------------------
Run OmniGlue on every row in the Clarity spreadsheet and test
inlier/raw-ratio thresholds against the 'Anomaly Status' column.

Example
-------
python batch_match_ratio_eval.py --csv data/Clarity.csv \
                                 --img_root data/images \
                                 --thr 0.4 0.55 0.7     \
                                 --save_vis
"""

# ────────────────── CLI & CONFIG ──────────────────────────────────────────
import argparse, random, warnings
from pathlib import Path

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",       required=True,  type=Path,
                   help="Clarity-test-images CSV file")
    p.add_argument("--img_root",  required=True,  type=Path,
                   help="folder containing all images")
    p.add_argument("--outdir",    default="vis",  type=Path,
                   help="where to save visualisations")
    p.add_argument("--seed",      default=42,     type=int)
    p.add_argument("--save_vis",  action="store_true",
                   help="save *.bmp match snapshots")
    p.add_argument("--thr",       type=float, nargs="+", default=[0.50],
                   help="ratio thresholds (anomaly if ratio < thr)")
    return p.parse_args()

# ────────────────── Imports ───────────────────────────────────────────────
import pandas as pd, numpy as np, cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import omniglue
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ────────────────── Helper functions ──────────────────────────────────────
def load_bmp(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:                       # grayscale BMP
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_bbox(txt: Path, shape):
    raw = txt.read_text().strip().replace(",", " ")
    vals = [float(v) for v in raw.split()[:4]]
    h, w = shape[:2]
    if all(v <= 1.0 for v in vals):         # normalised 0–1
        xmin, ymin, xmax, ymax = [int(round(v*s))
                                  for v, s in zip(vals, (w, h, w, h))]
    else:                                   # absolute pixels
        xmin, ymin, xmax, ymax = map(int, vals)
    xmin = max(0, min(xmin, w-1)); xmax = max(xmin+1, min(xmax, w))
    ymin = max(0, min(ymin, h-1)); ymax = max(ymin+1, min(ymax, h))
    return xmin, ymin, xmax, ymax

def resize_h(img, target_h):
    if img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*scale), target_h),
                      cv2.INTER_AREA)

def init_og():
    return omniglue.OmniGlue(
        og_export   ="E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\models\\og_export",
        sp_export   ="E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\models\\sp_v6",
        dino_export ="E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\models\\dinov2_vitb14_pretrain.pth",
    )

def match_pair(og, imgA, imgB):
    kp0, kp1, conf = og.FindMatches(imgA, imgB)
    raw = len(kp0)
    if raw < 4:
        return raw, 0, (kp0, kp1, conf)
    H, mask = cv2.findHomography(kp1, kp0, cv2.RANSAC, 3.0)
    inl = int(mask.sum()) if mask is not None else 0
    return raw, inl, (kp0, kp1, conf)

def status_to_label(val):
    """'True' → 1 (anomaly), 'False' → 0 (normal)."""
    if isinstance(val, bool):
        return int(val)
    s = str(val).strip().lower()
    return 1 if s == "true" or s == "1" else 0

# ────────────────── Main workflow ─────────────────────────────────────────
def main():
    args = get_args()
    random.seed(args.seed)

    df = pd.read_csv(args.csv)

    # 1 ─ pick ONE Grade-A reference per Item
    ref_map = {}
    for item, grp in df.groupby("Item"):
        ga = grp[grp["R1 Condition"].astype(str)
                 .str.contains("Grade A", na=False)]
        if ga.empty:
            continue
        ref_row = ga.sample(1, random_state=args.seed).iloc[0]
        ref_map[item] = (
            args.img_root / ref_row['Frontside ("FS") Scan Filename'],
            args.img_root / ref_row['Backside ("BS") Scan Filename'],
        )

    og = init_og()
    rows = []
    if args.save_vis:
        args.outdir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="rows"):
        item = row["Item"]
        if item not in ref_map:
            continue

        test_paths = {
            "front": args.img_root / row['Frontside ("FS") Scan Filename'],
            "back":  args.img_root / row['Backside ("BS") Scan Filename'],
        }
        ref_paths = {"front": ref_map[item][0], "back": ref_map[item][1]}

        for side in ("front", "back"):
            t_path, r_path = test_paths[side], ref_paths[side]
            if not (t_path.exists() and r_path.exists()):
                continue

            # crop test
            t_full = load_bmp(t_path)
            bbox_t = t_path.with_suffix(".txt")
            x0,y0,x1,y1 = read_bbox(bbox_t, t_full.shape) if bbox_t.exists() \
                          else (0,0,*t_full.shape[1::-1])
            t_crop = t_full[y0:y1, x0:x1].copy()

            # crop reference
            r_full = load_bmp(r_path)
            bbox_r = r_path.with_suffix(".txt")
            xr0,yr0,xr1,yr1 = read_bbox(bbox_r, r_full.shape) if bbox_r.exists() \
                              else (0,0,*r_full.shape[1::-1])
            r_crop = r_full[yr0:yr1, xr0:xr1].copy()

            raw, inl, (kp0, kp1, conf) = match_pair(og, t_crop, r_crop)
            ratio = inl / raw if raw else 0.0

            rows.append(dict(
                idx=idx, item=item, side=side,
                raw=raw, inliers=inl, ratio=ratio,
                gt=status_to_label(row.get("Anomaly Status", "False"))
            ))

            if args.save_vis:
                r_rs = resize_h(r_crop, t_crop.shape[0]); wL = t_crop.shape[1]
                vis  = np.hstack([t_crop, r_rs])
                for j in np.argsort(-conf)[:min(100, len(kp0))]:
                    x0,y0 = kp0[j]; x1,y1 = kp1[j]
                    cv2.circle(vis,(int(x0),int(y0)),3,(0,255,0),-1)
                    cv2.circle(vis,(int(x1)+wL,int(y1)),3,(0,255,0),-1)
                    cv2.line  (vis,(int(x0),int(y0)),
                                    (int(x1)+wL,int(y1)),(255,0,0),1)
                cv2.imwrite(str(args.outdir/f"{item}_{side}_{idx}.bmp"),
                            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    res = pd.DataFrame(rows)
    res.to_csv("all_match_metrics.csv", index=False)
    print(f"\nSaved {len(res)} rows → all_match_metrics.csv")

    y_true = res["gt"].values
    print("\n── 'Anomaly Status' counts ──")
    print(pd.Series(y_true).value_counts().rename({0:"normal",1:"anomaly"}))

    # 2 ─ evaluate thresholds
    print("\n── Inlier/raw ratio threshold evaluation ──")
    for thr in args.thr:
        y_pred = (res["ratio"].values < thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(y_true, y_pred)
        print(f"\nThreshold {thr:.2f}: anomaly if ratio < {thr}")
        print("Confusion matrix [rows: actual 0/1, cols: predicted 0/1]:")
        print(cm)
        print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
