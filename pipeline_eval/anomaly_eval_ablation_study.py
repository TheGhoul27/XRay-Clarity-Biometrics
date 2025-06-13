# anomaly_eval_ablation_study.py
# --------------------------------------------------------------------
# Run:
#   python anomaly_eval_ablation_study.py \
#          --csv   "Clarity-test-images-data-Sheet1.csv" \
#          --root  "E:/Masters_College_Work/RA_CyLab/X-Ray/results/Matches" \
#          --cos   0.88 0.90 0.92 0.94 0.96 \
#          --ssim  0.82 0.84 0.86 0.88 0.90 \
#          --device cuda
#
# Produces:
#   ablation_results.csv    – raw numbers (one row per pair)
#   ablation_results.md     – pretty Markdown table
# --------------------------------------------------------------------
import argparse, cv2, glob
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch, torchvision.transforms as T
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ───── Feature extractor (same as before) ────────────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_pre = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

def load_resnet(device: str):
    net = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
    net.fc = torch.nn.Identity()
    return net.to(device).eval()

@torch.inference_mode()
def embed(bgr: np.ndarray, model, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = _pre(rgb).unsqueeze(0).to(device)
    feat= model(t)
    return F.normalize(feat.squeeze(0), dim=0)

def cos_sim(a,b): return float(F.cosine_similarity(a,b,dim=0).cpu())

# ───── Pre-index the result folders once (fast) ──────────────────────
def index_instance_dirs(root: Path):
    mapping = {}
    for d in root.glob('*_cls*'):
        if d.is_dir():
            stem = d.name.split('_cls')[0]
            mapping.setdefault(stem, []).append(d)
    return mapping

# ───── Core evaluation loop for ONE threshold pair ───────────────────
def evaluate_pair(stem_to_dirs, df, resnet, device,
                  cos_thr, ssim_thr,
                  fs_col='Frontside ("FS") Scan Filename',
                  bs_col='Backside ("BS") Scan Filename'):
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        for col in (fs_col, bs_col):
            fname = row.get(col)
            if pd.isna(fname):    # missing cell
                continue
            stem = Path(fname).stem
            dirs = stem_to_dirs.get(stem, [])
            if not dirs:          # inference never ran?
                continue

            worst_cos, worst_ssim = 1.0, 1.0
            for d in dirs:
                br  = cv2.imread(str(d/'best_rotated.jpg'))
                ref = cv2.imread(str(d/'best_reference.jpg'))
                if br is None or ref is None:
                    continue
                cos  = cos_sim(embed(br,resnet,device),
                               embed(ref,resnet,device))
                h,w  = ref.shape[:2]
                br_r = cv2.resize(br,(w,h))
                ssim_val = ssim(cv2.cvtColor(br_r,cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(ref ,cv2.COLOR_BGR2GRAY))
                worst_cos  = min(worst_cos , cos)
                worst_ssim = min(worst_ssim, ssim_val)

            is_anom = (worst_cos < cos_thr) or (worst_ssim < ssim_thr)
            y_true.append(bool(row['Anomaly Status']))
            y_pred.append(is_anom)

    if not y_true:    # no data found
        return None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = dict(
        accuracy = accuracy_score (y_true, y_pred),
        precision= precision_score(y_true, y_pred),
        recall   = recall_score   (y_true, y_pred),
        f1       = f1_score       (y_true, y_pred),
        TP=tp, FP=fp, FN=fn, TN=tn
    )
    return metrics

# ───── Main: grid sweep & write outputs ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--csv',  required=True, type=Path)
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--cos',   nargs='+', type=float, required=True,
                    help='space-separated list of cosine thresholds')
    ap.add_argument('--ssim',  nargs='+', type=float, required=True,
                    help='space-separated list of SSIM thresholds')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='ablation_results')
    args = ap.parse_args()

    device  = args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu'
    resnet  = load_resnet(device)
    df      = pd.read_csv(args.csv)
    mapping = index_instance_dirs(args.root)

    rows = []
    for ct, st in product(args.cos, args.ssim):
        m = evaluate_pair(mapping, df, resnet, device, ct, st)
        if m is None: continue
        rows.append(dict(cos_thr=ct, ssim_thr=st, **m))
        print(f"[{ct:.3f}, {st:.3f}]  "
              f"Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}")

    results = pd.DataFrame(rows).sort_values(['cos_thr','ssim_thr'])
    csv_path = Path(f"{args.out}.csv")
    md_path  = Path(f"{args.out}.md")
    results.to_csv(csv_path, index=False)
    print(f"\nSaved raw grid to {csv_path}")

    # ── pretty Markdown – easier to paste into a report ──
    md = results.to_markdown(index=False, floatfmt=".3f")
    md_path.write_text("# Ablation grid (cos-sim vs. SSIM)\n\n" + md)
    print(f"Saved Markdown table to {md_path}")

if __name__ == '__main__':
    main()
