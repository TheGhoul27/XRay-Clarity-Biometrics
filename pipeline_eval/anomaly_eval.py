# anomaly_eval.py
# ---------------------------------------------------------------------
# Example:
#   python anomaly_eval.py ^
#          --csv  "Clarity-test-images-data-Sheet1.csv" ^
#          --root "E:\Masters_College_Work\RA_CyLab\X-Ray\results\Matches" ^
#          --cos-thresh 0.92 --ssim-thresh 0.88 --device cuda
# ---------------------------------------------------------------------
import argparse, cv2, glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torchvision.transforms as T
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ──────────────────────────── backbone ────────────────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_pre = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

def load_resnet(device):
    m = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
    m.fc = torch.nn.Identity()
    return m.to(device).eval()

@torch.inference_mode()
def embed(bgr, model, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = _pre(rgb).unsqueeze(0).to(device)
    feat= model(t)
    return F.normalize(feat.squeeze(0), dim=0)

def cos_sim(a, b): return float(F.cosine_similarity(a, b, dim=0).cpu())

# ──────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--csv',  required=True, type=Path, help='Clarity CSV')
    ap.add_argument('--root', required=True, type=Path, help='Matches root folder')
    ap.add_argument('--cos-thresh',  type=float, default=0.92)
    ap.add_argument('--ssim-thresh', type=float, default=0.88)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu'
    resnet = load_resnet(device)

    # index every *_cls* directory by image-stem
    stem_to_dirs = {}
    for d in args.root.glob('*_cls*'):
        if d.is_dir():
            stem = d.name.split('_cls')[0]
            stem_to_dirs.setdefault(stem, []).append(d)

    df = pd.read_csv(args.csv)
    fs_col = 'Frontside ("FS") Scan Filename'
    bs_col = 'Backside ("BS") Scan Filename'

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        for col in (fs_col, bs_col):
            fname = row.get(col)
            if pd.isna(fname):          # blank cell
                continue
            stem = Path(fname).stem
            dirs = stem_to_dirs.get(stem, [])
            if not dirs:
                print(f"⚠ results missing for {fname}")
                continue

            # aggregate worst similarity across all detections of this image
            worst_cos, worst_ssim = 1.0, 1.0
            for d in dirs:
                br  = cv2.imread(str(d/'best_rotated.jpg'))
                ref = cv2.imread(str(d/'best_reference.jpg'))
                if br is None or ref is None: continue

                cos  = cos_sim(embed(br, resnet, device), embed(ref, resnet, device))

                h,w  = ref.shape[:2]
                br_r = cv2.resize(br,(w,h))
                ssim_val = ssim(cv2.cvtColor(br_r,cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(ref ,cv2.COLOR_BGR2GRAY))
                worst_cos  = min(worst_cos , cos)
                worst_ssim = min(worst_ssim, ssim_val)

            is_anom = (worst_cos < args.cos_thresh) or (worst_ssim < args.ssim_thresh)
            y_true.append(bool(row['Anomaly Status']))
            y_pred.append(is_anom)

    # ─── metrics ───
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n===  Summary  ===")
    print(f"Samples   : {len(y_true)}")
    print(f"Accuracy  : {acc :.3f}")
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec :.3f}")
    print(f"F1 score  : {f1 :.3f}")
    print("\nConfusion matrix:")
    print(f"TP: {tp}  FP: {fp}")
    print(f"FN: {fn}  TN: {tn}")

if __name__ == '__main__':
    main()