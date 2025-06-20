# # pca_similarity_analysis.py  –  DINO-v2 (torch.hub, 256-token models)
# # --------------------------------------------------------------------
# import argparse, glob, cv2, torch, numpy as np, pandas as pd
# from pathlib import Path
# from PIL import Image
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # ─────── CLI ──────────────────────────────────────────────────────────
# ap = argparse.ArgumentParser()
# ap.add_argument("--root",  required=True, type=Path, help="results root")
# ap.add_argument("--csv",   required=True, type=Path, help="Clarity CSV")
# ap.add_argument("--out",   default="pca_out", type=Path, help="output folder")
# ap.add_argument("--model", default="dinov2_vits14",
#                 choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
#                 help="DINO-v2 hub ID (224 px models)")
# args = ap.parse_args()
# heat_root = args.out / "heatmaps"          # parent only
# heat_root.mkdir(parents=True, exist_ok=True)

# # ─────── load DINO-v2 (torch.hub) ─────────────────────────────────────
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model  = torch.hub.load("facebookresearch/dinov2", args.model)\
#                  .to(device).eval()

# IMG_SIZE = 224                                # all 3 models → 16×16 patches
# pre = T.Compose([
#     T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
#     T.CenterCrop(IMG_SIZE),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406],
#                 [0.229, 0.224, 0.225]),
# ])

# @torch.inference_mode()
# def patch_embed(img_bgr: np.ndarray):
#     rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     x   = pre(Image.fromarray(rgb)).unsqueeze(0).to(device)
#     # hub models return dict with key 'x_norm_patchtokens'
#     embs = model.forward_features(x)["x_norm_patchtokens"]  # [1,256,D]
#     return embs.squeeze(0).cpu()                            # [256,D]

# def patch_similarity(ref_bgr, rot_bgr):
#     e1, e2 = patch_embed(ref_bgr), patch_embed(rot_bgr)
#     return torch.nn.functional.cosine_similarity(e1, e2).numpy()  # [256]

# def save_outputs(stem: str, ref_bgr, rot_bgr, sims):
#     """
#     Writes 2 files in heatmaps/<stem>/ :
#       • <stem>_overlay.png  (reference | overlay)
#       • <stem>_hist.png     (histogram of 256 similarities)
#     """
#     dst = heat_root / stem
#     dst.mkdir(exist_ok=True)

#     # ── overlay ──────────────────────────────────────────────
#     hm   = sims.reshape(16, 16)
#     hm   = cv2.resize(hm, (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)
#     rotR = cv2.resize(rot_bgr, (IMG_SIZE, IMG_SIZE))
#     rgba = (plt.get_cmap('hot')(hm) * 255).astype(np.uint8)
#     overlay = cv2.addWeighted(rotR, 0.7, rgba[:, :, :3], 0.3, 0)

#     refR = cv2.resize(ref_bgr, (IMG_SIZE, IMG_SIZE))
#     side = cv2.hconcat([refR, overlay])
#     cv2.imwrite(str(dst / f"{stem}_overlay.png"),
#                 cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

#     # ── histogram ────────────────────────────────────────────
#     plt.figure(figsize=(6, 3))
#     plt.hist(sims, bins=50, color="steelblue",
#              edgecolor="k", alpha=0.85)
#     plt.title(f"Patch-similarity histogram · {stem}")
#     plt.xlabel("cosine(sim)"); plt.ylabel("# patches")
#     plt.tight_layout()
#     plt.savefig(dst / f"{stem}_hist.png", dpi=200)
#     plt.close()

# # ─────── CSV  →  label map ────────────────────────────────────────────
# df = pd.read_csv(args.csv)
# label_map = {}
# for _, row in df.iterrows():
#     for col in ['Frontside ("FS") Scan Filename',
#                 'Backside ("BS") Scan Filename']:
#         fname = row.get(col)
#         if pd.isna(fname):
#             continue
#         label_map[Path(fname).stem] = bool(row['Anomaly Status'])

# # ─────── iterate *_cls* folders ───────────────────────────────────────
# vectors, labels, stems = [], [], []

# for fld in glob.iglob(str(args.root / "**/*_cls*"), recursive=True):
#     folder = Path(fld)
#     ref_p, rot_p = folder / "best_reference.jpg", folder / "best_rotated.jpg"
#     if not (ref_p.exists() and rot_p.exists()):
#         continue
#     stem = folder.name.split("_cls")[0]
#     if stem not in label_map:
#         continue

#     ref = cv2.cvtColor(cv2.imread(str(ref_p)), cv2.COLOR_BGR2RGB)
#     rot = cv2.cvtColor(cv2.imread(str(rot_p)), cv2.COLOR_BGR2RGB)

#     sims = patch_similarity(ref, rot)          # 256-D
#     vectors.append(sims); labels.append(label_map[stem]); stems.append(stem)
#     save_outputs(stem, ref, rot, sims)

# if not vectors:          # guard: nothing matched
#     raise RuntimeError("No labelled images found – check --root / --csv paths")

# vectors = np.stack(vectors)
# labels  = np.array(labels)

# # ─────── PCA scatter & CSV ────────────────────────────────────────────
# X_std = StandardScaler().fit_transform(vectors)
# pcs   = PCA(n_components=2).fit_transform(X_std)

# args.out.mkdir(parents=True, exist_ok=True)
# plt.figure(figsize=(7, 6))
# plt.scatter(pcs[:, 0], pcs[:, 1],
#             c=np.where(labels, "red", "green"),
#             edgecolors='k', alpha=0.8)
# plt.title("PCA of patch-wise DINO-v2 similarities")
# plt.xlabel("PC1"); plt.ylabel("PC2")
# plt.legend(["Normal", "Anomaly"])
# plt.tight_layout()
# plt.savefig(args.out / "pca_scatter.png", dpi=300)
# plt.close()

# pd.DataFrame(dict(img=stems, pc1=pcs[:, 0], pc2=pcs[:, 1],
#                   anomaly=labels)).to_csv(args.out / "pca_coordinates.csv",
#                                           index=False)

# print(f"Finished.\n  Per-image outputs ➔ {heat_root}/<stem>/"
#       f"\n  PCA plot          ➔ {args.out/'pca_scatter.png'}")

# pca_similarity_analysis.py  –  DINO-v2 (torch.hub, 256-token models)
# --------------------------------------------------------------------
import argparse, glob, cv2, torch, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ───────────── CLI ────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--root",  required=True, type=Path, help="results root")
ap.add_argument("--csv",   required=True, type=Path, help="Clarity CSV")
ap.add_argument("--out",   default="pca_out", type=Path, help="output folder")
ap.add_argument("--model", default="dinov2_vits14",
                choices=["dinov2_vits14","dinov2_vitb14","dinov2_vitl14"],
                help="DINO-v2 hub ID (224 px models)")
args = ap.parse_args()
heat_root = args.out / "heatmaps"; heat_root.mkdir(parents=True, exist_ok=True)

# ───────────── load DINO-v2 (torch.hub) ───────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = torch.hub.load("facebookresearch/dinov2", args.model)\
                 .to(device).eval()

IMG = 224
pre = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@torch.inference_mode()
def patch_embed(bgr: np.ndarray) -> torch.Tensor:           # [256,D]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x   = pre(Image.fromarray(rgb)).unsqueeze(0).to(device)
    return model.forward_features(x)["x_norm_patchtokens"].squeeze(0).cpu()

def patch_distance(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    e1, e2 = patch_embed(ref), patch_embed(mov)        # [256, D] each

    dot_prod = torch.matmul(e1, e2.T)                  # [256, 256]
    diag = torch.diagonal(dot_prod)                    # [256]

    norm1 = torch.norm(e1, dim=1)                      # [256]
    norm2 = torch.norm(e2, dim=1)                      # [256]
    cos = diag / (norm1 * norm2 + 1e-8)               # avoid /0
    dist = 1.0 - cos                                   # [256]

    return dist.numpy()


def save_outputs(stem: str, ref_bgr, rot_bgr, dists):
    dst = heat_root / stem; dst.mkdir(exist_ok=True)

    # overlay
    hm   = dists.reshape(16,16)
    hm   = cv2.resize(hm, (IMG,IMG), cv2.INTER_LINEAR)
    rotR = cv2.resize(rot_bgr, (IMG,IMG))
    rgba = (plt.get_cmap('hot')(hm) * 255).astype(np.uint8)
    side = cv2.hconcat([cv2.resize(ref_bgr,(IMG,IMG)),
                        cv2.addWeighted(rotR,0.7,rgba[:,:,:3],0.3,0)])
    cv2.imwrite(str(dst/f"{stem}_overlay.png"),
                cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

    # histogram
    plt.figure(figsize=(6,3))
    plt.hist(dists, bins=50, color="steelblue", edgecolor="k", alpha=0.85)
    plt.title(f"Patch distance histogram · {stem}")
    plt.xlabel("1 − cosine(sim)"); plt.ylabel("# patches")
    plt.tight_layout(); plt.savefig(dst/f"{stem}_hist.png", dpi=200); plt.close()

# ───────────── CSV → label map ────────────────────────────────────────
df = pd.read_csv(args.csv)
label_map = {}
for _,row in df.iterrows():
    for col in ['Frontside ("FS") Scan Filename','Backside ("BS") Scan Filename']:
        fn=row.get(col);  # may be NaN
        if pd.isna(fn): continue
        label_map[Path(fn).stem] = bool(row['Anomaly Status'])

# ───────────── iterate folders ────────────────────────────────────────
vectors, labels, stems = [], [], []
for fld in glob.iglob(str(args.root / "**/*_cls*"), recursive=True):
    folder = Path(fld)
    ref_p, rot_p = folder/"best_reference.jpg", folder/"best_rotated.jpg"
    if not (ref_p.exists() and rot_p.exists()): continue
    stem = folder.name.split("_cls")[0]
    if stem not in label_map: continue

    ref = cv2.cvtColor(cv2.imread(str(ref_p)), cv2.COLOR_BGR2RGB)
    rot = cv2.cvtColor(cv2.imread(str(rot_p)), cv2.COLOR_BGR2RGB)

    dist = patch_distance(ref, rot)                 # 256-D
    vectors.append(dist); labels.append(label_map[stem]); stems.append(stem)
    save_outputs(stem, ref, rot, dist)

if not vectors:
    raise RuntimeError("No labelled images found – check paths")

vectors = np.stack(vectors); labels = np.array(labels)

# ───────────── PCA scatter & CSV ──────────────────────────────────────
# X_std = StandardScaler().fit_transform(vectors)
# pcs    = PCA(n_components=2).fit_transform(X_std)
pcs    = PCA(n_components=2).fit_transform(vectors)

args.out.mkdir(exist_ok=True)
plt.figure(figsize=(7,6))
plt.scatter(pcs[:,0], pcs[:,1],
            c=np.where(labels,"red","green"),
            edgecolors='k', alpha=0.8)
plt.title("PCA of patch-wise DINO-v2 distances")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(["Normal","Anomaly"]); plt.tight_layout()
plt.savefig(args.out/"pca_scatter.png", dpi=300); plt.close()

pd.DataFrame(dict(img=stems, pc1=pcs[:,0], pc2=pcs[:,1],
                  anomaly=labels)).to_csv(args.out/"pca_coordinates.csv",
                                          index=False)

print(f"Finished.\n  Per-image outputs ➔ {heat_root}/<stem>/"
      f"\n  PCA plot          ➔ {args.out/'pca_scatter.png'}")
