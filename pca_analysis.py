# @torch.inference_mode()
# def patch_embed(bgr: np.ndarray) -> torch.Tensor:
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     x   = pre(Image.fromarray(rgb)).unsqueeze(0).to(device)

#     out = model.forward_features(x)           # hub & fine-tuned both
#     if isinstance(out, dict):
#         if "x_norm_patchtokens" in out:
#             toks = out["x_norm_patchtokens"]
#         elif "x" in out:
#             toks = out["x"]
#         else:                                 # fallback: first tensor value
#             toks = next(v for v in out.values() if torch.is_tensor(v))
#     else:                                     # some scripted models
#         toks = out                            # tensor already

#     if toks.shape[1] == 257:                  # CLS + 256 patches
#         toks = toks[:, 1:, :]
#     return toks.squeeze(0).cpu()              # [256, D]

# pca_similarity_analysis.py – DINO-v2 patch-distance analysis
# --------------------------------------------------------------------
import argparse, glob, cv2, torch, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.manifold      import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ───────────── CLI ───────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--root",  required=True, type=Path, help="results root")
ap.add_argument("--csv",   required=True, type=Path, help="Clarity CSV")
ap.add_argument("--out",   default="pca_out", type=Path, help="output folder")
ap.add_argument("--model", default="dinov2_vits14",
                choices=["dinov2_vits14","dinov2_vitb14","dinov2_vitl14"])
ap.add_argument("--ckpt",  type=Path, help="local .pt (state_dict/full model)")
ap.add_argument("--arch",  choices=["vits14","vitb14","vitl14"],
                help="backbone of --ckpt (overrides auto-detect)")
args = ap.parse_args()
heat_root = args.out / "heatmaps"; heat_root.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────── DINO-v2 loader (hub or local) ─────────────────────────
def load_model():
    if not args.ckpt:
        mdl = torch.hub.load("facebookresearch/dinov2", args.model)
        print("✓ hub:", args.model); return mdl.to(device).eval()

    obj = torch.load(args.ckpt, map_location=device)
    if isinstance(obj, torch.nn.Module):
        print("✓ local full model:", args.ckpt); return obj.to(device).eval()

    sd = obj                              # state-dict
    if args.arch:
        hub_id = {"vits14":"dinov2_vits14",
                  "vitb14":"dinov2_vitb14",
                  "vitl14":"dinov2_vitl14"}[args.arch]
    else:
        hid = sd["cls_token"].shape[-1]
        hub_id = {384:"dinov2_vits14",768:"dinov2_vitb14",
                  1024:"dinov2_vitl14"}[hid]
        print(f"✓ auto-detected dim={hid} → {hub_id}")

    mdl = torch.hub.load("facebookresearch/dinov2", hub_id)
    mdl.load_state_dict(sd, strict=False)
    print("✓ state_dict loaded from", args.ckpt)
    return mdl.to(device).eval()

model = load_model()

# ───────────── preprocessing ─────────────────────────────────────────
IMG = 224
pre = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@torch.inference_mode()
def patch_embed(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x   = pre(Image.fromarray(rgb)).unsqueeze(0).to(device)

    out = model.forward_features(x)           # hub & fine-tuned both
    if isinstance(out, dict):
        if "x_norm_patchtokens" in out:
            toks = out["x_norm_patchtokens"]
        elif "x" in out:
            toks = out["x"]
        else:                                 # fallback: first tensor value
            toks = next(v for v in out.values() if torch.is_tensor(v))
    else:                                     # some scripted models
        toks = out                            # tensor already

    if toks.shape[1] == 257:                  # CLS + 256 patches
        toks = toks[:, 1:, :]
    return toks.squeeze(0).cpu()              # [256, D]

def patch_distance(ref, mov):
    e1, e2 = patch_embed(ref), patch_embed(mov)
    diag = torch.diagonal(e1 @ e2.T)
    cos  = diag / (torch.norm(e1,dim=1)*torch.norm(e2,dim=1)+1e-8)
    return (1.0 - cos).numpy()            # [256]

# ───────────── CSV → anomaly + item maps ────────────────────────────
df = pd.read_csv(args.csv)
label_map, item_map = {}, {}
for _,r in df.iterrows():
    item = str(r["Item"]).strip()
    for col in ['Frontside ("FS") Scan Filename',
                'Backside ("BS") Scan Filename']:
        fn = r.get(col)
        if pd.isna(fn): continue
        stem = Path(fn).stem
        label_map[stem] = bool(r['Anomaly Status'])
        item_map [stem] = item

# ───────────── collect vectors & visuals ─────────────────────────────
vectors, labels, stems, items = [], [], [], []
for fld in glob.iglob(str(args.root/"**/*_cls*"), recursive=True):
    folder = Path(fld)
    ref_p, rot_p = folder/"best_reference.jpg", folder/"best_rotated.jpg"
    if not (ref_p.exists() and rot_p.exists()): continue
    stem = folder.name.split("_cls")[0]
    if stem not in label_map: continue

    ref = cv2.cvtColor(cv2.imread(str(ref_p)), cv2.COLOR_BGR2RGB)
    rot = cv2.cvtColor(cv2.imread(str(rot_p)), cv2.COLOR_BGR2RGB)

    dist = patch_distance(ref, rot)
    vectors.append(dist); labels.append(label_map[stem])
    stems.append(stem);   items.append(item_map[stem])

    hm   = cv2.resize(dist.reshape(16,16), (IMG,IMG), cv2.INTER_LINEAR)
    rotR = cv2.resize(rot,(IMG,IMG)); refR=cv2.resize(ref,(IMG,IMG))
    rgba = (plt.cm.hot(hm)*255).astype(np.uint8)
    overlay = cv2.addWeighted(rotR,0.7,rgba[:,:,:3],0.3,0)
    dst = heat_root/stem; dst.mkdir(exist_ok=True)
    cv2.imwrite(str(dst/f"{stem}_overlay.png"),
                cv2.cvtColor(cv2.hconcat([refR,overlay]), cv2.COLOR_RGB2BGR))
    fig,ax=plt.subplots(1,3,figsize=(12,4));[a.axis("off") for a in ax]
    ax[0].imshow(refR);ax[0].set_title("Reference")
    ax[1].imshow(overlay);ax[1].set_title("Rot + Overlay")
    ax[2].imshow(rgba[:,:,:3]);ax[2].set_title("Overlay only")
    fig.tight_layout();fig.savefig(dst/f"{stem}_triplet.png",dpi=200);plt.close(fig)
    plt.figure(figsize=(6,3))
    plt.hist(dist,bins=50,color="steelblue",edgecolor="k",alpha=0.85)
    plt.xlabel("1−cosine");plt.ylabel("#patches");plt.tight_layout()
    plt.savefig(dst/f"{stem}_hist.png",dpi=200);plt.close()

if not vectors: raise RuntimeError("No labelled images found")

vectors = np.stack(vectors); labels = np.array(labels)
np.save(args.out/"vectors.npy", vectors)
np.save(args.out/"labels.npy",  labels)

# ───────────── PCA & t-SNE (item colour, anomaly edge) ──────────────
X_std = StandardScaler().fit_transform(vectors)
pcs   = PCA(n_components=2).fit_transform(X_std)
tsne  = TSNE(n_components=2, perplexity=30, init="pca",
             random_state=42).fit_transform(X_std)

uniq_items = sorted(set(items))
palette = plt.get_cmap("tab10", len(uniq_items))
item_col = {it: palette(i) for i,it in enumerate(uniq_items)}

def scatter(mat,fname,title,xl,yl):
    plt.figure(figsize=(7,6))
    for it in uniq_items:
        idx=[i for i,v in enumerate(items) if v==it]
        plt.scatter(mat[idx,0],mat[idx,1],
                    c=[item_col[it]]*len(idx),
                    edgecolors=np.where(labels[idx],"red","black"),
                    linewidths=1.1,alpha=0.85,label=it,s=50)
    plt.title(f"{title}\ncolour=item  edge=anomaly(red)")
    plt.xlabel(xl);plt.ylabel(yl);plt.legend(fontsize=7,ncol=2)
    plt.tight_layout();plt.savefig(args.out/fname,dpi=300);plt.close()

args.out.mkdir(exist_ok=True)
scatter(pcs,"pca_scatter.png","PCA","PC1","PC2")
scatter(tsne,"tsne_scatter.png","t-SNE","t-SNE-1","t-SNE-2")

pd.DataFrame(dict(img=stems, pc1=pcs[:,0], pc2=pcs[:,1],
                  ts1=tsne[:,0], ts2=tsne[:,1],
                  item=items, anomaly=labels))\
  .to_csv(args.out/"embeddings_coordinates.csv",index=False)

# ───────────── global train/test classifiers ─────────────────────────
Xtr,Xte,ytr,yte = train_test_split(X_std, labels, test_size=0.30,
                                   stratify=labels, random_state=42)
models = {
    "LogReg": LogisticRegression(max_iter=1000,solver="lbfgs"),
    "SVM-RBF": SVC(kernel="rbf",probability=False,class_weight="balanced"),
    "RandForest": RandomForestClassifier(n_estimators=200,random_state=42)
}

report_txt = []
for name, clf in models.items():
    clf.fit(Xtr,ytr); y_pred=clf.predict(Xte)
    rep = classification_report(yte,y_pred,target_names=["Normal","Anomaly"],
                                zero_division=0)
    cm  = confusion_matrix(yte,y_pred)
    print(f"\n=== {name} ===\n{rep}\nConfusion-matrix:\n{cm}\n")
    report_txt.append(f"=== {name} ===\n{rep}\nConfusion-matrix:\n{cm}\n")

with open(args.out/"classification_reports.txt","w") as f:
    f.write("\n".join(report_txt))

print(f"\nFinished."
      f"\n  vectors.npy / labels.npy saved"
      f"\n  PCA → {args.out/'pca_scatter.png'}"
      f"\n  t-SNE → {args.out/'tsne_scatter.png'}"
      f"\n  reports → {args.out/'classification_reports.txt'}")
