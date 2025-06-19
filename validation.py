# orientation_validator.py  (fixed)
# -------------------------------------------------------------
# • Searches one root recursively for *_cls* dirs
# • Skips any path containing tokens in --skip  (Matches by default)
# • Resizes best_rotated to reference height before hconcat => no cv2 error
# -------------------------------------------------------------
import argparse, cv2, glob
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ── helpers ────────────────────────────────────────────────────────────
def cv2_to_tk(img, max_h=500):
    h, w = img.shape[:2]
    s = min(1.0, max_h / h)
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) \
             .resize((int(w*s), int(h*s)), Image.BICUBIC)
    return ImageTk.PhotoImage(im)

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW, nH = int(h*sin + w*cos), int(h*cos + w*sin)
    M[0,2] += (nW/2) - w/2
    M[1,2] += (nH/2) - h/2
    return cv2.warpAffine(img, M, (nW, nH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

def gather(root: Path, skip_tokens):
    res = []
    patt = str(root / "**/*_cls*")
    for p in glob.iglob(patt, recursive=True):
        p = Path(p)
        if not p.is_dir():          continue
        if any(tok.lower() in str(p).lower() for tok in skip_tokens): continue
        if (p/'best_reference.jpg').exists() and (p/'best_rotated.jpg').exists():
            res.append(p)
    return sorted(res)

# ── main GUI class ─────────────────────────────────────────────────────
class Validator:
    def __init__(self, master, folders, step):
        self.master, self.step = master, step
        self.folders = iter(folders)
        self.total   = len(folders)
        self.wrong   = 0
        self.canvas  = tk.Label(master); self.canvas.pack()

        bt = tk.Frame(master); bt.pack(pady=6)
        tk.Button(bt, text="✓ OK", width=9,
                  command=self.accept).grid(row=0,column=0,padx=4)
        tk.Button(bt, text="⟲", width=6,
                  command=lambda:self.rotate(-step)).grid(row=0,column=1)
        tk.Button(bt, text="⟳", width=6,
                  command=lambda:self.rotate(step)).grid(row=0,column=2)
        tk.Button(bt, text="💾 Save / Next", width=12,
                  command=self.save_wrong).grid(row=0,column=3,padx=4)

        self.next()

    # ── navigation ───────────────────────────────────────────────
    def next(self):
        try: self.cur = next(self.folders)
        except StopIteration: return self.finish()
        self.ref = cv2.imread(str(self.cur/'best_reference.jpg'))
        self.mov = cv2.imread(str(self.cur/'best_rotated.jpg'))
        if self.ref is None or self.mov is None: return self.next()
        self.show()

    def accept(self): self.next()

    def save_wrong(self):
        cv2.imwrite(str(self.cur/'best_rotated.jpg'), self.mov)
        self.wrong += 1
        self.next()

    # ── UI helpers ───────────────────────────────────────────────
    def rotate(self, deg):
        self.mov = rotate(self.mov, deg); self.show()

    def show(self):
        h_ref = self.ref.shape[0]
        mov_r = cv2.resize(self.mov, (int(self.mov.shape[1]*h_ref/self.mov.shape[0]), h_ref))
        combo = cv2.hconcat([self.ref, mov_r])
        self.tkimg = cv2_to_tk(combo); self.canvas.configure(image=self.tkimg)

    def finish(self):
        pct = 100 * self.wrong / self.total if self.total else 0
        messagebox.showinfo("Summary",
            f"Total: {self.total}\nWrong fixed: {self.wrong}\n{pct:.1f}%")
        self.master.quit()

# ── CLI / entry point ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Orientation Validator")
    ap.add_argument("--root", type=Path, required=True,
                    help="root folder to scan recursively")
    ap.add_argument("--skip", nargs='*', default=["Matches"],
                    help="substrings to skip (case-insensitive)")
    ap.add_argument("--step", type=int, default=90,
                    help="rotate step (deg) for ⟲ / ⟳")
    args = ap.parse_args()

    folders = gather(args.root, args.skip)
    if not folders:
        print("No reviewable folders found."); return

    root = tk.Tk(); root.title("Orientation Validator")
    Validator(root, folders, args.step)
    root.mainloop()

if __name__ == "__main__":
    main()
