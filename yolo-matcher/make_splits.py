#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
from collections import defaultdict, Counter

def load_rows(products_json: Path, images_root: Path):
    with open(products_json, "r") as f:
        meta = json.load(f)

    rows = []
    for r in meta:
        rel = Path(r["path"])              # e.g. "mint/BAGGAGE_....jpg"
        img = (images_root / rel).resolve()
        rows.append({
            "abs_path": str(img),
            "rel_path": str(rel),
            "product": r.get("product_name", "unknown"),
            "condition": r.get("condition", "na"),
            "face": r.get("face", "na"),
            "brand": r.get("brand", "na"),
            "category": r.get("product_category", "na"),
        })
    return rows

def split_per_product(items, val_ratio=0.2, seed=42, keep_condition=True):
    """
    items: list[dict] for a single product.
    Guarantee: if len>=2, at least one sample goes to val.
    Try to preserve condition balance within the product.
    """
    n = len(items)
    if n == 1:
        return items, []  # all train

    # how many for val
    v = max(1, min(n - 1, round(n * val_ratio))) if n >= 5 else 1

    rng = random.Random(seed)

    if keep_condition:
        buckets = defaultdict(list)
        for it in items:
            buckets[it["condition"]].append(it)
        # proportional target per bucket
        targets = {}
        for k, lst in buckets.items():
            targets[k] = max(0, round(v * len(lst) / n))
        # fix rounding drift
        diff = v - sum(targets.values())
        keys_by_size = sorted(buckets, key=lambda k: len(buckets[k]), reverse=True)
        i = 0
        while diff != 0 and keys_by_size:
            k = keys_by_size[i % len(keys_by_size)]
            add = 1 if diff > 0 else -1
            new_t = max(0, min(len(buckets[k]), targets[k] + add))
            diff += (targets[k] - new_t)
            targets[k] = new_t
            i += 1

        val, train = [], []
        for k, lst in buckets.items():
            rng.shuffle(lst)
            cut = targets[k]
            val += lst[:cut]
            train += lst[cut:]
        if len(val) == 0:  # safety
            rng.shuffle(train)
            val.append(train.pop())
    else:
        items = items[:]
        rng.shuffle(items)
        val, train = items[:v], items[v:]

    return train, val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="dataset/ folder (contains images/scans and labels/scans)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emit-cls", action="store_true",
                    help="also write cls_train.txt/cls_val.txt with 'path label' for ResNet+ArcFace")
    args = ap.parse_args()

    root = args.root.resolve()
    images_root = root / "images" / "scans"
    labels_root = root / "labels" / "scans"
    products_json = images_root / "products.json"
    assert products_json.exists(), f"Missing {products_json}"

    rows = load_rows(products_json, images_root)

    # group by product
    by_product = defaultdict(list)
    for r in rows:
        by_product[r["product"]].append(r)

    rng = random.Random(args.seed)
    all_train, all_val = [], []
    for prod, items in by_product.items():
        tr, va = split_per_product(items, args.val_ratio, args.seed, keep_condition=True)
        all_train += tr
        all_val += va

    # quick report
    def report(name, items):
        c = Counter(x["condition"] for x in items)
        n = len(items)
        s = ", ".join(f"{k}:{v} ({v/n:.1%})" for k,v in sorted(c.items(), key=lambda kv: -kv[1]))
        print(f"{name}: {n} | {s}")

    print(f"Total images: {len(rows)}")
    report("Train", all_train)
    report("Val", all_val)

    # check labels exist and warn
    missing = []
    for r in rows:
        lbl = (labels_root / Path(r["rel_path"]).with_suffix(".txt")).resolve()
        if not lbl.exists():
            missing.append(str(lbl))
    if missing:
        print(f"[WARN] {len(missing)} missing label files (showing up to 10):")
        for m in missing[:10]:
            print("  ", m)

    # write lists for YOLO-seg (image-only lists are OK; Ultralytics resolves labels)
    outdir = root / "splits"
    outdir.mkdir(parents=True, exist_ok=True)
    seg_train = outdir / "seg_train.txt"
    seg_val   = outdir / "seg_val.txt"

    with open(seg_train, "w") as f:
        for r in all_train:
            f.write(r["abs_path"] + "\n")
    with open(seg_val, "w") as f:
        for r in all_val:
            f.write(r["abs_path"] + "\n")

    print(f"\nWrote:\n  {seg_train}\n  {seg_val}")

    # optional: emit classification splits for ResNet+ArcFace later
    if args.emit_cls:
        cls_train = outdir / "cls_train.txt"
        cls_val   = outdir / "cls_val.txt"
        with open(cls_train, "w") as f:
            for r in all_train:
                f.write(f"{r['abs_path']} {r['product']}\n")
        with open(cls_val, "w") as f:
            for r in all_val:
                f.write(f"{r['abs_path']} {r['product']}\n")
        print(f"Also wrote:\n  {cls_train}\n  {cls_val}")

if __name__ == "__main__":
    main()
