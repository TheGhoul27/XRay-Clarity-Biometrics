#!/usr/bin/env python3
import json, os, re, shutil, csv, random
from collections import defaultdict
from typing import Dict, Any

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")

def normalize_face(value: str) -> str | None:
    v = str(value or "").lower()
    v = re.sub(r"[^a-z]", "_", v)
    if "up" in v:
        return "face_up"
    if "down" in v:
        return "face_down"
    return None

def existing_path(preferred: str) -> str | None:
    if os.path.exists(preferred):
        return preferred
    root, ext = os.path.splitext(preferred)
    alt = root + (".jpg" if ext.lower() != ".jpg" else ".tif")
    return alt if os.path.exists(alt) else None

def main(
    dataset_root="dataset",
    json_rel="images/scans/products.json",
    output_rel="pairs",
    report_rel="pairs_report.csv",
):
    json_path   = os.path.join(dataset_root, json_rel)
    images_root = os.path.join(dataset_root, "images", "scans")
    out_root    = os.path.join(dataset_root, output_rel)
    report_path = os.path.join(dataset_root, report_rel)

    os.makedirs(out_root, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Collect ALL Face_Up / Face_Down per product (Mint only)
    groups: Dict[str, Dict[str, list[str]]] = defaultdict(lambda: {"face_up": [], "face_down": []})
    malformed = 0

    for r in records:
        try:
            if str(r.get("condition", "")).lower() != "mint":
                continue

            product = r.get("product_name")
            face_key = normalize_face(r.get("face"))
            relpath = r.get("path") or r.get("jpg_filename")

            if not product or not face_key or not relpath:
                malformed += 1
                continue

            groups[product][face_key].append(relpath)
        except Exception:
            malformed += 1

    copied = 0
    rows = []
    for product, faces in groups.items():
        safe = sanitize(product)
        dest_dir = os.path.join(out_root, safe)
        os.makedirs(dest_dir, exist_ok=True)

        # Randomly pick ONE candidate for each side if available
        src_up_rel   = random.choice(faces["face_up"]) if faces["face_up"] else None
        src_down_rel = random.choice(faces["face_down"]) if faces["face_down"] else None

        src_up_abs   = existing_path(os.path.join(images_root, src_up_rel)) if src_up_rel else None
        src_down_abs = existing_path(os.path.join(images_root, src_down_rel)) if src_down_rel else None

        dst_up = dst_down = None

        if src_up_abs:
            ext = os.path.splitext(src_up_abs)[1].lower()
            dst_up = os.path.join(dest_dir, f"face_up{ext}")
            shutil.copy2(src_up_abs, dst_up)
            copied += 1

        if src_down_abs:
            ext = os.path.splitext(src_down_abs)[1].lower()
            dst_down = os.path.join(dest_dir, f"face_down{ext}")
            shutil.copy2(src_down_abs, dst_down)
            copied += 1

        rows.append({
            "product_name": product,
            "has_face_up": bool(src_up_abs),
            "has_face_down": bool(src_down_abs),
            "dest_face_up": dst_up or "",
            "dest_face_down": dst_down or "",
        })

    # Write report CSV
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_name","has_face_up","has_face_down","dest_face_up","dest_face_down"])
        writer.writeheader()
        writer.writerows(rows)

    missing_count = sum(1 for r in rows if not (r["has_face_up"] and r["has_face_down"]))
    print(f"Done. Copied files: {copied}")
    print(f"Products processed: {len(rows)} | Products missing a side: {missing_count}")
    if malformed:
        print(f"Skipped malformed/unsupported records: {malformed}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main(
        dataset_root="cropped",
        json_rel="images/scans/products.json",
        output_rel="pairs",
        report_rel="pairs_report.csv",
    )
