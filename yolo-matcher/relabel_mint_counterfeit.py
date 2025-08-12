#!/usr/bin/env python3
"""
relabel_mint_counterfeit.py

Update YOLO label files so that:
  - labels/scans/mint/*.txt          -> class id 1
  - labels/scans/counterfeit/*.txt   -> class id 0

Works for:
  - YOLO segmentation (class + polygon coords)
  - YOLO boxes or OBB-like lines (class + numbers)
It simply replaces the FIRST token on each non-empty line.

Usage:
  python relabel_mint_counterfeit.py --root dataset
  python relabel_mint_counterfeit.py --root dataset --no-backup --dry-run
"""

import argparse
from pathlib import Path

def relabel_file(txt_path: Path, new_id: int, dry_run: bool = False, backup: bool = True) -> tuple[int, int]:
    """
    Return (lines_processed, lines_changed).
    """
    if not txt_path.is_file() or txt_path.suffix.lower() != ".txt":
        return (0, 0)

    try:
        lines = txt_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        # fallback encoding
        lines = txt_path.read_text(encoding="latin-1").splitlines()

    changed = 0
    out_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            out_lines.append(line)
            continue
        parts = s.split()
        # Replace only if first token looks like an int class id
        try:
            _ = int(parts[0])
            parts[0] = str(new_id)
            out_lines.append(" ".join(parts))
            changed += 1
        except ValueError:
            # First token isn't an integer — leave as-is
            out_lines.append(line)

    if not dry_run:
        if backup:
            bak = txt_path.with_suffix(txt_path.suffix + ".bak")
            if not bak.exists():
                bak.write_text("\n".join(lines), encoding="utf-8")
        txt_path.write_text("\n".join(out_lines) + ("\n" if lines else ""), encoding="utf-8")

    return (len(lines), changed)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root (the folder that contains images/ and labels/)")
    ap.add_argument("--labels_subdir", default="labels/scans",
                    help="Path under root to the labels (e.g., labels/scans)")
    ap.add_argument("--mint_dirname", default="mint", help="Subfolder name for mint labels")
    ap.add_argument("--counterfeit_dirname", default="counterfeit", help="Subfolder name for counterfeit labels")
    ap.add_argument("--mint_id", type=int, default=1, help="New class id for mint")
    ap.add_argument("--counterfeit_id", type=int, default=0, help="New class id for counterfeit")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    ap.add_argument("--no-backup", action="store_true", help="Do not write .bak backups")
    args = ap.parse_args()

    labels_root = (args.root / args.labels_subdir).resolve()
    mint_dir = labels_root / args.mint_dirname
    cf_dir   = labels_root / args.counterfeit_dirname

    if not mint_dir.exists() and not cf_dir.exists():
        raise SystemExit(f"Could not find '{mint_dir}' or '{cf_dir}'. Check --root / --labels_subdir / dir names.")

    total_files = total_lines = total_changed = 0

    def process_dir(d: Path, new_id: int, tag: str):
        nonlocal total_files, total_lines, total_changed
        if not d.exists():
            print(f"[WARN] {tag} labels folder not found: {d}")
            return
        files = sorted(d.rglob("*.txt"))
        print(f"[INFO] Processing {len(files)} files in {d} (set class -> {new_id})")
        for fp in files:
            ln, ch = relabel_file(fp, new_id, dry_run=args.dry_run, backup=not args.no_backup)
            total_files += 1
            total_lines += ln
            total_changed += ch

    process_dir(mint_dir, args.mint_id, "mint")
    process_dir(cf_dir, args.counterfeit_id, "counterfeit")

    mode = "DRY-RUN" if args.dry_run else "UPDATED"
    print(f"\n[{mode}] Files: {total_files} | Lines scanned: {total_lines} | Lines relabeled: {total_changed}")
    if not args.dry_run and not args.no_backup:
        print("Backups saved alongside files with .bak extension.")

if __name__ == "__main__":
    main()
