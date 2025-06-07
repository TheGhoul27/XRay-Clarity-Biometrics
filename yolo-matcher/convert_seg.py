#!/usr/bin/env python3
"""
convert_to_yolo8_seg.py  –  polygon-only → YOLO-v8 segmentation labels
Adds axis-aligned bbox (cx cy w h) in front of each polygon line.
"""
from pathlib import Path
import argparse, numpy as np, sys
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root',  type=Path, required=True,
                   help='dataset root (contains images/ and labels/)')
    p.add_argument('--split', default='train',
                   help='sub-folder to convert (train | val | test)')
    return p.parse_args()

def main():
    args      = parse_args()
    lbl_in_d  = args.root / 'labels'      / args.split
    lbl_out_d = args.root / 'labels-seg'  / args.split
    lbl_out_d.mkdir(parents=True, exist_ok=True)

    for txt in tqdm(sorted(lbl_in_d.glob('*.txt')), desc='converting'):
        with open(txt) as f_in, open(lbl_out_d / txt.name, 'w') as f_out:
            for ln in f_in:
                nums = [float(x) for x in ln.strip().split()]
                if len(nums) < 13 or (len(nums) - 1) % 2:
                    print(f'⚠  bad line in {txt}: {ln.strip()}', file=sys.stderr)
                    continue

                cls, seg = int(nums[0]), np.array(nums[1:]).reshape(-1, 2)

                # compute axis-aligned bbox
                xmin, ymin = seg.min(0)
                xmax, ymax = seg.max(0)
                cx, cy     = (xmin + xmax) / 2, (ymin + ymax) / 2
                w,  h      = xmax - xmin, ymax - ymin

                out_nums = [cx, cy, w, h] + seg.flatten().tolist()

                # write: integer class-id + floats
                f_out.write(
                    f'{cls:d} ' + ' '.join(f'{x:.6f}' for x in out_nums) + '\n'
                )

if __name__ == '__main__':
    main()
