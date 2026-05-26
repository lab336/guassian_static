#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-crop PLY point cloud files using ranges from crop_range.txt.

Usage:
    python crop_ply_batch.py  crop_range.txt  <input_folder>
    python crop_ply_batch.py  crop_range.txt  <input_folder>  <output_folder> --invert

If output_folder is omitted, a sibling folder named <input_folder>_cropped is created.
"""
import sys
import re
import argparse
from pathlib import Path
import numpy as np


# ───────────────────────────────────────────────────────────
#  Parse crop_range.txt
# ───────────────────────────────────────────────────────────
def parse_crop_range(txt_path):
    """Return dict {'x':(lo,hi), 'y':(lo,hi), 'z':(lo,hi)}."""
    text = Path(txt_path).read_text(encoding='utf-8')
    crop = {}
    for axis in 'XYZ':
        m = re.search(rf'{axis}:\s*\[([^\],]+),\s*([^\]]+)\]', text)
        if not m:
            raise ValueError(f"Cannot parse {axis} range in {txt_path}")
        crop[axis.lower()] = (float(m.group(1)), float(m.group(2)))
    return crop


# ───────────────────────────────────────────────────────────
#  PLY I/O  (no external dependencies)
# ───────────────────────────────────────────────────────────
_DT = {
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
    'int':   'i4', 'int32':   'i4', 'uint':   'u4', 'uint32':  'u4',
    'short': 'i2', 'int16':   'i2', 'ushort': 'u2', 'uint16':  'u2',
    'char':  'i1', 'int8':    'i1', 'uchar':  'u1', 'uint8':   'u1',
}


def load_ply(filepath):
    """Return (pts Nx3 float32, clr Nx3 float32 [0-1] or None)."""
    with open(filepath, 'rb') as f:
        n = 0; props = []; binary = False; le = True; in_v = False
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            tok  = line.split()
            if not tok:
                continue
            if tok[0] == 'format':
                binary = 'binary' in line
                le     = 'big_endian' not in line
            elif tok[0] == 'element' and len(tok) > 1:
                in_v = tok[1] == 'vertex'
                if in_v:
                    n = int(tok[2])
            elif tok[0] == 'property' and in_v:
                props.append((tok[1], tok[-1]))   # (type_str, name)
            elif line == 'end_header':
                break

        end = '<' if le else '>'
        if binary:
            dt  = np.dtype([(nm, end + _DT.get(tp, 'f4')) for tp, nm in props])
            raw = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
            g   = lambda nm: raw[nm].astype(np.float32)
            pts = np.stack([g('x'), g('y'), g('z')], axis=1)
            clr = (np.stack([g('red') / 255, g('green') / 255, g('blue') / 255], axis=1)
                   if 'red' in raw.dtype.names else None)
        else:
            rows = [f.readline().decode() for _ in range(n)]
            nms  = [p[1] for p in props]
            arr  = np.array([[float(v) for v in r.split()] for r in rows],
                            dtype=np.float32)
            d    = {nms[i]: arr[:, i] for i in range(len(nms))}
            pts  = np.stack([d['x'], d['y'], d['z']], axis=1)
            clr  = (np.stack([d['red'] / 255, d['green'] / 255, d['blue'] / 255], axis=1)
                    if 'red' in d else None)

    return pts, clr


def save_ply(filepath, pts, clr):
    """Save as binary_little_endian PLY.  clr: Nx3 float [0-1] or None."""
    n         = len(pts)
    has_color = clr is not None

    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header\n")
    header = "\n".join(lines).encode('ascii')

    pts32 = pts.astype(np.float32)

    if has_color:
        clr_u8 = (np.clip(clr, 0.0, 1.0) * 255).astype(np.uint8)
        # Pack as tightly-packed struct: 3×float32 + 3×uint8 = 15 bytes/vertex
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                       ('r', 'u1'),  ('g', 'u1'),  ('b', 'u1')])
        data       = np.empty(n, dtype=dt)
        data['x']  = pts32[:, 0]; data['y'] = pts32[:, 1]; data['z'] = pts32[:, 2]
        data['r']  = clr_u8[:, 0]; data['g'] = clr_u8[:, 1]; data['b'] = clr_u8[:, 2]
        body = data.tobytes()
    else:
        body = pts32.tobytes()

    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(body)


# ───────────────────────────────────────────────────────────
#  Crop a single file
# ───────────────────────────────────────────────────────────
def crop_one(src, dst, crop, invert=False):
    """
    Load src, apply crop, write dst.
    invert=True  -> keep points OUTSIDE the box (subtract region).
    Returns (n_total, n_kept).
    """
    pts, clr = load_ply(str(src))
    inside = (
        (pts[:, 0] >= crop['x'][0]) & (pts[:, 0] <= crop['x'][1]) &
        (pts[:, 1] >= crop['y'][0]) & (pts[:, 1] <= crop['y'][1]) &
        (pts[:, 2] >= crop['z'][0]) & (pts[:, 2] <= crop['z'][1])
    )
    mask   = ~inside if invert else inside
    n_kept = int(mask.sum())
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    save_ply(str(dst), pts[mask], clr[mask] if clr is not None else None)
    return len(pts), n_kept


# ───────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Batch-crop PLY files from crop_range.txt')
    parser.add_argument('crop_file', help='crop_range.txt path')
    parser.add_argument('input_dir', help='Folder containing .ply files')
    parser.add_argument('output_dir', nargs='?', default=None,
                        help='Output folder (default: <input_dir>_cropped / <input_dir>_inverted)')
    parser.add_argument('--invert', action='store_true',
                        help='Keep points OUTSIDE the crop box instead of inside')
    args = parser.parse_args()

    crop = parse_crop_range(args.crop_file)
    print("Crop range:")
    for ax, (lo, hi) in crop.items():
        print(f"  {ax.upper()}: [{lo:.6f}, {hi:.6f}]")

    in_dir  = Path(args.input_dir)
    suffix  = '_inverted' if args.invert else '_cropped'
    out_dir = (Path(args.output_dir) if args.output_dir
               else in_dir.parent / (in_dir.name + suffix))

    ply_files = sorted(in_dir.glob('*.ply'))
    if not ply_files:
        print(f"No .ply files found in: {in_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    mode_label = 'INVERT (keep outside)' if args.invert else 'CROP (keep inside)'
    print(f"\nMode: {mode_label}")
    print(f"{len(ply_files)} files  |  {in_dir}  ->  {out_dir}\n")

    total_in = total_out = 0
    for i, src in enumerate(ply_files, 1):
        dst = out_dir / src.name
        n_in, n_out = crop_one(src, dst, crop, invert=args.invert)
        total_in  += n_in
        total_out += n_out
        pct = n_out / max(n_in, 1) * 100
        print(f"  [{i:>4}/{len(ply_files)}]  {src.name:<28}  "
              f"{n_in:>8,} -> {n_out:>8,}  ({pct:.1f}%)")

    print(f"\nDone.  Total: {total_in:,} -> {total_out:,} points")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
