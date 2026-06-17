#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY Point Cloud Crop Viewer

Usage:
    python points/ply_viewer.py  path/to/file.ply
    python points/ply_viewer.py  path/to/file.ply  --max_display 80000

Interaction:
    Left-drag 3D view  : rotate
    Right-drag         : zoom
    Sliders            : crop the cloud in real time
                         (axes are fixed -- only a virtual "wall" moves)
    Export             : save current range to crop_range.txt
    Reset              : restore full cloud
"""
import sys
import argparse
import numpy as np

# ── Fix Chinese font on Windows before importing pyplot ──────────────
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'FangSong', 'KaiTi', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Fix Windows console UTF-8 output
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ═══════════════════════════════════════════════════════════
#  PLY loader  (open3d first, then built-in fallback)
# ═══════════════════════════════════════════════════════════
def load_ply(filepath: str):
    """Return (points Nx3 float32, colors Nx3 float32 or None)."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) > 0:
            clr = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
            print(f"[open3d] {len(pts):,} points")
            return pts, clr
        print("[open3d] returned 0 points, falling back to built-in parser ...")
    except Exception:
        pass

    print("Using built-in PLY parser ...")
    _DT = {
        'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
        'int':   'i4', 'int32':   'i4', 'uint':   'u4', 'uint32':  'u4',
        'short': 'i2', 'int16':   'i2', 'ushort': 'u2', 'uint16':  'u2',
        'char':  'i1', 'int8':    'i1', 'uchar':  'u1', 'uint8':   'u1',
    }
    with open(filepath, 'rb') as f:
        n = 0; props = []; binary = False; le = True; inv = False
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            tok  = line.split()
            if not tok: continue
            if tok[0] == 'format':
                binary = 'binary' in line; le = 'big_endian' not in line
            elif tok[0] == 'element' and len(tok) > 1:
                inv = tok[1] == 'vertex'
                if inv: n = int(tok[2])
            elif tok[0] == 'property' and inv:
                props.append((tok[1], tok[-1]))
            elif line == 'end_header':
                break
        end = '<' if le else '>'
        if binary:
            dt  = np.dtype([(nm, end + _DT.get(tp, 'f4')) for tp, nm in props])
            raw = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
            g   = lambda nm: raw[nm].astype('f4')
            pts = np.stack([g('x'), g('y'), g('z')], axis=1)
            clr = (np.stack([g('red')/255, g('green')/255, g('blue')/255], axis=1)
                   if 'red' in raw.dtype.names else None)
        else:
            rows = [f.readline().decode() for _ in range(n)]
            nms  = [p[1] for p in props]
            arr  = np.array([[float(v) for v in r.split()] for r in rows], dtype=np.float32)
            d    = {nms[i]: arr[:, i] for i in range(len(nms))}
            pts  = np.stack([d['x'], d['y'], d['z']], axis=1)
            clr  = (np.stack([d['red']/255, d['green']/255, d['blue']/255], axis=1)
                    if 'red' in d else None)
    print(f"[fallback] {len(pts):,} points")
    return pts, clr


# ═══════════════════════════════════════════════════════════
#  Main viewer
# ═══════════════════════════════════════════════════════════
def run(filepath: str, max_display: int = 80_000):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts, clr = load_ply(filepath)
    N = len(pts)

    if N == 0:
        raise RuntimeError(f"No points loaded from '{filepath}'. "
                           "Check that the file exists and is a valid PLY.")

    # Full (original) bounds — axis limits are locked to these forever
    XLO, XHI = float(pts[:,0].min()), float(pts[:,0].max())
    YLO, YHI = float(pts[:,1].min()), float(pts[:,1].max())
    ZLO, ZHI = float(pts[:,2].min()), float(pts[:,2].max())
    print(f"Total: {N:,}  X[{XLO:.3f},{XHI:.3f}]  Y[{YLO:.3f},{YHI:.3f}]  Z[{ZLO:.3f},{ZHI:.3f}]")

    # ── Build a fixed display subset (never changes size) ────────────
    if N > max_display:
        disp_idx = np.random.choice(N, max_display, replace=False)
    else:
        disp_idx = np.arange(N)
    dpts  = pts[disp_idx]                                   # (M, 3)
    dclr  = clr[disp_idx] if clr is not None else None      # (M, 3) or None
    M     = len(dpts)

    # Pre-compute stable base RGBA (alpha will be toggled per update)
    base_rgb = np.empty((M, 3), dtype=np.float32)
    if dclr is not None:
        base_rgb[:] = dclr
    else:
        zn = (dpts[:,2] - ZLO) / max(ZHI - ZLO, 1e-6)
        base_rgb[:,0] = zn
        base_rgb[:,1] = 0.45
        base_rgb[:,2] = 1.0 - zn
    ALPHA_VIS  = 0.75
    ALPHA_HIDE = 0.0
    rgba = np.ones((M, 4), dtype=np.float32)
    rgba[:, :3] = base_rgb
    rgba[:,  3] = ALPHA_VIS   # all visible at start

    # Crop state
    crop = dict(x=[XLO, XHI], y=[YLO, YHI], z=[ZLO, ZHI])

    # ── Figure layout ────────────────────────────────────────────────
    BG = '#1a1a1a'; FG = '#dddddd'; ACC = '#4a9eff'
    fig = plt.figure(figsize=(19, 9), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title('Point Cloud Crop Viewer')
    except Exception:
        pass

    # 3D view (left 62 %)
    ax3d = fig.add_axes([0.01, 0.04, 0.61, 0.93], projection='3d')
    ax3d.set_facecolor('#090912')
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#2a2a3a')
    ax3d.tick_params(colors='#888888', labelsize=7)
    ax3d.set_xlabel('X', color=FG, fontsize=9)
    ax3d.set_ylabel('Y', color=FG, fontsize=9)
    ax3d.set_zlabel('Z', color=FG, fontsize=9)

    # Draw scatter ONCE — same M points always; only alpha changes
    sc = ax3d.scatter(dpts[:,0], dpts[:,1], dpts[:,2],
                      c=rgba, s=0.9, linewidths=0, depthshade=False)

    # Lock axes to full bounds — NEVER change these again
    ax3d.set_xlim(XLO, XHI)
    ax3d.set_ylim(YLO, YHI)
    ax3d.set_zlim(ZLO, ZHI)

    # ── Right panel (6 sliders) ──────────────────────────────────────
    L, W = 0.66, 0.32
    TOP, GAP = 0.90, 0.108
    sls = {}
    DEFS = [
        ('x','lo','X  Min'), ('x','hi','X  Max'),
        ('y','lo','Y  Min'), ('y','hi','Y  Max'),
        ('z','lo','Z  Min'), ('z','hi','Z  Max'),
    ]
    for i, (axis, side, label) in enumerate(DEFS):
        lo = XLO if axis=='x' else YLO if axis=='y' else ZLO
        hi = XHI if axis=='x' else YHI if axis=='y' else ZHI
        init = lo if side == 'lo' else hi

        lax = fig.add_axes([L, TOP - i*GAP, 0.06, 0.030])
        lax.axis('off')
        lax.text(1.0, 0.5, label, ha='right', va='center', color=FG, fontsize=8.5)

        sax = fig.add_axes([L+0.065, TOP - i*GAP, W-0.07, 0.030], facecolor='#2c2c2c')
        step = (hi - lo) / 500 if hi != lo else 1e-4
        sl = Slider(sax, '', lo, hi, valinit=init, valstep=step, color=ACC)
        sl.valtext.set_color(FG)
        sl.valtext.set_fontsize(8)
        sls[f'{axis}_{side}'] = sl

    # Info panel
    iax = fig.add_axes([L, 0.24, W, 0.15])
    iax.axis('off')
    iax.patch.set_facecolor('#20202e')
    info_txt = iax.text(0.04, 0.96, '', va='top', color='#ffffff',
                         fontsize=8, fontfamily='monospace',
                         transform=iax.transAxes)

    # Buttons
    bex = fig.add_axes([L,          0.06, W*0.46, 0.065])
    brx = fig.add_axes([L+W*0.54,   0.06, W*0.46, 0.065])
    btn_exp = Button(bex, 'Export Range', color='#1f5c3a', hovercolor='#2e8b57')
    btn_rst = Button(brx, 'Reset',        color='#444444', hovercolor='#666666')
    for b in (btn_exp, btn_rst):
        b.label.set_color('#ffffff')
        b.label.set_fontsize(9)

    # ── Core update: toggle alpha only — axes NEVER change ───────────
    def _update(_=None):
        cx, cy, cz = crop['x'], crop['y'], crop['z']

        # Mask on the fixed display subset
        vis = ((dpts[:,0] >= cx[0]) & (dpts[:,0] <= cx[1]) &
               (dpts[:,1] >= cy[0]) & (dpts[:,1] <= cy[1]) &
               (dpts[:,2] >= cz[0]) & (dpts[:,2] <= cz[1]))

        rgba[:,3] = np.where(vis, ALPHA_VIS, ALPHA_HIDE)
        sc.set_facecolors(rgba)

        # Count in the full (non-subsampled) cloud for reporting
        vis_all = ((pts[:,0] >= cx[0]) & (pts[:,0] <= cx[1]) &
                   (pts[:,1] >= cy[0]) & (pts[:,1] <= cy[1]) &
                   (pts[:,2] >= cz[0]) & (pts[:,2] <= cz[1]))
        n_real = int(vis_all.sum())

        info_txt.set_text(
            f"  In range : {n_real:>8,}\n"
            f"  Displayed: {int(vis.sum()):>8,}\n"
            f"\n"
            f"  X [{cx[0]:+.4f},\n"
            f"      {cx[1]:+.4f}]\n"
            f"  Y [{cy[0]:+.4f},\n"
            f"      {cy[1]:+.4f}]\n"
            f"  Z [{cz[0]:+.4f},\n"
            f"      {cz[1]:+.4f}]"
        )
        fig.canvas.draw_idle()

    def _on_slider(_=None):
        for axis in 'xyz':
            lv = sls[f'{axis}_lo'].val
            hv = sls[f'{axis}_hi'].val
            if lv > hv:
                sls[f'{axis}_hi'].set_val(lv)   # triggers callback again
                return
            crop[axis] = [lv, hv]
        _update()

    def _on_export(_=None):
        vis = ((pts[:,0] >= crop['x'][0]) & (pts[:,0] <= crop['x'][1]) &
               (pts[:,1] >= crop['y'][0]) & (pts[:,1] <= crop['y'][1]) &
               (pts[:,2] >= crop['z'][0]) & (pts[:,2] <= crop['z'][1]))
        n = int(vis.sum())
        c = crop
        lines = [
            "===== Crop Range =====",
            f"X: [{c['x'][0]:.6f}, {c['x'][1]:.6f}]",
            f"Y: [{c['y'][0]:.6f}, {c['y'][1]:.6f}]",
            f"Z: [{c['z'][0]:.6f}, {c['z'][1]:.6f}]",
            f"Points in range: {n:,} / {N:,}",
            "======================",
        ]
        text = "\n".join(lines)
        print(f"\n{text}\n")
        with open("crop_range.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("Saved to crop_range.txt")

    def _on_reset(_=None):
        for axis, lo, hi in [('x',XLO,XHI), ('y',YLO,YHI), ('z',ZLO,ZHI)]:
            sls[f'{axis}_lo'].set_val(lo)
            sls[f'{axis}_hi'].set_val(hi)

    for sl in sls.values():
        sl.on_changed(_on_slider)
    btn_exp.on_clicked(_on_export)
    btn_rst.on_clicked(_on_reset)

    _update()   # initial render
    plt.show()


# ═══════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PLY Point Cloud Crop Viewer')
    parser.add_argument('ply', help='Path to .ply file')
    parser.add_argument('--max_display', type=int, default=80_000,
                        help='Max points shown (random subsample if exceeded, default 80000)')
    args = parser.parse_args()
    run(args.ply, args.max_display)
