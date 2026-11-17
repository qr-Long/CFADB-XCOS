#!/usr/bin/env python3
"""Draw Figure 2 from retained level tables in clean_data (no PDF input)."""
from pathlib import Path
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()
SEQUENCES = [(1, "H-like"), (2, "He-like"), (3, "Li-like"), (4, "Be-like"), (10, "Ne-like")]
DATABASES = ("atomdb313", "chianti1102", "spex30801")


def level_counts(clean_data):
    counts = np.zeros((30, 15), dtype=int)
    for z in range(1, 31):
        for s, (electrons, _) in enumerate(SEQUENCES):
            if z < electrons:
                continue
            path = clean_data / f"{electrons:02d}{z:02d}{z-electrons+1:02d}.h5"
            if not path.exists():
                continue
            with h5py.File(path, "r") as f:
                for d, db in enumerate(DATABASES):
                    name = f"lev_{db}"
                    if name in f:
                        counts[z-1, 3*s+d] = len(f[name])
    return counts


def draw(counts, output):
    # Dimensions in PDF points preserve the published table geometry.
    width, height = 614.418125, 613.44
    fig = plt.figure(figsize=(width/72, height/72))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    left, x0, right = 9.795168, 45.585020, 525.169035
    bottom, top, header, upper = 8.167524, 561.717929, 584.246143, 605.272476
    cw, rh = (right-x0)/15, (top-bottom)/30
    cmap, norm = plt.get_cmap("Blues"), LogNorm(vmin=1, vmax=max(2, int(counts.max())))

    def cell(x, y, w, h, face="white", lw=0.5, edge="0.6"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw))

    def label(x, y, text, size, bold=False, color="black"):
        ax.text(x, y, text, fontsize=size, weight="bold" if bold else "normal",
                color=color, ha="center", va="center")

    for row, element in enumerate(ELEMENTS):
        y = top-(row+1)*rh
        cell(left, y, x0-left, rh, lw=0.55)
        label((left+x0)/2, y+rh/2, element, 11, True)
        for col, n in enumerate(counts[row]):
            face = cmap(norm(n)) if n else "white"
            cell(x0+col*cw, y, cw, rh, face)
            label(x0+(col+0.5)*cw, y+rh/2, str(n), 9.8,
                  color="white" if n and norm(n) > 0.55 else "black")
    cell(left, top, x0-left, upper-top, lw=0.9, edge="black")
    label((left+x0)/2, (top+upper)/2, "Ion", 12.5, True)
    for col in range(15):
        cell(x0+col*cw, top, cw, header-top, lw=0.65)
        label(x0+(col+0.5)*cw, (top+header)/2, "ACS"[col % 3], 10.8, True)
    for s, (_, name) in enumerate(SEQUENCES):
        cell(x0+3*s*cw, header, 3*cw, upper-header, lw=0.9, edge="black")
        label(x0+(3*s+1.5)*cw, (header+upper)/2, name, 13.5, True)
    for x in [left] + [x0+3*s*cw for s in range(6)]:
        ax.plot([x, x], [bottom, upper], color="black", lw=0.85)
    for y, start in [(bottom, left), (top, left), (header, x0), (upper, left)]:
        ax.plot([start, right], [y, y], color="black", lw=0.85)
    cax = fig.add_axes([546.48/width, 99.36/height, 16.2/width, 414.72/height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Number of energy levels", fontsize=12.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data-dir", type=Path, default=HERE.parent/"clean_data")
    parser.add_argument("--output-dir", type=Path, default=HERE/"photos")
    args = parser.parse_args()
    if not args.clean_data_dir.is_dir():
        parser.error(f"Missing clean_data directory: {args.clean_data_dir}")
    counts = level_counts(args.clean_data_dir)
    if not counts.any():
        raise ValueError("No supported retained level datasets found")
    output = args.output_dir/"fig02_level_counts.pdf"
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
        draw(counts, output)
    print(output)


if __name__ == "__main__":
    main()
