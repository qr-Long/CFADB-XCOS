#!/usr/bin/env python3
"""Draw Figures 5(a,b): C IV energies and the asinh energy transformation."""
from pathlib import Path
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ENERGIES = (60, 360, 710)
COLORS = ("red", "green", "blue")


def plot_distribution(clean_data, output):
    width, height = 352.48, 279.98375
    fig = plt.figure(figsize=(width/72, height/72))
    ax = fig.add_axes([46.95/width, 37.54375/height, 298.33/width, 235.24/height])
    styles = [("atomdb313", "AtomDB v3.1.3", "o"),
              ("chianti1102", "CHIANTI v11.0.2", "s"),
              ("spex30801", "SPEX v3.08.01", "^")]
    with h5py.File(clean_data/"030604.h5", "r") as f:
        for i, (db, label, marker) in enumerate(styles):
            data = f[f"lev_{db}"][:]
            ax.scatter(data["idx"], data["eV"], s=24, marker=marker,
                       facecolors="none", edgecolors=f"C{i}", linewidths=1, label=label)
    ax.set_xlabel("index")
    ax.set_ylabel("Energy (eV)")
    # The original figure used the default 5% margins of these data.
    ax.margins(0.05)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    for energy, color in zip(ENERGIES, COLORS):
        ax.axhline(energy, color=color, linestyle="--", linewidth=1.2)
        ax.text(720, energy+12.3375, f"{energy} eV", color=color, fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, fancybox=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)


def plot_transform(output):
    width, height = 348.349206, 279.820625
    fig = plt.figure(figsize=(width/72, height/72))
    ax = fig.add_axes([45.10625/width, 37.54375/height, 288.916621/width, 231.28/height])
    x = np.linspace(0, 14.5, 500)
    ax.plot(x, np.arcsinh(x), color="C0", lw=1.8)
    ax.plot(x, x, color="black", ls="--", lw=1)
    ax.set(xlim=(0, 14.5), ylim=(0, 4), xlabel="x", ylabel=r"$\sinh^{-1}(x)$")
    box = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.7)
    ax.text(3.55, 3.76, "y=x", fontsize=9, bbox=box)
    for energy, color in zip(ENERGIES, COLORS):
        xx = energy/50
        yy = np.arcsinh(xx)
        ax.axvline(xx, color=color, ls="--", lw=1)
        ax.plot(xx, yy, "o", color=color, markersize=5)
        offset = -1.6 if energy == 710 else 0.18
        ax.text(xx+offset, yy+{60: 0.16, 360: 0.10, 710: 0.12}[energy], f"E={energy} eV", fontsize=9, bbox=box)
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data-dir", type=Path, default=HERE.parent/"clean_data")
    parser.add_argument("--output-dir", type=Path, default=HERE/"photos")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
        plot_distribution(args.clean_data_dir, args.output_dir/"fig05a_c_iv_energy.pdf")
        plot_transform(args.output_dir/"fig05b_energy_transform.pdf")
    print(args.output_dir/"fig05a_c_iv_energy.pdf")
    print(args.output_dir/"fig05b_energy_transform.pdf")


if __name__ == "__main__":
    main()
