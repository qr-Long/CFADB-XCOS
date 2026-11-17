#!/usr/bin/env python3
"""Draw only Figure 11 from completed weight-validation results."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

HERE = Path(__file__).resolve().parent


def plot_ternary(summary, output):
    totals = pd.to_numeric(summary["n_eval"], errors="raise").unique()
    if len(totals) != 1 or totals[0] <= 0:
        raise ValueError("Every weight point must cover the same nonempty set of cases")
    full_count = int(totals[0])
    # Fix draw order so archived and newly scanned tables give the same layout.
    df = summary.sort_values(['is_default','w_cfg','w_term','w_energy'], kind='stable').copy()
    df["x"] = df["w_term"] + 0.5*df["w_energy"]
    df["y"] = math.sqrt(3)/2*df["w_energy"]
    normal = df[~df["is_default"].astype(bool)]
    default = df[df["is_default"].astype(bool)]
    bounds = [v for v in range(0, full_count, 50) if v < full_count-0.5]
    bounds += [full_count-0.5, full_count+0.5]
    cmap = plt.get_cmap("viridis", len(bounds)-1)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([0, 1, 0.5, 0], [0, 0, math.sqrt(3)/2, 0], color="black", lw=1, zorder=0.5)
    sc = ax.scatter(normal["x"], normal["y"], c=normal["n_success"], s=400,
                    marker="o", cmap=cmap, norm=norm, edgecolors="none", zorder=3)
    complete = normal[np.isclose(normal["n_success"], full_count)]
    ax.scatter(complete["x"], complete["y"], s=400, marker="o", facecolors="none",
               edgecolors="black", linewidths=0.7, zorder=4)
    ax.scatter(default["x"], default["y"], c=default["n_success"], s=700, marker="*",
               cmap=cmap, norm=norm, edgecolors="black", linewidths=0.8, zorder=4)
    for row in df.itertuples():
        r, g, b, _ = cmap(norm(row.n_success))
        color = "black" if 0.2126*r + 0.7152*g + 0.0722*b >= 0.58 else "white"
        ax.text(row.x, row.y, str(int(round(row.n_success))), ha="center", va="center",
                fontsize=6, fontweight="bold", color=color, zorder=6, clip_on=False)
    ax.text(-0.035, -0.045, r"$w_{\rm cfg}=1$", ha="right", va="top", clip_on=False)
    ax.text(1.025, -0.045, r"$w_{\rm SLJP}=1$", ha="left", va="top", clip_on=False)
    ax.text(0.5, math.sqrt(3)/2+0.035, r"$w_E=1$", ha="center", va="bottom", clip_on=False)
    ax.set_xlim(-0.14, 1.20)
    ax.set_ylim(-0.10, math.sqrt(3)/2+0.10)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ticks = [v for v in range(0, full_count, 50)] + [full_count-3, full_count]
    cb.set_ticks(sorted(set(t for t in ticks if t >= 0)))
    cb.set_label("Successful cases")
    fig.tight_layout(pad=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE/"data")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    record = json.loads((args.input_dir/"run_config.json").read_text())
    path = args.input_dir/"weight_grid_results.csv"
    if record.get("status") != "complete":
        raise RuntimeError("Validation has not completed; finish or resume the analysis before plotting")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("data_sha256", {}).get(path.name):
        raise RuntimeError("Summary data do not match the completed run_config.json")
    summary = pd.read_csv(path)
    if len(summary) != record["n_grid_specs"] or not (summary["n_eval"] == record["n_cases_requested"]).all():
        raise RuntimeError("Summary does not cover all requested cases and weight points")
    out_dir = args.out_dir if args.out_dir is not None else HERE/"output"/"figures"
    output = out_dir/"fig11_weight_sensitivity.pdf"
    with plt.rc_context():
        plt.rcdefaults()
        plot_ternary(summary, output)
    print(output)


if __name__ == "__main__":
    main()
