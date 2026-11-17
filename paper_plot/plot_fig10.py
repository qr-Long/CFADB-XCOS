#!/usr/bin/env python3
"""Draw Figure 10; --rescan recomputes the ranges from local data and models."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()
SEQUENCES = ("h-like", "he-like", "li-like", "be-like", "ne-like")
DBS = ("atomdb313", "chianti1102", "spex30801")
PAIRS = ((DBS[0], DBS[1]), (DBS[0], DBS[2]), (DBS[1], DBS[2]))
sys.path.insert(0, str(ROOT))
from core.data import paper_parameters
SETTINGS = paper_parameters()
RULES = {seq: tuple(rule[key] for key in
    ('cfg_cos_thr','energy_rel_thr','energy_abs_thr','energy_rule_mode'))
    for seq, rule in SETTINGS['sequence_rules'].items()}
GRID = SETTINGS['threshold_grid']
SCAN = {'weights': SETTINGS['weights'], 'threshold_start': GRID['start'],
        'threshold_stop': GRID['stop'] + GRID['step']/10,
        'threshold_step': GRID['step'], 'rules': RULES}


def fingerprints(clean_data):
    """Hash the plotted/matched fields rather than HDF5 packaging or transitions."""
    result = {}
    for path in sorted(clean_data.glob("*.h5")):
        if path.stem[:2] not in ("01", "02", "03", "04", "10") or int(path.stem[2:4]) > 30:
            continue
        with h5py.File(path, "r") as f:
            for db in DBS:
                key = "lev_"+db
                if key not in f:
                    continue
                digest = hashlib.sha256()
                for field in ("idx", "cfg", "2S+1", "L", "J", "pi", "eV"):
                    array = f[key][field]
                    digest.update(field.encode())
                    digest.update(str(array.dtype).encode())
                    digest.update(array.tobytes())
                result[f"{path.name}/{key}"] = digest.hexdigest()
    for seq in SEQUENCES:
        path = ROOT/"models"/seq/"cfg_ts_model.pth"
        result[f"models/{seq}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[f"vocabularies/{seq}"] = hashlib.sha256(path.with_name("cfg_vocab.pkl").read_bytes()).hexdigest()
    return result


def save_scan(frame, csv_path, clean_data, source):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(csv_path)
    metadata = {"source": source, "scan": SCAN, "inputs": fingerprints(clean_data),
                "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    csv_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")


def load_scan(csv_path, clean_data):
    metadata = json.loads(csv_path.with_suffix(".json").read_text())
    if metadata["inputs"] != fingerprints(clean_data):
        raise ValueError("Saved ranges do not match the current levels/models. Run with --rescan.")
    if metadata["scan"] != json.loads(json.dumps(SCAN)):
        raise ValueError("Saved scan parameters differ. Run with --rescan.")
    if metadata["csv_sha256"] != hashlib.sha256(csv_path.read_bytes()).hexdigest():
        raise ValueError("Saved range CSV has changed. Run with --rescan.")
    return pd.read_csv(csv_path, dtype={"ion": str}, keep_default_na=False)


def rescan(clean_data,csv_path):
    sys.path.insert(0,str(ROOT))
    from core.data import available_cases,load_pair_from_h5,load_cfg_model
    from core.similarity import compute_xcos_parts
    from core.matching import paper_rules,threshold_results,acceptable_ranges
    records=[]; models={}
    for seq,ion,db1,db2 in available_cases(clean_data):
        if seq not in models: models[seq]=load_cfg_model(seq)
        pair=load_pair_from_h5(ion,db1,db2,seq=seq,clean_data_dir=clean_data,cfg_model=models[seq])
        sim,_=compute_xcos_parts(pair)
        rules=paper_rules(pair,seq); theoretical=int(rules['theoretical_overlap'])
        results=threshold_results(sim,rules,np.round(np.arange(GRID['start'], GRID['stop'] + GRID['step']/2, GRID['step']),6))
        records.append(dict(sequence=seq,ion=ion,db1=db1,db2=db2,theoretical=theoretical,success_ranges=acceptable_ranges(results,theoretical)))
        print(f'{len(records)}: {ion} {db1}/{db2}',flush=True)
    save_scan(pd.DataFrame(records),csv_path,clean_data,'Recomputed from the distributed clean_data and models.')


def table_values(frame):
    values = {}
    for row in frame.itertuples():
        z = int(row.ion[2:4])
        col = 3*SEQUENCES.index(row.sequence)+PAIRS.index((row.db1, row.db2))
        key = (z-1, col)
        if key in values:
            raise ValueError(f"Duplicate scan case: {row.ion} {row.db1} {row.db2}")
        ranges = []
        for part in str(row.success_ranges).split(","):
            if not part.strip():
                continue
            ends = [float(v) for v in part.strip().split("-")]
            lo, hi = ends[0], ends[-1]
            if not 0 <= lo <= hi <= 1:
                raise ValueError(f"Invalid range: {part}")
            ranges.append((lo, hi))
        values[key] = ranges
    return values


def draw(frame, output):
    values = table_values(frame)
    width, height = 618.68375, 613.44
    fig = plt.figure(figsize=(width/72, height/72))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    left, x0, right = 10.170596, 45.646093, 512.549404
    bottom, top, header, upper = 8.737970, 561.287430, 584.310324, 605.140562
    cw, rh = (right-x0)/15, (top-bottom)/30
    cmap, norm = plt.get_cmap("coolwarm"), plt.Normalize(0, 1)

    def cell(x, y, w, h, lw=0.5, edge="0.6", face="white"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))

    def label(x, y, text, size, bold=False):
        ax.text(x, y, text, fontsize=size, weight="bold" if bold else "normal",
                ha="center", va="center", zorder=4)

    for row, element in enumerate(ELEMENTS):
        y = top-(row+1)*rh
        cell(left, y, x0-left, rh, lw=0.55)
        label((left+x0)/2, y+rh/2, element, 11, True)
        for col in range(15):
            x = x0+col*cw
            ranges = values.get((row, col), [])
            cell(x, y, cw, rh, face="none" if ranges else "white")
            if not ranges:
                label(x+cw/2, y+rh/2, "-", 10)
                continue
            for i, (lo, hi) in enumerate(ranges):
                h = rh/len(ranges)
                yy = y+rh-(i+1)*h
                gradient = np.tile(np.linspace(lo, hi, 256), (2, 1))
                ax.imshow(gradient, extent=(x, x+cw, yy, yy+h), cmap=cmap, norm=norm,
                          aspect="auto", origin="lower", interpolation="bilinear", zorder=1)
                text = f"{lo:.2f}-{hi:.2f}".replace("0.", ".") if lo != hi else f"{lo:.2f}".lstrip("0")
                label(x+cw/2, yy+h/2, text, 8.2 if len(ranges) == 1 else 6)
    cell(left, top, x0-left, upper-top, lw=0.9, edge="black")
    label((left+x0)/2, (top+upper)/2, "Ion", 12.5, True)
    for col in range(15):
        cell(x0+col*cw, top, cw, header-top, lw=0.65)
        label(x0+(col+0.5)*cw, (top+header)/2, ("A-C", "A-S", "C-S")[col % 3], 10.2, True)
    for s, seq in enumerate(SEQUENCES):
        cell(x0+3*s*cw, header, 3*cw, upper-header, lw=0.9, edge="black")
        label(x0+(3*s+1.5)*cw, (header+upper)/2, seq.capitalize(), 13.2, True)
    for x in [left] + [x0+3*s*cw for s in range(6)]:
        ax.plot([x, x], [bottom, upper], color="black", lw=0.85, zorder=3)
    for y, start in [(bottom, left), (top, left), (header, x0), (upper, left)]:
        ax.plot([start, right], [y, y], color="black", lw=0.85, zorder=3)
    cax = fig.add_axes([546.48/width, 99.36/height, 16.2/width, 414.72/height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=np.arange(0, 1.01, .25))
    cb.set_label("Threshold", fontsize=12.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data-dir", type=Path, default=ROOT/"clean_data")
    parser.add_argument("--ranges", type=Path, default=HERE/"data"/"threshold_ranges.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE/"photos")
    parser.add_argument("--rescan", action="store_true", help="Recompute all five sequences before plotting (slower).")
    args = parser.parse_args()
    if args.rescan:
        rescan(args.clean_data_dir, args.ranges)
    frame = load_scan(args.ranges, args.clean_data_dir)
    output = args.output_dir/"fig10_threshold_ranges.pdf"
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
        draw(frame, output)
    print(output)


if __name__ == "__main__":
    main()
