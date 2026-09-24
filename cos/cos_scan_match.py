#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan similarity thresholds for cross-database atomic energy-level matching.

For each available database pair, the script compares thresholded Hungarian
matches with the rule-based theoretical one-to-one overlap and records the
thresholds for which all selected pairs satisfy the consistency rules.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


# SPEX v3.08.01 is used in the article; change this entry to "spex30803" to use SPEX v3.08.03 data.
TARGET_DBS = ("atomdb313", "chianti1102", "spex30801")
DEFAULT_THRESHOLDS = np.round(np.arange(0.80, 0.99, 0.01), 2)
VALID_ION_TYPES = ("h-like", "he-like", "li-like", "be-like", "ne-like")
VALID_ENERGY_RULE_MODES = ("rel_only", "abs_only", "rel_or_abs", "rel_and_abs")


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _project_root() -> Path:
    return _script_dir().parent


def _default_clean_data_dir() -> Path:
    return _project_root() / "clean_data"


def _default_thr_view_root() -> Path:
    return _project_root() / "map_data" / "thr_view"


def _default_cfg_model_path(ion_type: str) -> Path:
    return _project_root() / "map_data" / "cfg_ts" / ion_type / "cfg_ts_model.pth"


def _default_output_dir(ion_type: str) -> Path:
    return _project_root() / "cos_data" / "real_data_scan" / ion_type


def _add_import_paths() -> None:
    for p in (_script_dir(), _project_root(), _project_root() / "viewer", _project_root() / "thr_hung", _project_root() / "gnn", _project_root() / "cos") :
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _import_hungarian_module():
    for name in ("thr_hung.hungarian", "hungarian") :
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise ImportError("Could not import thr_hung.hungarian. Ensure that the thr_hung package is available from the CFADB project root.")


def _format_thr_list(thrs: Sequence[float]) -> str:
    return ", ".join(f"{float(t):.4f}" for t in thrs)


def _format_threshold_ranges_from_mask(thresholds: np.ndarray, mask: np.ndarray) -> str:
    """Compress consecutive successful thresholds, e.g. 0.9779-0.9889."""
    thresholds = np.asarray(thresholds, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return ""

    ranges: List[Tuple[int, int]] = []
    start = int(idxs[0])
    prev = int(idxs[0])
    for idx in idxs[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))

    parts: List[str] = []
    for a, b in ranges:
        if a == b:
            parts.append(f"{thresholds[a]:.4f}")
        else:
            parts.append(f"{thresholds[a]:.4f}-{thresholds[b]:.4f}")
    return ", ".join(parts)


def _available_dbs_in_h5(h5_path: Path, target_dbs: Sequence[str]) -> List[str]:
    lev_suffixes = set()
    with h5py.File(h5_path, "r") as f:
        def _visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            leaf = name.split("/")[-1]
            if leaf.startswith("lev_") and not leaf.endswith("_delete"):
                lev_suffixes.add(leaf.split("lev_", 1)[1])

        f.visititems(_visit)

    available = [db for db in target_dbs if db in lev_suffixes]
    return available


def _maximum_bipartite_matching_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        return 0

    n_left, n_right = mask.shape
    adj = [np.flatnonzero(mask[i]).astype(int).tolist() for i in range(n_left)]
    match_r = [-1] * n_right

    def _dfs(u: int, seen: List[bool]) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or _dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    matched = 0
    order = sorted(range(n_left), key=lambda u: len(adj[u]))
    for u in order:
        if not adj[u]:
            continue
        seen = [False] * n_right
        if _dfs(u, seen):
            matched += 1
    return matched


def _extract_theoretical_overlap(rules: Dict[str, Any]) -> int:
    if "theoretical_overlap" in rules:
        try:
            return int(rules["theoretical_overlap"])
        except Exception:
            pass
    all_pass = np.asarray(rules["all_pass"], dtype=bool)
    return _maximum_bipartite_matching_count(all_pass)


@contextlib.contextmanager
def _mute_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _candidate_dict(
    thresholds: np.ndarray,
    matches: np.ndarray,
    accmatches: np.ndarray,
    mismatches: np.ndarray,
    idx: Optional[int],
) -> Dict[str, Any]:
    if idx is None:
        return {"thr": None, "match": None, "accmatch": None, "mismatch": None}
    i = int(idx)
    return {
        "thr": float(thresholds[i]),
        "match": int(matches[i]),
        "accmatch": int(accmatches[i]),
        "mismatch": int(mismatches[i]),
    }


def _choose_idx_by_best_rule_quality(idxs: np.ndarray, accmatches: np.ndarray, mismatches: np.ndarray) -> int:
    """Among same-distance candidates, prefer more rule-consistent matches."""
    idxs = np.asarray(idxs, dtype=int)
    if len(idxs) == 1:
        return int(idxs[0])
    best = sorted(idxs.tolist(), key=lambda i: (-int(accmatches[i]), int(mismatches[i]), int(i)))[0]
    return int(best)


def _choose_below_candidate(
    thresholds: np.ndarray,
    matches: np.ndarray,
    accmatches: np.ndarray,
    mismatches: np.ndarray,
    theoretical: int,
) -> Dict[str, Any]:
    mask = matches < theoretical
    if not np.any(mask):
        return _candidate_dict(thresholds, matches, accmatches, mismatches, None)
    low_val = int(np.max(matches[mask]))
    idxs = np.where(mask & (matches == low_val))[0]
    chosen = _choose_idx_by_best_rule_quality(idxs, accmatches, mismatches)
    return _candidate_dict(thresholds, matches, accmatches, mismatches, chosen)


def _choose_above_candidate(
    thresholds: np.ndarray,
    matches: np.ndarray,
    accmatches: np.ndarray,
    mismatches: np.ndarray,
    theoretical: int,
) -> Dict[str, Any]:
    mask = matches > theoretical
    if not np.any(mask):
        return _candidate_dict(thresholds, matches, accmatches, mismatches, None)
    high_val = int(np.min(matches[mask]))
    idxs = np.where(mask & (matches == high_val))[0]
    chosen = _choose_idx_by_best_rule_quality(idxs, accmatches, mismatches)
    return _candidate_dict(thresholds, matches, accmatches, mismatches, chosen)


def _choose_best_failure_candidate(
    thresholds: np.ndarray,
    matches: np.ndarray,
    accmatches: np.ndarray,
    mismatches: np.ndarray,
    theoretical: int,
) -> Dict[str, Any]:
    if len(thresholds) == 0:
        return _candidate_dict(thresholds, matches, accmatches, mismatches, None)
    idxs = np.arange(len(thresholds), dtype=int)
    best = sorted(
        idxs.tolist(),
        key=lambda i: (
            abs(int(matches[i]) - int(theoretical)),
            int(mismatches[i]),
            -int(accmatches[i]),
            int(i),
        ),
    )[0]
    return _candidate_dict(thresholds, matches, accmatches, mismatches, best)


def _build_rules(
    tv,
    pair_data: Dict[str, Any],
    cfg_cos_thr: float,
    energy_rel_thr: float,
    energy_abs_thr: float,
    energy_rule_mode: str,
) -> Dict[str, Any]:
    return tv.build_rule_matrices(
        pair_data,
        check_cfg=True,
        cfg_cos_thr=float(cfg_cos_thr),
        check_2s1=True,
        check_l=True,
        check_j=True,
        check_parity=True,
        check_energy=True,
        energy_rel_thr=float(energy_rel_thr),
        energy_abs_thr=float(energy_abs_thr),
        energy_rule_mode=str(energy_rule_mode),
    )


def _compute_match_metrics_by_threshold(
    sim: np.ndarray,
    thresholds: np.ndarray,
    hungarian_module,
    rules: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return match, accmatch and mismatch counts for every threshold.

    accmatch means selected Hungarian pairs that also satisfy rules["all_pass"].
    """
    all_pass = np.asarray(rules["all_pass"], dtype=bool)
    matches: List[int] = []
    accmatches: List[int] = []
    mismatches: List[int] = []
    for thr in thresholds:
        _, (rows, cols) = hungarian_module.hungarian_binarize(sim, threshold=float(thr))
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        match = int(len(rows))
        if match == 0:
            accmatch = 0
        else:
            accmatch = int(np.sum(all_pass[rows, cols]))
        mismatch = int(match - accmatch)
        matches.append(match)
        accmatches.append(accmatch)
        mismatches.append(mismatch)
    return (
        np.asarray(matches, dtype=int),
        np.asarray(accmatches, dtype=int),
        np.asarray(mismatches, dtype=int),
    )


def _scan_one_pair(
    *,
    ion: str,
    db1: str,
    db2: str,
    seq: str,
    tv,
    hungarian_module,
    cfg_model,
    clean_data_dir: Path,
    thr_view_root: Path,
    cfg_model_path: Optional[Path],
    thresholds: np.ndarray,
    cfg_cos_thr: float,
    energy_rel_thr: float,
    energy_abs_thr: float,
    energy_rule_mode: str,
    edge_ablation: str,
    xcos_keep: Optional[str],
    struct_beta: float,
    struct_candk: int,
    struct_topk: int,
    struct_tol: float,
    struct_wav_mode: str,
    model_path: Optional[Path],
    quiet: bool,
) -> Dict[str, Any]:
    with _mute_stdout(quiet):
        pair_data = tv.load_pair_from_h5(
            ion=ion,
            db1=db1,
            db2=db2,
            clean_data_dir=clean_data_dir,
            cfg_model=cfg_model,
            seq=seq,
            cfg_model_path=cfg_model_path,
            thr_view_root=thr_view_root,
        )
        sim, _ = tv.compute_similarity_for_pair(
            pair_data,
            model_path=model_path,
            edge_ablation=edge_ablation,
            xcos_keep=xcos_keep,
            struct_beta=struct_beta,
            struct_candk=struct_candk,
            struct_topk=struct_topk,
            struct_tol=struct_tol,
            struct_wav_mode=struct_wav_mode,
            summarize=False,
        )

    pair_data["sim"] = sim
    rules = _build_rules(
        tv,
        pair_data,
        cfg_cos_thr=cfg_cos_thr,
        energy_rel_thr=energy_rel_thr,
        energy_abs_thr=energy_abs_thr,
        energy_rule_mode=energy_rule_mode,
    )
    theoretical = _extract_theoretical_overlap(rules)
    matches, accmatches, mismatches = _compute_match_metrics_by_threshold(
        sim, thresholds, hungarian_module, rules
    )

    # Success requires a nonempty theoretical overlap, complete recovery,
    # and no rule violations; an empty match is not a successful recovery.
    success_mask = (theoretical > 0) & (matches == theoretical) & (mismatches == 0)
    success_thresholds = thresholds[success_mask].tolist()
    success_threshold_ranges = _format_threshold_ranges_from_mask(thresholds, success_mask)
    status = "success" if len(success_thresholds) > 0 else "fail"

    best = _choose_best_failure_candidate(thresholds, matches, accmatches, mismatches, theoretical)
    below = _choose_below_candidate(thresholds, matches, accmatches, mismatches, theoretical)
    above = _choose_above_candidate(thresholds, matches, accmatches, mismatches, theoretical)

    return {
        "ion": ion,
        "db1": db1,
        "db2": db2,
        "status": status,
        "theoretical": int(theoretical),
        "cfg_cos_thr": float(cfg_cos_thr),
        "energy_rel_thr": float(energy_rel_thr),
        "energy_abs_thr": float(energy_abs_thr),
        "energy_rule_mode": str(energy_rule_mode),
        "success_thresholds": [float(x) for x in success_thresholds],
        "success_threshold_ranges": success_threshold_ranges,
        "best_thr": best["thr"],
        "best_match": best["match"],
        "best_accmatch": best["accmatch"],
        "best_mismatch": best["mismatch"],
        "below_thr": below["thr"],
        "below_match": below["match"],
        "below_accmatch": below["accmatch"],
        "below_mismatch": below["mismatch"],
        "above_thr": above["thr"],
        "above_match": above["match"],
        "above_accmatch": above["accmatch"],
        "above_mismatch": above["mismatch"],
        "matches": matches.tolist(),
        "accmatches": accmatches.tolist(),
        "mismatches": mismatches.tolist(),
        "thresholds": thresholds.tolist(),
    }


def _result_to_text(result: Dict[str, Any]) -> str:
    db1 = result["db1"]
    db2 = result["db2"]
    theoretical = result["theoretical"]
    if result["status"] == "success":
        thrs = result.get("success_threshold_ranges") or _format_thr_list(result["success_thresholds"])
        return f"[OK] {db1} - {db2}, success, theoretical={theoretical}, threshold={thrs};"

    pieces = [
        f"[FAIL] {db1} - {db2}, failure, theoretical={theoretical} "
        f"accmatch={result['best_accmatch']} match={result['best_match']} mismatch={result['best_mismatch']}"
    ]
    if result["below_thr"] is not None:
        pieces.append(
            f"below: threshold={result['below_thr']:.4f}, "
            f"match={result['below_match']} accmatch={result['below_accmatch']}"
        )
    else:
        pieces.append("below: None")
    if result["above_thr"] is not None:
        pieces.append(
            f"above: threshold={result['above_thr']:.4f}, "
            f"match={result['above_match']} accmatch={result['above_accmatch']}"
        )
    else:
        pieces.append("above: None")
    return ", ".join(pieces) + ";"


def run_scan(
    *,
    ion_type: str,
    ion_start: Optional[str],
    ion_end: Optional[str],
    clean_data_dir: Path,
    thr_view_root: Path,
    cfg_model_path: Optional[Path],
    thresholds: np.ndarray,
    cfg_cos_thr: float,
    energy_rel_thr: float,
    energy_abs_thr: float,
    energy_rule_mode: str,
    edge_ablation: str,
    xcos_keep: Optional[str],
    struct_beta: float,
    struct_candk: int,
    struct_topk: int,
    struct_tol: float,
    struct_wav_mode: str,
    model_path: Optional[Path],
    quiet: bool,
) -> Tuple[List[str], pd.DataFrame]:
    _add_import_paths()
    from viewer import threshold_viewer as tv
    from dataclean.seq_list import get_prefix_for_ion_type, get_seq_list

    hungarian_module = _import_hungarian_module()

    if str(edge_ablation).lower() != "xcos" and model_path is None:
        raise ValueError("--model-path is required when edge_ablation is not xcos.")

    prefix = get_prefix_for_ion_type(ion_type)
    if cfg_model_path is None:
        cfg_model_path = _default_cfg_model_path(ion_type)

    cfg_model = tv.load_cfg_model(seq=ion_type, cfg_model_path=cfg_model_path)

    seq_names = get_seq_list(
        folder_path=clean_data_dir,
        prefix=prefix,
        suffixes=".h5",
        start=ion_start,
        end=ion_end,
    )
    files = [clean_data_dir / f"{ion}.h5" for ion in seq_names]

    lines: List[str] = []
    rows: List[Dict[str, Any]] = []
    total_pairs = 0
    total_success = 0
    total_fail = 0

    range_text = "all"
    if ion_start is not None or ion_end is not None:
        range_text = f"{ion_start or '-inf'} to {ion_end or '+inf'}"
    lines.append(f"ion_type={ion_type}, prefix={prefix}, range={range_text}, files={len(files)}")
    lines.append(
        f"rules: cfg_cos_thr={float(cfg_cos_thr):.6g}, "
        f"energy_rel_thr={float(energy_rel_thr):.6g}, "
        f"energy_abs_thr={float(energy_abs_thr):.6g}, "
        f"energy_rule_mode={energy_rule_mode}"
    )
    lines.append("")

    for h5_path in files:
        ion = h5_path.stem
        available = _available_dbs_in_h5(h5_path, TARGET_DBS)
        pairs = list(itertools.combinations(available, 2))
        if not pairs:
            continue

        lines.append(f"{ion}:")
        for db1, db2 in pairs:
            total_pairs += 1
            try:
                result = _scan_one_pair(
                    ion=ion,
                    db1=db1,
                    db2=db2,
                    seq=ion_type,
                    tv=tv,
                    hungarian_module=hungarian_module,
                    cfg_model=cfg_model,
                    clean_data_dir=clean_data_dir,
                    thr_view_root=thr_view_root,
                    cfg_model_path=cfg_model_path,
                    thresholds=thresholds,
                    cfg_cos_thr=cfg_cos_thr,
                    energy_rel_thr=energy_rel_thr,
                    energy_abs_thr=energy_abs_thr,
                    energy_rule_mode=energy_rule_mode,
                    edge_ablation=edge_ablation,
                    xcos_keep=xcos_keep,
                    struct_beta=struct_beta,
                    struct_candk=struct_candk,
                    struct_topk=struct_topk,
                    struct_tol=struct_tol,
                    struct_wav_mode=struct_wav_mode,
                    model_path=model_path,
                    quiet=quiet,
                )
                lines.append(_result_to_text(result))

                row = {
                    "ion_type": ion_type,
                    "ion": ion,
                    "db1": db1,
                    "db2": db2,
                    "status": result["status"],
                    "theoretical": result["theoretical"],
                    "cfg_cos_thr": float(cfg_cos_thr),
                    "energy_rel_thr": float(energy_rel_thr),
                    "energy_abs_thr": float(energy_abs_thr),
                    "energy_rule_mode": str(energy_rule_mode),
                    "success_thresholds": "|".join(f"{t:.4f}" for t in result["success_thresholds"]),
                    "success_threshold_ranges": result["success_threshold_ranges"],
                    "best_thr": result["best_thr"],
                    "best_match": result["best_match"],
                    "best_accmatch": result["best_accmatch"],
                    "best_mismatch": result["best_mismatch"],
                    "below_thr": result["below_thr"],
                    "below_match": result["below_match"],
                    "below_accmatch": result["below_accmatch"],
                    "below_mismatch": result["below_mismatch"],
                    "above_thr": result["above_thr"],
                    "above_match": result["above_match"],
                    "above_accmatch": result["above_accmatch"],
                    "above_mismatch": result["above_mismatch"],
                }
                rows.append(row)

                if result["status"] == "success":
                    total_success += 1
                else:
                    total_fail += 1
            except Exception as e:
                total_fail += 1
                lines.append(f"[FAIL] {db1} - {db2}, error, error={type(e).__name__}: {e};")
                rows.append(
                    {
                        "ion_type": ion_type,
                        "ion": ion,
                        "db1": db1,
                        "db2": db2,
                        "status": "error",
                        "theoretical": None,
                        "cfg_cos_thr": float(cfg_cos_thr),
                        "energy_rel_thr": float(energy_rel_thr),
                        "energy_abs_thr": float(energy_abs_thr),
                        "energy_rule_mode": str(energy_rule_mode),
                        "success_thresholds": "",
                        "success_threshold_ranges": "",
                        "best_thr": None,
                        "best_match": None,
                        "best_accmatch": None,
                        "best_mismatch": None,
                        "below_thr": None,
                        "below_match": None,
                        "below_accmatch": None,
                        "below_mismatch": None,
                        "above_thr": None,
                        "above_match": None,
                        "above_accmatch": None,
                        "above_mismatch": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
        lines.append("")

    lines.append(f"Database pairs: total={total_pairs}, success={total_success}, failure={total_fail} pairs")
    df = pd.DataFrame(rows)
    return lines, df


def _save_outputs(
    lines: Sequence[str],
    df: pd.DataFrame,
    output_prefix: Path,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / output_prefix.with_suffix(".txt").name
    csv_path = output_dir / output_prefix.with_suffix(".csv").name
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return txt_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan database pairs for thresholds where match equals the theoretical overlap and mismatch is zero.")
    parser.add_argument("--ion-type", default=None, choices=VALID_ION_TYPES, help="Iso-electronic sequence. If omitted, use ion_type in main().")
    parser.add_argument("--ion-start", default=None, help="First ion identifier to scan, for example 030604. If omitted, start from the first available HDF5 file.")
    parser.add_argument("--ion-end", default=None, help="Last ion identifier to scan, for example 033028. If omitted, end at the last available HDF5 file.")
    parser.add_argument("--clean-data-dir", type=Path, default=_default_clean_data_dir())
    parser.add_argument("--thr-view-root", type=Path, default=_default_thr_view_root())
    parser.add_argument("--cfg-model-path", type=Path, default=None, help="Path to the configuration encoder. If omitted, use map_data/cfg_ts/<ion-type>/cfg_ts_model.pth.")
    parser.add_argument("--model-path", type=Path, default=None, help="Required when edge_ablation is not xcos.")
    parser.add_argument("--edge-ablation", default="xcos")
    parser.add_argument("--xcos-keep", default=None)
    parser.add_argument("--struct-beta", type=float, default=0.0)
    parser.add_argument("--struct-candk", type=int, default=100)
    parser.add_argument("--struct-topk", type=int, default=1000)
    parser.add_argument("--struct-tol", type=float, default=0.005)
    parser.add_argument("--struct-wav-mode", default="log")
    parser.add_argument("--cfg-cos-thr", type=float, default=0.76)
    parser.add_argument("--energy-rel-thr", type=float, default=0.023)
    parser.add_argument("--energy-abs-thr", type=float, default=4.4, help="Absolute energy-difference threshold in eV.")
    parser.add_argument(
        "--energy-rule-mode",
        default="rel_or_abs",
        choices=VALID_ENERGY_RULE_MODES,
        help="Energy-rule mode: rel_only, abs_only, rel_or_abs, or rel_and_abs."
    )
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-stop", type=float, default=0.991)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output-prefix", type=Path, default=_script_dir() / "threshold_theoretical_scan_report")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. If omitted, use cos_data/real_data_scan/<ion-type>.")
    parser.add_argument("--no-quiet", action="store_true", help="Show lower-level data-loading and inference messages.")
    return parser.parse_args()


def main() -> None:
    # Set the default iso-electronic sequence here.
    ion_type = "ne-like"

    # Set an optional ion range; None scans all available HDF5 files for the selected sequence.
    ion_start = None  # Example: "030604"
    ion_end = None    # Example: "033028"

    args = parse_args()
    if args.ion_type is not None:
        ion_type = str(args.ion_type)
    if args.ion_start is not None:
        ion_start = str(args.ion_start)
    if args.ion_end is not None:
        ion_end = str(args.ion_end)

    thresholds = np.round(np.arange(args.threshold_start, args.threshold_stop, args.threshold_step), 4)
    if len(thresholds) == 0:
        raise ValueError("The threshold array is empty. Check threshold-start, threshold-stop, and threshold-step.")

    cfg_model_path = Path(args.cfg_model_path) if args.cfg_model_path is not None else _default_cfg_model_path(ion_type)
    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir(ion_type)

    lines, df = run_scan(
        ion_type=ion_type,
        ion_start=ion_start,
        ion_end=ion_end,
        clean_data_dir=Path(args.clean_data_dir),
        thr_view_root=Path(args.thr_view_root),
        cfg_model_path=cfg_model_path,
        thresholds=thresholds,
        cfg_cos_thr=float(args.cfg_cos_thr),
        energy_rel_thr=float(args.energy_rel_thr),
        energy_abs_thr=float(args.energy_abs_thr),
        energy_rule_mode=str(args.energy_rule_mode),
        edge_ablation=str(args.edge_ablation),
        xcos_keep=args.xcos_keep,
        struct_beta=float(args.struct_beta),
        struct_candk=int(args.struct_candk),
        struct_topk=int(args.struct_topk),
        struct_tol=float(args.struct_tol),
        struct_wav_mode=str(args.struct_wav_mode),
        model_path=(None if args.model_path is None else Path(args.model_path)),
        quiet=(not args.no_quiet),
    )

    report_txt, report_csv = _save_outputs(lines, df, Path(args.output_prefix), output_dir=output_dir)
    print("\n".join(lines))
    print(f"\nReport saved to: {report_txt}")
    print(f"Detailed results saved to: {report_csv}")


if __name__ == "__main__":
    main()
