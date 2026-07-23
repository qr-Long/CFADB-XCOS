"""Threshold-scan utilities for cross-database atomic level assignment.

The module applies threshold-aware Hungarian matching over a sequence of
similarity thresholds, stores the selected assignments and similarity
statistics, and identifies rows and columns whose assignments change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

try:
    from thr_hung.hungarian import hungarian_binarize
except Exception:  # pragma: no cover
    from .hungarian import hungarian_binarize


def build_threshold_cube_from_sim(sim: np.ndarray, thresholds: Sequence[Union[int, float]]) -> Dict[str, Any]:
    """Run threshold-aware Hungarian matching and collect assignments and statistics."""
    thresholds_arr = np.array([float(x) for x in thresholds], dtype=float)
    thresholds_arr = np.unique(thresholds_arr)
    thresholds_arr.sort()
    n_thr = len(thresholds_arr)
    n1, n2 = sim.shape
    H = np.zeros((n_thr, n1, n2), dtype=np.uint8)
    M = np.zeros((n_thr, n1, n2), dtype=float)
    rows_list = []
    cols_list = []
    selected_stats = []
    for k, thr in enumerate(thresholds_arr):
        _, (r, c) = hungarian_binarize(sim, threshold=float(thr))
        H[k, r, c] = 1
        M[k] = H[k] * sim
        rows_list.append(np.array(r, dtype=int))
        cols_list.append(np.array(c, dtype=int))
        if len(r) > 0:
            sel = sim[r, c]
            stats = {
                "matched": int(len(r)),
                "sel_min": float(np.min(sel)),
                "sel_p10": float(np.quantile(sel, 0.1)),
                "sel_med": float(np.median(sel)),
                "sel_p90": float(np.quantile(sel, 0.9)),
            }
        else:
            stats = {"matched": 0, "sel_min": np.nan, "sel_p10": np.nan, "sel_med": np.nan, "sel_p90": np.nan}
        selected_stats.append(stats)
    return {
        "thresholds": thresholds_arr,
        "sim": sim,
        "H": H,
        "M": M,
        "match_rows": rows_list,
        "match_cols": cols_list,
        "selected_stats": selected_stats,
    }


def compute_changed_rows_cols(viewer_data: Dict[str, Any], start_thr_idx: int, end_thr_idx: int):
    """Identify assignment cells, rows, and columns that change across a threshold interval."""
    H = viewer_data["H"]
    t0 = int(min(start_thr_idx, end_thr_idx))
    t1 = int(max(start_thr_idx, end_thr_idx))
    seg = H[t0:t1 + 1]
    if len(seg) <= 1:
        changed_cells = np.zeros_like(seg[0], dtype=bool)
    else:
        changed_cells = np.any(np.diff(seg, axis=0) != 0, axis=0)
    changed_rows = changed_cells.any(axis=1)
    changed_cols = changed_cells.any(axis=0)
    return changed_cells, changed_rows, changed_cols
