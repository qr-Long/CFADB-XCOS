"""Interactive tools for inspecting cross-database atomic energy-level matching.

The module loads level pairs, computes similarity matrices, applies threshold-aware
Hungarian assignment, evaluates rule consistency, and provides diagnostic plots and
widgets for reproducing and examining the matching results.
"""

from __future__ import annotations

import base64
import html
import io
import math
import sys
import threading
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from thr_hung.rules import build_rule_matrices, evaluate_selected_matches


DEFAULT_CLEAN_DATA_DIR = Path("./clean_data")
DEFAULT_THR_VIEW_ROOT = Path("./map_data/thr_view")
DEFAULT_CFG_MODEL_ROOT = Path("./map_data/cfg_ts")


# ============================================================
# Imports / basic helpers
# ============================================================


def _import_project_modules():
    try:
        from thr_hung.hungarian import hungarian_binarize  # type: ignore
    except Exception as e:
        raise ImportError(
            "Cannot import thr_hung.hungarian.hungarian_binarize. "
            "Make sure the thr_hung package is available."
        ) from e

    try:
        from cfg.cfg_ts import CfgEncoder  # type: ignore
    except Exception as e:
        raise ImportError("Cannot import cfg.cfg_ts.CfgEncoder. Make sure cfg package is available.") from e

    return hungarian_binarize, CfgEncoder



def _register_cfg_encoder_main() -> None:
    _, CfgEncoder = _import_project_modules()
    sys.modules["__main__"].CfgEncoder = CfgEncoder



def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def _as_float_thresholds(thresholds: Sequence[Union[int, float]]) -> np.ndarray:
    arr = np.array([float(x) for x in thresholds], dtype=float)
    arr = np.unique(arr)
    arr.sort()
    return arr



def _pick_existing_cols(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]



def _safe_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)




def _clean_similarity(sim: np.ndarray) -> np.ndarray:
    """Return a finite float32 similarity matrix without changing valid values."""
    sim32 = np.asarray(sim, dtype=np.float32).copy()
    finite = np.isfinite(sim32)
    if finite.any():
        floor = np.float32(np.min(sim32[finite]) - np.float32(1.0))
    else:
        floor = np.float32(-1e6)
    sim32[~finite] = floor
    return sim32



@contextmanager
def _quiet_stdio(enabled: bool = True):
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        yield


@contextmanager
def _suppress_known_noise(enabled: bool = True):
    if not enabled:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Creating a tensor from a list of numpy\.ndarrays is extremely slow.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*pd\.concat.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*merged_data.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        yield



def _threshold_value_text(thresholds: np.ndarray, idx0: int, idx1: int) -> str:
    i0, i1 = int(min(idx0, idx1)), int(max(idx0, idx1))
    return f"[{thresholds[i0]:.6f}, {thresholds[i1]:.6f}]"


def _threshold_options(thresholds: np.ndarray) -> List[Tuple[str, int]]:
    return [(f"{float(t):.6f}", int(i)) for i, t in enumerate(np.asarray(thresholds, dtype=float))]


def _overview_html(viewer_data: Dict[str, Any]) -> str:
    thresholds = np.asarray(viewer_data["thresholds"], dtype=float)
    if len(thresholds) == 0:
        thr_txt = ""
    else:
        thr_txt = f"[{thresholds.min():.6f}, {thresholds.max():.6f}]"
    return (
        f"<b>ion</b>={html.escape(str(viewer_data['ion']))} &nbsp; "
        f"<b>{html.escape(str(viewer_data['db1']))}</b> vs <b>{html.escape(str(viewer_data['db2']))}</b> &nbsp; "
        f"<b>threshold min/max</b>={thr_txt}"
    )


def _idx_labels_from_info(info_df: pd.DataFrame, rows: np.ndarray, default_prefix: str = "row") -> List[int]:
    if "idx" in info_df.columns:
        vals = pd.to_numeric(info_df.iloc[rows]["idx"], errors="coerce").to_numpy()
        out: List[int] = []
        for i, v in enumerate(vals):
            if np.isfinite(v):
                out.append(int(v))
            else:
                out.append(int(rows[i]))
        return out
    return [int(r) for r in rows]



def _current_threshold_map_df(viewer_data: Dict[str, Any], thr_idx: int) -> pd.DataFrame:
    k = int(thr_idx)
    rows, cols, _, _ = _get_or_compute_assignment(viewer_data, k)
    sim = np.asarray(viewer_data["sim"], dtype=np.float32)
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]

    if len(rows) == 0:
        return pd.DataFrame(columns=["idx1", "idx2", "sim"])

    idx1 = _idx_labels_from_info(info1, rows, "row1")
    idx2 = _idx_labels_from_info(info2, cols, "row2")

    df = pd.DataFrame(
        {
            "idx1": np.asarray(idx1, dtype=int),
            "idx2": np.asarray(idx2, dtype=int),
            "sim": sim[rows, cols],
        }
    )
    return df.sort_values(["idx1", "idx2"]).reset_index(drop=True)



def _current_threshold_sim_matrix_df(viewer_data: Dict[str, Any]) -> pd.DataFrame:
    sim = np.asarray(viewer_data["sim"], dtype=float)
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]
    rows = np.arange(sim.shape[0], dtype=int)
    cols = np.arange(sim.shape[1], dtype=int)
    row_labels = _idx_labels_from_info(info1, rows, "row1")
    col_labels = _idx_labels_from_info(info2, cols, "row2")
    df = pd.DataFrame(sim, index=row_labels, columns=col_labels)
    df.index.name = "idx1"
    return df


def _change_sim_matrix_df(viewer_data: Dict[str, Any], change_span: Tuple[int, int]) -> pd.DataFrame:
    sim = np.asarray(viewer_data["sim"], dtype=float)
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]
    _, changed_rows, changed_cols = compute_changed_rows_cols(viewer_data, int(change_span[0]), int(change_span[1]))
    rows = np.where(changed_rows)[0].astype(int)
    cols = np.where(changed_cols)[0].astype(int)
    if len(rows) == 0 or len(cols) == 0:
        return pd.DataFrame()
    row_labels = _idx_labels_from_info(info1, rows, "row1")
    col_labels = _idx_labels_from_info(info2, cols, "row2")
    df = pd.DataFrame(sim[np.ix_(rows, cols)], index=row_labels, columns=col_labels)
    df.index.name = "idx1"
    return df


def _current_threshold_map_filename(viewer_data: Dict[str, Any], thr_idx: int) -> str:
    thr = float(np.asarray(viewer_data["thresholds"], dtype=float)[int(thr_idx)])
    return f"{viewer_data['db1']}_{viewer_data['db2']}_{thr:.6f}.dat"


def _current_threshold_sim_filename(viewer_data: Dict[str, Any], thr_idx: int) -> str:
    thr = float(np.asarray(viewer_data["thresholds"], dtype=float)[int(thr_idx)])
    return f"{viewer_data['db1']}_{viewer_data['db2']}_{thr:.6f}_sim.dat"


def _change_sim_filename(viewer_data: Dict[str, Any], thr_idx: int, change_span: Tuple[int, int]) -> str:
    thr = float(np.asarray(viewer_data["thresholds"], dtype=float)[int(thr_idx)])
    thresholds = np.asarray(viewer_data["thresholds"], dtype=float)
    t0 = float(thresholds[int(min(change_span[0], change_span[1]))])
    t1 = float(thresholds[int(max(change_span[0], change_span[1]))])
    return f"{viewer_data['db1']}_{viewer_data['db2']}_{thr:.6f}_change_{t0:.6f}_{t1:.6f}_sim.dat"


def _safe_filename_part(value: Any) -> str:
    text = str(value)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "value"


def _display_db_name(value: Any) -> str:
    name = str(value)
    mapping = {
        "atomdb313": "AtomDB v3.1.3",
        "chianti1102": "CHIANTI v11.0.2",
        "spex30801": "SPEX v3.08.01",
    }
    return mapping.get(name.lower(), name)


def _current_threshold_figure_pdf_filename(
    viewer_data: Dict[str, Any],
    thr_idx: int,
    *,
    mode: str,
    changed_mode: str,
    diff_highlight: str,
    change_span: Optional[Tuple[int, int]] = None,
) -> str:
    thr = float(np.asarray(viewer_data["thresholds"], dtype=float)[int(thr_idx)])
    parts = [
        _safe_filename_part(viewer_data.get("ion", "ion")),
        _safe_filename_part(viewer_data.get("db1", "db1")),
        "vs",
        _safe_filename_part(viewer_data.get("db2", "db2")),
        _safe_filename_part(str(mode).lower()),
        f"thr_{thr:.6f}",
    ]
    changed_mode_l = str(changed_mode).lower()
    diff_l = str(diff_highlight).lower()
    if changed_mode_l and changed_mode_l != "off":
        parts.append(f"change_{_safe_filename_part(changed_mode_l)}")
    if diff_l and diff_l != "off":
        parts.append(f"diff_{_safe_filename_part(diff_l)}")
    if change_span is not None and changed_mode_l != "off":
        thresholds = np.asarray(viewer_data["thresholds"], dtype=float)
        t0 = float(thresholds[int(min(change_span[0], change_span[1]))])
        t1 = float(thresholds[int(max(change_span[0], change_span[1]))])
        parts.append(f"span_{t0:.6f}_{t1:.6f}")
    return "_".join(parts) + ".pdf"


def _figure_to_pdf_bytes(fig: plt.Figure, *, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=int(dpi), bbox_inches="tight")
    return buf.getvalue()


def _df_to_dat_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(sep="	", index=True).encode("utf-8")


def _map_dat_bytes(viewer_data: Dict[str, Any], thr_idx: int) -> bytes:
    df = _current_threshold_map_df(viewer_data, thr_idx)
    return df.to_csv(sep="	", index=False).encode("utf-8")


def _sim_dat_bytes(viewer_data: Dict[str, Any]) -> bytes:
    return _df_to_dat_bytes(_current_threshold_sim_matrix_df(viewer_data))


def _change_sim_dat_bytes(viewer_data: Dict[str, Any], change_span: Tuple[int, int]) -> bytes:
    return _df_to_dat_bytes(_change_sim_matrix_df(viewer_data, change_span))



def summarize_sim_matrix(S: np.ndarray, name: str = "S", sample_pairs: int = 200000, seed: int = 0) -> Dict[str, Any]:
    """Print and return a compact summary of a similarity matrix."""
    rng = np.random.default_rng(seed)
    S = np.asarray(S)
    n, m = S.shape

    if n * m <= sample_pairs:
        flat = S.ravel()
    else:
        ii = rng.integers(0, n, size=sample_pairs)
        jj = rng.integers(0, m, size=sample_pairs)
        flat = S[ii, jj]

    qs = [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999]
    summary = {
        "shape": S.shape,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "quantiles": {q: float(np.quantile(flat, q)) for q in qs},
    }

    row_top1 = np.max(S, axis=1)
    row_top2 = np.partition(S, -2, axis=1)[:, -2] if S.shape[1] >= 2 else row_top1
    margin = row_top1 - row_top2

    summary.update(
        {
            "row_top1_q": np.quantile(row_top1, [0.1, 0.5, 0.9]),
            "margin_q": np.quantile(margin, [0.1, 0.5, 0.9]),
            "margin_lt_001": float(np.mean(margin < 0.01)),
            "margin_lt_002": float(np.mean(margin < 0.02)),
            "margin_lt_005": float(np.mean(margin < 0.05)),
        }
    )

    print(f"[{name}] shape={S.shape}")
    print(f"[{name}] min={summary['min']:.6f}  max={summary['max']:.6f}  mean={summary['mean']:.6f}")
    print(f"[{name}] quantiles: {summary['quantiles']}")
    print(f"[{name}] row_top1  q10/q50/q90 = {summary['row_top1_q']}")
    print(f"[{name}] margin    q10/q50/q90 = {summary['margin_q']}")
    print(
        f"[{name}] margin<0.01: {summary['margin_lt_001']:.3f}  "
        f"margin<0.02: {summary['margin_lt_002']:.3f}  "
        f"margin<0.05: {summary['margin_lt_005']:.3f}"
    )
    return summary


# ============================================================
# Level info / labels
# ============================================================


def _make_level_info_df(lev_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = _pick_existing_cols(
        lev_df,
        ["idx", "cfg", "2S+1", "L", "J", "pi", "eV", "pcm", "ryd"],
    )
    info = lev_df[base_cols].copy().reset_index(drop=True)
    info.insert(0, "row", np.arange(len(info), dtype=int))
    if "idx" in info.columns:
        info["idx"] = pd.to_numeric(info["idx"], errors="coerce")
    return info



def _label_for_row(row: pd.Series, mode: str = "idx") -> str:
    idx_txt = _safe_scalar(row.get("idx", row.get("row", "")))
    cfg_txt = _safe_scalar(row.get("cfg", ""))
    s_txt = _safe_scalar(row.get("2S+1", ""))
    l_txt = _safe_scalar(row.get("L", ""))
    j_txt = _safe_scalar(row.get("J", ""))
    e_val = row.get("eV", "")
    try:
        e_txt = f"{float(e_val):.1f}" if pd.notna(e_val) else ""
    except Exception:
        e_txt = _safe_scalar(e_val)
    term = " ".join([x for x in [s_txt, l_txt, j_txt] if x]).strip()

    mode = str(mode).lower()
    if mode == "idx":
        return idx_txt
    if mode == "cfg":
        return cfg_txt if cfg_txt else idx_txt
    if mode == "idx+cfg":
        return f"{idx_txt}|{cfg_txt}" if cfg_txt else idx_txt
    if mode == "idx+term":
        parts = [idx_txt]
        if cfg_txt:
            parts.append(cfg_txt)
        if term:
            parts.append(term)
        return "|".join([p for p in parts if p])
    if mode == "full":
        parts = [idx_txt]
        if cfg_txt:
            parts.append(cfg_txt)
        if term:
            parts.append(term)
        if e_txt:
            parts.append(e_txt)
        return "|".join([p for p in parts if p])
    return idx_txt



def _format_labels(info_df: pd.DataFrame, rows: Sequence[int], mode: str = "idx") -> List[str]:
    return [_label_for_row(info_df.iloc[int(r)], mode=mode) for r in rows]



def _row_from_idx(info_df: pd.DataFrame, idx_value: Optional[int]) -> Optional[int]:
    if idx_value is None:
        return None
    if "idx" not in info_df.columns:
        return None
    idx_series = pd.to_numeric(info_df["idx"], errors="coerce")
    hit = np.where(idx_series.to_numpy() == int(idx_value))[0]
    if len(hit) == 0:
        return None
    return int(hit[0])


def _idx_bounds(info_df: pd.DataFrame) -> Tuple[int, int]:
    if "idx" in info_df.columns and len(info_df) > 0:
        vals = pd.to_numeric(info_df["idx"], errors="coerce")
        vals = vals[np.isfinite(vals.to_numpy())]
        if len(vals) > 0:
            return int(vals.min()), int(vals.max())
    n = len(info_df)
    if n == 0:
        return 0, 0
    return 0, n - 1


def _rows_from_idx_range(info_df: pd.DataFrame, idx_range: Optional[Tuple[int, int]]) -> np.ndarray:
    n = len(info_df)
    if n == 0:
        return np.array([], dtype=int)
    if idx_range is None:
        return np.arange(n, dtype=int)
    a0, a1 = int(min(idx_range[0], idx_range[1])), int(max(idx_range[0], idx_range[1]))
    if "idx" in info_df.columns:
        idx_series = pd.to_numeric(info_df["idx"], errors="coerce").to_numpy()
        mask = np.isfinite(idx_series) & (idx_series >= a0) & (idx_series <= a1)
        rows = np.where(mask)[0].astype(int)
        if len(rows) > 0:
            return rows
        return np.array([], dtype=int)
    lo = max(0, a0)
    hi = min(n - 1, a1)
    if hi < lo:
        return np.array([], dtype=int)
    return np.arange(lo, hi + 1, dtype=int)


# ============================================================
# Pair loading / similarity / Hungarian cube
# ============================================================


def load_cfg_model(seq: str = "li-like", cfg_model_path: Optional[Union[str, Path]] = None):
    from common.data_loader import load_cfg_model as _load_cfg_model
    return _load_cfg_model(seq=seq, cfg_model_path=cfg_model_path)



def load_pair_from_h5(
    ion: str,
    db1: str,
    db2: str,
    *,
    clean_data_dir: Union[str, Path] = DEFAULT_CLEAN_DATA_DIR,
    cfg_model=None,
    seq: str = "li-like",
    cfg_model_path: Optional[Union[str, Path]] = None,
    thr_view_root: Union[str, Path] = DEFAULT_THR_VIEW_ROOT,
) -> Dict[str, Any]:
    from common.data_loader import load_pair_from_h5 as _load_pair_from_h5

    return _load_pair_from_h5(
        ion=ion,
        db1=db1,
        db2=db2,
        clean_data_dir=clean_data_dir,
        cfg_model=cfg_model,
        seq=seq,
        cfg_model_path=cfg_model_path,
        thr_view_root=thr_view_root,
    )



def compute_similarity_for_pair(
    pair_data: Dict[str, Any],
    *,
    model_path: Optional[Union[str, Path]] = None,
    edge_ablation: str = "xcos",
    xcos_keep: Optional[str] = None,
    struct_beta: Union[int, float, str] = 0,
    struct_candk: Union[int, str] = 100,
    struct_topk: Union[int, str] = 1000,
    struct_tol: Union[float, str] = 0.005,
    struct_wav_mode: str = "log",
    device: Optional[str] = None,
    summarize: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute the xcos similarity matrix used by the published workflow."""
    ab = str(edge_ablation).lower()
    if ab != "xcos":
        raise ValueError(
            "This publication package supports edge_ablation='xcos' only. "
            "GNN-based ablation modes are not part of the workflow described in the article."
        )

    from cos.xcos import compute_xcos_similarity

    print(f"[TV] edge_ablation={edge_ablation}, xcos_keep={xcos_keep}")
    sim, _ = compute_xcos_similarity(
        pair_data,
        keep=xcos_keep,
        device=device,
    )

    sim_clean = _clean_similarity(sim)
    sim_summary = summarize_sim_matrix(sim_clean, name="sim_clean") if summarize else {}
    return sim_clean, sim_summary




def build_threshold_cube(
    ion: str,
    db1: str,
    db2: str,
    *,
    thresholds: Sequence[Union[int, float]],
    model_path: Optional[Union[str, Path]] = None,
    clean_data_dir: Union[str, Path] = DEFAULT_CLEAN_DATA_DIR,
    thr_view_root: Union[str, Path] = DEFAULT_THR_VIEW_ROOT,
    seq: str = "li-like",
    cfg_model_path: Optional[Union[str, Path]] = None,
    edge_ablation: str = "xcos",
    xcos_keep: Optional[str] = None,
    struct_beta: Union[int, float, str] = 0,
    struct_candk: Union[int, str] = 100,
    struct_topk: Union[int, str] = 1000,
    struct_tol: Union[float, str] = 0.005,
    struct_wav_mode: str = "log",
    device: Optional[str] = None,
    summarize: bool = True,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Load one pair and initialize lazy threshold-assignment caches.

    The historical function name is retained for API compatibility. No dense
    threshold-by-row-by-column H or M cubes are created.
    """
    with _suppress_known_noise(quiet), _quiet_stdio(quiet):
        pair_data = load_pair_from_h5(
            ion=ion,
            db1=db1,
            db2=db2,
            clean_data_dir=clean_data_dir,
            seq=seq,
            cfg_model_path=cfg_model_path,
            thr_view_root=thr_view_root,
        )

        sim, sim_summary = compute_similarity_for_pair(
            pair_data,
            model_path=model_path,
            edge_ablation=edge_ablation,
            xcos_keep=xcos_keep,
            struct_beta=struct_beta,
            struct_candk=struct_candk,
            struct_topk=struct_topk,
            struct_tol=struct_tol,
            struct_wav_mode=struct_wav_mode,
            device=device,
            summarize=summarize,
        )

    thresholds_arr = _as_float_thresholds(thresholds)
    if len(thresholds_arr) == 0:
        raise ValueError("At least one threshold is required.")

    n_thr = len(thresholds_arr)
    pair_data.update(
        {
            "model_path": None if model_path is None else Path(model_path),
            "edge_ablation": edge_ablation,
            "xcos_keep": xcos_keep,
            "struct_beta": struct_beta,
            "struct_candk": struct_candk,
            "struct_topk": struct_topk,
            "struct_tol": struct_tol,
            "struct_wav_mode": struct_wav_mode,
            "thresholds": thresholds_arr,
            "sim": sim,
            "sim_summary": sim_summary,
            "match_rows": [None] * n_thr,
            "match_cols": [None] * n_thr,
            "match_sets": [None] * n_thr,
            "selected_stats": [None] * n_thr,
            "_assignment_lock": threading.RLock(),
        }
    )
    return pair_data


def _assignment_is_cached(viewer_data: Dict[str, Any], thr_idx: int) -> bool:
    k = int(thr_idx)
    return (
        0 <= k < len(viewer_data["thresholds"])
        and viewer_data["match_rows"][k] is not None
        and viewer_data["match_cols"][k] is not None
        and viewer_data["match_sets"][k] is not None
        and viewer_data["selected_stats"][k] is not None
    )


def _selected_stats_from_assignment(
    sim: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> Dict[str, Any]:
    if len(rows) == 0:
        return {
            "matched": 0,
            "sel_min": np.nan,
            "sel_p10": np.nan,
            "sel_med": np.nan,
            "sel_p90": np.nan,
        }

    selected = np.asarray(sim[rows, cols], dtype=float)
    return {
        "matched": int(len(rows)),
        "sel_min": float(np.min(selected)),
        "sel_p10": float(np.quantile(selected, 0.1)),
        "sel_med": float(np.median(selected)),
        "sel_p90": float(np.quantile(selected, 0.9)),
    }


def _get_or_compute_assignment(
    viewer_data: Dict[str, Any],
    thr_idx: int,
) -> Tuple[np.ndarray, np.ndarray, set, Dict[str, Any]]:
    """Return one threshold assignment, computing and caching it when needed."""
    k = int(thr_idx)
    thresholds = np.asarray(viewer_data["thresholds"], dtype=float)
    if k < 0 or k >= len(thresholds):
        raise IndexError(f"thr_idx out of range: {thr_idx}")

    if _assignment_is_cached(viewer_data, k):
        return (
            viewer_data["match_rows"][k],
            viewer_data["match_cols"][k],
            viewer_data["match_sets"][k],
            viewer_data["selected_stats"][k],
        )

    lock = viewer_data.setdefault("_assignment_lock", threading.RLock())
    with lock:
        if _assignment_is_cached(viewer_data, k):
            return (
                viewer_data["match_rows"][k],
                viewer_data["match_cols"][k],
                viewer_data["match_sets"][k],
                viewer_data["selected_stats"][k],
            )

        hungarian_binarize, _ = _import_project_modules()
        _, (rows, cols) = hungarian_binarize(
            viewer_data["sim"],
            threshold=float(thresholds[k]),
            return_matrix=False,
        )
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        pair_set = {(int(r), int(c)) for r, c in zip(rows, cols)}
        stats = _selected_stats_from_assignment(viewer_data["sim"], rows, cols)

        viewer_data["match_rows"][k] = rows
        viewer_data["match_cols"][k] = cols
        viewer_data["match_sets"][k] = pair_set
        viewer_data["selected_stats"][k] = stats
        return rows, cols, pair_set, stats


def _ensure_assignments(
    viewer_data: Dict[str, Any],
    threshold_indices: Sequence[int],
    progress_callback: Optional[Any] = None,
) -> None:
    """Compute only missing assignments in the requested threshold order."""
    indices = [int(k) for k in threshold_indices]
    total = len(indices)
    for done, k in enumerate(indices, start=1):
        _get_or_compute_assignment(viewer_data, k)
        if progress_callback is not None:
            progress_callback(done, total, k)



# ============================================================
# Rule consistency / mismatch checking
# ============================================================

# Rule construction and selected-match evaluation are centralized in
# thr_hung.rules and imported above.


# ============================================================
# Queries / traces / changed regions
# ============================================================


def query_levels(
    viewer_data: Dict[str, Any],
    *,
    db_side: str = "db1",
    idx_values: Optional[Sequence[int]] = None,
    rows: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    info = viewer_data["info1"] if str(db_side).lower() == "db1" else viewer_data["info2"]
    if rows is not None:
        rows = [int(x) for x in rows]
        return info[info["row"].isin(rows)].copy().reset_index(drop=True)
    if idx_values is not None:
        if "idx" not in info.columns:
            raise ValueError("This info table does not contain an idx column.")
        idxs = [int(x) for x in idx_values]
        idx_series = pd.to_numeric(info["idx"], errors="coerce")
        return info[idx_series.isin(idxs)].copy().reset_index(drop=True)
    raise ValueError("Provide either idx_values or rows.")




def compute_changed_rows_cols(
    viewer_data: Dict[str, Any],
    start_thr_idx: int,
    end_thr_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cells, rows, and columns changed across adjacent assignments."""
    thresholds = viewer_data["thresholds"]
    t0 = int(min(start_thr_idx, end_thr_idx))
    t1 = int(max(start_thr_idx, end_thr_idx))
    if t0 < 0 or t1 >= len(thresholds):
        raise IndexError(f"Threshold span out of range: {(start_thr_idx, end_thr_idx)}")

    _ensure_assignments(viewer_data, range(t0, t1 + 1))
    n_rows, n_cols = np.asarray(viewer_data["sim"]).shape
    changed_cells = np.zeros((n_rows, n_cols), dtype=bool)

    for k in range(t0 + 1, t1 + 1):
        previous = viewer_data["match_sets"][k - 1]
        current = viewer_data["match_sets"][k]
        for row, col in previous.symmetric_difference(current):
            changed_cells[int(row), int(col)] = True

    changed_rows = changed_cells.any(axis=1)
    changed_cols = changed_cells.any(axis=0)
    return changed_cells, changed_rows, changed_cols




def select_view_window(
    viewer_data: Dict[str, Any],
    *,
    db1_range: Optional[Tuple[int, int]] = None,
    db2_range: Optional[Tuple[int, int]] = None,
    changed_mode: str = "off",
    change_span: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]

    rows = _rows_from_idx_range(info1, db1_range)
    cols = _rows_from_idx_range(info2, db2_range)

    mode = str(changed_mode).lower()
    if mode != "subset":
        return rows, cols

    if change_span is None:
        raise ValueError("changed_mode='subset' requires change_span=(start_thr_idx, end_thr_idx)")

    _, changed_rows, changed_cols = compute_changed_rows_cols(viewer_data, int(change_span[0]), int(change_span[1]))
    keep_rows = rows[changed_rows[rows]] if len(rows) > 0 else rows
    keep_cols = cols[changed_cols[cols]] if len(cols) > 0 else cols
    if len(keep_rows) == 0:
        keep_rows = rows
    if len(keep_cols) == 0:
        keep_cols = cols
    return keep_rows, keep_cols



def trace_match_by_idx(
    viewer_data: Dict[str, Any],
    *,
    db_side: str,
    idx_value: int,
    rules: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    thresholds = viewer_data["thresholds"]
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]
    sim = viewer_data["sim"]
    _ensure_assignments(viewer_data, range(len(thresholds)))

    side = str(db_side).lower()
    if side not in {"db1", "db2"}:
        raise ValueError("db_side must be 'db1' or 'db2'")

    if side == "db1":
        src_info, tgt_info = info1, info2
        src_pos = _row_from_idx(src_info, int(idx_value))
        fixed_key, other_key = "db1_idx", "db2_idx"
    else:
        src_info, tgt_info = info2, info1
        src_pos = _row_from_idx(src_info, int(idx_value))
        fixed_key, other_key = "db2_idx", "db1_idx"

    if src_pos is None:
        return pd.DataFrame()

    src_row = src_info.iloc[src_pos]
    records: List[Dict[str, Any]] = []
    prev_other = None
    prev_matched = None

    for k, thr in enumerate(thresholds):
        rows = viewer_data["match_rows"][k]
        cols = viewer_data["match_cols"][k]
        if side == "db1":
            hit = np.where(rows == src_pos)[0]
        else:
            hit = np.where(cols == src_pos)[0]

        rec: Dict[str, Any] = {
            "threshold": float(thr),
            fixed_key: int(idx_value),
            "src_cfg": src_row.get("cfg", None),
            "src_2S+1": src_row.get("2S+1", None),
            "src_L": src_row.get("L", None),
            "src_J": src_row.get("J", None),
            "src_pi": src_row.get("pi", None),
            "src_eV": src_row.get("eV", None),
        }

        if len(hit) == 0:
            rec.update({
                other_key: np.nan,
                "sim": np.nan,
                "cfg_cos": np.nan,
                "energy_rel": np.nan,
                "matched": False,
                "cfg": np.nan,
                "2S+1": np.nan,
                "L": np.nan,
                "J": np.nan,
                "parity": np.nan,
                "E": np.nan,
                "all": np.nan,
            })
        else:
            if side == "db1":
                row = src_pos
                col = int(cols[hit[0]])
            else:
                row = int(rows[hit[0]])
                col = src_pos
            tgt_row = tgt_info.iloc[col if side == "db1" else row]
            rec.update({
                other_key: tgt_row.get("idx", np.nan),
                "sim": float(sim[row, col]),
                "matched": True,
                "cfg_cos": np.nan,
                "energy_rel": np.nan,
                "cfg": np.nan,
                "2S+1": np.nan,
                "L": np.nan,
                "J": np.nan,
                "parity": np.nan,
                "E": np.nan,
                "all": np.nan,
            })
            if rules is not None:
                cfg_cos = rules["cfg_cos"][row, col]
                energy_rel = rules["energy_rel"][row, col]
                rec.update({
                    "cfg_cos": float(cfg_cos) if np.isfinite(cfg_cos) else np.nan,
                    "energy_rel": float(energy_rel) if np.isfinite(energy_rel) else np.nan,
                    "cfg": bool(rules["cfg_pass"][row, col]),
                    "2S+1": bool(rules["pass_2s1"][row, col]),
                    "L": bool(rules["pass_l"][row, col]),
                    "J": bool(rules["pass_j"][row, col]),
                    "parity": bool(rules["pass_parity"][row, col]),
                    "E": bool(rules["pass_energy"][row, col]),
                    "all": bool(rules["all_pass"][row, col]),
                })

        current_other = rec.get(other_key, np.nan)
        current_matched = bool(rec["matched"])
        if prev_matched is None:
            rec["changed"] = False
        else:
            same_other = (pd.isna(current_other) and pd.isna(prev_other)) or (current_other == prev_other)
            rec["changed"] = (current_matched != prev_matched) or (not same_other)
        prev_other = current_other
        prev_matched = current_matched
        records.append(rec)

    first_cols = ["threshold", fixed_key, other_key, "matched", "changed", "sim", "cfg_cos", "energy_rel", "cfg", "2S+1", "L", "J", "parity", "E", "all"]
    keep_cols = [c for c in first_cols if c in records[0]] + [c for c in records[0] if c not in first_cols and not c.startswith("src_")]
    return pd.DataFrame(records)[keep_cols]



# ============================================================
# Plotting
# ============================================================


def _make_tick_positions_labels(
    info_df: pd.DataFrame,
    rows: np.ndarray,
    *,
    label_mode: str,
    max_tick_labels: int,
    force_all: bool,
) -> Tuple[np.ndarray, List[str]]:
    if force_all or len(rows) <= max_tick_labels:
        pos = np.arange(len(rows))
        labels = _format_labels(info_df, rows, mode=label_mode)
        return pos, labels
    step = max(1, math.ceil(len(rows) / max_tick_labels))
    pos = np.arange(0, len(rows), step)
    labels = _format_labels(info_df, rows[::step], mode="idx")
    return pos, labels




def plot_threshold_slice(
    viewer_data: Dict[str, Any],
    *,
    thr_idx: int,
    mode: str = "overlay",
    db1_range: Optional[Tuple[int, int]] = None,
    db2_range: Optional[Tuple[int, int]] = None,
    changed_mode: str = "off",
    change_span: Optional[Tuple[int, int]] = None,
    diff_highlight: str = "off",
    rules: Optional[Dict[str, Any]] = None,
    label_mode: str = "idx",
    max_tick_labels: int = 30,
    figsize: Tuple[float, float] = (10.0, 10.0),
    show_title: bool = False,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    thresholds = viewer_data["thresholds"]
    k = int(thr_idx)
    if k < 0 or k >= len(thresholds):
        raise IndexError(f"thr_idx out of range: {thr_idx}")

    sim = viewer_data["sim"]
    selected_rows, selected_cols, _, _ = _get_or_compute_assignment(viewer_data, k)
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]
    db1 = viewer_data["db1"]
    db2 = viewer_data["db2"]

    rows, cols = select_view_window(
        viewer_data,
        db1_range=db1_range,
        db2_range=db2_range,
        changed_mode=changed_mode,
        change_span=change_span,
    )

    sim_sub = sim[np.ix_(rows, cols)]
    row_positions = {int(row): pos for pos, row in enumerate(rows)}
    col_positions = {int(col): pos for pos, col in enumerate(cols)}
    H_sub = np.zeros((len(rows), len(cols)), dtype=np.uint8)
    for global_row, global_col in zip(selected_rows, selected_cols):
        local_row = row_positions.get(int(global_row))
        local_col = col_positions.get(int(global_col))
        if local_row is not None and local_col is not None:
            H_sub[local_row, local_col] = 1

    changed_cells_full = None
    if str(changed_mode).lower() != "off" and change_span is not None:
        changed_cells_full, _, _ = compute_changed_rows_cols(
            viewer_data,
            int(change_span[0]),
            int(change_span[1]),
        )
        changed_cells_sub = changed_cells_full[np.ix_(rows, cols)]
    else:
        changed_cells_sub = np.zeros((len(rows), len(cols)), dtype=bool)

    fig, ax = plt.subplots(figsize=figsize)
    custom_colors = [
        "#3A4750",
        "#048998",
        "#3bb4c1",
        "#9CDEC9",
        "#E8FFF8",
    ]

    cmap10 = ListedColormap(custom_colors)
    n_color_bins = len(custom_colors)

    def _make_discrete_norm(arr, n_bins=10):
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None, None
        vmin = float(vals.min())
        vmax = float(vals.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        bounds = np.linspace(vmin, vmax, n_bins + 1)
        norm = BoundaryNorm(bounds, cmap10.N, clip=True)
        return norm, bounds

    mode_l = str(mode).lower()
    if mode_l == "h":
        im = ax.imshow(H_sub, aspect="auto", interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2f")
    elif mode_l == "m":
        M_sub = np.where(H_sub == 1, sim_sub, 0.0)
        m_plot = np.ma.masked_where(M_sub == 0, M_sub)
        norm, bounds = _make_discrete_norm(M_sub[M_sub != 0], n_bins=n_color_bins)
        im = ax.imshow(
            m_plot,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap10,
            norm=norm,
        )
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            boundaries=bounds,
            format="%.2f",
        )
    elif mode_l == "overlay":
        norm, bounds = _make_discrete_norm(sim_sub, n_bins=n_color_bins)
        im = ax.imshow(
            sim_sub,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap10,
            norm=norm,
        )
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            boundaries=bounds,
            format="%.2f",
        )
    else:
        raise ValueError("mode must be 'H', 'M', or 'overlay'")

    if diff_highlight == "off" or rules is None:
        rr, cc = np.where(H_sub == 1)
        if len(rr) > 0:
            ax.scatter(
                cc,
                rr,
                s=34,
                facecolors="none",
                edgecolors="#AB0000FF",
                linewidths=1.0,
            )
    else:
        all_pass_sub = rules["all_pass"][np.ix_(rows, cols)]
        selected_consistent = (H_sub == 1) & all_pass_sub
        selected_mismatch = (H_sub == 1) & (~all_pass_sub)
        consistent_unselected = (H_sub == 0) & all_pass_sub

        rr, cc = np.where(selected_consistent)
        if len(rr) > 0:
            ax.scatter(
                cc,
                rr,
                s=32,
                facecolors="none",
                edgecolors="#FF9900FF",
                linewidths=1.0,
            )
        rr, cc = np.where(selected_mismatch)
        if len(rr) > 0:
            ax.scatter(cc, rr, s=30, c="#FF0000", marker="x", linewidths=1.0)
        rr, cc = np.where(consistent_unselected)
        if len(rr) > 0 and len(rr) <= 3000:
            ax.scatter(
                cc,
                rr,
                s=28,
                c="#8B00FC",
                marker="+",
                linewidths=1.0,
                alpha=0.7,
            )

    if str(changed_mode).lower() == "highlight" and change_span is not None:
        rr, cc = np.where(changed_cells_sub)
        if len(rr) > 0:
            ax.scatter(
                cc,
                rr,
                s=40,
                facecolors="none",
                edgecolors="#0D00FCFF",
                linewidths=1.0,
            )

    thr = thresholds[k]
    if show_title:
        title = (
            f"ion={viewer_data['ion']} | {db1} vs {db2} | "
            f"thr={thr:.6f} | mode={mode_l} | view={len(rows)}x{len(cols)}"
        )
        if str(changed_mode).lower() != "off" and change_span is not None:
            title += (
                " | change="
                + _threshold_value_text(
                    thresholds,
                    int(change_span[0]),
                    int(change_span[1]),
                )
            )
        ax.set_title(title)

    ax.set_xlabel(_display_db_name(db2))
    ax.set_ylabel(_display_db_name(db1))

    force_all_ticks = str(changed_mode).lower() == "subset"
    xt, xl = _make_tick_positions_labels(
        info2,
        cols,
        label_mode=label_mode,
        max_tick_labels=max_tick_labels,
        force_all=force_all_ticks,
    )
    yt, yl = _make_tick_positions_labels(
        info1,
        rows,
        label_mode=label_mode,
        max_tick_labels=max_tick_labels,
        force_all=force_all_ticks,
    )
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=90, fontsize=8)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8)

    fig.tight_layout()
    context = {
        "rows": rows,
        "cols": cols,
        "changed_cells_sub": changed_cells_sub,
    }
    return fig, ax, context



# ============================================================
# Saving figures
# ============================================================


def save_threshold_pngs(
    viewer_data: Dict[str, Any],
    *,
    modes: Sequence[str] = ("overlay",),
    db1_range: Optional[Tuple[int, int]] = None,
    db2_range: Optional[Tuple[int, int]] = None,
    changed_mode: str = "off",
    change_span: Optional[Tuple[int, int]] = None,
    diff_highlight: str = "off",
    rules: Optional[Dict[str, Any]] = None,
    label_mode: str = "idx",
    max_tick_labels: int = 30,
    figsize: Tuple[float, float] = (10.0, 10.0),
    dpi: int = 180,
    save_dir: Optional[Union[str, Path]] = None,
) -> Path:
    thresholds = viewer_data["thresholds"]
    base_dir = Path(save_dir) if save_dir is not None else viewer_data["out_dir"]
    base_dir = _ensure_dir(base_dir)

    suffix_bits = [str(changed_mode).lower(), str(diff_highlight).lower()]
    if change_span is not None:
        suffix_bits.append(f"change_{thresholds[int(change_span[0])]:.6f}_{thresholds[int(change_span[1])]:.6f}")
    suffix = "__".join([x for x in suffix_bits if x and x != "off"])

    for mode in modes:
        mode_dir = _ensure_dir(base_dir / str(mode).lower())
        if suffix:
            mode_dir = _ensure_dir(mode_dir / suffix)
        for k, thr in enumerate(thresholds):
            fig, _, _ = plot_threshold_slice(
                viewer_data,
                thr_idx=k,
                mode=mode,
                db1_range=db1_range,
                db2_range=db2_range,
                changed_mode=changed_mode,
                change_span=change_span,
                diff_highlight=diff_highlight,
                rules=rules,
                label_mode=label_mode,
                max_tick_labels=max_tick_labels,
                figsize=figsize,
            )
            out_path = mode_dir / f"thr_{thr:.6f}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[SAVE] {out_path}")
    return base_dir


def save_threshold_pdfs(
    viewer_data: Dict[str, Any],
    *,
    modes: Sequence[str] = ("overlay",),
    db1_range: Optional[Tuple[int, int]] = None,
    db2_range: Optional[Tuple[int, int]] = None,
    changed_mode: str = "off",
    change_span: Optional[Tuple[int, int]] = None,
    diff_highlight: str = "off",
    rules: Optional[Dict[str, Any]] = None,
    label_mode: str = "idx",
    max_tick_labels: int = 30,
    figsize: Tuple[float, float] = (10.0, 10.0),
    dpi: int = 300,
    save_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Save all threshold figures as high-quality PDF files.

    Figures are saved without titles by default to match the interactive viewer.
    """
    thresholds = viewer_data["thresholds"]
    base_dir = Path(save_dir) if save_dir is not None else viewer_data["out_dir"]
    base_dir = _ensure_dir(base_dir)

    suffix_bits = [str(changed_mode).lower(), str(diff_highlight).lower()]
    if change_span is not None:
        suffix_bits.append(f"change_{thresholds[int(change_span[0])]:.6f}_{thresholds[int(change_span[1])]:.6f}")
    suffix = "__".join([x for x in suffix_bits if x and x != "off"])

    for mode in modes:
        mode_dir = _ensure_dir(base_dir / str(mode).lower())
        if suffix:
            mode_dir = _ensure_dir(mode_dir / suffix)
        for k, thr in enumerate(thresholds):
            fig, _, _ = plot_threshold_slice(
                viewer_data,
                thr_idx=k,
                mode=mode,
                db1_range=db1_range,
                db2_range=db2_range,
                changed_mode=changed_mode,
                change_span=change_span,
                diff_highlight=diff_highlight,
                rules=rules,
                label_mode=label_mode,
                max_tick_labels=max_tick_labels,
                figsize=figsize,
                show_title=False,
            )
            out_path = mode_dir / f"thr_{thr:.6f}.pdf"
            fig.savefig(out_path, format="pdf", dpi=int(dpi), bbox_inches="tight")
            plt.close(fig)
            print(f"[SAVE] {out_path}")
    return base_dir


# ============================================================
# Viewer UI helpers
# ============================================================


def _html_escape(x: Any) -> str:
    return html.escape(_safe_scalar(x))



def _fmt_num(x: Any, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:
        return _safe_scalar(x)
    if np.isnan(xf):
        return ""
    return f"{xf:.{digits}f}"



def _bool_mark(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return "✓" if bool(x) else "✗"



def _compact_metric_table(rows: Sequence[Tuple[str, Any]]) -> str:
    body = []
    for k, v in rows:
        body.append(
            "<tr>"
            f"<td style='padding:2px 8px 2px 0; white-space:nowrap;'><b>{html.escape(str(k))}</b></td>"
            f"<td style='padding:2px 0; white-space:nowrap;'>{html.escape(str(v))}</td>"
            "</tr>"
        )
    return "<table style='border-collapse:collapse; font-size:12px;'>" + "".join(body) + "</table>"




def _pair_detail_html(
    viewer_data: Dict[str, Any],
    *,
    thr_idx: int,
    db1_idx: Optional[int],
    db2_idx: Optional[int],
    rules: Optional[Dict[str, Any]],
) -> str:
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]
    db1 = viewer_data["db1"]
    db2 = viewer_data["db2"]
    sim = viewer_data["sim"]
    _, _, selected_pairs, _ = _get_or_compute_assignment(viewer_data, int(thr_idx))

    row = _row_from_idx(info1, db1_idx)
    col = _row_from_idx(info2, db2_idx)
    if db1_idx is None and db2_idx is None:
        return "No idx selected."

    def _level_block(
        title: str,
        info_df: pd.DataFrame,
        pos: Optional[int],
        idx_value: Optional[int],
    ) -> str:
        if pos is None:
            return (
                f"<b>{html.escape(title)}</b>"
                f"<br>idx={_html_escape(idx_value)} not found"
            )
        level = info_df.iloc[pos]
        rows = [
            ("idx", _safe_scalar(level.get("idx"))),
            ("cfg", _safe_scalar(level.get("cfg"))),
            (
                "term",
                " / ".join(
                    [
                        _safe_scalar(level.get("2S+1")),
                        _safe_scalar(level.get("L")),
                        _safe_scalar(level.get("J")),
                    ]
                ).strip(" /"),
            ),
            ("pi", _safe_scalar(level.get("pi"))),
            ("eV", _fmt_num(level.get("eV"), 6)),
        ]
        return f"<b>{html.escape(title)}</b>" + _compact_metric_table(rows)

    left_html = _level_block(db1, info1, row, db1_idx)
    right_html = _level_block(db2, info2, col, db2_idx)

    middle_rows: List[Tuple[str, Any]] = []
    if row is not None and col is not None:
        middle_rows.extend(
            [
                ("sim", _fmt_num(sim[row, col], 6)),
                ("selected", _bool_mark((int(row), int(col)) in selected_pairs)),
            ]
        )
        if rules is not None:
            middle_rows.extend(
                [
                    ("cfg_cos", _fmt_num(rules["cfg_cos"][row, col], 4)),
                    ("E_rel", _fmt_num(rules["energy_rel"][row, col], 4)),
                    ("cfg", _bool_mark(rules["cfg_pass"][row, col])),
                    ("2S+1", _bool_mark(rules["pass_2s1"][row, col])),
                    ("L", _bool_mark(rules["pass_l"][row, col])),
                    ("J", _bool_mark(rules["pass_j"][row, col])),
                    ("parity", _bool_mark(rules["pass_parity"][row, col])),
                    ("E", _bool_mark(rules["pass_energy"][row, col])),
                    ("all", _bool_mark(rules["all_pass"][row, col])),
                ]
            )
    else:
        middle_rows.append(("pair", "Need valid idx on both sides"))

    middle_html = "<b>Pair</b>" + _compact_metric_table(middle_rows)
    return (
        "<table style='border-collapse:collapse; font-size:12px;'>"
        f"<tr><td style='vertical-align:top; padding-right:16px;'>{left_html}</td>"
        f"<td style='vertical-align:top; padding-right:16px;'>{middle_html}</td>"
        f"<td style='vertical-align:top;'>{right_html}</td></tr>"
        "</table>"
    )




def _trace_html(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or len(df) == 0:
        return "No idx found."

    show_df = df.copy().head(max_rows)
    for col in ["threshold"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].map(lambda x: _fmt_num(x, 6))
    for col in ["sim", "cfg_cos", "energy_rel"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].map(lambda x: _fmt_num(x, 4))
    for col in ["matched", "changed", "cfg", "2S+1", "L", "J", "parity", "E", "all"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].map(_bool_mark)

    rename_map = {
        "matched": "m",
        "changed": "chg",
        "energy_rel": "E_rel",
    }
    show_df = show_df.rename(columns={k: v for k, v in rename_map.items() if k in show_df.columns})
    return (
        f"<div style='font-size:12px; color:#666; margin-bottom:4px;'>(top {min(len(df), max_rows)})</div>"
        + show_df.to_html(index=False, escape=False, border=0)
    )



def _summary_html(
    viewer_data: Dict[str, Any],
    *,
    thr_idx: int,
    rules: Optional[Dict[str, Any]],
    selected_eval: Optional[Dict[str, Any]],
    change_span: Tuple[int, int],
    change_result: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> str:
    k = int(thr_idx)
    _, _, current_set, stats = _get_or_compute_assignment(viewer_data, k)

    if k == 0:
        changed_cells_from_prev: Any = 0
    elif _assignment_is_cached(viewer_data, k - 1):
        previous_set = viewer_data["match_sets"][k - 1]
        changed_cells_from_prev = len(previous_set.symmetric_difference(current_set))
    else:
        changed_cells_from_prev = "not computed"

    if change_result is None:
        changed_cells = changed_rows = changed_cols = None
    else:
        changed_cells, changed_rows, changed_cols = change_result
    theoretical_overlap = ""
    if rules is not None:
        theoretical_overlap = rules.get("theoretical_overlap", "")

    def _fmt_energy_summary_value(x: Any, decimals: int = 5) -> str:
        try:
            xf = float(x)
        except Exception:
            return ""
        if not np.isfinite(xf):
            return ""
        factor = 10 ** int(decimals)
        # Use a fixed-width truncated display for non-negative values.
        xf = math.floor(xf * factor) / factor if xf >= 0 else math.ceil(xf * factor) / factor
        return f"{xf:.{int(decimals)}f}"

    def _selected_energy_pair_values() -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned relative and absolute energy differences for selected pairs."""
        rows = np.asarray(viewer_data["match_rows"][k], dtype=int)
        cols = np.asarray(viewer_data["match_cols"][k], dtype=int)
        if len(rows) == 0 or len(cols) == 0:
            empty = np.array([], dtype=float)
            return empty, empty

        rel_vals = np.array([], dtype=float)
        abs_vals = np.array([], dtype=float)
        if rules is not None:
            if "energy_rel" in rules:
                try:
                    rel_vals = np.asarray(rules["energy_rel"], dtype=float)[rows, cols]
                except Exception:
                    rel_vals = np.array([], dtype=float)
            if "energy_abs" in rules:
                try:
                    abs_vals = np.asarray(rules["energy_abs"], dtype=float)[rows, cols]
                except Exception:
                    abs_vals = np.array([], dtype=float)

        # Compute from eV columns when energy_abs is unavailable in the rule matrices.
        need_fallback = rel_vals.size != len(rows) or abs_vals.size != len(rows)
        if need_fallback:
            if "eV" not in viewer_data.get("lev_df1", pd.DataFrame()).columns:
                empty = np.array([], dtype=float)
                return empty, empty
            if "eV" not in viewer_data.get("lev_df2", pd.DataFrame()).columns:
                empty = np.array([], dtype=float)
                return empty, empty
            e1 = pd.to_numeric(viewer_data["lev_df1"]["eV"], errors="coerce").to_numpy(dtype=float)
            e2 = pd.to_numeric(viewer_data["lev_df2"]["eV"], errors="coerce").to_numpy(dtype=float)
            a = e1[rows]
            b = e2[cols]
            abs_vals = np.abs(a - b)
            denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
            rel_vals = abs_vals / denom

        return np.asarray(rel_vals, dtype=float), np.asarray(abs_vals, dtype=float)

    def _selected_energy_values(key: str) -> np.ndarray:
        rel_vals, abs_vals = _selected_energy_pair_values()
        if key == "energy_rel":
            vals = rel_vals
        elif key == "energy_abs":
            vals = abs_vals
        else:
            return np.array([], dtype=float)
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)] if vals.size > 0 else vals
        return np.sort(vals)

    def _selected_energy_cutoff_text(
        key: str,
        decimals: int = 5,
        *,
        corr_abs_for_rel_max: bool = False,
    ) -> str:
        vals = _selected_energy_values(key)
        if vals.size == 0:
            return "N/A"

        cutoffs = []
        for q in (0.90, 0.95, 0.98, 1.00):
            idx = int(math.ceil(q * vals.size)) - 1
            idx = max(0, min(int(vals.size) - 1, idx))
            cutoffs.append(_fmt_energy_summary_value(vals[idx], decimals))
        text = " / ".join(cutoffs)

        if corr_abs_for_rel_max:
            rel_vals, abs_vals = _selected_energy_pair_values()
            mask = np.isfinite(rel_vals) & np.isfinite(abs_vals)
            if np.any(mask):
                rel_f = rel_vals[mask]
                abs_f = abs_vals[mask]
                max_rel_idx = int(np.argmax(rel_f))
                text += f"(corr_E_rel:{_fmt_energy_summary_value(abs_f[max_rel_idx], decimals)} eV)"
            else:
                text += "(corr_E_rel:N/A)"
        return text

    top_rows = [
        ("matched", stats["matched"]),
        ("mismatch", selected_eval["mismatch"] if selected_eval is not None else ""),
        ("acc", _fmt_num(selected_eval["acc"], 4) if selected_eval is not None else ""),
        ("theoretical", theoretical_overlap),
        ("sel(min/med/p90)", f"{_fmt_num(stats['sel_min'],4)} / {_fmt_num(stats['sel_med'],4)} / {_fmt_num(stats['sel_p90'],4)}"),
    ]
    energy_rows = [
        ("E_rel(p90/95/98/100)", _selected_energy_cutoff_text("energy_rel", 4, corr_abs_for_rel_max=True)),
        ("E_abs(eV)(p90/95/98/100)", _selected_energy_cutoff_text("energy_abs", 4)),
    ]
    if change_result is None:
        change_rows = [
            ("chg(prev)", changed_cells_from_prev),
            ("chg(span)", "not computed"),
            ("chg(cells)", "not computed"),
            (f"chg({viewer_data['db1']})", "not computed"),
            (f"chg({viewer_data['db2']})", "not computed"),
        ]
    else:
        change_rows = [
            ("chg(prev)", changed_cells_from_prev),
            ("chg(span)", _threshold_value_text(viewer_data["thresholds"], int(change_span[0]), int(change_span[1]))),
            ("chg(cells)", int(np.sum(changed_cells))),
            (f"chg({viewer_data['db1']})", int(np.sum(changed_rows))),
            (f"chg({viewer_data['db2']})", int(np.sum(changed_cols))),
        ]

    html_parts = [
        "<div style='display:flex; gap:18px; align-items:flex-start;'>",
        _compact_metric_table(top_rows),
        _compact_metric_table(change_rows),
        "</div>",
        "<div style='margin-top:4px; display:block;'>",
        _compact_metric_table(energy_rows),
        "</div>",
    ]

    if selected_eval is not None:
        fail_rows = [
            ("cfg_fail", selected_eval["cfg_fail"]),
            ("2S+1_fail", selected_eval["s1_fail"]),
            ("L_fail", selected_eval["l_fail"]),
            ("J_fail", selected_eval["j_fail"]),
            ("parity_fail", selected_eval["parity_fail"]),
            ("E_fail", selected_eval["energy_fail"]),
        ]
        html_parts.extend([
            "<div style='margin-top:6px;'>",
            _compact_metric_table(fail_rows),
            "</div>",
        ])
        if rules is not None:
            all_pass = rules["all_pass"]
            consistent_unselected = int(np.sum(all_pass)) - int(selected_eval["selected_consistent"])
            html_parts.append(
                f"<div style='margin-top:4px; font-size:12px;'>selected ok / mismatch = {selected_eval['selected_consistent']} / {selected_eval['selected_mismatch']} &nbsp;|&nbsp; unselected but rule-ok = {consistent_unselected}</div>"
            )

    return "".join(html_parts)


# ============================================================
# Notebook viewer
# ============================================================




def launch_threshold_viewer(
    viewer_data: Dict[str, Any],
    *,
    default_mode: str = "overlay",
    default_label_mode: str = "idx",
    default_threshold: Optional[float] = 0.90,
    figsize: Tuple[float, float] = (10.0, 10.0),
    max_tick_labels: int = 30,
):
    """Launch the lazy, Voila-compatible threshold viewer."""
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except Exception as e:
        raise ImportError(
            "launch_threshold_viewer requires ipywidgets and IPython.display"
        ) from e

    thresholds = np.asarray(viewer_data["thresholds"], dtype=float)
    if len(thresholds) == 0:
        raise ValueError("The viewer requires at least one threshold.")

    db1 = viewer_data["db1"]
    db2 = viewer_data["db2"]
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]

    def _local_idx_bounds(info_df):
        if "idx" not in info_df.columns or len(info_df) == 0:
            return 0, 0
        idx_vals = (
            pd.to_numeric(info_df["idx"], errors="coerce")
            .dropna()
            .astype(int)
            .to_numpy()
        )
        if len(idx_vals) == 0:
            return 0, 0
        return int(idx_vals.min()), int(idx_vals.max())

    def _default_idx_window(info_df, n_default=80):
        if "idx" not in info_df.columns or len(info_df) == 0:
            return (0, 0)
        idx_vals = (
            pd.to_numeric(info_df["idx"], errors="coerce")
            .dropna()
            .astype(int)
            .to_numpy()
        )
        if len(idx_vals) == 0:
            return (0, 0)
        idx_vals = idx_vals[: min(n_default, len(idx_vals))]
        return int(idx_vals.min()), int(idx_vals.max())

    db1_idx_min, db1_idx_max = _local_idx_bounds(info1)
    db2_idx_min, db2_idx_max = _local_idx_bounds(info2)
    db1_idx_default = _default_idx_window(info1, n_default=80)
    db2_idx_default = _default_idx_window(info2, n_default=80)

    if default_threshold is None:
        default_thr_idx = 0
    else:
        default_thr_idx = int(np.argmin(np.abs(thresholds - float(default_threshold))))

    thr_options = _threshold_options(thresholds)
    thr_slider = widgets.SelectionSlider(
        options=thr_options,
        value=default_thr_idx,
        description="thr",
        continuous_update=False,
        layout=widgets.Layout(width="420px"),
    )

    play_first_btn = widgets.Button(description="|<", tooltip="first threshold", layout=widgets.Layout(width="44px"))
    play_prev_btn = widgets.Button(description="<", tooltip="previous threshold", layout=widgets.Layout(width="44px"))
    play_toggle_btn = widgets.ToggleButton(description="▶", tooltip="play / pause", layout=widgets.Layout(width="56px"))
    play_next_btn = widgets.Button(description=">", tooltip="next threshold", layout=widgets.Layout(width="44px"))
    play_last_btn = widgets.Button(description=">|", tooltip="last threshold", layout=widgets.Layout(width="44px"))
    play_controls = widgets.HBox(
        [play_first_btn, play_prev_btn, play_toggle_btn, play_next_btn, play_last_btn],
        layout=widgets.Layout(align_items="center", gap="10px", width="260px"),
    )
    play_state: Dict[str, Any] = {"running": False, "thread": None}

    mode_dd = widgets.Dropdown(
        options=[("H (0/1 selected)", "H"), ("M (selected sim)", "M"), ("S + H overlay", "overlay")],
        value=default_mode,
        description="plot",
        layout=widgets.Layout(width="210px"),
    )
    label_dd = widgets.Dropdown(
        options=[("idx", "idx"), ("cfg", "cfg"), ("idx+cfg", "idx+cfg"), ("idx+term", "idx+term"), ("full", "full")],
        value=default_label_mode,
        description="labels",
        layout=widgets.Layout(width="170px"),
    )
    changed_mode_dd = widgets.Dropdown(
        options=[("off", "off"), ("subset changed rows/cols", "subset"), ("highlight changed cells", "highlight")],
        value="off",
        description="change",
        layout=widgets.Layout(width="190px"),
    )
    diff_dd = widgets.Dropdown(
        options=[("off", "off"), ("selected vs rules", "rules")],
        value="off",
        description="diff",
        layout=widgets.Layout(width="160px"),
    )

    db1_range = widgets.IntRangeSlider(
        value=db1_idx_default,
        min=db1_idx_min,
        max=db1_idx_max,
        step=1,
        description=f"{db1} idx",
        continuous_update=False,
        layout=widgets.Layout(width="580px"),
    )
    db2_range = widgets.IntRangeSlider(
        value=db2_idx_default,
        min=db2_idx_min,
        max=db2_idx_max,
        step=1,
        description=f"{db2} idx",
        continuous_update=False,
        layout=widgets.Layout(width="580px"),
    )
    change_range = widgets.SelectionRangeSlider(
        options=thr_options,
        value=(0, len(thresholds) - 1),
        description="span",
        continuous_update=False,
        layout=widgets.Layout(width="360px"),
    )

    cb_style = {"description_width": "initial"}
    check_cfg = widgets.Checkbox(value=True, description="cfg", style=cb_style)
    check_2s1 = widgets.Checkbox(value=True, description="2S+1", style=cb_style)
    check_l = widgets.Checkbox(value=True, description="L", style=cb_style)
    check_j = widgets.Checkbox(value=True, description="J", style=cb_style)
    check_parity = widgets.Checkbox(value=True, description="parity", style=cb_style)
    check_energy = widgets.Checkbox(value=True, description="energy", style=cb_style)
    cfg_cos_thr = widgets.FloatText(value=0.8, description="cfg cos", layout=widgets.Layout(width="150px"))
    energy_rel_thr = widgets.FloatText(value=0.05, description="E rel", layout=widgets.Layout(width="150px"))
    energy_abs_thr = widgets.FloatText(value=3.0, description="E abs/eV", layout=widgets.Layout(width="150px"))
    energy_rule_mode = widgets.Dropdown(
        options=[
            ("rel only", "rel_only"),
            ("abs only", "abs_only"),
            ("rel OR abs", "rel_or_abs"),
            ("rel AND abs", "rel_and_abs"),
        ],
        value="rel_or_abs",
        description="E mode",
        layout=widgets.Layout(width="175px"),
    )
    fig_w = widgets.FloatSlider(
        value=min(float(figsize[0]), 10.0),
        min=3.0,
        max=10.0,
        step=0.2,
        description="fig W",
        readout_format=".1f",
        continuous_update=False,
        layout=widgets.Layout(width="220px"),
    )
    fig_h = widgets.FloatSlider(
        value=min(float(figsize[1]), 10.0),
        min=3.0,
        max=10.0,
        step=0.2,
        description="fig H",
        readout_format=".1f",
        continuous_update=False,
        layout=widgets.Layout(width="220px"),
    )

    db1_idx_box = widgets.IntText(
        value=int(info1.iloc[0]["idx"]) if "idx" in info1.columns and len(info1) else 0,
        description=f"{db1} idx",
    )
    db2_idx_box = widgets.IntText(
        value=int(info2.iloc[0]["idx"]) if "idx" in info2.columns and len(info2) else 0,
        description=f"{db2} idx",
    )
    detail_btn = widgets.Button(description="Update details")
    trace_side = widgets.Dropdown(options=[(db1, "db1"), (db2, "db2")], value="db1", description="trace db")
    trace_idx_box = widgets.IntText(
        value=int(info1.iloc[0]["idx"]) if "idx" in info1.columns and len(info1) else 0,
        description="trace idx",
    )
    trace_btn = widgets.Button(description="Update trace")
    change_btn = widgets.Button(description="Update changes")
    prepare_sim_btn = widgets.Button(
        description="Prepare similarity download",
        layout=widgets.Layout(width="210px"),
    )
    prepare_change_btn = widgets.Button(
        description="Prepare change download",
        layout=widgets.Layout(width="190px"),
    )
    prepare_pdf_btn = widgets.Button(
        description="Prepare PDF",
        layout=widgets.Layout(width="110px"),
    )

    map_download_html = widgets.HTML()
    sim_download_html = widgets.HTML()
    change_download_html = widgets.HTML()
    figure_download_html = widgets.HTML()
    header_html = widgets.HTML()
    status_html = widgets.HTML()
    change_status_html = widgets.HTML()
    summary_html = widgets.HTML()
    detail_html = widgets.HTML()
    trace_html = widgets.HTML(value="Click Update trace to compute the complete threshold trace.")
    out = widgets.Output(layout=widgets.Layout(width="840px"))

    state: Dict[str, Any] = {
        "rules": None,
        "rules_key": None,
        "selected_eval": None,
        "change_result": None,
        "change_span": None,
        "busy": False,
        "rendering": False,
    }

    heavy_buttons = [
        trace_btn,
        change_btn,
        prepare_sim_btn,
        prepare_change_btn,
        prepare_pdf_btn,
    ]

    def _set_busy(busy: bool, message: str = "") -> None:
        state["busy"] = bool(busy)
        for button in heavy_buttons:
            button.disabled = bool(busy)
        play_toggle_btn.disabled = bool(busy)
        if message:
            status_html.value = f"<span style='font-size:12px;'>{html.escape(message)}</span>"
        elif not busy:
            status_html.value = ""

    def _rules_key() -> Tuple[Any, ...]:
        return (
            bool(check_cfg.value),
            float(cfg_cos_thr.value),
            bool(check_2s1.value),
            bool(check_l.value),
            bool(check_j.value),
            bool(check_parity.value),
            bool(check_energy.value),
            float(energy_rel_thr.value),
            float(energy_abs_thr.value),
            str(energy_rule_mode.value),
        )

    def _build_rules() -> Dict[str, Any]:
        key = _rules_key()
        if state.get("rules") is not None and state.get("rules_key") == key:
            return state["rules"]
        rules = build_rule_matrices(
            viewer_data,
            check_cfg=key[0],
            cfg_cos_thr=key[1],
            check_2s1=key[2],
            check_l=key[3],
            check_j=key[4],
            check_parity=key[5],
            check_energy=key[6],
            energy_rel_thr=key[7],
            energy_abs_thr=key[8],
            energy_rule_mode=key[9],
        )
        state["rules"] = rules
        state["rules_key"] = key
        return rules

    def _make_download_link(filename: str, payload: bytes, label: str, mime: str = "text/plain") -> str:
        b64 = base64.b64encode(payload).decode("ascii")
        safe_filename = html.escape(filename, quote=True)
        safe_label = html.escape(label)
        safe_mime = html.escape(mime, quote=True)
        return (
            f"<a download='{safe_filename}' target='_blank' rel='noopener' "
            f"href='data:{safe_mime};base64,{b64}' "
            "style='display:inline-block; padding:4px 9px; border:1px solid #bdbdbd; "
            "border-radius:4px; text-decoration:none; color:#222; background:#f7f7f7; "
            f"font-size:12px; line-height:18px; white-space:nowrap;'>{safe_label}</a>"
        )

    def _make_pdf_open_link(filename: str, payload: bytes, label: str = "Open PDF") -> str:
        b64 = base64.b64encode(payload).decode("ascii")
        safe_filename = html.escape(filename, quote=True)
        safe_label = html.escape(label)
        return (
            f"<a target='_blank' rel='noopener' title='Open {safe_filename}' "
            f"href='data:application/pdf;base64,{b64}' "
            "style='display:inline-block; padding:4px 9px; border:1px solid #bdbdbd; "
            "border-radius:4px; text-decoration:none; color:#222; background:#f7f7f7; "
            f"font-size:12px; line-height:18px; white-space:nowrap;'>{safe_label}</a>"
        )

    def _current_figsize() -> Tuple[float, float]:
        return (float(fig_w.value), float(fig_h.value))

    def _change_is_current() -> bool:
        return (
            state.get("change_result") is not None
            and state.get("change_span") == tuple(change_range.value)
        )

    def _active_change_settings() -> Tuple[str, Optional[Tuple[int, int]]]:
        requested = str(changed_mode_dd.value)
        if requested == "off":
            return "off", None
        if _change_is_current():
            return requested, tuple(change_range.value)
        return "off", None

    def _invalidate_downloads() -> None:
        sim_download_html.value = ""
        change_download_html.value = ""
        figure_download_html.value = ""

    def _invalidate_change_analysis() -> None:
        state["change_result"] = None
        state["change_span"] = None
        change_download_html.value = ""
        change_status_html.value = (
            "<span style='font-size:12px; color:#666;'>"
            "Click Update changes to analyze the selected span."
            "</span>"
        )

    def _refresh_map_download() -> None:
        k = int(thr_slider.value)
        map_download_html.value = _make_download_link(
            _current_threshold_map_filename(viewer_data, k),
            _map_dat_bytes(viewer_data, k),
            "Download map",
        )

    def _refresh_details(*_):
        k = int(thr_slider.value)
        detail_html.value = _pair_detail_html(
            viewer_data,
            thr_idx=k,
            db1_idx=int(db1_idx_box.value) if db1_idx_box.value is not None else None,
            db2_idx=int(db2_idx_box.value) if db2_idx_box.value is not None else None,
            rules=state.get("rules"),
        )

    def _refresh_trace(*_):
        if state["busy"]:
            return
        _set_busy(True, "Computing remaining threshold assignments for trace...")
        trace_html.value = "Computing trace..."
        try:
            def _progress(done, total, _k):
                status_html.value = (
                    f"<span style='font-size:12px;'>Computing trace assignments: {done}/{total}</span>"
                )

            _ensure_assignments(viewer_data, range(len(thresholds)), _progress)
            rules = _build_rules()
            tdf = trace_match_by_idx(
                viewer_data,
                db_side=str(trace_side.value),
                idx_value=int(trace_idx_box.value),
                rules=rules,
            )
            trace_html.value = _trace_html(tdf)
        except Exception as exc:
            trace_html.value = f"<span style='color:#b00020;'>{html.escape(str(exc))}</span>"
        finally:
            _set_busy(False)

    def _update_changes(*_):
        if state["busy"]:
            return
        span = tuple(change_range.value)
        _set_busy(True, "Computing threshold-change assignments...")
        change_status_html.value = "Computing changes..."
        try:
            def _progress(done, total, _k):
                status_html.value = (
                    f"<span style='font-size:12px;'>Computing change assignments: {done}/{total}</span>"
                )

            t0, t1 = int(min(span)), int(max(span))
            _ensure_assignments(viewer_data, range(t0, t1 + 1), _progress)
            state["change_result"] = compute_changed_rows_cols(viewer_data, t0, t1)
            state["change_span"] = span
            change_status_html.value = (
                "<span style='font-size:12px; color:#2e6b2e;'>Change analysis is ready.</span>"
            )
            change_download_html.value = ""
        except Exception as exc:
            _invalidate_change_analysis()
            change_status_html.value = f"<span style='color:#b00020;'>{html.escape(str(exc))}</span>"
        finally:
            _set_busy(False)
        _render()

    def _prepare_similarity_download(*_):
        if state["busy"]:
            return
        _set_busy(True, "Preparing the full similarity DAT file...")
        try:
            k = int(thr_slider.value)
            sim_download_html.value = _make_download_link(
                _current_threshold_sim_filename(viewer_data, k),
                _sim_dat_bytes(viewer_data),
                "Download sim",
            )
        except Exception as exc:
            sim_download_html.value = f"<span style='color:#b00020;'>{html.escape(str(exc))}</span>"
        finally:
            _set_busy(False)

    def _prepare_change_download(*_):
        if state["busy"]:
            return
        if not _change_is_current():
            change_download_html.value = (
                "<span style='color:#b00020;'>Run Update changes first.</span>"
            )
            return
        _set_busy(True, "Preparing the changed-region DAT file...")
        try:
            k = int(thr_slider.value)
            span = tuple(change_range.value)
            change_download_html.value = _make_download_link(
                _change_sim_filename(viewer_data, k, span),
                _change_sim_dat_bytes(viewer_data, span),
                "Download change sim",
            )
        except Exception as exc:
            change_download_html.value = f"<span style='color:#b00020;'>{html.escape(str(exc))}</span>"
        finally:
            _set_busy(False)

    def _prepare_pdf(*_):
        if state["busy"]:
            return
        _set_busy(True, "Preparing the current figure as PDF...")
        try:
            k = int(thr_slider.value)
            active_changed_mode, active_span = _active_change_settings()
            rules = _build_rules()
            fig, _, _ = plot_threshold_slice(
                viewer_data,
                thr_idx=k,
                mode=mode_dd.value,
                db1_range=tuple(db1_range.value),
                db2_range=tuple(db2_range.value),
                changed_mode=active_changed_mode,
                change_span=active_span,
                diff_highlight=diff_dd.value,
                rules=rules,
                label_mode=label_dd.value,
                max_tick_labels=max_tick_labels,
                figsize=_current_figsize(),
                show_title=False,
            )
            filename = _current_threshold_figure_pdf_filename(
                viewer_data,
                k,
                mode=mode_dd.value,
                changed_mode=active_changed_mode,
                diff_highlight=diff_dd.value,
                change_span=active_span,
            )
            payload = _figure_to_pdf_bytes(fig, dpi=300)
            plt.close(fig)
            figure_download_html.value = (
                "<div style='display:flex; gap:6px; align-items:center; flex-wrap:wrap;'>"
                + _make_download_link(filename, payload, "Download PDF", mime="application/pdf")
                + _make_pdf_open_link(filename, payload, "Open PDF")
                + "</div>"
            )
        except Exception as exc:
            figure_download_html.value = f"<span style='color:#b00020;'>{html.escape(str(exc))}</span>"
        finally:
            _set_busy(False)

    def _render(*_):
        if state["rendering"]:
            return
        state["rendering"] = True
        try:
            k = int(thr_slider.value)
            _get_or_compute_assignment(viewer_data, k)
            rules = _build_rules()
            selected_eval = evaluate_selected_matches(viewer_data, rules, k)
            state["selected_eval"] = selected_eval

            active_changed_mode, active_span = _active_change_settings()
            if str(changed_mode_dd.value) != "off" and active_span is None:
                change_status_html.value = (
                    "<span style='font-size:12px; color:#8a5a00;'>"
                    "Click Update changes before using subset or highlight mode."
                    "</span>"
                )

            header_html.value = _overview_html(viewer_data)
            _refresh_map_download()
            summary_html.value = _summary_html(
                viewer_data,
                thr_idx=k,
                rules=rules,
                selected_eval=selected_eval,
                change_span=tuple(change_range.value),
                change_result=state.get("change_result") if _change_is_current() else None,
            )
            _refresh_details()

            with out:
                clear_output(wait=True)
                fig, _, _ = plot_threshold_slice(
                    viewer_data,
                    thr_idx=k,
                    mode=mode_dd.value,
                    db1_range=tuple(db1_range.value),
                    db2_range=tuple(db2_range.value),
                    changed_mode=active_changed_mode,
                    change_span=active_span,
                    diff_highlight=diff_dd.value,
                    rules=rules,
                    label_mode=label_dd.value,
                    max_tick_labels=max_tick_labels,
                    figsize=_current_figsize(),
                    show_title=False,
                )
                plt.show()
                plt.close(fig)
        except Exception as exc:
            status_html.value = f"<span style='color:#b00020;'>{html.escape(str(exc))}</span>"
        finally:
            state["rendering"] = False

    def _set_thr_index(value: int) -> None:
        thr_slider.value = int(max(0, min(len(thresholds) - 1, value)))

    def _play_first(_button):
        _set_thr_index(0)

    def _play_prev(_button):
        _set_thr_index(int(thr_slider.value) - 1)

    def _play_next(_button):
        _set_thr_index(int(thr_slider.value) + 1)

    def _play_last(_button):
        _set_thr_index(len(thresholds) - 1)

    def _stop_playback() -> None:
        play_state["running"] = False
        if bool(play_toggle_btn.value):
            play_toggle_btn.value = False
        play_toggle_btn.description = "▶"

    def _play_loop():
        import time

        while play_state["running"]:
            if state["busy"]:
                time.sleep(0.1)
                continue
            k = int(thr_slider.value)
            if k >= len(thresholds) - 1:
                _stop_playback()
                break
            _set_thr_index(k + 1)
            time.sleep(0.7)

    def _toggle_play(change):
        if bool(change["new"]):
            if state["busy"]:
                play_toggle_btn.value = False
                return
            play_toggle_btn.description = "⏸"
            play_state["running"] = True
            thread = threading.Thread(target=_play_loop, daemon=True)
            play_state["thread"] = thread
            thread.start()
        else:
            play_state["running"] = False
            play_toggle_btn.description = "▶"

    def _on_render_change(change=None):
        _invalidate_downloads()
        _render()

    def _on_change_span(change=None):
        _invalidate_change_analysis()
        figure_download_html.value = ""
        _render()

    play_first_btn.on_click(_play_first)
    play_prev_btn.on_click(_play_prev)
    play_next_btn.on_click(_play_next)
    play_last_btn.on_click(_play_last)
    play_toggle_btn.observe(_toggle_play, names="value")

    detail_btn.on_click(_refresh_details)
    trace_btn.on_click(_refresh_trace)
    change_btn.on_click(_update_changes)
    prepare_sim_btn.on_click(_prepare_similarity_download)
    prepare_change_btn.on_click(_prepare_change_download)
    prepare_pdf_btn.on_click(_prepare_pdf)

    for widget in [
        thr_slider,
        mode_dd,
        label_dd,
        changed_mode_dd,
        diff_dd,
        db1_range,
        db2_range,
        check_cfg,
        check_2s1,
        check_l,
        check_j,
        check_parity,
        check_energy,
        cfg_cos_thr,
        energy_rel_thr,
        energy_abs_thr,
        energy_rule_mode,
        fig_w,
        fig_h,
    ]:
        widget.observe(_on_render_change, names="value")
    change_range.observe(_on_change_span, names="value")

    def _group_box(title: str, children: List[Any], *, width: str = "100%"):
        return widgets.VBox(
            [
                widgets.HTML(
                    "<div style='font-weight:700; font-size:12px; margin-bottom:4px;'>"
                    + html.escape(str(title))
                    + "</div>"
                ),
                *children,
            ],
            layout=widgets.Layout(
                width=width,
                border="1px solid #cfcfcf",
                padding="5px 7px 5px 7px",
                margin="0",
            ),
        )

    compact_row = widgets.Layout(align_items="center", gap="6px")
    playback_grid = widgets.Layout(
        width="100%",
        display="grid",
        grid_template_columns="280px minmax(430px, 1fr)",
        column_gap="24px",
        align_items="center",
    )
    play_wrap = widgets.Box([play_controls], layout=widgets.Layout(width="280px"))
    thr_wrap = widgets.Box([thr_slider], layout=widgets.Layout(width="100%"))

    overview_box = _group_box(
        "Overview / Threshold",
        [
            widgets.HBox(
                [widgets.Box([header_html], layout=widgets.Layout(flex="1 1 auto")), map_download_html],
                layout=widgets.Layout(align_items="center", gap="8px", width="100%"),
            ),
            widgets.GridBox([play_wrap, thr_wrap], layout=playback_grid),
            widgets.HBox([prepare_sim_btn, sim_download_html], layout=compact_row),
            status_html,
        ],
    )
    display_box = _group_box(
        "Display",
        [
            widgets.HBox([mode_dd, label_dd, prepare_pdf_btn, figure_download_html], layout=compact_row),
            db1_range,
            db2_range,
            widgets.HBox([fig_w, fig_h], layout=compact_row),
        ],
    )
    change_box = _group_box(
        "Change",
        [
            widgets.HBox([change_range, changed_mode_dd, change_btn], layout=compact_row),
            widgets.HBox([prepare_change_btn, change_download_html], layout=compact_row),
            change_status_html,
        ],
    )
    rules_box = _group_box(
        "Rule / Diff",
        [
            widgets.HBox([check_cfg, check_2s1, check_l, check_j, check_parity, check_energy], layout=compact_row),
            widgets.HBox([cfg_cos_thr, energy_rel_thr, energy_abs_thr, energy_rule_mode, diff_dd], layout=compact_row),
        ],
    )
    figure_box = _group_box("Figure", [out])

    left_panel = widgets.VBox(
        [overview_box, display_box, change_box, rules_box, figure_box],
        layout=widgets.Layout(width="840px", min_width="840px", gap="0px"),
    )
    summary_box = _group_box("Summary", [summary_html], width="560px")
    detail_box = _group_box(
        "Pair details (by database idx)",
        [widgets.HBox([db1_idx_box, db2_idx_box, detail_btn], layout=compact_row), detail_html],
        width="560px",
    )
    trace_box = _group_box(
        "Trace",
        [widgets.HBox([trace_side, trace_idx_box, trace_btn], layout=compact_row), trace_html],
        width="560px",
    )
    right_panel = widgets.VBox(
        [summary_box, detail_box, trace_box],
        layout=widgets.Layout(width="560px", min_width="560px", gap="0px"),
    )

    ui = widgets.HBox(
        [left_panel, right_panel],
        layout=widgets.Layout(align_items="flex-start", gap="6px"),
    )
    _invalidate_change_analysis()
    display(ui)
    _render()

    return {
        "thr_slider": thr_slider,
        "play": play_toggle_btn,
        "play_controls": play_controls,
        "mode_dd": mode_dd,
        "label_dd": label_dd,
        "changed_mode_dd": changed_mode_dd,
        "diff_dd": diff_dd,
        "db1_range": db1_range,
        "db2_range": db2_range,
        "change_range": change_range,
        "change_btn": change_btn,
        "check_cfg": check_cfg,
        "check_2s1": check_2s1,
        "check_l": check_l,
        "check_j": check_j,
        "check_parity": check_parity,
        "check_energy": check_energy,
        "cfg_cos_thr": cfg_cos_thr,
        "energy_rel_thr": energy_rel_thr,
        "energy_abs_thr": energy_abs_thr,
        "energy_rule_mode": energy_rule_mode,
        "fig_w": fig_w,
        "fig_h": fig_h,
        "db1_idx_box": db1_idx_box,
        "db2_idx_box": db2_idx_box,
        "trace_side": trace_side,
        "trace_idx_box": trace_idx_box,
        "prepare_sim_btn": prepare_sim_btn,
        "prepare_change_btn": prepare_change_btn,
        "prepare_pdf_btn": prepare_pdf_btn,
        "map_download_html": map_download_html,
        "sim_download_html": sim_download_html,
        "change_download_html": change_download_html,
        "figure_download_html": figure_download_html,
        "status_html": status_html,
        "output": out,
    }



# ============================================================
# Quick entry point
# ============================================================



def quick_start_viewer(
    ion: str,
    db1: str,
    db2: str,
    *,
    thresholds: Sequence[Union[int, float]],
    model_path: Optional[Union[str, Path]] = None,
    clean_data_dir: Union[str, Path] = DEFAULT_CLEAN_DATA_DIR,
    thr_view_root: Union[str, Path] = DEFAULT_THR_VIEW_ROOT,
    seq: str = "li-like",
    cfg_model_path: Optional[Union[str, Path]] = None,
    edge_ablation: str = "xcos",
    xcos_keep: Optional[str] = None,
    struct_beta: Union[int, float, str] = 0,
    struct_candk: Union[int, str] = 100,
    struct_topk: Union[int, str] = 1000,
    struct_tol: Union[float, str] = 0.005,
    struct_wav_mode: str = "log",
    summarize: bool = True,
    launch: bool = True,
    quiet: bool = True,
    default_threshold: Optional[float] = 0.90,
) -> Dict[str, Any]:
    viewer_data = build_threshold_cube(
        ion=ion,
        db1=db1,
        db2=db2,
        thresholds=thresholds,
        model_path=model_path,
        clean_data_dir=clean_data_dir,
        thr_view_root=thr_view_root,
        seq=seq,
        cfg_model_path=cfg_model_path,
        edge_ablation=edge_ablation,
        xcos_keep=xcos_keep,
        struct_beta=struct_beta,
        struct_candk=struct_candk,
        struct_topk=struct_topk,
        struct_tol=struct_tol,
        struct_wav_mode=struct_wav_mode,
        summarize=summarize,
        quiet=quiet,
    )
    if launch:
        viewer_data["viewer_ui"] = launch_threshold_viewer(
            viewer_data,
            default_threshold=default_threshold,
        )
    return viewer_data

