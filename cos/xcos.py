"""Compute the XCOS similarity matrix for cross-database energy levels.

XCOS combines configuration-embedding cosine similarity, atomic-term and parity
consistency, and energy-feature cosine similarity using normalized block weights.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from cos.term_score import compute_term_score_matrix
from cos.energy_features import (
    build_pair_energy_context,
    build_energy_features,
    ENERGY_ASINH_DIV,
    ENERGY_W1,
    ENERGY_W2,
    ENERGY_W3,
)


# Block selection and weight helpers.


def _parse_xcos_keep(
    keep: Optional[Iterable[str]] = None,
    default=("cfg", "term", "energy"),
) -> Tuple[str, ...]:
    """Parse the selected XCOS blocks.

    Accepted forms include ``None``, comma-separated strings, iterables, and
    ``all``. The alias ``phys`` maps to the ``term`` block.
    """
    valid = {"cfg", "term", "energy"}
    alias = {"phys": "term"}

    if keep is None:
        s = os.environ.get("XCOS_KEEP", "").strip().lower()
        if not s or s == "all":
            vals = tuple(default)
        else:
            vals = tuple(x.strip() for x in s.split(",") if x.strip())

    elif isinstance(keep, str):
        s = keep.strip().lower()
        if not s or s == "all":
            vals = tuple(default)
        else:
            vals = tuple(x.strip() for x in s.split(",") if x.strip())

    else:
        vals = tuple(str(x).strip().lower() for x in keep if str(x).strip())

    vals = tuple(alias.get(x, x) for x in vals)

    bad = [x for x in vals if x not in valid]
    if bad:
        raise ValueError(f"Invalid XCOS_KEEP entries {bad}; valid={sorted(valid)}")

    # Remove duplicates while preserving order.
    out = []
    for x in vals:
        if x not in out:
            out.append(x)

    return tuple(out)


def _get_xcos_keep(default=("cfg", "term", "energy")) -> Tuple[str, ...]:
    """Return the XCOS block selection from the environment."""
    return _parse_xcos_keep(None, default=default)


def _clone_with_block_mask(data, keep=("cfg", "term", "energy")):
    """Mask inactive feature blocks in graph data objects."""
    data_m = data.clone()
    x = data.x.clone()

    if not hasattr(data, "x_slices"):
        raise AttributeError(
            "Data object has no x_slices. "
            "This helper requires graph objects created by preprocess_data()."
        )

    keep_set = set(_parse_xcos_keep(keep))

    for name, sl in data.x_slices.items():
        canonical = "term" if name == "phys" else name
        if canonical not in keep_set:
            x[:, sl] = 0.0

    data_m.x = x
    return data_m


def _env_float(name: str, default: float) -> float:
    s = os.environ.get(name, "")
    if s is None or str(s).strip() == "":
        return float(default)
    return float(s)


def _active_weights(
    keep,
    *,
    cfg_weight: float,
    term_weight: float,
    energy_weight: float,
):
    """Normalize weights across the selected XCOS blocks only.

    For example, selecting only ``cfg`` assigns it unit weight, while selecting
    ``cfg,term`` renormalizes their supplied weights over those two blocks.
    """
    keep_set = set(keep)

    wc = float(cfg_weight) if "cfg" in keep_set else 0.0
    wt = float(term_weight) if "term" in keep_set else 0.0
    we = float(energy_weight) if "energy" in keep_set else 0.0

    total = wc + wt + we
    if total <= 0:
        raise ValueError(
            f"No active xcos blocks. keep={keep}, "
            f"weights={(cfg_weight, term_weight, energy_weight)}"
        )

    return wc / total, wt / total, we / total


# Configuration-embedding cosine similarity.


def _cfg_numeric_id(col: str) -> int:
    try:
        return int(str(col).split("_", 1)[1])
    except Exception:
        return 10**9


def _cfg_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if str(c).startswith("cfg_")]
    return sorted(cols, key=_cfg_numeric_id)


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def compute_cfg_cos_matrix(
    lev_df1: pd.DataFrame,
    lev_df2: pd.DataFrame,
) -> np.ndarray:
    """Compute cosine similarity using the shared ``cfg_*`` columns."""
    cols1 = _cfg_cols(lev_df1)
    cols2 = _cfg_cols(lev_df2)

    common = [c for c in cols1 if c in set(cols2)]
    if not common:
        raise KeyError(
            "No common cfg_ columns found. "
            "Make sure load_pair_from_h5() has already run process_cfg()."
        )

    x1 = (
        lev_df1[common]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    x2 = (
        lev_df2[common]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    x1 = _l2_normalize_rows(x1)
    x2 = _l2_normalize_rows(x2)

    return (x1 @ x2.T).astype(np.float32)

def compute_energy_feature_cos_matrix(
    lev_df1: pd.DataFrame,
    lev_df2: pd.DataFrame,
) -> np.ndarray:
    """Compute cosine similarity between the encoded energy features.

    This block compares normalized energy-feature vectors rather than applying
    a direct pairwise energy-difference penalty.
    """
    if "eV" not in lev_df1.columns or "eV" not in lev_df2.columns:
        raise KeyError("Missing required energy column: eV")

    e1 = pd.to_numeric(lev_df1["eV"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    e2 = pd.to_numeric(lev_df2["eV"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    energy_ctx = build_pair_energy_context(e1, e2)

    x1, _ = build_energy_features(
        e1,
        energy_ctx=energy_ctx,
        asinh_div=ENERGY_ASINH_DIV,
        w1=ENERGY_W1,
        w2=ENERGY_W2,
        w3=ENERGY_W3,
    )

    x2, _ = build_energy_features(
        e2,
        energy_ctx=energy_ctx,
        asinh_div=ENERGY_ASINH_DIV,
        w1=ENERGY_W1,
        w2=ENERGY_W2,
        w3=ENERGY_W3,
    )

    x1 = _l2_normalize_rows(x1)
    x2 = _l2_normalize_rows(x2)

    return (x1 @ x2.T).astype(np.float32)


# XCOS similarity assembly.


def compute_xcos_parts(
    pair_data: Dict,
    *,
    keep: Optional[Iterable[str]] = None,
    cfg_weight: Optional[float] = None,
    term_weight: Optional[float] = None,
    energy_weight: Optional[float] = None,
    energy_mode: str = "none",
):
    """Compute the weighted XCOS matrix and its component matrices.

    ``C_ij`` is the configuration cosine, ``T_ij`` is the pairwise term score,
    and ``E_ij`` is the energy-feature cosine. The combined similarity is
    ``S_ij = wc*C_ij + wt*T_ij + we*E_ij``.
    """
    keep = _parse_xcos_keep(keep)

    if cfg_weight is None:
        cfg_weight = _env_float("XCOS_CFG_W", 0.47)
    if term_weight is None:
        term_weight = _env_float("XCOS_TERM_W", 0.25)
    if energy_weight is None:
        energy_weight = _env_float("XCOS_ENERGY_W", 0.28)

    wc, wt, we = _active_weights(
        keep,
        cfg_weight=cfg_weight,
        term_weight=term_weight,
        energy_weight=energy_weight,
    )

    lev_df1 = pair_data["lev_df1"]
    lev_df2 = pair_data["lev_df2"]

    n1, n2 = len(lev_df1), len(lev_df2)
    zero = np.zeros((n1, n2), dtype=np.float32)

    if wc > 0:
        cfg_cos = compute_cfg_cos_matrix(lev_df1, lev_df2)
    else:
        cfg_cos = zero

    if wt > 0:
        term_score = compute_term_score_matrix(lev_df1, lev_df2)
    else:
        term_score = zero

    if we > 0:
        energy_score = compute_energy_feature_cos_matrix(lev_df1, lev_df2)
    else:
        energy_score = zero

    sim = wc * cfg_cos + wt * term_score + we * energy_score
    sim = np.nan_to_num(sim, nan=-1e6, posinf=1.0, neginf=-1e6)

    parts = {
        "cfg_cos": cfg_cos,
        "term_score": term_score,
        "energy_cos": energy_score,
        "weights": {
            "cfg": wc,
            "term": wt,
            "energy": we,
        },
        "keep": keep,
        "energy_mode": "energy_features_cos",
    }

    return sim.astype(np.float32), parts


def compute_xcos_similarity(
    pair_data: Dict,
    keep: Optional[Iterable[str]] = None,
    device: Optional[str] = None,
):
    """Return the XCOS matrix through the interface used by viewers and scans.

    ``device`` is accepted by the interface but is not used because the
    computation is performed with NumPy arrays.
    """
    sim, parts = compute_xcos_parts(
        pair_data,
        keep=keep,
    )

    print(
        "[XCOS] mode=cfg_term "
        f"keep={parts['keep']} "
        f"weights={parts['weights']} "
        f"energy_mode={parts['energy_mode']}"
    )

    return sim, sim.shape