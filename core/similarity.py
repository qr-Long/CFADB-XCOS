"""Configuration, term/parity and energy components of the XCOS score."""
from __future__ import annotations
import os
from typing import Iterable, Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd

ENERGY_ASINH_DIV = 50.0


ENERGY_TBAND_MIN = 20.0


ENERGY_TBAND_Q = 0.05


ENERGY_W1 = 0.4


ENERGY_W2 = 0.2


ENERGY_W3 = 0.4


ENERGY_EPS = 1e-8


def build_pair_energy_context(
    e_values1,
    e_values2,
    asinh_div=ENERGY_ASINH_DIV,
    tband_min=ENERGY_TBAND_MIN,
    tband_q=ENERGY_TBAND_Q,
):
    """Build shared energy bands and an absolute scaling range for two databases."""
    e1 = np.asarray(e_values1, dtype=np.float64).reshape(-1)
    e2 = np.asarray(e_values2, dtype=np.float64).reshape(-1)

    e_all = np.concatenate([e1, e2])
    e_all = e_all[np.isfinite(e_all)]
    if len(e_all) == 0:
        raise ValueError("Cannot build the energy context because no finite eV values are available.")

    e_all = np.sort(np.unique(e_all))

    # Tband = max(20 eV, 0.05 * E95)
    e95 = float(np.quantile(e_all, 0.95))
    tband = max(float(tband_min), float(tband_q) * e95)

    # Identify gaps that separate energy bands.
    if len(e_all) == 1:
        split_idx = np.array([], dtype=int)
    else:
        gaps = np.diff(e_all)
        split_idx = np.where(gaps >= tband)[0]

    # Define each band by its observed minimum and maximum.
    starts = np.concatenate(([0], split_idx + 1))
    ends = np.concatenate((split_idx, [len(e_all) - 1]))

    segments = []
    for s, t in zip(starts, ends):
        L = float(e_all[s])
        U = float(e_all[t])
        segments.append((L, U))

    # Shared scaling range for the absolute-energy component.
    a_all = np.arcsinh(e_all / asinh_div)
    a_min = float(a_all.min())
    a_max = float(a_all.max())

    return {
        "segments": segments,   # [(L1,U1), (L2,U2), ...]
        "a_min": a_min,
        "a_max": a_max,
        "tband": tband,
    }


def build_energy_features(
    e_values,
    energy_ctx=None,
    asinh_div=ENERGY_ASINH_DIV,
    w1=ENERGY_W1,
    w2=ENERGY_W2,
    w3=ENERGY_W3,
):
    """Return energy features and their names.

    The output matrix has shape ``[N, 1 + K + 1]`` and contains normalized
    absolute energy, a one-hot energy-band encoding, and normalized position
    within the assigned band.
    """
    e = np.asarray(e_values, dtype=np.float64).reshape(-1)
    n = len(e)

    if energy_ctx is None:
        # Build a self-consistent context when no pairwise context is supplied.
        energy_ctx = build_pair_energy_context(e, e, asinh_div=asinh_div)

    segments = energy_ctx["segments"]
    a_min = float(energy_ctx["a_min"])
    a_max = float(energy_ctx["a_max"])

    n_bands = len(segments)

    # 1) Absolute energy: apply asinh and scale jointly to [0, 1].
    a_raw = np.arcsinh(e / asinh_div)
    if abs(a_max - a_min) < ENERGY_EPS:
        a = np.zeros_like(a_raw)
    else:
        a = (a_raw - a_min) / (a_max - a_min)
    a = np.clip(a, 0.0, 1.0).reshape(-1, 1)

    # 2) One-hot energy-band membership.
    right_edges = np.array([seg[1] for seg in segments], dtype=np.float64)
    band_idx = np.searchsorted(right_edges, e, side="left")
    band_idx = np.clip(band_idx, 0, n_bands - 1)

    band_onehot = np.zeros((n, n_bands), dtype=np.float64)
    band_onehot[np.arange(n), band_idx] = 1.0

    # 3) Position within the observed band range, scaled to [0, 1].
    r = np.zeros(n, dtype=np.float64)
    for k, (L, U) in enumerate(segments):
        mask = band_idx == k
        if not np.any(mask):
            continue
        if abs(U - L) < ENERGY_EPS:
            # Assign the midpoint when a band contains a single unique energy.
            r[mask] = 0.5
        else:
            r[mask] = (e[mask] - L) / (U - L)

    r = np.clip(r, 0.0, 1.0).reshape(-1, 1)

    energy_features = np.concatenate([
        w1 * a,
        w2 * band_onehot,
        w3 * r,
    ], axis=1)

    energy_feature_names = (
        ["energy_abs"]
        + [f"energy_band_{i}" for i in range(n_bands)]
        + ["energy_rel"]
    )

    return energy_features, energy_feature_names


TERM_COLS = ("2S+1", "L", "J")


PI_COL = "pi"


DEFAULT_TERM_WEIGHTS = (0.3, 0.3, 0.4)


def _numeric_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Missing required term column: {col}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)


def _parity_col(df: pd.DataFrame, col: str = PI_COL) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Missing required parity column: {col}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)


def term_kernel(
    diff,
    *,
    score0: float = 1.0,
    score1: float = 0.1,
    score2: float = 0.01,
    tol: float = 1e-8,
) -> np.ndarray:
    """Score absolute term differences using configurable values.

    Differences of 0, 1, and 2 receive ``score0``, ``score1``, and ``score2``;
    all other or non-finite differences receive zero. This helper is intended
    for L and J, while spin multiplicity uses ``spin_multiplicity_kernel``.
    """
    d = np.abs(np.asarray(diff, dtype=np.float64))
    out = np.zeros_like(d, dtype=np.float64)

    finite = np.isfinite(d)
    out[finite & np.isclose(d, 0.0, atol=tol)] = score0
    out[finite & np.isclose(d, 1.0, atol=tol)] = score1
    out[finite & np.isclose(d, 2.0, atol=tol)] = score2

    return out


def spin_multiplicity_kernel(
    diff,
    *,
    score0: float = 1.0,
    score2: float = 0.1,
    tol: float = 1e-8,
) -> np.ndarray:
    """Score spin-multiplicity differences.

    Differences of 0 and 2 receive ``score0`` and ``score2``. Other or
    non-finite differences receive zero because valid multiplicities for a
    given ion normally change in steps of two.
    """
    d = np.abs(np.asarray(diff, dtype=np.float64))
    out = np.zeros_like(d, dtype=np.float64)

    finite = np.isfinite(d)
    out[finite & np.isclose(d, 0.0, atol=tol)] = score0
    out[finite & np.isclose(d, 2.0, atol=tol)] = score2

    return out


def valid_j_mask(
    lev_df: pd.DataFrame,
    *,
    cols: Tuple[str, str, str] = TERM_COLS,
    tol: float = 1e-8,
) -> np.ndarray:
    """Return whether each level has a physically allowed J value.

    With ``S = ((2S+1) - 1) / 2``, valid levels satisfy
    ``|L-S| <= J <= L+S``. Rows with missing or non-numeric values are invalid.
    """
    s_mult = _numeric_col(lev_df, cols[0])
    l_vals = _numeric_col(lev_df, cols[1])
    j_vals = _numeric_col(lev_df, cols[2])

    s_vals = (s_mult - 1.0) / 2.0
    lower = np.abs(l_vals - s_vals)
    upper = l_vals + s_vals

    finite = np.isfinite(s_mult) & np.isfinite(l_vals) & np.isfinite(j_vals) & np.isfinite(s_vals)
    return finite & (j_vals >= lower - float(tol)) & (j_vals <= upper + float(tol))


def compute_term_score_matrix(
    lev_df1: pd.DataFrame,
    lev_df2: pd.DataFrame,
    *,
    weights: Sequence[float] = DEFAULT_TERM_WEIGHTS,
    cols: Tuple[str, str, str] = TERM_COLS,
    pi_col: str = PI_COL,
    normalize_weights: bool = True,
    enforce_j_validity: bool = True,
    invalid_score: float = 0.0,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute the pairwise term-score matrix between two databases.

    Scores are assigned as follows:
      1. Equal 2S+1, L, J, and parity: 1.0.
      2. Equal J and parity, with only 2S+1 differing by 2: 0.1.
      3. Equal J and parity, with only L differing by 1: 0.1.
      4. Equal J and parity, with 2S+1 differing by 2 and L by 1: 0.01.
      5. All other pairs: 0.0.

    The ``pi`` column is read from the cleaned level tables, where 0 denotes
    even parity, 1 denotes odd parity, and 2 denotes a limit level. ``weights`` and ``normalize_weights``
    are accepted by the public interface but are not used by this discrete score.
    """
    # These interface parameters are intentionally unused by the discrete score.
    _ = weights
    _ = normalize_weights

    s1 = _numeric_col(lev_df1, cols[0])
    l1 = _numeric_col(lev_df1, cols[1])
    j1 = _numeric_col(lev_df1, cols[2])
    p1 = _parity_col(lev_df1, pi_col)

    s2 = _numeric_col(lev_df2, cols[0])
    l2 = _numeric_col(lev_df2, cols[1])
    j2 = _numeric_col(lev_df2, cols[2])
    p2 = _parity_col(lev_df2, pi_col)

    ds = np.abs(s1[:, None] - s2[None, :])
    dl = np.abs(l1[:, None] - l2[None, :])
    dj = np.abs(j1[:, None] - j2[None, :])

    same_s = np.isfinite(ds) & np.isclose(ds, 0.0, atol=tol)
    same_l = np.isfinite(dl) & np.isclose(dl, 0.0, atol=tol)
    same_j = np.isfinite(dj) & np.isclose(dj, 0.0, atol=tol)

    spin_diff_2 = np.isfinite(ds) & np.isclose(ds, 2.0, atol=tol)
    l_diff_1 = np.isfinite(dl) & np.isclose(dl, 1.0, atol=tol)

    valid_p1 = np.isfinite(p1) & ((p1 == 0) | (p1 == 1) | (p1 == 2))
    valid_p2 = np.isfinite(p2) & ((p2 == 0) | (p2 == 1) | (p2 == 2))
    valid_parity_pair = valid_p1[:, None] & valid_p2[None, :]
    same_parity = valid_parity_pair & (p1[:, None] == p2[None, :])

    same_j_and_parity = same_j & same_parity

    score = np.zeros((len(lev_df1), len(lev_df2)), dtype=np.float64)

    full_match = same_j_and_parity & same_s & same_l
    double_small_change = same_j_and_parity & spin_diff_2 & l_diff_1
    one_small_change = same_j_and_parity & (
        (spin_diff_2 & same_l) |
        (same_s & l_diff_1)
    )

    # Assign lower scores first so that exact matches retain the highest score.
    score[double_small_change] = 0.01
    score[one_small_change] = 0.1
    score[full_match] = 1.0

    if enforce_j_validity:
        valid1 = valid_j_mask(lev_df1, cols=cols, tol=tol)
        valid2 = valid_j_mask(lev_df2, cols=cols, tol=tol)
        valid_pair = valid1[:, None] & valid2[None, :]
        score = np.where(valid_pair, score, float(invalid_score))

    score = np.where(valid_parity_pair, score, float(invalid_score))

    return score.astype(np.float32)


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
        cfg_weight = _paper_weights()[0]
    if term_weight is None:
        term_weight = _paper_weights()[1]
    if energy_weight is None:
        energy_weight = _paper_weights()[2]

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

def _paper_weights():
    from .data import paper_parameters
    return paper_parameters()["weights"]
