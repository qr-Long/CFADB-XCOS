"""Compute pairwise atomic-term similarity using term labels and parity.

The scoring matrix compares spin multiplicity, orbital angular momentum, total
angular momentum, and the parity column produced during data cleaning.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


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
