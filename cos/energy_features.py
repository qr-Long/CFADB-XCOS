"""Build shared energy-feature vectors for cross-database level similarity.

The module defines common energy bands for a database pair and encodes each
level using normalized absolute energy, band membership, and within-band position.
"""

from __future__ import annotations

import numpy as np

# Energy-feature parameters used by cosine and XCOS similarity.
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
