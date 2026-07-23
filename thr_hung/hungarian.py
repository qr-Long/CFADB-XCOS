"""Hungarian assignment utilities for cross-database atomic energy-level matching.

The module supports full one-to-one assignment, threshold-aware partial
assignment, and visualization of the resulting binary assignment matrix.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_assignment_matrix(bin_mat, title='Hungarian Assignment (0/1)', out_path=None):
    bin_mat = np.array(bin_mat, dtype=int)
    plt.figure(figsize=(6, 5))
    # Use fixed binary limits so the assignment states remain visually distinct.
    plt.imshow(bin_mat, vmin=0, vmax=1, aspect='auto')
    plt.title(title)
    plt.colorbar(label='assignment (0/1)')
    plt.xlabel('Graph2 nodes')
    plt.ylabel('Graph1 nodes')
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
        plt.show()

import torch

def hungarian_binarize(sim_matrix, threshold=None, return_matrix=True):
    """
    Hungarian assignment with optional threshold-aware partial matching.

    Semantics
    ---------
    1) threshold is None:
       - Solve the standard global maximum-sum one-to-one assignment
         on the original (n_rows x n_cols) matrix.

    2) threshold is not None:
       - Solve a GLOBAL partial matching problem with unmatched baseline = threshold.
       - Equivalently, maximize sum(sim_ij - threshold) over selected real pairs.
       - A real pair is kept only if it is worth at least as much as leaving it unmatched.
       - This is implemented by expanding to a (n_rows + n_cols) x (n_rows + n_cols)
         assignment matrix with zero-profit dummy matches.

    Parameters
    ----------
    return_matrix : bool, optional
        When True, return the complete 0/1 assignment matrix. When False,
        skip that allocation and return None in its place.

    Returns
    -------
    bin_mat : np.ndarray or None
        Complete 0/1 matrix of chosen real-real pairs when ``return_matrix``
        is True; otherwise None.
    (row_ind, col_ind) : tuple[np.ndarray, np.ndarray]
        Row/col indices of chosen real-real pairs.
    """
    # --- to numpy ---
    if torch.is_tensor(sim_matrix):
        sim = sim_matrix.detach().cpu().numpy()
    else:
        sim = np.asarray(sim_matrix, dtype=float)

    if sim.ndim != 2:
        raise ValueError(f"sim_matrix must be 2D, got shape={sim.shape}")

    n_rows, n_cols = sim.shape

    # Read the threshold from the environment when no argument is provided.
    if threshold is None:
        s = os.environ.get("HUG_THRESHOLD", "").strip()
        if s != "":
            threshold = float(s)

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise ImportError(
            "SciPy is required for globally optimal Hungarian assignment. "
            "Please install scipy (e.g., pip install scipy)."
        ) from e

    # Empty matrix
    if n_rows == 0 or n_cols == 0:
        bin_mat = np.zeros((n_rows, n_cols), dtype=int) if return_matrix else None
        return bin_mat, (np.array([], dtype=int), np.array([], dtype=int))

    # Finite mask
    finite = np.isfinite(sim)

    # -----------------------------
    # Case 1: standard full matching
    # -----------------------------
    if threshold is None:
        # Maximize the total similarity on the original matrix.
        # Convert to minimization cost, avoiding inf directly.
        if np.any(finite):
            finite_vals = sim[finite]
            sim_max = float(np.max(finite_vals))
            sim_min = float(np.min(finite_vals))
            penalty = (sim_max - sim_min + 1.0) * (n_rows + n_cols + 1)
        else:
            # Return an empty assignment when no finite edge is available.
            bin_mat = np.zeros((n_rows, n_cols), dtype=int) if return_matrix else None
            return bin_mat, (np.array([], dtype=int), np.array([], dtype=int))

        cost = np.where(finite, sim_max - sim, penalty)
        row_ind, col_ind = linear_sum_assignment(cost)

        bin_mat = np.zeros((n_rows, n_cols), dtype=int) if return_matrix else None
        keep_r, keep_c = [], []
        for r, c in zip(row_ind, col_ind):
            if r < n_rows and c < n_cols and finite[r, c]:
                if bin_mat is not None:
                    bin_mat[r, c] = 1
                keep_r.append(r)
                keep_c.append(c)

        return bin_mat, (np.array(keep_r, dtype=int), np.array(keep_c, dtype=int))

    # -----------------------------------------
    # Case 2: threshold-aware GLOBAL partial matching
    # unmatched baseline = threshold
    # maximize sum(sim_ij - threshold) over chosen real pairs
    # -----------------------------------------
    threshold = float(threshold)

    # Profit of choosing a real pair instead of leaving it unmatched
    # invalid edges should never be selected
    if np.any(finite):
        finite_vals = sim[finite]
        sim_max = float(np.max(finite_vals))
        sim_min = float(np.min(finite_vals))
        scale = max(1.0, sim_max - sim_min + abs(threshold) + 1.0)
    else:
        # Leave all nodes unmatched when no finite real edge is available.
        bin_mat = np.zeros((n_rows, n_cols), dtype=int) if return_matrix else None
        return bin_mat, (np.array([], dtype=int), np.array([], dtype=int))

    neg_large = -scale * (n_rows + n_cols + 1) * 1000.0

    profit_real = np.where(finite, sim - threshold, neg_large)

    # Expanded square matrix:
    # [ real-real profit      row-unmatched(0) ]
    # [ col-unmatched(0)      dummy-dummy(0)   ]
    N = n_rows + n_cols
    profit = np.zeros((N, N), dtype=float)
    profit[:n_rows, :n_cols] = profit_real

    # Convert max-profit assignment to min-cost assignment
    profit_max = float(np.max(profit))
    cost = profit_max - profit

    r_all, c_all = linear_sum_assignment(cost)

    bin_mat = np.zeros((n_rows, n_cols), dtype=int) if return_matrix else None
    real_r, real_c = [], []

    for r, c in zip(r_all, c_all):
        # Keep only real-real matches
        if r < n_rows and c < n_cols and finite[r, c]:
            # Only keep matches that are truly selected as profitable/non-worse than unmatched
            if sim[r, c] >= threshold:
                if bin_mat is not None:
                    bin_mat[r, c] = 1
                real_r.append(r)
                real_c.append(c)

    return bin_mat, (np.array(real_r, dtype=int), np.array(real_c, dtype=int))
    

