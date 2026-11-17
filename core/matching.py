"""Threshold-constrained assignment and physical validation of selected pairs."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
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


def _cfg_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c).startswith("cfg_")]
    cols.sort(key=lambda x: int(str(x).split("_")[-1]) if str(x).split("_")[-1].isdigit() else str(x))
    return cols


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return X / norms


def _pairwise_eq_matrix(vals1: np.ndarray, vals2: np.ndarray) -> np.ndarray:
    return vals1[:, None] == vals2[None, :]


def _pairwise_rel_diff_matrix(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(v1, dtype=float)[:, None]
    b = np.asarray(v2, dtype=float)[None, :]
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
    return np.abs(a - b) / denom


def _maximum_bipartite_matching_count(mask: np.ndarray) -> int:
    """Return the maximum one-to-one match count for a boolean bipartite mask."""
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


def build_rule_matrices(
    viewer_data: Dict[str, Any],
    *,
    check_cfg: bool = False,
    cfg_cos_thr: float = 0.8,
    check_2s1: bool = True,
    check_l: bool = True,
    check_j: bool = True,
    check_parity: bool = True,
    check_energy: bool = True,
    energy_rel_thr: float = 0.05,
    energy_abs_thr: Optional[float] = None,
    energy_rule_mode: str = "rel_only",
) -> Dict[str, Any]:
    """Build pairwise rule-consistency matrices for the whole db1 x db2 grid."""
    lev_df1 = viewer_data["lev_df1"]
    lev_df2 = viewer_data["lev_df2"]
    n1, n2 = len(lev_df1), len(lev_df2)

    true_all = np.ones((n1, n2), dtype=bool)
    energy_rule_mode = str(energy_rule_mode).lower()
    if energy_rule_mode not in {"rel_only", "abs_only", "rel_or_abs", "rel_and_abs"}:
        raise ValueError(
            "energy_rule_mode must be one of: rel_only, abs_only, rel_or_abs, rel_and_abs"
        )

    results: Dict[str, Any] = {
        "check_cfg": bool(check_cfg),
        "cfg_cos_thr": float(cfg_cos_thr),
        "check_2s1": bool(check_2s1),
        "check_l": bool(check_l),
        "check_j": bool(check_j),
        "check_parity": bool(check_parity),
        "check_energy": bool(check_energy),
        "energy_rel_thr": float(energy_rel_thr),
        "energy_abs_thr": None if energy_abs_thr is None else float(energy_abs_thr),
        "energy_rule_mode": energy_rule_mode,
    }

    # cfg cosine
    cfg_cols1 = _cfg_feature_cols(lev_df1)
    cfg_cols2 = _cfg_feature_cols(lev_df2)
    common_cfg_cols = [c for c in cfg_cols1 if c in cfg_cols2]
    if len(common_cfg_cols) > 0:
        cfg1 = lev_df1[common_cfg_cols].to_numpy(dtype=float)
        cfg2 = lev_df2[common_cfg_cols].to_numpy(dtype=float)
        cfg_cos = _normalize_rows(cfg1) @ _normalize_rows(cfg2).T
    else:
        cfg_cos = np.full((n1, n2), np.nan, dtype=float)
    cfg_pass = np.ones((n1, n2), dtype=bool)
    if check_cfg and len(common_cfg_cols) > 0:
        cfg_pass = cfg_cos >= float(cfg_cos_thr)
    results["cfg_cos"] = cfg_cos
    results["cfg_pass"] = cfg_pass

    # 2S+1, L, J, parity
    def _get_col(df: pd.DataFrame, name: str) -> Optional[np.ndarray]:
        if name not in df.columns:
            return None
        return pd.Series(df[name]).astype(str).to_numpy()

    s1_1 = _get_col(lev_df1, "2S+1")
    s1_2 = _get_col(lev_df2, "2S+1")
    l1 = _get_col(lev_df1, "L")
    l2 = _get_col(lev_df2, "L")
    j1 = _get_col(lev_df1, "J")
    j2 = _get_col(lev_df2, "J")

    pass_2s1 = true_all if (not check_2s1 or s1_1 is None or s1_2 is None) else _pairwise_eq_matrix(s1_1, s1_2)
    pass_l = true_all if (not check_l or l1 is None or l2 is None) else _pairwise_eq_matrix(l1, l2)
    pass_j = true_all if (not check_j or j1 is None or j2 is None) else _pairwise_eq_matrix(j1, j2)

    if check_parity:
        if "pi" not in lev_df1.columns or "pi" not in lev_df2.columns:
            raise KeyError("Missing required parity column: pi")
        p1 = pd.to_numeric(lev_df1["pi"], errors="coerce").to_numpy(dtype=float)
        p2 = pd.to_numeric(lev_df2["pi"], errors="coerce").to_numpy(dtype=float)
        valid_p1 = np.isfinite(p1) & np.isin(p1, [0, 1, 2])
        valid_p2 = np.isfinite(p2) & np.isin(p2, [0, 1, 2])
        pass_parity = (
            valid_p1[:, None]
            & valid_p2[None, :]
            & _pairwise_eq_matrix(p1, p2)
        )
    else:
        pass_parity = true_all

    results["pass_2s1"] = pass_2s1
    results["pass_l"] = pass_l
    results["pass_j"] = pass_j
    results["pass_parity"] = pass_parity

    # energy consistency: relative / absolute / combined
    if check_energy and ("eV" in lev_df1.columns) and ("eV" in lev_df2.columns):
        e1 = pd.to_numeric(lev_df1["eV"], errors="coerce").to_numpy(dtype=float)
        e2 = pd.to_numeric(lev_df2["eV"], errors="coerce").to_numpy(dtype=float)

        a = e1[:, None]
        b = e2[None, :]

        e_abs = np.abs(a - b)
        e_rel = _pairwise_rel_diff_matrix(e1, e2)

        rel_ok = np.isfinite(e_rel) & (e_rel <= float(energy_rel_thr))

        if energy_abs_thr is None:
            abs_ok = np.zeros((n1, n2), dtype=bool)
        else:
            abs_ok = np.isfinite(e_abs) & (e_abs <= float(energy_abs_thr))

        if energy_rule_mode == "rel_only":
            pass_energy = rel_ok
        elif energy_rule_mode == "abs_only":
            pass_energy = abs_ok
        elif energy_rule_mode == "rel_or_abs":
            pass_energy = rel_ok | abs_ok
        elif energy_rule_mode == "rel_and_abs":
            pass_energy = rel_ok & abs_ok
        else:
            raise ValueError(f"Unknown energy_rule_mode: {energy_rule_mode}")

    else:
        e_rel = np.full((n1, n2), np.nan, dtype=float)
        e_abs = np.full((n1, n2), np.nan, dtype=float)
        pass_energy = true_all

    results["energy_rel"] = e_rel
    results["energy_abs"] = e_abs
    results["pass_energy"] = pass_energy

    all_pass = cfg_pass & pass_2s1 & pass_l & pass_j & pass_parity & pass_energy
    results["all_pass"] = all_pass
    results["theoretical_overlap"] = _maximum_bipartite_matching_count(all_pass)
    return results


def evaluate_selected_matches(
    viewer_data: Dict[str, Any],
    rules: Dict[str, Any],
    thr_idx: int,
) -> Dict[str, Any]:
    """Evaluate selected matches at one threshold with compact idx-first diagnostics."""
    k = int(thr_idx)
    rows = viewer_data["match_rows"][k]
    cols = viewer_data["match_cols"][k]
    sim = viewer_data["sim"]
    info1 = viewer_data["info1"]
    info2 = viewer_data["info2"]

    empty_cols = [
        "db1_idx", "db1_cfg", "db1_2S+1", "db1_L", "db1_J", "db1_pi", "db1_eV",
        "db2_idx", "db2_cfg", "db2_2S+1", "db2_L", "db2_J", "db2_pi", "db2_eV",
        "sim", "cfg_cos", "energy_rel", "cfg_pass", "pass_2s1", "pass_l", "pass_j", "pass_parity", "pass_energy", "all_pass",
    ]
    if len(rows) == 0:
        empty_df = pd.DataFrame(columns=empty_cols)
        return {
            "pairs_df": empty_df,
            "matched": 0,
            "mismatch": 0,
            "acc": np.nan,
            "cfg_fail": 0,
            "s1_fail": 0,
            "l_fail": 0,
            "j_fail": 0,
            "parity_fail": 0,
            "energy_fail": 0,
            "selected_consistent": 0,
            "selected_mismatch": 0,
        }

    df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "sim": sim[rows, cols],
        "cfg_cos": rules["cfg_cos"][rows, cols],
        "energy_rel": rules["energy_rel"][rows, cols],
        "cfg_pass": rules["cfg_pass"][rows, cols],
        "pass_2s1": rules["pass_2s1"][rows, cols],
        "pass_l": rules["pass_l"][rows, cols],
        "pass_j": rules["pass_j"][rows, cols],
        "pass_parity": rules["pass_parity"][rows, cols],
        "pass_energy": rules["pass_energy"][rows, cols],
        "all_pass": rules["all_pass"][rows, cols],
    })

    for src_name, src_df, prefix, keys in [
        ("db1", info1, "db1", ["idx", "cfg", "2S+1", "L", "J", "pi", "eV"]),
        ("db2", info2, "db2", ["idx", "cfg", "2S+1", "L", "J", "pi", "eV"]),
    ]:
        pos_col = "row" if src_name == "db1" else "col"
        subset = src_df.reset_index(drop=True)
        for key in keys:
            if key in subset.columns:
                df[f"{prefix}_{key}"] = subset.loc[df[pos_col].astype(int), key].to_numpy()

    mismatch = int(np.sum(~df["all_pass"].to_numpy(dtype=bool)))
    matched = int(len(df))
    acc = float((matched - mismatch) / matched) if matched > 0 else np.nan

    ordered = [c for c in empty_cols if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered and c not in {"row", "col"}]]

    return {
        "pairs_df": df,
        "matched": matched,
        "mismatch": mismatch,
        "acc": acc,
        "cfg_fail": int(np.sum(~df["cfg_pass"].to_numpy(dtype=bool))),
        "s1_fail": int(np.sum(~df["pass_2s1"].to_numpy(dtype=bool))),
        "l_fail": int(np.sum(~df["pass_l"].to_numpy(dtype=bool))),
        "j_fail": int(np.sum(~df["pass_j"].to_numpy(dtype=bool))),
        "parity_fail": int(np.sum(~df["pass_parity"].to_numpy(dtype=bool))),
        "energy_fail": int(np.sum(~df["pass_energy"].to_numpy(dtype=bool))),
        "selected_consistent": int(np.sum(df["all_pass"].to_numpy(dtype=bool))),
        "selected_mismatch": int(np.sum(~df["all_pass"].to_numpy(dtype=bool))),
    }


def paper_rules(pair, sequence):
    from .data import paper_parameters
    return build_rule_matrices(pair, check_cfg=True, check_2s1=True, check_l=True,
        check_j=True, check_parity=True, check_energy=True, **paper_parameters()['sequence_rules'][sequence])

def threshold_results(sim, rules, thresholds):
    records = []
    theoretical = int(rules['theoretical_overlap'])
    for threshold in thresholds:
        _, (rows, cols) = hungarian_binarize(sim, threshold=float(threshold), return_matrix=False)
        valid = int(rules['all_pass'][rows,cols].sum())
        records.append((float(threshold),len(rows),valid,len(rows)-valid))
    return pd.DataFrame(records, columns=['threshold','matched','accepted','mismatched'])

def acceptable_ranges(frame, theoretical):
    ok = (frame.matched.to_numpy()==theoretical) & (frame.accepted.to_numpy()==theoretical) & (frame.mismatched.to_numpy()==0) & (theoretical>0)
    idx = np.flatnonzero(ok)
    groups = np.split(idx, np.flatnonzero(np.diff(idx)>1)+1) if len(idx) else []
    return ','.join(f'{frame.threshold.iloc[g[0]]:.2f}-{frame.threshold.iloc[g[-1]]:.2f}' for g in groups if len(g))
