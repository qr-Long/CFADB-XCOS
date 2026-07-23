"""Diagnose mismatched atomic level assignments and compare their associated transition properties."""

import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Any, Iterable


def _resolve_sim(sim_df, idx1, idx2):
    """Return the similarity for ``(idx1, idx2)`` or ``NaN`` when it is unavailable.

    The input table is expected to contain the columns ``idx1``, ``idx2``, and
    ``sim``.
    """
    import numpy as np
    import pandas as pd
    if sim_df is None or idx2 is None:
        return np.nan
    if not {'idx1','idx2','sim'}.issubset(sim_df.columns):
        return np.nan
    hit = sim_df[(sim_df['idx1']==idx1) & (sim_df['idx2']==idx2)]
    if hit.empty:
        return np.nan
    try:
        return float(hit.iloc[0]['sim'])
    except Exception:
        return np.nan


def save_wrong_points_rows(
    bin_mat: np.ndarray | None,
    idx: tuple[np.ndarray, np.ndarray] | None,
    lev_df1: pd.DataFrame, tran_df1: pd.DataFrame,
    lev_df2: pd.DataFrame, tran_df2: pd.DataFrame,
    map_df: pd.DataFrame,   # Reference assignments before remapping
    out_dir: str = "./map_data/gnn_mp/wrong_point",
    sim_df: pd.DataFrame | None = None   # Optional table with ['idx1', 'idx2', 'sim']
):
    """Save reference and predicted mappings together with mismatch diagnostics.

    The function writes ``map_pre.dat`` and ``map_ture.dat`` with similarity
    values, records changed assignments in ``dmap.dat``, and exports the
    affected level and transition subsets for both databases. It returns
    ``(post_map_df, dmap_like, lchange)``.
    """
    import os, numpy as np, pandas as pd

    if idx is None and bin_mat is None:
        raise ValueError("Either idx or bin_mat must be provided.")
    os.makedirs(out_dir, exist_ok=True)

    # Reference assignment pairs
    expected_pairs = set(map(tuple, map_df[['idx1','idx2']].astype(int).to_numpy()))
    exp_map = {a: b for a, b in expected_pairs}  # Map idx1 to idx2

    # Predicted assignment pairs, using idx when available
    if idx is not None:
        rows, cols = (np.asarray(idx[0], dtype=int), np.asarray(idx[1], dtype=int))
    else:
        rr, cc = np.where(np.asarray(bin_mat) > 0)
        rows, cols = rr.astype(int), cc.astype(int)

    n1, n2 = len(lev_df1), len(lev_df2)
    post_pairs = []
    for r, c in zip(rows, cols):
        if 0 <= r < n1 and 0 <= c < n2:
            id1 = int(lev_df1.iloc[r]['idx'])
            id2 = int(lev_df2.iloc[c]['idx'])
            post_pairs.append((id1, id2))
    post_pairs = sorted(set(post_pairs), key=lambda x: (x[0], x[1]))
    post_map_df = pd.DataFrame(post_pairs, columns=['idx1','idx2'])
    after_map = {a: b for a, b in post_pairs}

    # Attach similarity values to predicted and reference mappings
    def _attach_sim(df):
        if sim_df is None:
            df = df.copy()
            df['sim'] = np.nan
            return df
        sims = [_resolve_sim(sim_df, int(x), int(y)) for x, y in df[['idx1','idx2']].itertuples(index=False)]
        out = df.copy()
        out['sim'] = sims
        return out

    map_pre  = _attach_sim(post_map_df)                       # Predicted mapping
    map_ture = _attach_sim(map_df[['idx1','idx2']].astype(int))  # Reference mapping
    map_pre.to_csv(os.path.join(out_dir, "map_pre.dat"),  sep="\t", index=False)
    map_ture.to_csv(os.path.join(out_dir, "map_ture.dat"), sep="\t", index=False)

    # Record only idx1 entries whose assignments changed
    changed_idx1 = sorted(set(exp_map.keys()) | set(after_map.keys()))
    rows_dmap = []
    for a in changed_idx1:
        b_before = exp_map.get(a, None)
        b_after  = after_map.get(a, None)
        if b_before == b_after:
            continue
        before_sim = _resolve_sim(sim_df, a, b_before)
        after_sim  = _resolve_sim(sim_df, a, b_after)
        dsim = before_sim - after_sim if (pd.notna(before_sim) and pd.notna(after_sim)) else np.nan
        rows_dmap.append({
            "before_idx1": a,
            "before_idx2": b_before,
            "before_sim":  before_sim,
            "after_idx1":  a,
            "after_idx2":  b_after,
            "after_sim":   after_sim,
            "dsim":        dsim
        })
    dmap_full = pd.DataFrame(rows_dmap, columns=[
        "before_idx1","before_idx2","before_sim",
        "after_idx1","after_idx2","after_sim","dsim"
    ])
    dmap_full.to_csv(os.path.join(out_dir, "dmap.dat"), sep="\t", index=False)

    # Predicted-only differences returned to callers
    predicted  = set(post_pairs)
    diff_pred  = sorted(list(predicted - expected_pairs))     # Predicted but absent from the reference
    dmap_like  = pd.DataFrame(diff_pred, columns=['idx1','idx2'])

    # Identify candidate swapped assignments
    missing = expected_pairs - predicted
    print('right pair: ', list(missing))
    print('predict pair:', diff_pred)
    lchange = find_swapped_pairs(list(missing), diff_pred)

    # Extract affected level and transition subsets
    changed_set      = {row["before_idx1"] for row in rows_dmap}                       # Changed idx1 values
    before_idx2_set  = {row["before_idx2"] for row in rows_dmap if row["before_idx2"] is not None}
    after_idx2_set   = {row["after_idx2"]  for row in rows_dmap if row["after_idx2"]  is not None}

    # Database 1 levels use the same changed idx1 set before and after remapping
    before_lev_wrong1 = _reorder(_drop_cfg_num_cols(lev_df1[lev_df1['idx'].isin(changed_set)].copy()),
                                 ["idx","cfg","2S+1","L","J","E"])
    after_lev_wrong1  = before_lev_wrong1.copy()

    # Database 2 levels use the corresponding before and after idx2 sets
    before_lev_wrong2 = _reorder(_drop_cfg_num_cols(lev_df2[lev_df2['idx'].isin(before_idx2_set)].copy()),
                                 ["idx","cfg","2S+1","L","J","E"])
    after_lev_wrong2  = _reorder(_drop_cfg_num_cols(lev_df2[lev_df2['idx'].isin(after_idx2_set)].copy()),
                                 ["idx","cfg","2S+1","L","J","E"])

    # Database 1 transitions are selected by changed idx1 values
    before_tran_wrong1 = _subset_tran_by_idx(tran_df1, changed_set)
    after_tran_wrong1  = before_tran_wrong1.copy()

    # Database 2 transitions are selected by the before and after idx2 sets
    before_tran_wrong2 = _subset_tran_by_idx(tran_df2, before_idx2_set)
    after_tran_wrong2  = _subset_tran_by_idx(tran_df2, after_idx2_set)

    # Save diagnostic tables
    before_lev_wrong1.to_csv(os.path.join(out_dir, "before_lev_wrong1.dat"), sep="\t", index=False)
    after_lev_wrong1.to_csv (os.path.join(out_dir, "after_lev_wrong1.dat"),  sep="\t", index=False)
    before_lev_wrong2.to_csv(os.path.join(out_dir, "before_lev_wrong2.dat"), sep="\t", index=False)
    after_lev_wrong2.to_csv (os.path.join(out_dir, "after_lev_wrong2.dat"),  sep="\t", index=False)

    before_tran_wrong1.to_csv(os.path.join(out_dir, "before_tran_wrong1.dat"), sep="\t", index=False)
    after_tran_wrong1.to_csv (os.path.join(out_dir, "after_tran_wrong1.dat"),  sep="\t", index=False)
    before_tran_wrong2.to_csv(os.path.join(out_dir, "before_tran_wrong2.dat"), sep="\t", index=False)
    after_tran_wrong2.to_csv (os.path.join(out_dir, "after_tran_wrong2.dat"),  sep="\t", index=False)



    return post_map_df, dmap_like, lchange


# ------------------------------
# Shared helpers
# ------------------------------
_CFG_NUM_RE = re.compile(r"^cfg_\d+$")
def _drop_cfg_num_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: 
        return df
    keep = [c for c in df.columns if not _CFG_NUM_RE.match(str(c))]
    return df[keep].copy()

def _reorder(df: pd.DataFrame, prefer: list[str]) -> pd.DataFrame:
    cols = list(df.columns)
    front = [c for c in prefer if c in cols]
    tail  = [c for c in cols if c not in front]
    return df[front + tail].copy()

def _subset_tran_by_idx(df: pd.DataFrame, idx_set: set[int]) -> pd.DataFrame:
    """Return transitions whose lower or upper level belongs to ``idx_set``."""
    if df is None or df.empty: 
        return df.iloc[0:0].copy()
    if not {'lo','up'}.issubset(df.columns):
        return df.iloc[0:0].copy()
    tdf = _drop_cfg_num_cols(df.copy())
    lo = pd.to_numeric(tdf['lo'], errors='coerce')
    up = pd.to_numeric(tdf['up'], errors='coerce')
    mask = lo.isin(idx_set) | up.isin(idx_set)
    tdf = tdf[mask]
    return _reorder(tdf, ["lo","up","wav","A","loggf"])

# ------------------------------
# Identify candidate swapped assignments
# ------------------------------
def find_swapped_pairs(before: List[Tuple[Any, Any]],
                       after:  List[Tuple[Any, Any]]) -> List[List[List[Any]]]:
    """Group predicted-only edges into candidate pair swaps.

    ``before`` contains reference pairs that are missing from the prediction,
    while ``after`` contains predicted-only pairs. Each predicted-only edge
    ``(x, y)`` is mapped to the reference pair containing ``x`` and the
    reference pair containing ``y``. A self-pair is used when one endpoint has
    no reference pair. Duplicate groups are removed irrespective of order.
    """

    # Map each index to its reference pair
    idx2pair: dict[Any, Tuple[Any, Any]] = {}
    for a, b in before:
        pair = (a, b)
        idx2pair[a] = pair
        idx2pair[b] = pair

    groups: List[List[List[Any]]] = []
    seen_keys = set()

    # Process each predicted-only edge
    for x, y in after:
        # Skip edges unrelated to all missing reference pairs
        if x not in idx2pair and y not in idx2pair:
            continue

        # Map endpoints to their reference pairs, using self-pairs when absent
        pair_x = idx2pair.get(x, (x, x))
        pair_y = idx2pair.get(y, (y, y))

        # Use an order-independent key to remove duplicate groups
        key = tuple(sorted((pair_x, pair_y)))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        groups.append([
            [pair_x[0], pair_x[1]],
            [pair_y[0], pair_y[1]],
        ])

    return groups





# ------------------------------
# Normalize candidate swap representations
# ------------------------------
def _as_lchange_list(x) -> List[List[int]]:
    import pandas as pd

    # DataFrame representation
    if isinstance(x, pd.DataFrame):
        if not {'idx1','idx2'}.issubset(x.columns):
            raise KeyError("The change-like DataFrame must contain idx1 and idx2 columns.")
        if len(x) < 2:
            raise ValueError("The change-like DataFrame must contain at least two rows.")
        i1 = int(x.iloc[0]['idx1']); i2 = int(x.iloc[1]['idx1'])
        j1 = int(x.iloc[0]['idx2']); j2 = int(x.iloc[1]['idx2'])
        return [[i1, i2], [j1, j2]]

    # Pair-list representation: [[i1, j1], [i2, j2]] -> [[i1, i2], [j1, j2]]
    if isinstance(x, (list, tuple)) and len(x) == 2 and \
       all(isinstance(v, (list, tuple)) and len(v) == 2 for v in x):
        i1, j1 = map(int, x[0])
        i2, j2 = map(int, x[1])
        return [[i1, i2], [j1, j2]]

    raise TypeError("Expected [[i1, j1], [i2, j2]] or a DataFrame containing idx1 and idx2.")


def ctran_list(change_like, tran: pd.DataFrame, map_df: pd.DataFrame, order_name: str="1", abs_wav: bool=False):
    """Select transition subsets for the two levels in a candidate swap.

    For database 1, the function selects ``i1`` and ``i2`` and constructs
    ``lo_note`` and ``up_note`` from ``map_df['idx1']``. For database 2, it
    selects ``j1`` and ``j2`` and uses ``map_df['idx2']``.
    """
    if order_name not in ("1","2"):
        raise ValueError("order_name must be either '1' or '2'.")
    lchg = _as_lchange_list(change_like)
    idxs = lchg[0] if order_name=="1" else lchg[1]

    key_col = f"idx{order_name}"
    if key_col not in map_df.columns:
        raise KeyError(f"map_df does not contain the required column '{key_col}'.")
    idx_to_note = {map_df.iloc[i][key_col]: i+1 for i in range(len(map_df))}

    out = []
    for idx in idxs:
        # Return an empty table when the level is absent from the database
        tdf = _subset_tran_by_idx(tran, {idx})
        if tdf is None or tdf.empty:
            out.append(pd.DataFrame(columns=['lo','up','lo_note','up_note']))
            continue

        # Place lower and upper level indices first
        other = [c for c in tdf.columns if c not in ('lo','up')]
        tdf = tdf[['lo','up'] + other]
        # Map level indices to reference positions; unavailable values remain NaN
        tdf['lo_note'] = tdf['lo'].map(idx_to_note)
        tdf['up_note'] = tdf['up'].map(idx_to_note)
        if abs_wav and 'wav' in tdf.columns:
            tdf['wav'] = tdf['wav'].abs()
        out.append(tdf)
    return out

def cad_AE2_log2(file: pd.DataFrame) -> pd.DataFrame:
    file = file.copy()
    if {'wav1','wav2','A1','A2','loggf1','loggf2'}.issubset(file.columns):
        file['AE_wav']   = file['wav1'] - file['wav2']
        file['AE_logA']  = np.log10(np.abs(file['A1'])) - np.log10(np.abs(file['A2']))
        file['AE_loggf'] = file['loggf1'] - file['loggf2']
    else:
        # Add empty result columns when required transition fields are unavailable
        for c in ['AE_wav','AE_logA','AE_loggf']:
            if c not in file.columns: file[c] = np.nan
    return file

def com_ltran(uc_ltran1: List[pd.DataFrame], c_ltran: List[pd.DataFrame], change_like, dchange: List[int]):
    """Compare unchanged and swapped transition combinations.

    ``uc_ltran1`` and ``c_ltran`` contain two transition subsets from database
    1 and database 2, respectively. ``change_like`` identifies the candidate
    level swap, and ``dchange`` defines the cross-pair replacement order. The
    returned value is ``[ucl, cl]`` for unchanged and swapped combinations.
    """
    lchg = _as_lchange_list(change_like)         # [[i1, i2], [j1, j2]]
    left_pair = lchg[0]                           # [i1, i2]
    num = len(dchange)

    # Merge corresponding transition subsets without swapping assignments
    ucl = [None]*num
    for k in range(num):
        alltran = pd.concat([uc_ltran1[k][['lo_note','up_note']], 
                             c_ltran[k][['lo_note','up_note']]], ignore_index=True)
        alltran = alltran.drop_duplicates()
        uc_idx_tran0 = pd.merge(alltran, uc_ltran1[k], on=['lo_note','up_note'], how='left', suffixes=('1','2'))
        com_idx_tran1 = pd.merge(uc_idx_tran0, c_ltran[k],  on=['lo_note','up_note'], how='left', suffixes=('1','2'))
        com_idx_tran1.fillna(0, inplace=True)
        com_idx_tran1 = cad_AE2_log2(com_idx_tran1)
        ucl[k] = com_idx_tran1

    # Replace database 2 assignment positions to evaluate the swapped combination
    cl = [None]*num
    ch_ltran = [None]*num
    for i in range(num):
        cplace = i + dchange[i]
        pridx = left_pair[i]
        toidx = left_pair[cplace]
        ch_ltran[cplace] = copy.deepcopy(c_ltran[i])
        if not ch_ltran[cplace].empty:
            ch_ltran[cplace]['lo_note'] = ch_ltran[cplace]['lo_note'].replace(pridx, toidx)
            ch_ltran[cplace]['up_note'] = ch_ltran[cplace]['up_note'].replace(pridx, toidx)

    for c in range(num):
        alltran = pd.concat([uc_ltran1[c][['lo_note','up_note']], 
                             ch_ltran[c][['lo_note','up_note']]], ignore_index=True)
        alltran = alltran.drop_duplicates()
        uc_idx_tran0 = pd.merge(alltran, uc_ltran1[c], on=['lo_note','up_note'], how='left', suffixes=('1','2'))
        com_idx_tran2 = pd.merge(uc_idx_tran0, ch_ltran[c],  on=['lo_note','up_note'], how='left', suffixes=('1','2'))
        com_idx_tran2.fillna(0, inplace=True)
        com_idx_tran2 = cad_AE2_log2(com_idx_tran2)
        cl[c] = com_idx_tran2

    uc_c = [ucl, cl]
    return uc_c

def de_nosame(uc_c: List[List[pd.DataFrame]]):
    """Remove rows containing non-finite transition-difference values."""
    numc = len(uc_c[0])
    uc_cc = [[None]*numc, [None]*numc]
    for i in range(2):
        for j in range(numc):
            df = uc_c[i][j]
            if df is None or df.empty:
                uc_cc[i][j] = pd.DataFrame(columns=['AE_wav','AE_logA','AE_loggf'])
                continue
            bad = ~np.isfinite(df.get('AE_logA', np.nan))
            if 'AE_wav' in df.columns:
                bad |= ~np.isfinite(df['AE_wav'])
            if 'AE_loggf' in df.columns:
                bad |= ~np.isfinite(df['AE_loggf'])
            uc_cc[i][j] = df.loc[~bad].copy()
    return uc_cc

def tran_samepa_plt_box(uc_cc, lchange, dchange, pa, path='./map_data/gnn_mp/wrong_point/box'):
    """Plot transition-difference boxplots for reference and swapped pairings."""
    lchg = _as_lchange_list(lchange)
    # Create the output directory for this candidate swap
    folder_path = os.path.join(path, f"{lchg[0][0]}_{lchg[0][1]}&{lchg[1][0]}_{lchg[1][1]}")
    os.makedirs(folder_path, exist_ok=True)

    fig, axes = plt.subplots(1, len(uc_cc) * len(uc_cc[0]), figsize=(2 * 1.6, 3.34), sharey=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])

    cnum = len(uc_cc[0])
    palette = sns.color_palette("viridis", n_colors=cnum*2)
    colors = palette.as_hex()
    y_ranges = []
    texts_to_draw = []

    labels = ['', '']
    for i in range(2):
        group_ns = []
        for j in range(cnum):
            if i == 0:
                xname = f"{lchg[0][j]}_{lchg[1][j]}"
            else:
                xname = f"{lchg[0][j]}_{lchg[1][(j + dchange[j]) % cnum]}"
            ax = axes[i*cnum + j]
            df = uc_cc[i][j]
            # Use an empty table when no transition comparison is available
            if df is None:
                df = pd.DataFrame(columns=[pa])
            try:
                sns.boxplot(data=df, y=pa, color=colors[i*cnum + j], width=0.2, ax=ax)
            except Exception as e:
                print(f"[warn] boxplot failed: {e}")

            n_j = int(df.get(pa, pd.Series(dtype=float)).notna().sum())
            group_ns.append(n_j)
            mean = float(df.get(pa, pd.Series(dtype=float)).mean()) if n_j>0 else 0.0
            texts_to_draw.append((i*cnum + j, -0.4, mean, f"{mean:.3g}", colors[i*cnum + j]))

            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            if j > 0:
                ax.tick_params(left=False, labelleft=False)
                ax.spines['left'].set_visible(False)
            ax.set_ylabel('')
            y_ranges.append(ax.get_ylim())
            ax.set_xlabel(xname)
        ns_str = ", ".join(str(v) for v in group_ns)
        axes[cnum * i].set_title(f'{labels[i]} (n={ns_str})')

    y_min = min(y[0] for y in y_ranges) if y_ranges else 0
    y_max = max(y[1] for y in y_ranges) if y_ranges else 1
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    plt.subplots_adjust(wspace=-0.1)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88)
    plt.savefig(os.path.join(folder_path, f"{pa}_boxplot.png"), dpi=500)
    plt.close(fig)

def plt_all(lchange_list, map_df, lev_df1, lev_df2, tran_df1, tran_df2):
    """Run transition diagnostics and generate boxplots for each candidate swap."""
    dchange = [1, -1]
    print("Generating boxplots for transition-property differences...")
    for lchg in lchange_list:
        ltran_df1 = ctran_list(lchg, tran_df1, map_df, order_name="1")
        ltran_df2 = ctran_list(lchg, tran_df2, map_df, order_name="2")
        uc_c  = com_ltran(ltran_df1, ltran_df2, lchg, dchange)
        uc_cc = de_nosame(uc_c)
        for pa in ['AE_wav', 'AE_logA', 'AE_loggf']:
            tran_samepa_plt_box(uc_cc, lchg, dchange, pa)




