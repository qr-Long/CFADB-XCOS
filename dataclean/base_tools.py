"""
Provide shared utilities for cleaning atomic transition tables and deriving
level parity from electron-configuration strings.
"""

import pandas as pd
import numpy as np
import re

ORBITAL_L = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
    "h": 5,
    "i": 6,
    "k": 7,
    "l": 8,
    "m": 9,
    "n": 10,
    "o": 11,
    "q": 12,
    "r": 13,
    "t": 14,
    "u": 15,
    "v": 16,
    "w": 17,
    "x": 18,
    "y": 19,
    "z": 20,
}

_SUBSHELL_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(\d+)\s*"
    r"([spdfghiklmnoqrtuvwxyzSPDFGHIKLMNOQRTUVWXYZ])"
    r"\s*\^?\s*"
    r"(\d*)"
)


def _remove_parent_state_labels(cfg: str) -> str:
    """Remove parent-term labels so they are not parsed as electron orbitals."""

    def repl(match: re.Match[str]) -> str:
        inside = match.group(1).strip()
        if inside.isdigit():
            return inside
        return ""

    text = re.sub(r"\(([^()]*)\)", repl, cfg)
    text = re.sub(r"\[[^\[\]]*\]", "", text)
    text = re.sub(r"\{[^{}]*\}", "", text)
    return text


def configuration_parity_value(cfg) -> int:
    """
    Calculate parity from an electron-configuration string.

    Returns
    -------
    int
        ``0`` for even parity, ``1`` for odd parity, ``2`` for configurations
        containing ``limit``, and ``-1`` when parity cannot be determined.
    """
    if pd.isna(cfg):
        return -1

    text = str(cfg).strip()
    if not text:
        return -1

    if re.search(r"\blimit\b", text, flags=re.IGNORECASE):
        return 2

    text = _remove_parent_state_labels(text)

    total = 0
    found = False

    for match in _SUBSHELL_RE.finditer(text):
        orbital = match.group(2).lower()
        occ_txt = match.group(3)

        if orbital not in ORBITAL_L:
            continue

        occ = int(occ_txt) if occ_txt else 1
        total += occ * ORBITAL_L[orbital]
        found = True

    if not found:
        return -1

    return total % 2


def add_pi_from_cfg(
    lev_df: pd.DataFrame,
    *,
    cfg_col: str = "cfg",
    pi_col: str = "pi",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Calculate parity from the configuration column and add it as ``pi``.

    The parity value is ``0`` for even levels, ``1`` for odd levels, ``2``
    for configurations containing ``limit``, and ``-1`` when it cannot be
    determined.
    """
    if not inplace:
        df = lev_df.copy()
    else:
        df = lev_df

    if cfg_col not in df.columns:
        raise KeyError(f"Missing required configuration column: {cfg_col}")

    df[pi_col] = df[cfg_col].apply(configuration_parity_value).astype(np.int8)

    preferred_cols = ["idx", cfg_col, "2S+1", "L", "J", pi_col, "P", "pcm", "ryd", "eV"]
    ordered_cols = [c for c in preferred_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining_cols]


def clean_transitions_by_levels(
    tran_df: pd.DataFrame,
    lev_df: pd.DataFrame,
    *,
    level_col: str = "idx",
    lo_col: str = "lo",
    up_col: str = "up",
    wav_col: str = "wav",
    make_wav_positive: bool = True,
    coerce_numeric: bool = True,
    inplace: bool = False,
    verbose: bool = True,
):
    """
    Retain transitions whose lower and upper levels both exist in the level table.

    Wavelengths can also be converted to positive values after filtering.
    """
    if not inplace:
        df = tran_df.copy()
    else:
        df = tran_df

    lev_idx = lev_df[level_col]
    if coerce_numeric:
        lev_idx = pd.to_numeric(lev_idx, errors="coerce")
        df[lo_col] = pd.to_numeric(df[lo_col], errors="coerce")
        df[up_col] = pd.to_numeric(df[up_col], errors="coerce")
        if make_wav_positive and (wav_col in df.columns):
            df[wav_col] = pd.to_numeric(df[wav_col], errors="coerce")

    valid_levels = set(lev_idx.dropna().astype(np.int64).tolist())

    lo_valid = df[lo_col].isin(valid_levels)
    up_valid = df[up_col].isin(valid_levels)
    both_valid = lo_valid & up_valid

    before = len(df)
    invalid_rows = df.loc[~both_valid].copy()
    reason = np.where(
        ~lo_valid & ~up_valid,
        "lo&up_invalid",
        np.where(
            ~lo_valid,
            "lo_invalid",
            np.where(~up_valid, "up_invalid", "unknown"),
        ),
    )
    invalid_rows["invalid_reason"] = reason[~both_valid.values]

    df = df.loc[both_valid].copy()

    neg_before = None
    if make_wav_positive and (wav_col in df.columns):
        s = df[wav_col]
        neg_before = int((s < 0).sum()) if pd.api.types.is_numeric_dtype(s) else None
        df[wav_col] = df[wav_col].abs()

    df[lo_col] = df[lo_col].astype(np.int64)
    df[up_col] = df[up_col].astype(np.int64)

    after = len(df)
    dropped = before - after

    if verbose:
        print(f"[clean_transitions_by_levels] valid_levels: {len(valid_levels)}")
        print(
            f"[clean_transitions_by_levels] invalid transitions removed: "
            f"{dropped} / {before} ({dropped / before:.2%})"
            if before > 0
            else "[clean_transitions_by_levels] transition table is empty."
        )
        if make_wav_positive and (wav_col in df.columns) and (neg_before is not None):
            print(
                f"[clean_transitions_by_levels] negative wavelengths converted "
                f"to positive values: {neg_before}"
            )
        if dropped > 0:
            disp_cols = [
                c
                for c in [lo_col, up_col, "invalid_reason"]
                if c in invalid_rows.columns
            ]
            print(invalid_rows[disp_cols].head(10))

    stats = {
        "valid_levels": len(valid_levels),
        "before": before,
        "after": after,
        "dropped": dropped,
        "drop_rate": (dropped / before) if before else 0.0,
        "lo_only_invalid": int((~lo_valid & up_valid).sum()),
        "up_only_invalid": int((lo_valid & ~up_valid).sum()),
        "both_invalid": int((~lo_valid & ~up_valid).sum()),
    }
    if make_wav_positive and (wav_col in df.columns) and (neg_before is not None):
        stats["wav_neg_fixed"] = neg_before

    return df, stats, invalid_rows


def clean_transition_numeric_fields(
    tran_df: pd.DataFrame,
    *,
    wav_col: str = "wav",
    A_col: str = "A",
    loggf_col: str = "loggf",
    abs_wav: bool = True,
    drop_nonfinite: bool = True,
    A_policy: str = "drop",
    A_min: float = 1e-30,
    inplace: bool = False,
    verbose: bool = True,
):
    """
    Clean numeric transition fields before they are used as edge attributes.

    Wavelengths can be made positive, non-positive Einstein A coefficients can
    be removed or clipped, and non-finite values can be discarded.
    """
    df = tran_df if inplace else tran_df.copy()

    report = {"n_in": len(df)}
    cols = [c for c in [wav_col, A_col, loggf_col] if c in df.columns]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if abs_wav and wav_col in df.columns:
        df[wav_col] = df[wav_col].abs()

    if A_col in df.columns:
        if A_policy == "drop":
            mask_A = df[A_col].notna() & (df[A_col] > 0)
            report["drop_A_nonpos_or_nan"] = int((~mask_A).sum())
            df = df.loc[mask_A]
        elif A_policy == "clip":
            mask_A = df[A_col].notna()
            report["drop_A_nan"] = int((~mask_A).sum())
            df = df.loc[mask_A]
            df[A_col] = df[A_col].clip(lower=A_min)
            report["clip_A_to_min"] = int((df[A_col] <= A_min).sum())
        else:
            raise ValueError("A_policy must be 'drop' or 'clip'")

    if drop_nonfinite and len(df) > 0:
        finite_mask = np.ones(len(df), dtype=bool)
        for c in cols:
            v = df[c].to_numpy(dtype=np.float64, copy=False)
            finite_mask &= np.isfinite(v)
        report["drop_nonfinite_any"] = int((~finite_mask).sum())
        df = df.loc[finite_mask]

    df = df.reset_index(drop=True)
    report["n_out"] = len(df)

    if verbose:
        print(
            f"[clean_transition_numeric_fields] in={report['n_in']} out={report['n_out']} "
            f"dropA={report.get('drop_A_nonpos_or_nan', report.get('drop_A_nan', 0))} "
            f"dropNonFinite={report.get('drop_nonfinite_any', 0)}"
        )

    return df, report
