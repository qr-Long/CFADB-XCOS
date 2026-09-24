"""
Provide reversible level-only cleaning and configuration/parity utilities.
Legacy transition helpers remain available but are not called by XCOS cleaners.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

import h5py


def build_level_tables(original, keep_mask=None):
    """Partition original records without changing any existing field except cfg.

    Deleted rows retain their source positions; both outputs keep original_cfg
    last. Source dtypes (including string widths) are retained without a
    DataFrame round trip. An existing source pi is preserved, not overwritten.
    """
    names = original.dtype.names
    if names is None or "cfg" not in names:
        raise ValueError("Expected a structured level table containing cfg")
    if {"original_cfg", "original_row"}.intersection(names):
        raise ValueError("Use atomic_data as input, not an already cleaned table")
    if original.dtype.hasobject:
        raise ValueError("Variable-length/object fields require explicit archival support")
    if keep_mask is None:
        keep_mask = np.ones(len(original), dtype=bool)
    keep_mask = np.asarray(keep_mask)
    if keep_mask.dtype != np.bool_ or keep_mask.shape != (len(original),):
        raise ValueError("keep_mask must be one Boolean per original row")

    kept = original[keep_mask]
    cfg = pd.DataFrame({"cfg": kept["cfg"].astype(str)})
    cfg = clean_configuration_strings(cfg)
    derived_pi = "pi" not in names
    if derived_pi:
        cfg = add_pi_from_cfg(cfg)
    clean_fields = []
    for name in names:
        if name == "pcm" and derived_pi:
            clean_fields.append(("pi", np.dtype("i1")))
        clean_fields.append((name, original.dtype.fields[name][0]))
    if derived_pi and "pcm" not in names:
        clean_fields.append(("pi", np.dtype("i1")))
    clean_fields.append(("original_cfg", original.dtype.fields["cfg"][0]))
    cleaned = np.empty(len(kept), dtype=clean_fields)
    for name in names:
        cleaned[name] = kept[name]
    cleaned["cfg"] = cfg["cfg"].to_numpy()
    cleaned["original_cfg"] = kept["cfg"]
    if derived_pi:
        cleaned["pi"] = cfg["pi"].to_numpy()

    row_dtype = "u4" if len(original) <= np.iinfo(np.uint32).max else "u8"
    deleted_fields = [(n, original.dtype.fields[n][0]) for n in names if n != "cfg"]
    deleted_fields.extend([
        ("original_row", row_dtype),
        ("original_cfg", original.dtype.fields["cfg"][0]),
    ])
    deleted = np.empty(np.count_nonzero(~keep_mask), dtype=deleted_fields)
    for name in names:
        target = "original_cfg" if name == "cfg" else name
        deleted[target] = original[name][~keep_mask]
    deleted["original_row"] = np.flatnonzero(~keep_mask)
    return cleaned, deleted


def restore_level_records(cleaned, deleted, source_dtype, source_count):
    """Reconstruct the pre-clean level records, including their original order."""
    kept_count = 0 if cleaned is None else len(cleaned)
    deleted_count = 0 if deleted is None else len(deleted)
    if kept_count + deleted_count != source_count:
        raise ValueError("Kept and deleted row counts do not match the source")
    positions = (np.empty(0, dtype=np.uint64) if deleted is None
                 else deleted["original_row"])
    if (np.any(positions >= source_count) or
            len(np.unique(positions)) != len(positions)):
        raise ValueError("Invalid or duplicate original_row values")
    keep_mask = np.ones(source_count, dtype=bool)
    keep_mask[positions] = False
    restored = np.empty(source_count, dtype=source_dtype)
    for name in restored.dtype.names:
        archived_name = "original_cfg" if name == "cfg" else name
        if cleaned is not None:
            restored[name][keep_mask] = cleaned[archived_name]
        if deleted is not None:
            restored[name][positions] = deleted[archived_name]
    return restored


def _assert_original_levels(original, restored):
    """Fail if any source field changes, including NaN bits or string bytes."""
    if original.dtype != restored.dtype or original.shape != restored.shape:
        raise ValueError("Restored level structure differs from the source")
    for name in original.dtype.names:
        if original[name].tobytes() != restored[name].tobytes():
            raise ValueError(f"Non-reversible level cleaning: field {name!r} changed")


def restore_level_table(path, lev_path):
    """Return original records, treating absent retained/deleted tables as empty.

    Source metadata is present on either nonempty table. An originally empty
    source is recorded as an empty typed file attribute, not a dataset.
    """
    with h5py.File(path, "r") as h5f:
        cleaned_ds = h5f.get(lev_path)
        deleted_ds = h5f.get(lev_path + "_delete")
        if cleaned_ds is None and deleted_ds is None:
            return h5f.attrs["empty_level:" + lev_path]
        ds = cleaned_ds if cleaned_ds is not None else deleted_ds
        names = list(ds.attrs["source_field_names"])
        # Read field types from HDF5 itself, preserving string-encoding metadata.
        source_dtype = np.dtype({
            "names": names,
            "formats": [ds.dtype.fields["original_cfg" if n == "cfg" else n][0]
                        for n in names],
            "offsets": list(ds.attrs["source_field_offsets"]),
            "itemsize": int(ds.attrs["source_itemsize"]),
        })
        return restore_level_records(
            None if cleaned_ds is None else cleaned_ds[:],
            None if deleted_ds is None else deleted_ds[:], source_dtype,
            int(ds.attrs["source_row_count"]),
        )


def clean_level_table(path_ad, lev_path, path_c, *, filter_atomdb_terms=False):
    """Write a reversible level-only archive; never read or clean transitions."""
    if Path(path_ad).resolve() == Path(path_c).resolve():
        raise ValueError("Source atomic_data and destination clean_data must differ")
    if not lev_path.startswith("lev_") or lev_path.endswith("_delete"):
        raise ValueError("Expected an original lev_<database> table")
    with h5py.File(path_ad, "r") as src:
        original = src[lev_path][:]
        source_attrs = dict(src[lev_path].attrs)
    keep_mask = None
    if filter_atomdb_terms:
        keep_mask = (original["2S+1"] >= 0) & (original["L"] >= 0)
    cleaned, deleted = build_level_tables(original, keep_mask)
    _assert_original_levels(original, restore_level_records(
        cleaned, deleted, original.dtype, len(original),
    ))
    Path(path_c).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path_c, "a") as dst:
        # Append mode must not leave transitions from an earlier cleaner run.
        stale_transitions = []
        dst.visititems(lambda name, obj: stale_transitions.append(name)
                       if isinstance(obj, h5py.Dataset)
                       and name.split("/")[-1].startswith("tran_") else None)
        for name in stale_transitions:
            del dst[name]
        # Keep enough metadata to restore a genuinely empty source without
        # creating a zero-row dataset. Clear an older marker on repeated runs.
        empty_key = "empty_level:" + lev_path
        empty_attr_prefix = "empty_source_attr:" + lev_path + ":"
        for attr in list(dst.attrs):
            if attr == empty_key or attr.startswith(empty_attr_prefix):
                del dst.attrs[attr]
        if not len(original):
            dst.attrs[empty_key] = original
            for attr, value in source_attrs.items():
                dst.attrs[empty_attr_prefix + attr] = value
        for name, records in ((lev_path, cleaned), (lev_path + "_delete", deleted)):
            if name in dst:
                del dst[name]
            if not len(records):
                continue
            ds = dst.create_dataset(name, data=records, compression="gzip")
            for attr, value in source_attrs.items():
                ds.attrs["source_attr:" + attr] = value
            ds.attrs["columns"] = list(records.dtype.names)
            ds.attrs["source_field_names"] = list(original.dtype.names)
            ds.attrs["source_field_offsets"] = [original.dtype.fields[n][1]
                                                for n in original.dtype.names]
            ds.attrs["source_itemsize"] = original.dtype.itemsize
            ds.attrs["source_row_count"] = len(original)
            ds.attrs["level_clean_schema"] = 3
        if lev_path in dst:
            dst[lev_path].attrs["derived_pi"] = "pi" not in original.dtype.names
    # Verify the actual serialized archive as well as the in-memory partition.
    _assert_original_levels(original, restore_level_table(path_c, lev_path))
    return {"original": len(original), "kept": len(cleaned), "deleted": len(deleted)}


def default_atomic_data_dir(project_root):
    """Prefer local inputs, otherwise use the sibling cfadb inputs read-only."""
    local = Path(project_root) / "atomic_data"
    return local if local.is_dir() else Path(project_root).parent / "cfadb" / "atomic_data"


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


def clean_configuration_strings(
    lev_df: pd.DataFrame,
    *,
    cfg_col: str = "cfg",
    inplace: bool = False,
) -> pd.DataFrame:
    """Remove trailing dot padding from electronic configurations.

    Internal separators and parent-term labels are preserved: ``1s.2s...``
    becomes ``1s.2s``, while ``1s.2p(3P)`` is unchanged. Whitespace after
    trailing dots is removed with the padding. Missing values stay missing,
    and applying this function again leaves the result unchanged.
    """
    df = lev_df if inplace else lev_df.copy()
    if cfg_col not in df.columns:
        raise KeyError(f"Missing required configuration column: {cfg_col}")

    df[cfg_col] = df[cfg_col].str.replace(r"\.+\s*$", "", regex=True)
    return df


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
