"""
data_loader.py

Load atomic level and transition data from HDF5 or text files, attach learned
configuration embeddings, and prepare database pairs for the CFADB
cross-database energy-level matching workflow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import h5py
from itertools import combinations
from io import StringIO

try:
    from datacollect import h5_tools  # type: ignore
    from tbl import show_tbl_cflev  # type: ignore
except Exception:  # pragma: no cover
    h5_tools = None
    show_tbl_cflev = None

try:
    from common.cfg_embed import process_cfg, load_cfg_ts_model
except Exception:  # pragma: no cover
    from .cfg_embed import process_cfg, load_cfg_ts_model

DEFAULT_CLEAN_DATA_DIR = Path("./clean_data")
DEFAULT_THR_VIEW_ROOT = Path("./map_data/thr_view")
DEFAULT_CFG_MODEL_ROOT = Path("./map_data/cfg_ts")


def _ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory when needed and return it as a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pick_existing_cols(df: pd.DataFrame, candidates):
    """Return candidate column names that are present in a DataFrame."""
    return [c for c in candidates if c in df.columns]


def _make_level_info_df(lev_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact level-information table for matching diagnostics."""
    base_cols = _pick_existing_cols(lev_df, ["idx", "cfg", "2S+1", "L", "J", "pi", "eV", "pcm", "ryd"])
    info = lev_df[base_cols].copy().reset_index(drop=True)
    info.insert(0, "row", np.arange(len(info), dtype=int))
    if "idx" in info.columns:
        info["idx"] = pd.to_numeric(info["idx"], errors="coerce")
    return info


def _load_optional_transition_df(h5_path: Union[str, Path], dataset_name: str) -> pd.DataFrame:
    """Load a transition table or return an empty table when it is unavailable."""
    with h5py.File(h5_path, "r") as h5f:
        if dataset_name not in h5f:
            return pd.DataFrame(columns=["lo", "up", "wav", "A", "loggf"])
    return h5_tools.ds_to_df(str(h5_path), dataset_name)


def load_cfg_model(seq: str = "li-like", cfg_model_path: Optional[Union[str, Path]] = None):
    """Load the configuration encoder for the requested isoelectronic sequence."""
    if cfg_model_path is not None:
        model_path = Path(cfg_model_path)
    else:
        model_path = DEFAULT_CFG_MODEL_ROOT / seq / "cfg_ts_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find cfg model for seq={seq!r}: {model_path}")
    return load_cfg_ts_model(model_path=str(model_path))


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
    """Load one database pair from an ion HDF5 file and prepare matching inputs."""
    try:
        from datacollect import h5_tools  # type: ignore
        from tbl import show_tbl_cflev  # type: ignore
    except Exception as e:
        raise ImportError("Cannot import datacollect.h5_tools or tbl.show_tbl_cflev. Check the project environment.") from e

    clean_data_dir = Path(clean_data_dir)
    h5_path = clean_data_dir / f"{ion}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Cannot find HDF5 file: {h5_path}")

    if cfg_model is None:
        cfg_model = load_cfg_model(seq=seq, cfg_model_path=cfg_model_path)

    lev_name1 = f"lev_{db1}"
    tran_name1 = f"tran_{db1}"
    lev_name2 = f"lev_{db2}"
    tran_name2 = f"tran_{db2}"

    lev_df1 = h5_tools.ds_to_df(str(h5_path), lev_name1)
    tran_df1 = _load_optional_transition_df(h5_path, tran_name1)
    lev_df2 = h5_tools.ds_to_df(str(h5_path), lev_name2)
    tran_df2 = _load_optional_transition_df(h5_path, tran_name2)

    if cfg_model_path is not None:
        cfg_vocab_path = Path(cfg_model_path).with_name("cfg_vocab.pkl")
    else:
        cfg_vocab_path = DEFAULT_CFG_MODEL_ROOT / seq / "cfg_vocab.pkl"
    if not cfg_vocab_path.exists():
        raise FileNotFoundError(f"Cannot find cfg vocab for seq={seq!r}: {cfg_vocab_path}")

    lev_df1 = process_cfg(lev_df1, cfg_model, vocab_path=str(cfg_vocab_path), max_len=30)
    lev_df2 = process_cfg(lev_df2, cfg_model, vocab_path=str(cfg_vocab_path), max_len=30)

    merged = show_tbl_cflev.levmerged_db2([lev_df1, lev_df2], [1, 2], "eV")
    map_df = merged[["idx1", "idx2"]].drop_duplicates().reset_index(drop=True)

    out_dir = _ensure_dir(Path(thr_view_root) / ion / f"{db1}_{db2}")
    train_item = (lev_df1, tran_df1, lev_df2, tran_df2, map_df)
    return {
        "ion": ion, "db1": db1, "db2": db2, "h5_path": h5_path, "out_dir": out_dir,
        "lev_df1": lev_df1, "tran_df1": tran_df1, "lev_df2": lev_df2, "tran_df2": tran_df2,
        "map_df": map_df, "train_item": train_item, "train_data_list": [train_item],
        "info1": _make_level_info_df(lev_df1), "info2": _make_level_info_df(lev_df2),
    }


def process_hdf5_file(hdf5_path, model):
    from gnn.graph_build import preprocess_data

    """
    Load every level table in an HDF5 file and build all database pairs.

    Parameters
    ----------
    hdf5_path : str or path-like
        Path to an HDF5 file containing ``lev_*`` and ``tran_*`` datasets.
    model : torch.nn.Module
        Trained configuration encoder.

    Returns
    -------
    tuple
        Prepared database pairs, feature-column names, and database suffix
        combinations.
    """
    lev_suffixes = []
    dataset_paths = {}

    with h5py.File(hdf5_path, 'r') as f:
        def find_lev_datasets(name, obj):
            if (isinstance(obj, h5py.Dataset)
                    and name.split('/')[-1].startswith('lev_')
                    and not name.endswith('_delete')):
                suffix = name.split('lev_')[-1]
                lev_suffixes.append(suffix)
                dataset_paths[f'lev_{suffix}'] = name

                tran_name = name.replace('lev_', 'tran_')
                if tran_name in f:
                    dataset_paths[f'tran_{suffix}'] = tran_name

        f.visititems(find_lev_datasets)

    print(f"Level-table suffixes: {lev_suffixes}")

    all_combinations = list(combinations(lev_suffixes, 2))
    print(f"Number of pairwise combinations: {len(all_combinations)}")
    print(f"Number of generated combinations: {len(all_combinations)}")

    results = []
    feature_cols = None
    filtered_combinations = []
    for combo in all_combinations:
        if len(combo) != 2:
            continue
        else:
            feature_cols = None
        suffix1, suffix2 = combo
        lev_name1 = f'lev_{suffix1}'
        tran_name1 = f'tran_{suffix1}'
        lev_name2 = f'lev_{suffix2}'
        tran_name2 = f'tran_{suffix2}'
        with h5py.File(hdf5_path, 'r') as f:
            lev_df1 = h5_tools.ds_to_df(hdf5_path, lev_name1)
            tran_df1 = _load_optional_transition_df(hdf5_path, tran_name1)
            lev_df2 = h5_tools.ds_to_df(hdf5_path, lev_name2)
            tran_df2 = _load_optional_transition_df(hdf5_path, tran_name2)
            lev_df1 = process_cfg(lev_df1, model)
            lev_df2 = process_cfg(lev_df2, model)
            _, feature_cols = preprocess_data(lev_df1, tran_df1)

        min_len = min(len(lev_df1), len(lev_df2))
        merged = show_tbl_cflev.levmerged_db2 ([lev_df1, lev_df2], [1, 2], 'eV')
        map_df = merged[["idx1", "idx2"]].drop_duplicates().reset_index(drop=True)

        results.append((lev_df1, tran_df1, lev_df2, tran_df2, map_df))
        filtered_combinations = [combo for combo in all_combinations if len(combo) == 2]

    return results, feature_cols, filtered_combinations


def _fix_indices(tran_df, n_levels, assume_one_based=True):
    """Convert transition indices to valid zero-based integer indices."""
    tran_df = tran_df.copy()
    tran_df["lo"] = pd.to_numeric(tran_df["lo"], errors="coerce")
    tran_df["up"] = pd.to_numeric(tran_df["up"], errors="coerce")
    tran_df = tran_df.dropna(subset=["lo", "up"])

    if assume_one_based and (tran_df["lo"].min() >= 1) and (tran_df["up"].min() >= 1):
        tran_df["lo"] = tran_df["lo"] - 1
        tran_df["up"] = tran_df["up"] - 1

    tran_df = tran_df[
        (tran_df["lo"] >= 0) & (tran_df["lo"] < n_levels) &
        (tran_df["up"] >= 0) & (tran_df["up"] < n_levels)
    ].copy()
    tran_df["lo"] = tran_df["lo"].astype("int64")
    tran_df["up"] = tran_df["up"].astype("int64")
    return tran_df


def _read_dat_auto(path, encoding="utf-8", allow_commas=True):
    """
    Read a ``.dat`` table after removing blank lines and comment lines.

    The first remaining line is interpreted as the table header.
    """
    with open(path, "r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    filtered = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        filtered.append(ln.rstrip("\n"))

    text = "\n".join(filtered)

    df = pd.read_csv(
        StringIO(text),
        delim_whitespace=True,
        header=0,
        engine="python"
    )
    return df


def load_dat_folders(root_dir, model=None, encoding="utf-8", verbose=True):
    from gnn.graph_build import preprocess_data

    """
    Load matching inputs from valid subdirectories beneath ``root_dir``.

    Each accepted directory must contain ``lev1.dat``, ``lev2.dat``,
    ``tran1.dat``, ``tran2.dat``, and ``map1.dat``.

    Returns
    -------
    results : list
        Tuples containing two level tables, two transition tables, and one
        reference mapping table.
    feature_cols : list
        Feature-column names inferred by ``preprocess_data``.
    filtered_combinations : list
        Names of the accepted data directories.
    """
    expected_files = ["lev1.dat", "lev2.dat", "tran1.dat", "tran2.dat", "map1.dat"]

    subdirs = []
    for entry in sorted(os.scandir(root_dir), key=lambda e: e.name):
        if entry.is_dir():
            files_here = set(os.listdir(entry.path))
            if all(f in files_here for f in expected_files):
                subdirs.append(entry)
            elif verbose:
                missing = [f for f in expected_files if f not in files_here]
                print(f"[Skipped] {entry.name}; missing files: {missing}")

    if verbose:
        print(f"Number of valid data directories: {len(subdirs)} -> {[d.name for d in subdirs]}")

    results = []
    feature_cols = None

    for idx, d in enumerate(subdirs):
        folder = d.path
        if verbose:
            print(f"\nReading directory: {d.name}")

        lev1_path = os.path.join(folder, "lev1.dat")
        lev2_path = os.path.join(folder, "lev2.dat")
        tran1_path = os.path.join(folder, "tran1.dat")
        tran2_path = os.path.join(folder, "tran2.dat")
        map1_path = os.path.join(folder, "map1.dat")

        lev_df1 = pd.read_csv(lev1_path,
                                sep=r"\s+",
                                comment="#",
                                dtype={"idx": int, "cfg": str, '2S+1': int, 'L': int, 'J': float, 'pcm': float})
        lev_df1['ryd'] = unit_cnvt.convert_units(lev_df1['pcm'], 'pcm', 'ryd')
        lev_df1['eV'] = unit_cnvt.convert_units(lev_df1['pcm'], 'pcm', 'eV')
        lev_df2 = pd.read_csv(lev2_path,
                                sep=r"\s+",
                                comment="#",
                                dtype={"idx": int, "cfg": str, '2S+1': int, 'L': int, 'J': float, 'pcm': float})
        lev_df2['ryd'] = unit_cnvt.convert_units(lev_df2['pcm'], 'pcm', 'ryd')
        lev_df2['eV'] = unit_cnvt.convert_units(lev_df2['pcm'], 'pcm', 'eV')
        tran_df1 = pd.read_csv(tran1_path,
                                sep=r"\s+",
                                comment="#",
                                dtype={"lo": int, "up": int, 'wav': float, 'A': float, 'loggf': float})
        tran_df2 = pd.read_csv(tran2_path,
                                sep=r"\s+",
                                comment="#",
                                dtype={"lo": int, "up": int, 'wav': float, 'A': float, 'loggf': float})
        map_df = pd.read_csv(map1_path,
                                sep=r"\s+",
                                comment="#",
                                dtype={"idx1": int, "idx2": int})

        lev_df1 = process_cfg(lev_df1, model)
        lev_df2 = process_cfg(lev_df2, model)
        _, feature_cols = preprocess_data(lev_df1, tran_df1)

        results.append((lev_df1, tran_df1, lev_df2, tran_df2, map_df))

    filtered_combinations = [d.name for d in subdirs]

    if verbose:
        print(f"\nData directory names: {filtered_combinations}")

    return results, feature_cols, filtered_combinations
