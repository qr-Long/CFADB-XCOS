"""
Clean CHIANTI 11.0.2 level and transition tables for the cross-database
atomic energy-level mapping workflow.
"""

import pandas as pd
import numpy as np
import sys
import h5py
import os

# Add the project root directory to the module search path.
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)

from tbl import show_tbl_lev, show_tbl_tran
from datacollect import h5_tools
import base_tools
from dataclean.seq_list import get_seq_list_by_ion_type


def cdb_df_del_bracket(df):
    """Remove parenthetical labels from the configuration column."""
    df['cfg'] = df["cfg"].str.replace(r"\([^)]*\)", "", regex=True)
    return df


def cdb_df_del_inf(df):
    """Remove transition rows with negative-infinite log(gf) values."""
    df = df[~np.isneginf(df['loggf'].values)].copy()
    return df


def cdb_df_clean(path_ad, lev_path, tran_path, path_c):
    """Clean one pair of CHIANTI level and transition tables."""
    with h5py.File(path_ad, "r") as fsrc, h5py.File(path_c, "a") as fdst:
        if lev_path in fdst:
            del fdst[lev_path]
        if tran_path in fdst:
            del fdst[tran_path]

        fsrc.copy(lev_path, fdst)
        has_transition_data = tran_path in fsrc
        if has_transition_data:
            fsrc.copy(tran_path, fdst)

    lev_df = show_tbl_lev.lev_df(path_ad, lev_path)
    lev_df = base_tools.add_pi_from_cfg(lev_df)

    tran_df = None
    if has_transition_data:
        tran_df = show_tbl_tran.tran_df(path_ad, tran_path)
        tran_df = cdb_df_del_inf(tran_df)
        tran_df, _ = base_tools.clean_transition_numeric_fields(tran_df, verbose=True)
        tran_df, _, _ = base_tools.clean_transitions_by_levels(tran_df, lev_df)

    with h5py.File(path_c, 'a') as h5f:
        grp = h5f
        lev_ds_path = lev_path
        tran_ds_path = tran_path

        if lev_ds_path in grp:
            del grp[lev_ds_path]
        if tran_ds_path in grp:
            del grp[tran_ds_path]

        dset = grp.create_dataset(
            lev_ds_path,
            data=h5_tools.df_to_hdf5grp(lev_df),
            compression="gzip",
        )
        if tran_df is not None:
            dset = grp.create_dataset(
                tran_ds_path,
                data=h5_tools.df_to_hdf5grp(tran_df),
                compression="gzip",
            )


def main():
    """Clean all CHIANTI files for the selected iso-electronic sequence."""
    ion_type = "h-like"  # h-like / he-like / li-like / be-like / ne-like

    atomic_data_dir = os.path.join(directory, "atomic_data")
    clean_data_dir = os.path.join(directory, "clean_data")

    ions = get_seq_list_by_ion_type(
        folder_path=atomic_data_dir,
        ion_type=ion_type,
        suffixes=".h5",
    )

    for ion in ions:
        path_ad = './atomic_data/' + ion + '.h5'
        lev_path = 'lev_chianti1102'
        tran_path = 'tran_chianti1102'
        path_c = './clean_data/' + ion + '.h5'

        try:
            cdb_df_clean(path_ad, lev_path, tran_path, path_c)
        except Exception as e:
            print(f"Error processing {ion}: {e}")


if __name__ == "__main__":
    main()
