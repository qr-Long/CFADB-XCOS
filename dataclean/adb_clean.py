"""
Clean AtomDB 3.1.3 level tables for the cross-database
atomic energy-level mapping workflow.
"""

import sys
import os

# Add the project root directory to the module search path.
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)

from dataclean import base_tools
from dataclean.seq_list import get_seq_list_by_ion_type


def adb_df_clean(path_ad, lev_path, tran_path, path_c):
    """Clean levels only; tran_path remains an unused compatibility argument."""
    # XCOS uses levels only. Transition copying, cleaning and saving are disabled:
    # fsrc.copy(tran_path, fdst)
    # tran_df = show_tbl_tran.tran_df(path_ad, tran_path)
    # tran_df, _ = base_tools.clean_transition_numeric_fields(tran_df, verbose=True)
    # tran_df, _, _ = base_tools.clean_transitions_by_levels(tran_df, lev_df)
    # h5f.create_dataset(tran_path, data=h5_tools.df_to_hdf5grp(tran_df))
    return base_tools.clean_level_table(
        path_ad, lev_path, path_c, filter_atomdb_terms=True,
    )


def main():
    """Clean all AtomDB files for the selected iso-electronic sequence."""
    ion_type = "ne-like"  # h-like / he-like / li-like / be-like / ne-like

    atomic_data_dir = base_tools.default_atomic_data_dir(directory)
    clean_data_dir = os.path.join(directory, "clean_data")

    ions = get_seq_list_by_ion_type(
        folder_path=atomic_data_dir,
        ion_type=ion_type,
        suffixes=".h5",
    )

    for ion in ions:
        path_ad = os.path.join(atomic_data_dir, ion + ".h5")
        lev_path = 'lev_atomdb313'
        tran_path = 'tran_atomdb313'
        path_c = os.path.join(clean_data_dir, ion + ".h5")

        try:
            adb_df_clean(path_ad, lev_path, tran_path, path_c)
        except Exception as e:
            print(f"Error processing {ion}: {e}")


if __name__ == "__main__":
    main()
