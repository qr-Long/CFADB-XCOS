"""Provide HDF5 conversion, inspection, reading, and merge utilities for atomic data."""

import os
import h5py
import sys
import shutil
import numpy as np
import pandas as pd

# Add the project root directory to sys.path.
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)
from tbl import show_tbl_lev, show_tbl_tran


def get_parser_by_filename(filename):
    """Select the level or transition table parser from a file-name prefix."""
    if filename.startswith('lev_'):
        return show_tbl_lev.lev_df
    elif filename.startswith('tran_'):
        return show_tbl_tran.tran_df
    else:
        return None


def df_to_hdf5grp(df):
    """Convert a DataFrame into an HDF5-compatible structured NumPy array."""
    structured_dtype = []
    for col in df.columns:
        if df[col].dtype.kind in ('O', 'U'):
            structured_dtype.append((col, h5py.string_dtype(encoding='utf-8', length=256)))
        else:
            structured_dtype.append((col, df[col].dtype))

    structured_array = np.empty(len(df), dtype=structured_dtype)

    for col in df.columns:
        if df[col].dtype.kind in ('O', 'U'):
            structured_array[col] = df[col].astype(str)
        else:
            structured_array[col] = df[col].values
    return structured_array


def ds_to_df(file_path, dataset_path):
    """Read an HDF5 dataset and return it as a DataFrame."""
    with h5py.File(file_path, 'r') as f:
        ds = f[dataset_path]
        data = ds[:]

        decoded_data = {}
        for name in data.dtype.names:
            col = data[name]
            if col.dtype.kind == 'S':
                decoded_data[name] = col.astype(str)
            else:
                decoded_data[name] = col

        df = pd.DataFrame(decoded_data)
        df.columns = ds.attrs.get('columns', df.columns)

    return df


def convert_atomic_data_to_hdf5(root_folder, hdf5_output_path):
    """Convert existing level and transition .lis files into ion-specific HDF5 files."""
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if not fname.endswith('.lis'):
                continue
            ion = os.path.basename(dirpath)
            file_path = os.path.join(dirpath, fname)
            file_output_path = os.path.join(hdf5_output_path, ion+'.h5')
            with h5py.File(file_output_path, 'a') as h5f:
                grp = h5f
                if ion in grp:
                    del grp[ion]

                if fname.startswith('lev_'):
                    df = show_tbl_lev.lev_df(file_path, 0)
                elif fname.startswith('tran_'):
                    df = show_tbl_tran.tran_df (file_path, 0, unit='A')
                else:
                    df = None

                if df is not None:
                    dsname = os.path.splitext(fname)[0]
                    if dsname in grp:
                        del grp[dsname]
                    dset = grp.create_dataset(dsname, data=df_to_hdf5grp(df), compression="gzip")

                dset.attrs['ion'] = ion
                dset.attrs['database'] = ['chianti1102']
                dset.attrs['columns'] = df.columns.tolist()


def explore_hdf5_structure(h5_path):
    """Print the group and dataset structure of an HDF5 file."""
    with h5py.File(h5_path, 'r') as h5f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📂 Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"📄 Dataset: {name} | shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"🔍 Unknown object: {name} | type={type(obj)}")
        h5f.visititems(print_structure)


def merge_hdf5_keep_new(file_src, file_dest):
    """Merge source HDF5 content into a destination file, replacing name conflicts."""
    def merge_groups(src, dest):
        for name, item in src.items():
            if isinstance(item, h5py.Group):
                if name not in dest:
                    dest.copy(item, name)
                else:
                    merge_groups(item, dest[name])
            elif isinstance(item, h5py.Dataset):
                if name in dest:
                    del dest[name]
                dest.create_dataset(name, data=item[()])

    with h5py.File(file_src, 'r') as src, h5py.File(file_dest, 'r+') as dest:
        merge_groups(src, dest)


def merge_h5_folders(folder_src, folder_dest):
    """Merge HDF5 files from a source folder into matching destination files."""
    files_src = {f for f in os.listdir(folder_src) if f.endswith('.h5')}
    files_dest = {f for f in os.listdir(folder_dest) if f.endswith('.h5')}

    for f in files_src:
        path_src = os.path.join(folder_src, f)
        path_dest = os.path.join(folder_dest, f)

        if f in files_dest:
            print(f"Merging {f} ...")
            merge_hdf5_keep_new(path_src, path_dest)
        else:
            shutil.copy2(path_src, path_dest)
            print(f"Copied {f} to destination")


def main():
    # AtomDB v3.1.3, CHIANTI v11.0.2, SPEX v3.08.01 is used in the article; change this path to
    # './database/spex30803' when working with SPEX v3.08.03 data.
    folder_src = './database/spex30801'
    folder_dest = './atomic_data'
    os.makedirs(folder_dest, exist_ok=True)
    merge_h5_folders(folder_src, folder_dest)


if __name__ == "__main__":
    main()
