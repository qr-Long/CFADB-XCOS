"""Convert CHIANTI 11.0.2 level and transition text files into HDF5 tables."""

import pandas as pd
import numpy as np
import os
import h5py
from io import StringIO
import sys

# Add the project root directory to sys.path.
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)
from datacollect import h5_tools
from util import ptbl, quantbl, unit_cnvt


def cdb_file_todf(filepath, columns, lev_trantype):
    """Convert a CHIANTI .elvlc or .wgfa file into a DataFrame."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # CHIANTI data records end at the line containing -1.
    cutoff = next((i for i, line in enumerate(lines) if line.strip() == '-1'), len(lines))
    data_lines = lines[:cutoff]
    data_text = ''.join(data_lines)

    if lev_trantype == 'lev':
        colspecs = [
            (0, 7),
            (7, 37),
            (37, 45),
            (45, 47),
            (47, 52),
            (52, 57),
            (57, 72),
            (72, 85),
        ]
        df = pd.read_fwf(StringIO(data_text), colspecs=colspecs, header=None)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        df = df.drop(df.columns[[2, 6]], axis=1)
        df.columns = columns
        df['cfg'] = df['cfg'].str.replace(' ', '.', regex=False)
        df['L'] = quantbl.L_letter_to_num(df, col_name='L')
        df = df.astype({'idx': 'i4', 'cfg': 'string', '2S+1': 'i4', 'L': 'i4', 'J': 'f4', 'pcm': 'f4'})
        df['ryd'] = unit_cnvt.convert_units(df['pcm'], 'pcm', 'ryd')
        df['eV'] = unit_cnvt.convert_units(df['pcm'], 'pcm', 'eV')
        df = df.astype({'ryd': 'f4', 'eV': 'f4'})

    elif lev_trantype == 'tran':
        colspecs = [
            (0, 5),
            (5, 10),
            (10, 25),
            (25, 40),
            (40, 55),
        ]
        df = pd.read_fwf(StringIO(data_text), colspecs=colspecs, header=None)
        df.columns = columns
        # The source column contains gf values; store their base-10 logarithm as loggf.
        df['loggf'] = np.log10(df['loggf'])
        df = df.astype({'lo': 'i4', 'up': 'i4', 'wav': 'f4', 'loggf': 'f4', 'A': 'f4'})

    return df


def cdb_file_toh5(df, filename, levtran_type, save_path='./database/chianti1102'):
    """Store a CHIANTI 11.0.2 level or transition DataFrame in HDF5 format."""
    element = filename.split('_')[0]
    element_num = ptbl.element_to_atomic_number(element)
    roman_num = int(filename[filename.find('_')+1 : filename.find('.')])
    seq_num = element_num - roman_num + 1
    ion = f"{seq_num:02}{element_num:02}{roman_num:02}"

    filepath = os.path.join(save_path, ion + '.h5')
    dataset_name= f'{levtran_type}_chianti1102'

    with h5py.File(filepath, 'a') as h5f:
        grp = h5f
        if dataset_name in grp:
            del grp[dataset_name]

        dset = grp.create_dataset(dataset_name, data=h5_tools.df_to_hdf5grp(df), compression="gzip")
        dset.attrs['ion'] = ion
        dset.attrs['database'] = ['chianti1102']
        dset.attrs['columns'] = df.columns.tolist()


def cdb_all_toh5(folder_path, save_path='./database/chianti1102'):
    """Convert all CHIANTI 11.0.2 .elvlc and .wgfa files under a directory tree."""
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)

            if filename.endswith('.elvlc'):
                try:
                    columns = ['idx', 'cfg', '2S+1', 'L', 'J', 'pcm']
                    df = cdb_file_todf(filepath, columns, 'lev')
                    cdb_file_toh5(df, filename, 'lev', save_path)
                except Exception as e:
                    print(f"❌ Failed to process {filename}: {e}")

            elif filename.endswith('.wgfa'):
                try:
                    columns = ['lo', 'up', 'wav', 'loggf', 'A']
                    df = cdb_file_todf(filepath, columns, 'tran')
                    df = df[['lo', 'up', 'wav', 'A', 'loggf']]
                    cdb_file_toh5(df, filename, 'tran', save_path)
                except Exception as e:
                    print(f"❌ Failed to process {filename}: {e}")


def main():
    folder_path = 'path_to_CHIANTI/CHIANTI_11.0.2_database'
    save_path = './database/chianti1102'
    os.makedirs(save_path, exist_ok=True)
    cdb_all_toh5(folder_path, save_path)


if __name__ == "__main__":
    main()
