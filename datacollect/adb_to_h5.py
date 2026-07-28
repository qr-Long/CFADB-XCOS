"""Convert AtomDB 3.1.3 level and transition FITS data into HDF5 tables."""

import os
import pyatomdb
import pandas as pd
import numpy as np
from astropy.io import fits
import h5py
import re
import sys

# Add the project root directory to sys.path.
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)
from util import ptbl, unit_cnvt
from datacollect import h5_tools


def clean_cfg_1(cfg):
    """Remove a trailing occupancy value of 1 from each configuration segment."""
    parts = cfg.split('.')
    cleaned = []
    for part in parts:
        new_part = re.sub(r'([a-zA-Z])1$', r'\1', part)
        cleaned.append(new_part)
    return '.'.join(cleaned)


def adb_levfits_to_df(fits_data):
    """Convert an AtomDB LV FITS table into a level DataFrame."""
    data = fits_data[1].data

    cols_wanted = ['ELEC_CONFIG', 'S_QUAN', 'L_QUAN', 'LEV_DEG', 'ENERGY']
    df = pd.DataFrame({col: data[col] for col in cols_wanted})

    df.rename(columns={
        'ELEC_CONFIG': 'cfg',
        'S_QUAN': 'S',
        'L_QUAN': 'L',
        'LEV_DEG': 'g',
        'ENERGY': 'eV'
    }, inplace=True)

    # Use one-based level indices.
    df.insert(0, 'idx', range(1, len(df) + 1))

    df['cfg'] = df['cfg'].str.replace(' ', '.', regex=False)
    df['cfg'] = df['cfg'].apply(clean_cfg_1)
    df['J'] = (df['g'] - 1) / 2
    df['pcm'] = unit_cnvt.convert_units(df['eV'], 'eV', 'pcm')
    df['ryd'] = unit_cnvt.convert_units(df['pcm'], 'pcm', 'ryd')
    df['2S+1'] = 2 * df['S'] + 1

    for col in ['L', 'g']:
        arr = df[col].values
        df[col] = arr.byteswap().newbyteorder()

    df = df[['idx', 'cfg', '2S+1', 'L', 'J', 'pcm', 'ryd', 'eV']]

    df = df.astype({'idx': 'i4', 'cfg': 'string', '2S+1': 'i4',
                    'L': 'i4', 'J': 'f4', 'pcm': 'f4', 'ryd': 'f4', 'eV': 'f4'})

    return df


def adb_tranfits_to_df(fits_data, lev_df):
    """Convert an AtomDB LA FITS table into a transition DataFrame."""
    data = fits_data[1].data

    cols_wanted = ['UPPER_LEV', 'LOWER_LEV', 'WAVELEN', 'EINSTEIN_A']
    df = pd.DataFrame({col: data[col] for col in cols_wanted})

    df.rename(columns={'UPPER_LEV': 'up', 'LOWER_LEV': 'lo', 'WAVELEN': 'wav', 'EINSTEIN_A': 'A'}, inplace=True)

    idx_to_J = dict(zip(lev_df['idx'], lev_df['J']))
    constant = 1.499e-16
    df['gf'] = constant * (2*df['up'].map(idx_to_J)+1) * (df['wav'] ** 2) * df['A']
    df['loggf'] = np.log10(df['gf'])

    # Convert FITS columns to native byte order.
    for col in ['up', 'lo', 'wav', 'A']:
        arr = df[col].values
        df[col] = arr.byteswap().newbyteorder()

    df = df[['lo', 'up', 'wav', 'A', 'loggf']]
    df = df.astype({'lo': 'i4', 'up': 'i4', 'wav': 'f4', 'loggf': 'f4', 'A': 'f4'})

    return df


def adb_df_to_h5(z, roman, dtype, df, aped_path, save_path):
    """Store an AtomDB level or transition DataFrame in an ion-specific HDF5 file."""
    ion = f"{z-roman+1:02d}{z:02d}{roman:02d}"
    version_str = '313'
    dataset_name = f"lev_atomdb{version_str}" if dtype == "LV" else f"tran_atomdb{version_str}"

    filepath = os.path.join(save_path, ion + '.h5')
    with h5py.File(filepath, 'a') as h5f:
        grp = h5f
        if dataset_name in grp:
            del grp[dataset_name]

        dset = grp.create_dataset(dataset_name, data=h5_tools.df_to_hdf5grp(df), compression="gzip")
        dset.attrs['ion'] = ion
        dset.attrs['database'] = [f'atomdb{version_str}']
        dset.attrs['columns'] = df.columns.tolist()


def adb_all_to_h5(z_left, z_right, aped_path, save_path):
    """Convert all available AtomDB ions in the requested atomic-number range."""
    for z in range(z_left, z_right):
        for roman in range(1, z+1):
            lev_fits = pyatomdb.atomdb.get_data(z, roman, 'LV')
            if isinstance(lev_fits, fits.HDUList):
                lev_df = adb_levfits_to_df(lev_fits)
                adb_df_to_h5(z, roman, 'LV', lev_df, aped_path, save_path)
                print(f"✅ Saved lev for Z={z}, roman={roman} to HDF5.")
            else:
                print(f"❌ Warning: No lev data for Z={z}, roman={roman}. Skipping...")

            tran_fits = pyatomdb.atomdb.get_data(z, roman, 'LA')
            if isinstance(lev_fits, fits.HDUList) and isinstance(tran_fits, fits.HDUList):
                tran_df = adb_tranfits_to_df(tran_fits, lev_df)
                adb_df_to_h5(z, roman, 'LA', tran_df, aped_path, save_path)
                print(f"✅ Saved tran for Z={z}, roman={roman} to HDF5.")
            else:
                print(f"❌ Warning: No lev or tran data for Z={z}, roman={roman}. Skipping...")


def main():
    z_left = 1
    z_right = 36

    aped_path = '/Users/longqirui/Desktop/science/databases/atomdb313'
    aped_path = os.path.join(aped_path, 'APED')
    save_path = os.path.join(directory, 'database', 'atomdb313')
    os.makedirs(save_path, exist_ok=True)
    adb_all_to_h5(z_left, z_right, aped_path, save_path)


if __name__ == "__main__":
    main()
