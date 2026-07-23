"""Convert SPEX 3.08.01 level and oscillator FITS data into HDF5 tables."""

import os
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.constants import h, c
import h5py
import re
import sys

# Add the project root directory to sys.path.
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)
from util import quantbl, unit_cnvt
from datacollect import h5_tools


def spex_levfits_to_df(data):
    """Convert a SPEX LEV FITS table into a level DataFrame."""
    cols_wanted = ['config', 'cterm', 'w', 'energy']
    df = pd.DataFrame({col: data[col] for col in cols_wanted})

    df.rename(columns={'config': 'cfg', 'energy': 'keV'}, inplace=True)
    df.insert(0, 'idx', range(1, len(df) + 1))

    df['cfg'] = df['cfg'].str.strip()
    df['cfg'] = df['cfg'].str.replace(' ', '.', regex=False)

    df['J'] = (df['w'] - 1) / 2
    df['eV'] = df['keV'] * 1000
    df['pcm'] = unit_cnvt.convert_units(df['eV'], 'eV', 'pcm')
    df['ryd'] = unit_cnvt.convert_units(df['pcm'], 'pcm', 'ryd')
    df = quantbl.split_SL_column(df, 'cterm')

    df = df[['idx', 'cfg', '2S+1', 'L', 'J', 'pcm', 'ryd', 'eV']]
    df = df.astype({'idx': 'i4', 'cfg': 'string', '2S+1': 'i4',
                    'L': 'i4', 'J': 'f4', 'pcm': 'f4', 'ryd': 'f4', 'eV': 'f4'})

    return df


def spex_tranfits_to_df(data, lev_df):
    """Convert a SPEX OSC FITS table into a transition DataFrame."""
    cols_wanted = ['il', 'iu', 'f', 'a']
    df = pd.DataFrame({col: data[col] for col in cols_wanted})

    df.rename(columns={'il': 'lo', 'iu': 'up', 'a': 'A'}, inplace=True)

    idx_to_J = dict(zip(lev_df['idx'], lev_df['J']))
    df['loggf'] = np.log10((2*df['up'].map(idx_to_J)+1) * df['f'])

    idx_to_eV = lev_df.set_index('idx')['eV']
    up_eV = df['up'].map(idx_to_eV)
    lo_eV = df['lo'].map(idx_to_eV)
    delta_eV = up_eV - lo_eV
    df['wav'] = (h * c).to(u.eV * u.AA) / delta_eV

    # Convert FITS columns to native byte order.
    for col in ['up', 'lo', 'f', 'A']:
        arr = df[col].values
        df[col] = arr.byteswap().newbyteorder()

    df = df[['lo', 'up', 'wav', 'A', 'loggf']]
    df = df.astype({'lo': 'i4', 'up': 'i4', 'wav': 'f4', 'loggf': 'f4', 'A': 'f4'})

    return df


def spex_df_to_h5(ion, dtype, df, save_path, version_str):
    """Store a SPEX level or transition DataFrame in an ion-specific HDF5 file."""
    dataset_name = f"lev_spex{version_str}" if dtype == "lev" else f"tran_spex{version_str}"

    filepath = os.path.join(save_path, ion + '.h5')
    with h5py.File(filepath, 'a') as h5f:
        grp = h5f
        if dataset_name in grp:
            del grp[dataset_name]

        dset = grp.create_dataset(dataset_name, data=h5_tools.df_to_hdf5grp(df), compression="gzip")
        dset.attrs['ion'] = ion
        dset.attrs['database'] = [f'spex{version_str}']
        dset.attrs['columns'] = df.columns.tolist()


def spex_all_to_h5(linear_path, save_path, version_str):
    """Convert all SPEX FITS files in a linpar directory into HDF5 tables."""
    for fits_data in os.listdir(linear_path):
        if not fits_data.endswith('.fits'):
            continue

        ion = os.path.splitext(fits_data)[0]
        fits_path = os.path.join(linear_path, fits_data)

        with fits.open(fits_path) as hdul:
            if "LEV" in hdul:
                lev_data = hdul["LEV"].data
                lev_df = spex_levfits_to_df(lev_data)
                spex_df_to_h5(ion, 'lev', lev_df, save_path, version_str)
                print(f"✅ Processed LEV data of {ion} to HDF5.")
            else:
                print(f"❌  Skipping {ion}: No LEV extension found.")

            if "OSC" in hdul:
                tran_data = hdul["OSC"].data
                tran_df = spex_tranfits_to_df(tran_data, lev_df)
                spex_df_to_h5(ion, 'tran', tran_df, save_path, version_str)
                print(f"✅ Processed OSC data of {ion} to HDF5.")
            else:
                print(f"❌  Skipping {ion}: No LEV or OSC extension found.")


def main():
    # SPEX v3.08.01 is used in the article. Set version_str to '30803' and
    # update spex_root to process SPEX v3.08.03 with the same workflow.
    version_str = '30801'

    spex_root = 'path_to_SPEX/SPEX-3.08.01-Source'
    linear_path = os.path.join(spex_root, 'data', 'linpar')
    save_path = os.path.join(directory, 'database', 'spex'+version_str)
    os.makedirs(save_path, exist_ok=True)
    spex_all_to_h5(linear_path, save_path, version_str)


if __name__ == "__main__":
    main()
