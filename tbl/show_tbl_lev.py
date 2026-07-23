"""Read atomic energy-level tables and export standardized level data.

Both whitespace-delimited level lists and HDF5 datasets are converted to a common
DataFrame schema containing level indices, configurations, term quantum numbers,
and energies in pcm, rydberg, and electron-volt units.
"""

import pandas as pd
import os
import sys
import h5py

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(directory)

from datacollect import h5_tools
from util import unit_cnvt, quantbl


def lev_df(file_path, dataset_path):
    """Load a level table and return the standardized columns."""
    if file_path.endswith('.lis'):
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, names=["idx", "cfg", "SL", "J", "ryd", "pcm"])
        df = df.drop("ryd", axis=1)
        df = quantbl.split_SL_column(df, 'SL')

        df = df.astype({'idx': 'i4', 'cfg': 'string', '2S+1': 'i4', 'L': 'i4', 'J': 'f4', 'pcm': 'f4'})
        df['ryd'] = unit_cnvt.convert_units(df['pcm'], 'pcm', 'ryd')
        df['eV'] = unit_cnvt.convert_units(df['pcm'], 'pcm', 'eV')
        df = df.astype({'ryd': 'f4', 'eV': 'f4'})

    elif file_path.endswith('.h5'):
        df = h5_tools.ds_to_df(file_path, dataset_path)

    df = df[['idx', 'cfg', '2S+1', 'L', 'J', 'pcm', 'ryd', 'eV']]

    return df


def main():
    """Export an example atomic energy-level table as a tab-delimited file."""

    ion = '020605'

    db = 'spex30801'

    file_path_atdata = os.path.join(directory, 'atomic_data', ion+'.h5')
    dataset_path = '/lev_' + db
    df = lev_df(file_path_atdata, dataset_path)

    directory_d = os.path.join(directory, "data", 'seq'+ion[:2], ion)
    if not os.path.exists(directory_d):
        os.makedirs(directory_d)

    file_path_save = os.path.join(directory_d, 'lev_'+db+'.dat')

    with open(file_path_save, "w", encoding="utf-8") as f:
        f.write(f"# ion: {ion}\n")
        f.write(f"# database: {db}\n")

    df.to_csv(file_path_save, index=False, sep='\t', mode='a', header=True)


if __name__ == "__main__":
    main()
