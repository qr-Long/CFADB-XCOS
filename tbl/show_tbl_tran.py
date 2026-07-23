"""Read atomic transition tables and export standardized transition data.

Transition records can be loaded from whitespace-delimited lists or HDF5 datasets.
The wavelength column is converted to the requested wavelength or energy unit.
"""

import pandas as pd
import os
from scipy.constants import h, c, e
import sys
import h5py

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(directory)
from datacollect import h5_tools
from util import unit_cnvt


def tran_df (tran_path, dataset_path, unit='A'):
    """Load transitions and convert the wavelength column to the requested unit."""
    if tran_path.endswith('.h5'):
        tran_df = h5_tools.ds_to_df(tran_path, dataset_path)
    elif tran_path.endswith('.lis'):
        tran_df = pd.read_csv(tran_path, sep=r'\s+', comment='#', header=None, names=['lo', 'up', 'wav', 'A', 'loggf'])

    tran_df['wav'] = tran_df['wav'].apply(lambda x: unit_cnvt.wav_to_E(x, unit))

    tran_df = tran_df[['lo', 'up', 'wav', 'A', 'loggf']]

    tran_df = tran_df.astype({'lo': 'i4', 'up': 'i4', 'wav': 'f4', 'loggf': 'f4', 'A': 'f4'})

    return tran_df


def main():
    """Export an example atomic transition table as a tab-delimited file."""

    ion = '020605'

    db = 'atomdb313'
    unit = 'eV'

    folder_path = os.path.join(directory, 'atomic_data')
    file_path_atdata = os.path.join(folder_path, ion+'.h5')
    dataset_path = '/tran_' + db
    df = tran_df(file_path_atdata, dataset_path, unit)

    directory_d = os.path.join(directory, "data", 'seq'+ion[:2], ion)
    if not os.path.exists(directory_d):
        os.makedirs(directory_d)

    file_path_save = os.path.join(directory_d, 'tran_'+db+'.dat')

    with open(file_path_save, "w", encoding="utf-8") as f:
        f.write(f"# ion: {ion}\n")
        f.write(f"# database: {db}\n")
        f.write(f"# unit of wavelength: {unit}, if it is an energy unit, it means the \"wav\" column means the corresponding dE\n\n")

    df.to_csv(file_path_save, index=False, sep='\t', mode='a', header=True)


if __name__ == "__main__":
    main()
