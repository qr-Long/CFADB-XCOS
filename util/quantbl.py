"""
Provide orbital-angular-momentum lookup tables and helpers for parsing atomic
term-symbol columns in level tables.
"""

import pandas as pd


L_letter2num = {
    'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6,
    'K': 7, 'L': 8, 'M': 9, 'N': 10, 'O': 11, 'Q': 12, 'R': 13,
    'T': 14, 'U': 15, 'V': 16, 'W': 17, 'X': 18, 'Y': 19, 'Z': 20
}

L_num2letter = {v: k for k, v in L_letter2num.items()}


def L_letter_to_num(df, col_name='L'):
    """Convert orbital-angular-momentum letters in a DataFrame column to integers."""
    df['L'] = df[col_name].replace(L_letter2num).infer_objects(copy=False).astype('int64')
    return df['L']


def split_SL_column(df, col_name):
    """Split a combined ``2S+1 L`` column into integer ``2S+1`` and ``L`` columns."""
    df['2S+1'] = df[col_name].str.extract(r'(\d+)').astype('int64')
    df['L'] = df[col_name].str.extract(r'([A-Za-z])')
    df['L'] = L_letter_to_num(df)

    return df
