"""CFADB-XCOS 0.3.0: validate and rebuild reversible level-only archives."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import h5py
import numpy as np

# Add the project root directory to the module search path.
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)

from dataclean import base_tools
from dataclean.seq_list import get_seq_list_by_ion_type
from datacollect.cdb_to_h5 import (
    XCOS_VERSION, ENERGY_DATASET, cdb_file_todf, cdb_file_toh5,
    validate_energy_bundle,
)
from util import ptbl

DATABASES = ('atomdb313', 'chianti1102', 'spex30801', 'spex30803')
SEQUENCES = {'h-like': '01', 'he-like': '02', 'li-like': '03',
             'be-like': '04', 'ne-like': '10'}


def cdb_df_del_bracket(df):
    """Remove parenthetical labels from the configuration column."""
    df['cfg'] = df['cfg'].str.replace(r'\([^)]*\)', '', regex=True)
    return df


def cdb_df_clean(path_ad, lev_path, tran_path, path_c):
    """Validate selected energies, then clean levels; tran_path is unused."""
    if lev_path != 'lev_chianti1102':
        raise ValueError('Expected lev_chianti1102')
    with h5py.File(path_ad, 'r') as src:
        validate_energy_bundle(src)
    return base_tools.clean_level_table(
        path_ad, lev_path, path_c, filter_atomdb_terms=False,
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild_archives(source_atomic_dir, chianti_source, output_root):
    """Build all five sequences in a fresh directory, without transitions.

    The four database/ion coverages are inherited from source_atomic_dir.
    Non-CHIANTI levels are copied losslessly; CHIANTI is regenerated from raw
    .elvlc files. Existing output folders are never overwritten by this command.
    """
    source_atomic_dir = Path(source_atomic_dir).resolve()
    output_root = Path(output_root).resolve()
    atomic_dir, clean_dir = output_root / 'atomic_data', output_root / 'clean_data'
    if atomic_dir.exists() or clean_dir.exists():
        raise FileExistsError('Use a fresh output root; atomic_data/clean_data already exists')
    files = sorted(p for p in source_atomic_dir.glob('*.h5')
                   if re.fullmatch(r'\d{6}', p.stem) and p.stem[:2] in SEQUENCES.values())
    if not files:
        raise FileNotFoundError('No five-sequence atomic input files')
    raw_index = {}
    for ep in sorted(Path(chianti_source).rglob('*.elvlc')):
        match = re.fullmatch(r'([a-z]+)_(\d+)', ep.stem)
        if not match:  # Independent dielectronic satellite files are excluded.
            continue
        z, stage = ptbl.element_to_atomic_number(match[1]), int(match[2])
        ion = f'{z-stage+1:02}{z:02}{stage:02}'
        if ion in raw_index:
            raise ValueError(f'Duplicate raw CHIANTI ion: {ion}')
        raw_index[ion] = ep
    source_hashes = {p.name: _sha256(p) for p in files}
    raw_hashes = {}
    for p in files:
        with h5py.File(p, 'r') as f:
            if 'lev_chianti1102' in f:
                if p.stem not in raw_index:
                    raise FileNotFoundError(f'Missing raw CHIANTI levels for {p.stem}')
                ep = raw_index[p.stem]
                raw_hashes[str(ep.resolve())] = _sha256(ep)
    atomic_dir.mkdir(parents=True)
    clean_dir.mkdir()
    totals = Counter()
    for p in files:
        atomic_path, clean_path = atomic_dir / p.name, clean_dir / p.name
        with h5py.File(p, 'r') as src, h5py.File(atomic_path, 'w') as dst:
            has_chianti = 'lev_chianti1102' in src
            dst.attrs['cfadb_xcos_version'] = XCOS_VERSION
            dst.attrs['source_atomic_file_sha256'] = source_hashes[p.name]
            dst.attrs['source_atomic_file'] = p.name
            for db in DATABASES:
                name = 'lev_' + db
                if name in src and db != 'chianti1102':
                    src.copy(name, dst, name=name)
        if has_chianti:
            ep = raw_index[p.stem]
            df = cdb_file_todf(ep, ['idx', 'cfg', '2S+1', 'L', 'J', 'pcm'], 'lev')
            cdb_file_toh5(df, ep.name, 'lev', atomic_dir)
            with h5py.File(atomic_path, 'r') as f:
                energy, _ = validate_energy_bundle(f)
                obs = energy['pcm_obs']
                totals['chianti_observed_levels'] += int((np.isfinite(obs) & (obs >= 0)).sum())
                totals['chianti_levels'] += len(energy)
                totals['chianti_tables'] += 1
        with h5py.File(atomic_path, 'r') as f:
            names = [name for name in f if name.startswith('lev_')]
        if not names:
            raise ValueError(f'No supported level tables: {p.name}')
        for name in names:
            if name == 'lev_chianti1102':
                counts = cdb_df_clean(atomic_path, name, None, clean_path)
            else:
                counts = base_tools.clean_level_table(
                    atomic_path, name, clean_path, filter_atomdb_terms=name == 'lev_atomdb313')
            totals['level_tables'] += 1
            for key, count in counts.items():
                totals[key] += count
        with h5py.File(clean_path, 'a') as f:
            f.attrs['cfadb_xcos_version'] = XCOS_VERSION
            f.attrs['atomic_file_sha256'] = _sha256(atomic_path)
            f.attrs['atomic_file'] = p.name
            f.attrs['level_only'] = True
        totals['ion_files'] += 1
    # Source changes during a build must not be silently accepted.
    if source_hashes != {p.name: _sha256(p) for p in files}:
        raise RuntimeError('Atomic inputs changed during rebuild')
    if any(_sha256(p) != digest for p, digest in raw_hashes.items()):
        raise RuntimeError('Raw CHIANTI inputs changed during rebuild')
    summary = dict(cfadb_xcos_version=XCOS_VERSION, totals=dict(totals),
                   source_atomic_dir=str(source_atomic_dir), source_atomic_sha256=source_hashes,
                   raw_elvlc_sha256=raw_hashes)
    (output_root / 'rebuild_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true', help='Build atomic_data and clean_data for all five sequences')
    parser.add_argument('--source-atomic-data', type=Path, help='Standardized source tables for all four databases')
    parser.add_argument('--chianti-source', type=Path, help='Raw CHIANTI 11.0.2 database directory')
    parser.add_argument('--output-root', type=Path, help='Fresh output root for a complete rebuild')
    parser.add_argument('--atomic-data', type=Path, default=base_tools.default_atomic_data_dir(directory))
    parser.add_argument('--output', type=Path, default=Path(directory) / 'clean_data')
    parser.add_argument('--ion-type', choices=['all', *SEQUENCES], default='all')
    args = parser.parse_args()
    if args.rebuild:
        if not all((args.source_atomic_data, args.chianti_source, args.output_root)):
            parser.error('--rebuild requires --source-atomic-data, --chianti-source and --output-root')
        print(json.dumps(rebuild_archives(args.source_atomic_data, args.chianti_source,
                                         args.output_root)['totals'], indent=2))
        return
    sequences = SEQUENCES if args.ion_type == 'all' else [args.ion_type]
    count = 0
    for seq in sequences:
        ions = get_seq_list_by_ion_type(folder_path=args.atomic_data, ion_type=seq, suffixes='.h5')
        for ion in ions:
            path = args.atomic_data / (ion + '.h5')
            with h5py.File(path, 'r') as f:
                if 'lev_chianti1102' not in f:
                    continue
            cdb_df_clean(path, 'lev_chianti1102', None, args.output / path.name)
            count += 1
    print(f'Validated and cleaned {count} CHIANTI level tables')


if __name__ == '__main__':
    main()
