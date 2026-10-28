"""CFADB-XCOS 0.3.0: collect CHIANTI levels with observed-first energies."""

import pandas as pd
import numpy as np
import os
import h5py
from io import StringIO
import sys
import argparse
import hashlib
import json
from pathlib import Path
import re

# Add the project root directory to sys.path.
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)
from datacollect import h5_tools
from util import ptbl, quantbl, unit_cnvt

XCOS_VERSION = "0.3.0"

ENERGY_DATASET = "energy_chianti1102"
ENERGY_DTYPE = np.dtype([("idx", "<i4"), ("pcm_obs", "<f8"), ("pcm_theory", "<f8")])
ENERGY_SCHEMA = 1
LEVEL_POLICY = "observed_if_finite_nonnegative_else_theoretical"
PROVENANCE_KEYS = ("energy_schema", "energy_policy", "energy_sha256", "source_elvlc_sha256")
OVERRIDE_POLICY = LEVEL_POLICY + "_with_explicit_theoretical_overrides"
# Explicit curation for CHIANTI 11.0.2 only; never apply a general outlier filter.
THEORY_OVERRIDES = {
    "031109": {
        81: dict(pcm_obs=1048314.0, pcm_theory=10464952.0,
                 reason="Documented Na IX idx 81 exception: observed energy is about "
                        "one tenth of theory; theory agrees with AtomDB idx 81."),
    },
}


def ion_from_filename(filename):
    """Return the six-digit ion identifier for a regular CHIANTI level file."""
    match = re.fullmatch(r'([a-z]+)_(\d+)\.elvlc', Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported CHIANTI level filename: {filename}")
    z = ptbl.element_to_atomic_number(match[1])
    stage = int(match[2])
    return f"{z-stage+1:02}{z:02}{stage:02}"


def energy_override_records(energy, *, ion):
    """Identify curated rows by ion and idx, verifying the known source values."""
    records = []
    for idx, override in THEORY_OVERRIDES.get(ion, {}).items():
        rows = energy[energy['idx'] == idx]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one CHIANTI exception row: {ion}/{idx}")
        row = rows[0]
        if any(row[key] != override[key] for key in ('pcm_obs', 'pcm_theory')):
            raise ValueError(f"CHIANTI exception source changed for {ion}/{idx}; review required")
        records.append(dict(database='chianti1102', ion=ion, idx=idx,
                            selected_source='theoretical', **override))
    return records


def energy_policy_metadata(energy, *, ion):
    records = energy_override_records(energy, ion=ion)
    if not records:
        return dict(energy_policy=LEVEL_POLICY)
    return dict(energy_policy=OVERRIDE_POLICY,
                energy_overrides=json.dumps(records, sort_keys=True))


def observed_energy_mask(energy, *, ion):
    mask = usable_energy(energy['pcm_obs'])
    for record in energy_override_records(energy, ion=ion):
        mask[energy['idx'] == record['idx']] = False
    return mask


def usable_energy(values):
    """Zero is a valid energy; negative sentinels and nonfinite values are not."""
    return np.isfinite(values) & (values >= 0)


def _ids(values):
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("CHIANTI idx must be a one-dimensional integer array")
    if len(np.unique(values)) != len(values):
        raise ValueError("Duplicate CHIANTI idx values")
    return values


def align_energy(energy, ids, *, exact=False):
    """Align by idx, rejecting missing/duplicate IDs; source rows may be reordered."""
    if (energy is None or energy.ndim != 1 or not energy.dtype.names
            or not set(ENERGY_DTYPE.names).issubset(energy.dtype.names)):
        raise ValueError("CHIANTI requires energy_chianti1102 with idx/pcm_obs/pcm_theory")
    for name in ("pcm_obs", "pcm_theory"):
        if energy.dtype[name].kind not in "fiu":
            raise ValueError(f"Non-numeric CHIANTI {name}")
    source_ids, ids = _ids(energy["idx"]), _ids(np.asarray(ids))
    order = np.argsort(source_ids)
    pos = np.searchsorted(source_ids[order], ids)
    if np.any(pos >= len(source_ids)) or np.any(source_ids[order][pos] != ids):
        raise ValueError("Missing CHIANTI energy idx")
    if exact and len(ids) != len(source_ids):
        raise ValueError("CHIANTI level and energy idx sets differ")
    return energy[order[pos]]


def selected_pcm(energy, *, ion):
    return np.where(observed_energy_mask(energy, ion=ion),
                    energy["pcm_obs"], energy["pcm_theory"])


def energy_sha256(energy):
    """Hash a canonical, sorted representation, independent of HDF5 layout."""
    ordered = align_energy(energy, np.sort(energy["idx"]), exact=True)
    canonical = np.empty(len(ordered), dtype=ENERGY_DTYPE)
    for name in canonical.dtype.names:
        canonical[name] = ordered[name]
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def validate_primary_energy(levels, energy, *, ion, exact=False):
    aligned = align_energy(energy, levels["idx"], exact=exact)
    # The collector rounds pcm to float32 before deriving the other two units.
    # Select on the source table before aligning, also supporting retained subsets.
    source_pcm = selected_pcm(energy, ion=ion)
    order = np.argsort(energy['idx'])
    pcm = source_pcm[order[np.searchsorted(energy['idx'][order], levels['idx'])]].astype("f4")
    for name in ("pcm", "ryd", "eV"):
        if name not in levels.dtype.names:
            continue
        expected = pcm if name == "pcm" else unit_cnvt.convert_units(pcm, "pcm", name)
        expected = expected.astype(levels.dtype[name])
        if not np.array_equal(levels[name], expected, equal_nan=True):
            raise ValueError(f"CHIANTI {name} disagrees with the energy policy for {ion}")
    return aligned


def validate_energy_bundle(group):
    """Check the complete atomic_data bundle and return energy plus metadata."""
    if ENERGY_DATASET not in group or "lev_chianti1102" not in group:
        raise ValueError("Missing CHIANTI level/energy bundle; regenerate atomic_data")
    aux, lev = group[ENERGY_DATASET], group["lev_chianti1102"]
    energy = aux[:]
    ion = lev.attrs.get('ion', '')
    if not re.fullmatch(r'\d{6}', str(ion)) or aux.attrs.get('ion') != ion:
        raise ValueError("Missing/inconsistent CHIANTI ion identity")
    metadata = dict(energy_schema=ENERGY_SCHEMA, **energy_policy_metadata(energy, ion=ion),
                    energy_sha256=energy_sha256(energy))
    source_hash = aux.attrs.get("source_elvlc_sha256", "")
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(c not in "0123456789abcdef" for c in source_hash)):
        raise ValueError("Missing/invalid CHIANTI source hash")
    metadata["source_elvlc_sha256"] = source_hash
    for ds in (aux, lev):
        if 'energy_overrides' in ds.attrs and 'energy_overrides' not in metadata:
            raise ValueError(f"Unexpected CHIANTI energy override: {ds.name}")
        for key, value in metadata.items():
            if ds.attrs.get(key) != value:
                raise ValueError(f"CHIANTI energy provenance mismatch: {ds.name}/{key}")
    validate_primary_energy(lev[:], energy, ion=ion, exact=True)
    return energy, metadata


def cdb_file_todf(filepath, columns, lev_trantype):
    """Read .elvlc levels; transition collection is outside XCOS scope."""
    if lev_trantype != "lev":
        raise ValueError("CFADB-XCOS collects energy levels only")
    ion = ion_from_filename(filepath)
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
            (72, 87),  # Read the full 15-character theoretical-energy field.
        ]
        df = pd.read_fwf(StringIO(data_text), colspecs=colspecs, header=None)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        energy = np.empty(len(df), dtype=ENERGY_DTYPE)
        energy['idx'] = df[0].to_numpy()
        energy['pcm_obs'] = df[6].to_numpy(dtype='f8')
        energy['pcm_theory'] = df[7].to_numpy(dtype='f8')
        df = df.drop(df.columns[[2, 6]], axis=1)
        df.columns = columns
        df['pcm'] = selected_pcm(energy, ion=ion)
        df['cfg'] = df['cfg'].str.replace(' ', '.', regex=False)
        df['L'] = quantbl.L_letter_to_num(df, col_name='L')
        df = df.astype({'idx': 'i4', 'cfg': 'string', '2S+1': 'i4', 'L': 'i4', 'J': 'f4', 'pcm': 'f4'})
        df['ryd'] = unit_cnvt.convert_units(df['pcm'], 'pcm', 'ryd')
        df['eV'] = unit_cnvt.convert_units(df['pcm'], 'pcm', 'eV')
        df = df.astype({'ryd': 'f4', 'eV': 'f4'})
        df.attrs['energy_records'] = energy
        df.attrs['ion'] = ion
        df.attrs['source_elvlc_sha256'] = hashlib.sha256(Path(filepath).read_bytes()).hexdigest()

    return df


def cdb_file_toh5(df, filename, levtran_type, save_path='./database/chianti1102'):
    """Write selected energies and their observed/theoretical provenance."""
    if levtran_type != "lev":
        raise ValueError("CFADB-XCOS stores energy levels only")
    ion = ion_from_filename(filename)
    if df.attrs.get('ion') != ion:
        raise ValueError("CHIANTI input ion does not match output filename")

    filepath = os.path.join(save_path, ion + '.h5')
    dataset_name= f'{levtran_type}_chianti1102'

    records = h5_tools.df_to_hdf5grp(df)
    if levtran_type == 'lev':
        energy = df.attrs['energy_records']
        validate_primary_energy(records, energy, ion=ion, exact=True)
        metadata = dict(cfadb_xcos_version=XCOS_VERSION, energy_schema=ENERGY_SCHEMA,
                        **energy_policy_metadata(energy, ion=ion),
                        energy_sha256=energy_sha256(energy),
                        source_elvlc_sha256=df.attrs['source_elvlc_sha256'])
    if not len(records):
        raise ValueError(f"Empty source table: {filename}")
    os.makedirs(save_path, exist_ok=True)
    with h5py.File(filepath, 'a') as h5f:
        grp = h5f
        if any(name.startswith('tran_') for name in grp):
            raise ValueError('Output contains transitions; use a fresh XCOS collection directory')
        h5f.attrs['cfadb_xcos_version'] = XCOS_VERSION
        if dataset_name in grp:
            del grp[dataset_name]

        dset = grp.create_dataset(dataset_name, data=records, compression="gzip")
        dset.attrs['ion'] = ion
        dset.attrs['database'] = ['chianti1102']
        dset.attrs['columns'] = df.columns.tolist()
        if levtran_type == 'lev':
            if ENERGY_DATASET in grp:
                del grp[ENERGY_DATASET]
            aux = grp.create_dataset(ENERGY_DATASET, data=energy, compression='gzip')
            aux.attrs['columns'] = list(energy.dtype.names)
            aux.attrs['units'] = 'cm^-1'
            aux.attrs['ion'] = ion
            aux.attrs['database'] = ['chianti1102']
            for ds in (dset, aux):
                for key, value in metadata.items():
                    ds.attrs[key] = value
    return Path(filepath)


def cdb_all_toh5(folder_path, save_path='./database/chianti1102'):
    """Convert regular ions; fail on errors rather than publishing partial data.

    Separate dielectronic satellite (*d) files are not regular ion tables and
    cannot share their six-digit IDs. They remain outside this collector's scope.
    Use a fresh output directory for a complete reproducible batch.
    """
    files = sorted(Path(folder_path).rglob('*.elvlc'))
    if not files:
        raise FileNotFoundError(f"No CHIANTI elvlc files in {folder_path}")
    counts = dict(level_files=0, skipped_satellite_files=0)
    for ep in files:
        if re.fullmatch(r'[a-z]+_\d+d', ep.stem):
            counts['skipped_satellite_files'] += 1
            continue
        if not re.fullmatch(r'[a-z]+_\d+', ep.stem):
            raise ValueError(f"Unsupported CHIANTI filename: {ep}")
        df = cdb_file_todf(ep, ['idx', 'cfg', '2S+1', 'L', 'J', 'pcm'], 'lev')
        cdb_file_toh5(df, ep.name, 'lev', save_path)
        counts['level_files'] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='CHIANTI 11.0.2 text database')
    parser.add_argument('--output', type=Path, default=Path(directory) / 'database/chianti1102')
    args = parser.parse_args()
    print(cdb_all_toh5(args.source, args.output))


if __name__ == "__main__":
    main()
