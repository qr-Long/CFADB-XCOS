"""Read the distributed level tables and prepare pairs for matching."""
from pathlib import Path
from itertools import permutations
import json
import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parents[1]
ION_TYPE_PREFIX = {'h-like':'01', 'he-like':'02', 'li-like':'03', 'be-like':'04', 'ne-like':'10'}

def paper_parameters():
    return json.loads((ROOT/'paper_parameters.json').read_text())

def load_cfg_model(seq='li-like', cfg_model_path=None):
    from .encoder import load_cfg_ts_model
    path = Path(cfg_model_path) if cfg_model_path else ROOT/'models'/seq/'cfg_ts_model.pth'
    return load_cfg_ts_model(path)

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

def restore_level_records(cleaned, deleted, source_dtype, source_count):
    """Reconstruct the pre-clean level records, including their original order."""
    kept_count = 0 if cleaned is None else len(cleaned)
    deleted_count = 0 if deleted is None else len(deleted)
    if kept_count + deleted_count != source_count:
        raise ValueError("Kept and deleted row counts do not match the source")
    positions = (np.empty(0, dtype=np.uint64) if deleted is None
                 else deleted["original_row"])
    if (np.any(positions >= source_count) or
            len(np.unique(positions)) != len(positions)):
        raise ValueError("Invalid or duplicate original_row values")
    keep_mask = np.ones(source_count, dtype=bool)
    keep_mask[positions] = False
    restored = np.empty(source_count, dtype=source_dtype)
    for name in restored.dtype.names:
        archived_name = "original_cfg" if name == "cfg" else name
        if cleaned is not None:
            restored[name][keep_mask] = cleaned[archived_name]
        if deleted is not None:
            restored[name][positions] = deleted[archived_name]
    return restored


def restore_level_table(path, lev_path):
    """Return original records, treating absent retained/deleted tables as empty.

    Source metadata is present on either nonempty table. An originally empty
    source is recorded as an empty typed file attribute, not a dataset.
    """
    with h5py.File(path, "r") as h5f:
        cleaned_ds = h5f.get(lev_path)
        deleted_ds = h5f.get(lev_path + "_delete")
        if cleaned_ds is None and deleted_ds is None:
            return h5f.attrs["empty_level:" + lev_path]
        ds = cleaned_ds if cleaned_ds is not None else deleted_ds
        names = list(ds.attrs["source_field_names"])
        # Read field types from HDF5 itself, preserving string-encoding metadata.
        source_dtype = np.dtype({
            "names": names,
            "formats": [ds.dtype.fields["original_cfg" if n == "cfg" else n][0]
                        for n in names],
            "offsets": list(ds.attrs["source_field_offsets"]),
            "itemsize": int(ds.attrs["source_itemsize"]),
        })
        return restore_level_records(
            None if cleaned_ds is None else cleaned_ds[:],
            None if deleted_ds is None else deleted_ds[:], source_dtype,
            int(ds.attrs["source_row_count"]),
        )

def levmerged_db2 (lenergy, ldname, unit):
    """Merge level tables by cfg, 2S+1, L, and J.

    Duplicate states are paired by minimizing the summed squared energy
    differences.
    """
    dbname1 = ldname[0]
    dbname2 = ldname[1]

    data1 = lenergy[ldname.index(dbname1)].copy()
    data2 = lenergy[ldname.index(dbname2)].copy()

    data1 = data1[['cfg', '2S+1', 'L', 'J', 'idx', unit]]
    data2 = data2[['cfg', '2S+1', 'L', 'J', 'idx', unit]]

    data1 = data1.rename(columns={'idx': 'idx1', unit: unit+'1'})
    data2 = data2.rename(columns={'idx': 'idx2', unit: unit+'2'})

    merged_states = pd.merge(data2[['cfg', '2S+1', 'L', 'J']], data1[['cfg', '2S+1', 'L', 'J']], on=['cfg', '2S+1', 'L', 'J'], how='outer')
    merged_states = merged_states.drop_duplicates()
    merged_states = merged_states.reset_index(drop=True)

    merged_data = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in [('cfg', 'string'), ('2S+1', 'i4'), ('L', 'i4'), ('J', 'f4'), ('idx1', 'i4'), (unit+'1', 'f4'), ('idx2', 'i4'), (unit+'2', 'f4')]})

    for i in range(len(merged_states)):
        state = merged_states.iloc[i:i+1]
        everystate1 = pd.merge(state, data1, on=['cfg', '2S+1', 'L', 'J'], how='inner')
        everystate2 = pd.merge(state, data2, on=['cfg', '2S+1', 'L', 'J'], how='inner')

        l1 = len(everystate1)
        l2 = len(everystate2)

        if l1>1 or l2>1:
            if l1>=l2:
                ee = 0
                everystate11 = everystate1
                everystate22 = everystate2
                end11 = '1'
                end22 = '2'
            else:
                ee = 1
                everystate22 = everystate1
                everystate11 = everystate2
                end11 = '2'
                end22 = '1'

            dmin = min(l1, l2)
            dmax = max(l1, l2)

            ddata1 = list(range(dmax))
            ddata2 = list(range(dmin))

            result_combinations = []
            current_combination = []

            def find_combinations(ddata1, dmin, result_combinations, current_combination, start_index):
                """Enumerate index combinations of the requested length."""
                if len(current_combination) == dmin:
                    result_combinations.append(current_combination[:])
                    return

                for i in range(start_index, len(ddata1)):
                    current_combination.append(ddata1[i])
                    find_combinations(ddata1, dmin, result_combinations, current_combination, i+1)
                    current_combination.pop()

            find_combinations(ddata1, dmin, result_combinations, current_combination, 0)

            all_permutations = list(permutations(ddata2))

            min_sumscore = float('inf')

            for a in all_permutations:
                for c in result_combinations:
                    score_list = []
                    for i in range(dmin):
                        score = (everystate11.loc[c[i], unit+end11]-everystate22.loc[a[i], unit+end22])**2
                        score_list.append(score)
                    sumscore = sum(score_list)
                    if sumscore<min_sumscore:
                        min_sumscore = sumscore
                        min_a = a
                        min_c = c

            nnewline = pd.DataFrame(columns=['cfg', '2S+1', 'L', 'J', 'idx1', unit+'1', 'idx2', unit+'2'])

            for i in range(dmin):
                newline11 = everystate11.iloc[min_c[i]].to_frame().T
                newline22 = everystate22.iloc[min_a[i]].to_frame().T

                if ee == 0:
                    newline = pd.merge(newline11, newline22, on=['cfg', '2S+1', 'L', 'J'], how='inner')
                    newline1 = newline11
                else:
                    newline = pd.merge(newline22, newline11, on=['cfg', '2S+1', 'L', 'J'], how='inner')
                    newline1 = newline22

                nnewline = pd.concat([nnewline, newline], ignore_index=True)

            merged_data = pd.concat([merged_data, nnewline], ignore_index=True)

        elif l1 == 1 and l2 == 1:
            newline = pd.merge(everystate1, everystate2, on=['cfg', '2S+1', 'L', 'J'], how='inner')
            merged_data = pd.concat([merged_data, newline], ignore_index=True)
        else:
            continue

    merged_data = merged_data.sort_values(by=unit+'1')
    merged_data.reset_index(drop=True, inplace=True)

    return merged_data


def load_pair_from_h5(ion, db1, db2, *, clean_data_dir=ROOT/'clean_data', cfg_model=None, seq='li-like', cfg_model_path=None, thr_view_root=None):
    from .encoder import process_cfg
    path = Path(clean_data_dir)/f'{ion}.h5'
    model_path = Path(cfg_model_path) if cfg_model_path else ROOT/'models'/seq/'cfg_ts_model.pth'
    model = cfg_model if cfg_model is not None else load_cfg_model(seq, model_path)
    left = process_cfg(ds_to_df(path, 'lev_'+db1), model, vocab_path=model_path.with_name('cfg_vocab.pkl'))
    right = process_cfg(ds_to_df(path, 'lev_'+db2), model, vocab_path=model_path.with_name('cfg_vocab.pkl'))
    if left.empty or right.empty:
        raise ValueError('Both databases must contain retained levels for this ion.')
    aligned = levmerged_db2([left, right], [1, 2], 'eV')
    reference = aligned[['idx1', 'idx2']].drop_duplicates().reset_index(drop=True)
    def info(df):
        cols = [c for c in ['idx','cfg','2S+1','L','J','pi','eV','pcm','ryd'] if c in df]
        result = df[cols].copy().reset_index(drop=True)
        result.insert(0, 'row', np.arange(len(result)))
        return result
    return dict(ion=str(ion), db1=db1, db2=db2, h5_path=path,
                out_dir=Path(thr_view_root) if thr_view_root is not None else ROOT/"viewer/output", lev_df1=left, lev_df2=right, map_df=reference, info1=info(left), info2=info(right))

def available_cases(clean_data_dir=ROOT/'clean_data', sequences=None, databases=None):
    from itertools import combinations
    settings = paper_parameters()
    sequences = sequences or list(ION_TYPE_PREFIX)
    databases = databases or settings['databases']
    for seq in sequences:
        for path in sorted(Path(clean_data_dir).glob(ION_TYPE_PREFIX[seq]+'*.h5')):
            if int(path.stem[2:4]) > settings['max_atomic_number']:
                continue
            with h5py.File(path, 'r') as f:
                dbs = [d for d in databases if 'lev_'+d in f and len(f['lev_'+d]) > 0]
            for a,b in combinations(dbs,2):
                yield seq,path.stem,a,b
