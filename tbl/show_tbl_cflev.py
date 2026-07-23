"""Generate aligned cross-database atomic energy-level tables and index maps.

Levels sharing the same spectroscopic identifiers are paired by minimizing their
energy differences when duplicate states occur. The resulting tables support
cross-database comparison and downstream transition analysis.
"""

import pandas as pd
import os
from itertools import permutations
from tbl import show_tbl_lev
from list import list_database
import h5py
from datacollect import h5_tools


def levmerged_db2_past (lenergy, ldname, unit):
    """Merge level tables by cfg, SL, and J.

    Duplicate states are paired by minimizing the summed squared energy
    differences.
    """
    dbname1 = ldname[0]
    dbname2 = ldname[1]

    data1 = lenergy[ldname.index(dbname1)].copy()
    data2 = lenergy[ldname.index(dbname2)].copy()

    data1 = data1.rename(columns={'idx': 'idx1', unit: unit+'1'})
    data2 = data2.rename(columns={'idx': 'idx2', unit: unit+'2'})

    merged_states = pd.merge(data2[['cfg', 'SL', 'J']], data1[['cfg', 'SL', 'J']], on=['cfg', 'SL', 'J'], how='outer')
    merged_states = merged_states.drop_duplicates()
    merged_states = merged_states.reset_index(drop=True)

    merged_data = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in [('cfg', 'object'), ('SL', 'object'), ('J', 'float64'), ('idx1', 'int64'), (unit+'1', 'float64'), ('idx2', 'int64'), (unit+'2', 'float64')]})

    for i in range(len(merged_states)):
        state = merged_states.iloc[i:i+1]
        everystate1 = pd.merge(state, data1, on=['cfg', 'SL', 'J'], how='inner')
        everystate2 = pd.merge(state, data2, on=['cfg', 'SL', 'J'], how='inner')

        l1 = len(everystate1)
        l2 = len(everystate2)

        if l1>1 or l2>1:
            if l1>=l2:
                ee = 0
                everystate11 = everystate1
                everystate22 = everystate2
                dbname22 = dbname2
                dbname11 = dbname1
            else:
                ee = 1
                everystate22 = everystate1
                everystate11 = everystate2
                dbname11 = dbname2
                dbname22 = dbname1

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
                        score = (everystate11.loc[c[i], dbname11+'_'+unit]-everystate22.loc[a[i], dbname22+'_'+unit])**2
                        score_list.append(score)
                    sumscore = sum(score_list)
                    if sumscore<min_sumscore:
                        min_sumscore = sumscore
                        min_a = a
                        min_c = c

            nnewline = pd.DataFrame(columns=['cfg', 'SL', 'J', 'idx1', unit+'1', 'idx2', unit+'2'])

            for i in range(dmin):
                newline11 = everystate11.iloc[min_c[i]].to_frame().T
                newline22 = everystate22.iloc[min_a[i]].to_frame().T

                if ee == 0:
                    newline = pd.merge(newline11, newline22, on=['cfg', 'SL', 'J'], how='inner')
                    newline1 = newline11
                else:
                    newline = pd.merge(newline22, newline11, on=['cfg', 'SL', 'J'], how='inner')
                    newline1 = newline22

                nnewline = pd.concat([nnewline, newline], ignore_index=True)

            merged_data = pd.concat([merged_data, nnewline], ignore_index=True)

        elif l1 == 1 and l2 == 1:
            newline = pd.merge(everystate1, everystate2, on=['cfg', 'SL', 'J'], how='inner')
            merged_data = pd.concat([merged_data, newline], ignore_index=True)
        else:
            continue

    merged_data = merged_data.sort_values(by=unit+'1')
    merged_data.reset_index(drop=True, inplace=True)

    return merged_data


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


def main():
    """Generate an example cross-database level table and its index map."""

    ion = '020605'
    dbname1 = 'chianti1001'
    dbname2 = 'spex306'

    unit = 'eV'

    ldbname = [dbname1, dbname2]

    dictory = os.path.dirname(os.path.abspath(__file__))

    directory = os.path.dirname(dictory)

    lenergy = []
    for i in range(len(ldbname)):
        file_path_atdata = os.path.join(directory, "atomic_data", ion+'.h5')
        dataset_path = '/lev_' + ldbname[i]
        data = show_tbl_lev.lev_df(file_path_atdata, dataset_path)
        data = data[['idx', 'cfg', '2S+1', 'L', 'J', unit]]
        print(data.dtypes)

        lenergy.append(data)

    lev_merged = levmerged_db2 (lenergy, ldbname, unit)
    print(lev_merged.dtypes)

    directory_d = os.path.join(directory, "data", 'seq'+ion[:2], ion, ldbname[0]+'_'+ldbname[1])
    if not os.path.exists(directory_d):
        os.makedirs(directory_d)

    file_path_save0 = os.path.join(directory_d, 'cflev_'+'map0.dat')

    with open(file_path_save0, "w", encoding="utf-8") as f:
        f.write(f"# ion: {ion}\n")
        f.write(f"# database1, database2: {ldbname[0]}, {ldbname[1]}\n")
        f.write(f"# unit: {unit}\n\n")

    lev_merged.to_csv(file_path_save0, index=False, sep='\t', mode='a', header=True)

    file_path_save1 = os.path.join(directory_d, 'map0.dat')
    with open(file_path_save1, "w", encoding="utf-8") as f:
        f.write(f"# ion: {ion}\n")
        f.write(f"# database1, database2: {ldbname[0]}, {ldbname[1]}\n\n")
    lev_merged[['idx1', 'idx2']].to_csv(file_path_save1, index=False, sep='\t', mode='a', header=True)


if __name__ == "__main__":
    main()
