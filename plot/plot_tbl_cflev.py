"""Plot and compare atomic energy levels from two or more databases.

The module groups levels by electron configuration and energy range, visualizes
energy distributions and category overlap, and generates pairwise energy and
energy-difference plots for cross-database level comparisons.
"""
import os
import pandas as pd
import sys

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(directory)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from util import quantbl
from tbl import show_tbl_lev
import re


def seq_ele_sort (prefix_num):
    """Return configuration fields and grouping patterns for an isoelectronic sequence."""

    if prefix_num == 2:
        cfg_part = ['cfg1', 'cfg2']
        set_name = ['1s2', '2s2', '2p2']
        lsort_name = ['1s.', '2s.', '2p.']
        return cfg_part, set_name, lsort_name

    elif prefix_num == 3:
        cfg_part = ['cfg1', 'cfg2', 'cfg3']
        set_name = ['1s.2s2', '1s.2p2']
        lsort_name = ['1s2.', '1s.2s.', '1s.2p.']
        return cfg_part, set_name, lsort_name
    elif prefix_num == 6:
        cfg_part = ['cfg1', 'cfg2', 'cfg3']
        set_name = ['2s2.2p2', '2s.2p3', '2p4', '1s.2s2.2p3', '1s.2s.2p4', '1s.2p5']
        lsort_name = ['2s2.2p.', '2s.2p2.', '1s.2p2.']
        return cfg_part, set_name, lsort_name
    else:
        print(f'The sequence ({prefix_num}) does not exist. ')


def clean_cfg_sha(s):
    """Normalize the final orbital label used for configuration-category grouping."""
    if s[-1].isalpha():
        return s[:-1] + 'l'
    else:
        return s


def extract_leading_digits(s):
    """Extract the leading principal quantum number from an orbital label."""
    digits = ''
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            break
    return digits


def levelnum_table (ldname, lenergy, cfg_part, set_name, lsort_name, unit):
    """Build the configuration-category count table and reorder levels for plotting."""
    lchart = []

    for i in range(len(ldname)):
        data = lenergy[i]
        dname = ldname[i]
        chart = pd.DataFrame(columns=['category', dname])
        chartl = []

        data_columns = data.columns
        data_new = pd.DataFrame(columns=data_columns)
        used_index = set()

        for setc in range(len(set_name)):

            row_ssetc = data
            for ssetc in range(len(set_name[setc].split('.'))):
                row_ssetc = row_ssetc[(row_ssetc[cfg_part[ssetc]] == set_name[setc].split('.')[ssetc])]
            chartl.append(len(row_ssetc))
            data_new = pd.concat([data_new, row_ssetc], ignore_index=True)
            used_index.update(row_ssetc.index)

        unset_name = []

        for bc in range(len(lsort_name)):
            row_sc = data
            nums_part = len(lsort_name[bc].split('.'))

            for sc in range(nums_part-1):
                row_sc = row_sc[(row_sc[cfg_part[sc]] == lsort_name[bc].split('.')[sc])]

            unique_bc = row_sc[cfg_part[nums_part-1]].unique()

            aln_bc = [extract_leading_digits(s) for s in unique_bc]
            aln_bc = [n for n in aln_bc if n is not None and n.isdigit()]
            aln_bc = list(set(aln_bc))
            aln_bc = sorted(aln_bc, key=int)
            print('aln_bc:', aln_bc)

            for n in aln_bc:
                print('n:', n)
                unset_name.append(lsort_name[bc]+n)
                row_sc1 = row_sc[row_sc[cfg_part[nums_part-1]].str.startswith(n)]
                n_bc = len(row_sc1)
                print('n_bc:', n_bc)
                chartl.append(n_bc)

                row_sc1 = row_sc1.sort_values(by=unit+str(i+1))
                data_new = pd.concat([data_new, row_sc1], ignore_index=True)
                used_index.update(row_sc1.index)

        other_data = data[~data.index.isin(used_index)]
        other_data = other_data.sort_values(by=unit+str(i+1))
        data_new = pd.concat([data_new, other_data], ignore_index=True)

        if  'level_idx'+str(i+1) not in data_new.columns:
            lenergy[i] = data_new.reset_index().rename(columns={'index': 'level_idx'+str(i+1)})

        print(chartl)
        print(sum(chartl))
        chart['category'] = set_name + unset_name + ['other'] + ['total']
        chart[dname] = chartl + [len(data)-sum(chartl)] + [len(data)]
        lchart.append(chart)

    sum_table = lchart[0]
    for i in range(1, len(lchart)):

        sum_table['original_index'] = sum_table.index

        sum_table = pd.merge(sum_table, lchart[i], on='category', how='outer')

        sum_table = sum_table.sort_values(by='original_index').drop(columns='original_index')

        sum_table = sum_table.reset_index(drop=True)

    for column in ldname:
        sum_table[column] = sum_table[column].fillna(0).astype(int)

    sum_table0 = sum_table.loc[:len(set_name)-1]

    for i in range(len(lsort_name)):
        sortpart = sum_table[sum_table['category'].str[:-1] == lsort_name[i]]

        sortpart['last_char'] = sortpart['category'].apply(lambda x: int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else -1)

        sortpart['category'] = sortpart['category']+"l"

        numeric_rows = sortpart.sort_values(by='last_char', key=lambda x: x.astype(int))
        sum_table0 = pd.concat([sum_table0, numeric_rows], ignore_index=True)

    sum_table = pd.concat([sum_table0, sum_table[sum_table['category'] == 'other'],
                           sum_table[sum_table['category'] == 'total']], ignore_index=True)

    return sum_table


def levelnum_table_range (sum_table, lenergy, ldname, unit):
    """Add minimum and maximum energies for each category and database."""
    for i in range(len(ldname)):
        dname = ldname[i]
        data = lenergy[i]
        sum_table['min_'+unit+str(i+1)] = float('nan')
        sum_table['max_'+unit+str(i+1)] = float('nan')
        for j in range(len(sum_table)-1):
            shaca_data = data.loc[data['cfg_c'] == sum_table.loc[j, 'category'], unit+str(i+1)]

            sum_table.loc[j, 'min_'+unit+str(i+1)] = shaca_data.min()
            sum_table.loc[j, 'max_'+unit+str(i+1)] = shaca_data.max()
    return sum_table


def slevelnum_table (dbname1, dbname2, ldname, lenergy, sum_table, unit):
    """Merge categories whose energy ranges overlap in a two-database comparison."""
    last_n = 0

    sum_table_nototal = sum_table.drop(sum_table.index[-1]).copy()
    sum_table_nototal = sum_table_nototal[(sum_table_nototal[dbname1] != 0) | (sum_table_nototal[dbname2] != 0)]

    sum_table_nototal['min_'+unit] = sum_table_nototal[['min_'+unit+str(1), 'min_'+unit+str(2)]].min(axis=1)
    sum_table_nototal['max_'+unit] = sum_table_nototal[['max_'+unit+str(1), 'max_'+unit+str(2)]].max(axis=1)

    sum_table_nototal = sum_table_nototal.sort_values(by='min_'+unit).reset_index(drop=True)
    line_num = len(sum_table_nototal)

    lall_ca = sum_table_nototal['category'].drop_duplicates().to_frame()

    data1 = lenergy[ldname.index(dbname1)].copy()
    data2 = lenergy[ldname.index(dbname2)].copy()

    lshaca = []
    lsha_num1 = []
    lsha_num2 = []

    i = 0

    while i < line_num-1:

        shaca = sum_table_nototal.loc[i, 'category']
        sha_num1 = len(data1[data1['cfg_c'] == shaca])
        sha_num2 = len(data2[data2['cfg_c'] == shaca])

        sha_max_E = sum_table_nototal.loc[i, 'max_'+unit]

        if sha_max_E < sum_table_nototal.loc[i+1, 'min_'+unit]:
            lshaca.append(shaca)
            lsha_num1.append(sha_num1)
            lsha_num2.append(sha_num2)
            i += 1

        else:
            print('sum_table_nototal:', sum_table_nototal)
            while i < line_num-1 and sha_max_E > sum_table_nototal.loc[i+1, 'min_'+unit]:
                print(f"Overlap scan progress: {i}/10")
                more_shaca = sum_table_nototal.loc[i+1, 'category']
                shaca = shaca + '_' + more_shaca
                sha_num1 += len(data1[data1['cfg_c'] == more_shaca])
                sha_num2 += len(data2[data2['cfg_c'] == more_shaca])

                if sum_table_nototal.loc[i+1, 'max_'+unit] > sha_max_E:
                    sha_max_E = sum_table_nototal.loc[i+1, 'max_'+unit]

                i += 1
                print('sha_max_E:', sha_max_E)
                print('sum_table_nototal.loc[i+1, min_+unit]:', sum_table_nototal.loc[i+1, 'min_'+unit])
            i += 1

            lshaca.append(shaca)
            lsha_num1.append(sha_num1)
            lsha_num2.append(sha_num2)
            if i == line_num:
                last_n = 1

    if last_n != 1:
        lastca_name = sum_table_nototal.loc[line_num-1, 'category']
        if sum_table_nototal.loc[line_num-1, 'category'] == 'other':
            lastsha_num1 = len(data1[~data1['cfg_c'].isin(lall_ca)])
            lastsha_num2 = len(data2[~data2['cfg_c'].isin(lall_ca)])
        else:
            lastsha_num1 = len(data1[data1['cfg_c'] == lastca_name])
            lastsha_num2 = len(data2[data2['cfg_c'] == lastca_name])

        if sum_table_nototal.loc[line_num-2, 'max_'+unit] < sum_table_nototal.loc[line_num-1, 'min_'+unit]:
            shaca = lastca_name
            sha_num1 = lastsha_num1
            sha_num2 = lastsha_num2

            lshaca.append(shaca)
            lsha_num1.append(sha_num1)
            lsha_num2.append(sha_num2)
        else:
            shaca = shaca + '_' + lastca_name
            sha_num1 += lastsha_num1
            sha_num2 += lastsha_num2

            lshaca.append(shaca)
            lsha_num1.append(sha_num1)
            lsha_num2.append(sha_num2)

    sha_table = pd.DataFrame({'category': lshaca, dbname1: lsha_num1, dbname2: lsha_num2})

    return sha_table


def levcate(ion, ldname, lenergy, cfg_part, set_name, lsort_name, unit):
    """Run the default configuration and energy-range classification workflow."""
    dbname0 = ldname[0]
    dbname1 = ldname[1]

    sum_table = levelnum_table(ldname, lenergy, cfg_part, set_name, lsort_name, unit)

    try:
        sum_table = sum_table.drop('last_char', axis=1)
    except KeyError:
        print("'last_char' was not found and could not be removed")

    mask = (sum_table['category'] == 'other') & (sum_table.drop('category', axis=1).ne(0).any(axis=1))
    result = sum_table[mask]
    if len(result) != 0:
        print(ion)
        print(sum_table)

    sum_table1 = levelnum_table_range (sum_table.copy(), lenergy, ldname, unit)

    sha_table = slevelnum_table(dbname0, dbname1, ldname, lenergy, sum_table1, unit)
    return sum_table, sha_table


def _get_category_data(data, category, sum_table, col):
    """Select and sort levels by category name from the summary table."""
    if category == 'other':
        known_categories = set(sum_table.loc[sum_table['category'] != 'total', 'category'])
        known_categories.discard('other')
        cat_data = data.loc[~data['cfg_c'].isin(known_categories)].copy()
    else:
        cat_data = data.loc[data['cfg_c'] == category].copy()
    return cat_data.sort_values(by=col).reset_index(drop=True)


def _make_gap_compressor(ldname, lenergy, sum_table, unit, cut_big_gap=False, max_gap=10):
    """Create a transform that compresses energy gaps shared by all databases."""
    def identity(y):
        return y

    if not cut_big_gap:
        return identity, None

    intervals = []
    for i in range(len(ldname)):
        dname = ldname[i]
        col = unit + str(i + 1)
        data = lenergy[i]
        for j in range(len(sum_table) - 1):
            category = sum_table.loc[j, 'category']
            level_num = int(sum_table.loc[j, dname])
            if level_num != 0:
                cat_data = _get_category_data(data, category, sum_table, col)
                ys = pd.to_numeric(cat_data[col], errors='coerce').dropna()
                if len(ys) > 0:
                    intervals.append((float(ys.min()), float(ys.max())))

    if len(intervals) < 2:
        return identity, None

    intervals.sort(key=lambda x: x[0])
    merged = []
    for y0, y1 in intervals:
        if not merged or y0 > merged[-1][1]:
            merged.append([y0, y1])
        else:
            merged[-1][1] = max(merged[-1][1], y1)

    steps = []
    for k in range(len(merged) - 1):
        gap_start = merged[k][1]
        gap_end = merged[k + 1][0]
        gap = gap_end - gap_start
        if gap > max_gap:
            steps.append((gap_start, gap_end, gap))

    if len(steps) == 0:
        return identity, None

    def compress_scalar(y):
        if pd.isna(y):
            return y
        y = float(y)
        shift = 0.0
        for gap_start, gap_end, gap in steps:
            if y <= gap_start:
                break
            if y < gap_end:
                return gap_start - shift + (y - gap_start) * max_gap / gap
            shift += gap - max_gap
        return y - shift

    def compress(y):
        if hasattr(y, 'apply'):
            return y.apply(compress_scalar)
        if isinstance(y, (list, tuple)):
            return [compress_scalar(v) for v in y]
        return compress_scalar(y)

    return compress, steps


def _set_compressed_energy_ticks(axes, transform_y, ldname, lenergy, unit, nticks=8):
    """Display physical energy values on a compressed energy coordinate."""
    vals = []
    for i in range(len(ldname)):
        col = unit + str(i + 1)
        vals.extend(pd.to_numeric(lenergy[i][col], errors='coerce').dropna().tolist())

    if len(vals) == 0:
        return

    ymin = min(vals)
    ymax = max(vals)
    if ymin == ymax:
        ticks = [ymin]
    else:
        ticks = [ymin + (ymax - ymin) * k / (nticks - 1) for k in range(nticks)]

    tick_pos = [transform_y(t) for t in ticks]
    tick_lab = [f'{t:.6g}' for t in ticks]
    for ax in axes:
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)


def plt_levelnum (ldname, lenergy, sum_table, n_colors, plt_path, unit, cut_big_gap=False, max_gap=10):
    """Plot category-resolved energy levels for one ion across databases.

When enabled, shared empty energy intervals larger than ``max_gap`` are
compressed to the specified display length.
"""
    fig1, axes = plt.subplots(1, len(ldname), figsize=(len(ldname) * 2, 10), sharey=True)
    axes = list(axes) if hasattr(axes, '__len__') else [axes]

    palette = sns.color_palette("husl", max(n_colors, len(sum_table)-1))
    colors = palette.as_hex()
    transform_y, compress_steps = _make_gap_compressor(ldname, lenergy, sum_table, unit, cut_big_gap, max_gap)
    y_ranges = []

    for i in range(len(ldname)):
        dname = ldname[i]
        data = lenergy[i]
        col = unit + str(i+1)
        vals = pd.to_numeric(data[col], errors='coerce').dropna()
        if len(vals) > 0:
            y_ranges.append((transform_y(vals.min()), transform_y(vals.max())))

        prev_ymax = None
        for j in range(len(sum_table)-1):
            category = sum_table.loc[j, 'category']
            level_num = int(sum_table.loc[j, dname])
            if level_num == 0:
                continue

            cat_data = _get_category_data(data, category, sum_table, col)
            ys = pd.to_numeric(cat_data[col], errors='coerce').dropna()
            if len(ys) == 0:
                print(f"Warning: {dname} category {category} has count {level_num} but no matched rows.")
                continue

            y0, y1 = float(ys.min()), float(ys.max())
            axes[i].hlines(transform_y(ys), 0, 1, colors[j], linewidth=1.1)
            axes[i].axhspan(transform_y(y0), transform_y(y1), color=colors[j], alpha=0.15, xmin=0.05, xmax=0.95)

            if prev_ymax is not None:
                axes[i].hlines(transform_y(prev_ymax), 0, 1, colors='k', linestyles='--', linewidth=1)
                axes[i].hlines(transform_y(y0), 0, 1, colors='k', linestyles='--', linewidth=1)
            prev_ymax = y1

        axes[i].set_xticks([0.5])
        axes[i].set_xticklabels([dname])

        if i > 0:
            axes[i].tick_params(left=False, labelleft=False)
            axes[i].set_ylabel('')
            axes[i].spines['left'].set_visible(False)

    nonzero_counts = (sum_table.drop('category', axis=1) != 0).sum()
    maxno0_cname = nonzero_counts.idxmax()
    order = ldname.index(maxno0_cname) + 1
    ref_data = lenergy[order-1]
    ref_col = unit + str(order)
    maxno0 = sum_table[(sum_table[maxno0_cname] != 0) & (sum_table['category'] != 'total')]

    for row_idx, row in maxno0.iterrows():
        category = row['category']
        cat_data = _get_category_data(ref_data, category, sum_table, ref_col)
        ys = pd.to_numeric(cat_data[ref_col], errors='coerce').dropna()
        if len(ys) == 0:
            continue
        y_text = (float(ys.min()) + float(ys.max())) / 2
        axes[-1].text(1.1, transform_y(y_text), category, va='center', ha='left', color=colors[row_idx], fontsize=12)

    axes[0].set_ylabel('Energy ('+unit+')')
    plt.subplots_adjust(wspace=-0.01)

    if compress_steps is not None and len(y_ranges) > 0:
        ymin = min([r[0] for r in y_ranges])
        ymax = max([r[1] for r in y_ranges])
        for ax in axes:
            ax.set_ylim(ymin, ymax)
        _set_compressed_energy_ticks(axes, transform_y, ldname, lenergy, unit)
    elif len(y_ranges) >= 2:
        sorted_y_ranges = sorted(y_ranges, key=lambda x: x[1] - x[0], reverse=True)
        max_y_range = sorted_y_ranges[0]
        second_max_y_range = sorted_y_ranges[1]
        if max_y_range[1] > 2*second_max_y_range[1]:
            for i in range(len(ldname)):
                if y_ranges[i] != sorted_y_ranges[0]:
                    axes[i].set_ylim(0, second_max_y_range[1])

    fig1.subplots_adjust(left=0.2, right=0.75, top=0.95, bottom=0.05)
    plt_path_lev = os.path.join(plt_path, ('Evsdb_num'+'.png'))
    plt.savefig(plt_path_lev, dpi=200)
    plt.show()


def plt_leveloverlap (ldname, lenergy, sum_table, n_colors, plt_path, unit, cut_big_gap=False, max_gap=10):
    """Plot category energy ranges to show overlap across databases.

When enabled, shared empty energy intervals larger than ``max_gap`` are
compressed to the specified display length.
"""
    fig2, axes2 = plt.subplots(1, len(ldname), figsize=(len(ldname) * 2, 10), sharey=True)
    axes2 = list(axes2) if hasattr(axes2, '__len__') else [axes2]

    palette = sns.color_palette("husl", max(n_colors, len(sum_table)-1))
    colors = palette.as_hex()
    transform_y, compress_steps = _make_gap_compressor(ldname, lenergy, sum_table, unit, cut_big_gap, max_gap)
    y_ranges = []

    for i in range(len(ldname)):
        dname = ldname[i]
        data = lenergy[i]
        col = unit + str(i+1)
        vals = pd.to_numeric(data[col], errors='coerce').dropna()
        if len(vals) > 0:
            y_ranges.append((transform_y(vals.min()), transform_y(vals.max())))

        for j in range(len(sum_table)-1):
            category = sum_table.loc[j, 'category']
            level_num = int(sum_table.loc[j, dname])
            if level_num == 0:
                continue

            cat_data = _get_category_data(data, category, sum_table, col)
            ys = pd.to_numeric(cat_data[col], errors='coerce').dropna()
            if len(ys) == 0:
                print(f"Warning: {dname} category {category} has count {level_num} but no matched rows.")
                continue

            axes2[i].axhspan(transform_y(float(ys.min())), transform_y(float(ys.max())), color=colors[j], alpha=0.35, xmin=0.05, xmax=0.95)

        axes2[i].set_xticks([0.5])
        axes2[i].set_xticklabels([dname])

        if i > 0:
            axes2[i].tick_params(left=False, labelleft=False)
            axes2[i].set_ylabel('')
            axes2[i].spines['left'].set_visible(False)

    nonzero_counts = (sum_table.drop('category', axis=1) != 0).sum()
    maxno0_cname = nonzero_counts.idxmax()
    order = ldname.index(maxno0_cname) + 1
    ref_data = lenergy[order-1]
    ref_col = unit + str(order)
    maxno0 = sum_table[(sum_table[maxno0_cname] != 0) & (sum_table['category'] != 'total')]

    for row_idx, row in maxno0.iterrows():
        category = row['category']
        cat_data = _get_category_data(ref_data, category, sum_table, ref_col)
        ys = pd.to_numeric(cat_data[ref_col], errors='coerce').dropna()
        if len(ys) == 0:
            continue
        y_text = (float(ys.min()) + float(ys.max())) / 2
        axes2[-1].text(1.1, transform_y(y_text), category, va='center', ha='left', color=colors[row_idx], fontsize=12)

    axes2[0].set_ylabel('Energy ('+unit+')')
    plt.subplots_adjust(wspace=-0.01)

    if compress_steps is not None and len(y_ranges) > 0:
        ymin = min([r[0] for r in y_ranges])
        ymax = max([r[1] for r in y_ranges])
        for ax in axes2:
            ax.set_ylim(ymin, ymax)
        _set_compressed_energy_ticks(axes2, transform_y, ldname, lenergy, unit)
    elif len(y_ranges) >= 2:
        sorted_y_ranges = sorted(y_ranges, key=lambda x: x[1] - x[0], reverse=True)
        max_y_range = sorted_y_ranges[0]
        second_max_y_range = sorted_y_ranges[1]
        if max_y_range[1] > 2*second_max_y_range[1]:
            for i in range(len(ldname)):
                if y_ranges[i] != sorted_y_ranges[0]:
                    axes2[i].set_ylim(0, second_max_y_range[1])

    fig2.subplots_adjust(left=0.2, right=0.75, top=0.95, bottom=0.05)
    plt_path_lev = os.path.join(plt_path, ('Evsdb_lap'+'.png'))
    plt.savefig(plt_path_lev, dpi=200)


def plt_slevelnum_table_EdElev_hold (filename, nonan, shaca, dbname1, dbname2, unit):
    '''plt E vs lev'''

    x = nonan['state_math']
    y1 = nonan[unit+'1']
    y2 = nonan[unit+'2']
    y3 = nonan["d_"+unit]

    ture_num = len(x)

    if len(x)>0 and len(y1)>0 and len(y2)>0:
        if ture_num < 50:
            size = 6.5
        else:
            size = ture_num * 0.125

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(size, 5), sharex=True)

        ax1.set_position([0.1, 0.5, 0.8, 0.4])
        ax1.scatter(x, y1, label=dbname1, color=(1.0, 0.4, 0.0))
        ax1.scatter(x, y2, label=dbname2, color='g')
        ax1.set_ylabel('E '+unit)
        ax1.legend()

        ax2.set_position([0.1, 0.1, 0.8, 0.4])
        y3_mean = y3.mean()

        ax2.scatter(x[ y3 >= y3_mean], y3[ y3 >= y3_mean], color='r')
        ax2.scatter(x[ y3 < y3_mean], y3[ y3 < y3_mean], color='b')

        ax2.axhline(y=y3_mean, color='black', linestyle='--')

        ax2.text(2, y3_mean, f'Mean={y3_mean:.2f}', ha='right', va='center', backgroundcolor='white')

        ax2.set_ylabel(r'$\Delta E$ '+unit)

        plt.xticks(rotation=90)

        fig.subplots_adjust(hspace=0)

        fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.31)

        plt_path_lev = os.path.join(filename, ('Evslev_'+'{}.png').format(shaca))
        plt.savefig(plt_path_lev, dpi=200)


def plt_slevelnum_table_EdElev(filename, nonan, shaca, dbname1, dbname2, unit):
    """Plot energy against the level index in a single panel."""

    x = nonan['state_math']
    y1 = nonan[unit+'1']
    y2 = nonan[unit+'2']

    ture_num = len(x)

    if len(x) > 0 and len(y1) > 0 and len(y2) > 0:
        if ture_num < 50:
            size = 4
        else:
            size = ture_num * 0.125

        fig, ax1 = plt.subplots(1, 1, figsize=(size, 3.3))

        ax1.scatter(x, y1, label=dbname1, color=(1.0, 0.4, 0.0))
        ax1.scatter(x, y2, label=dbname2, color='g')
        ax1.set_ylabel('E ' + unit)
        ax1.legend()

        plt.xticks(rotation=90)

        fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.31)

        plt_path_lev = os.path.join(filename, f"Evslev_{shaca}.png")
        plt.savefig(plt_path_lev, dpi=200)


def plt_slevelnum_table_EdE_ (filename, nonan, shaca, unit):
    """Plot energy differences against the reference-database energy and fit a trend."""
    num2 = len(nonan)

    degree = 1

    x = nonan[unit+'1']
    y = nonan["d_"+unit]

    def func1(x, a, b):
        return a * x + b

    if len(x)>0 and len(y)>0:

        fig, ax = plt.subplots(figsize=(7, 4))

        if num2 < 20:

            y_mean = y.mean()
            ax.scatter(x[ y >= y_mean], y[ y >= y_mean], color='r')
            ax.scatter(x[ y < y_mean], y[ y < y_mean], color='b')

            ax.axhline(y=y_mean, color='black', linestyle='--')

            ax.text(x.mean(), y_mean, f'Mean={y_mean:.2f}', ha='right', va='center', backgroundcolor='white')
        else:

            popt, pcov = curve_fit(eval(f'func{degree}'), x, y)
            y_pre = eval(f'func{degree}')(x, *popt)
            ax.scatter(x[ y >= y_pre], y[ y >= y_pre], color='r')
            ax.scatter(x[ y < y_pre], y[ y < y_pre], color='b')

            ax.plot(x, func1(x, *popt), 'k--', label=f'Fit: y = %.2fx + %.2f' % (popt[0], popt[1]))
            ax.legend()

        ax.set_xlabel('E$_1$ '+unit)
        ax.set_ylabel(r'$\Delta$E '+unit)

        fig.subplots_adjust(hspace=0)

        fig.subplots_adjust(left=0.13, right=0.92)

        plt_path_lev = os.path.join(filename, ('dEvsE_'+'{}.png').format(shaca))
        plt.savefig(plt_path_lev, dpi=200)


def cfg_mathform(cfg_str):
    """Convert a configuration string to a Matplotlib-compatible mathematical label."""

    parts = cfg_str.split('.')

    for i in range(len(parts)):
        part = parts[i]

        match = re.search(r'(\d*[A-Za-z]+)(\d+)$', part)
        if match:
            prefix = match.group(1)
            number = match.group(2)

            parts[i] = prefix + r'$^{' + number + r'}$'

    return ''.join(parts)


def lev_add_state (df):
    '''add state '1s.2s_3S1' and state math formation '1s2s ^3S_1 in a DaraFrame' '''
    df['state'] = df['cfg']+"_"+df['2S+1'].astype(str)+df['L'].map(quantbl.L_num2letter)+"_"+df['J'].astype(str)

    df['state_math'] = df['cfg'].apply(cfg_mathform)+" "+((df['2S+1'].astype(str) + df['L'].map(quantbl.L_num2letter)).apply(lambda x: re.sub(r'^(\d+)([A-Za-z]+)', r'$^\1$\2', x))+ "$_{" + df['J'].apply(lambda x: str(int(x)) if float(x).is_integer() else str(x)) + "}$")

    return df


def plt_user_E_ca (dbname1, dbname2, merged_data, luser_E, user_E_name, path, unit, Elev=1, EdE=1):
    """Generate two-panel comparison plots for user-defined energy ranges."""
    for i in range(len(luser_E)):

        range_rows = merged_data[(merged_data[unit+user_E_name] >= luser_E[i][0]) & (merged_data[unit+user_E_name] <= luser_E[i][1])]

        range_rows = range_rows.reset_index(drop=True)

        range_rows = lev_add_state(range_rows)

        range_rows["d_"+unit] = range_rows[unit+'2'] - range_rows[unit+'1']

        if Elev == 1:
            plt_slevelnum_table_EdElev (path, range_rows, 'E'+str(luser_E[i]), dbname1, dbname2, unit)
        if EdE == 1:
            plt_slevelnum_table_EdE_ (path, range_rows, 'E'+str(luser_E[i]), unit)


def plt_user_cfg_ca (dbname1, dbname2, merged_data, luser_cfg, path, unit, Elev=1, EdE=1):
    """Generate two-panel plots for user-selected configuration and term groups."""
    for i in range(len(luser_cfg)):
        lrow = []
        for j in range(len(luser_cfg[i])):

            row = merged_data[(merged_data['cfg'] == luser_cfg[i][j][0]) & (merged_data['2S+1'] == luser_cfg[i][j][1]) & (merged_data['L'] == luser_cfg[i][j][2]) & (merged_data['J'] == luser_cfg[i][j][3])]

            lrow.append(row)

        rows = pd.concat(lrow, ignore_index=True)
        rows = rows.reset_index(drop=True)
        rows = lev_add_state(rows)

        rows["d_"+unit] = rows[unit+'2'] - rows[unit+'1']

        if Elev == 1:
            plt_slevelnum_table_EdElev (path, rows, 'cfg'+str(i+1), dbname1, dbname2, unit)
        if EdE == 1:
            plt_slevelnum_table_EdE_ (path, rows, 'cfg'+str(i+1), unit)


def plt_user_idx_ca (dbname1, dbname2, merged_data, luser_idx, use_idx_name, path, unit, Elev=1, EdE=1):
    """Generate two-panel comparison plots for user-selected level indices."""
    for i in range(len(luser_idx)):
        lrow = []
        for j in range(len(luser_idx[i])):

            row = merged_data[merged_data['idx'+use_idx_name] == luser_idx[i][j]]

            lrow.append(row)

        rows = pd.concat(lrow, ignore_index=True)
        rows = rows.reset_index(drop=True)
        rows = lev_add_state(rows)

        rows["d_"+unit] = rows[unit+'2'] - rows[unit+'1']

        if Elev == 1:
            plt_slevelnum_table_EdElev (path, rows, 'idx'+str(luser_idx[i]), dbname1, dbname2, unit)
        if EdE == 1:
            plt_slevelnum_table_EdE_ (path, rows, 'idx'+str(luser_idx[i]), unit)


def plt_slevelnum_ca (dbname1, dbname2, merged_data, sha_table, path, unit, Elev=1, EdE=1):
    """Generate category-based energy and energy-difference comparison plots."""

    merged_data = lev_add_state(merged_data)

    merged_data["d_"+unit] = merged_data[unit+'2'] - merged_data[unit+'1']

    merged_data_ = merged_data.copy()

    for i in range(len(sha_table)-1):
        shaca = sha_table.loc[i, 'category']

        lca = shaca.split('_')

        ca_data = merged_data.loc[merged_data['cfg_c'].isin(lca)]
        merged_data_ = merged_data_.loc[~merged_data['cfg_c'].isin(lca)]

        if Elev == 1:
            plt_slevelnum_table_EdElev (path, ca_data, shaca, dbname1, dbname2, unit)
        if EdE == 1:
            plt_slevelnum_table_EdE_ (path, ca_data, shaca, unit)

    shaca = sha_table.loc[len(sha_table)-1, 'category']
    lca = shaca.split('_')
    if 'other' in lca:
        plt_slevelnum_table_EdElev (path, merged_data_, shaca, dbname1, dbname2, unit)
        plt_slevelnum_table_EdE_ (path, merged_data_, shaca, unit)
    else:

        ca_data = merged_data.loc[merged_data['cfg_c'].isin(lca)]
        if Elev == 1:
            plt_slevelnum_table_EdElev (path, ca_data, shaca, dbname1, dbname2, unit)
        if EdE == 1:
            plt_slevelnum_table_EdE_ (path, ca_data, shaca, unit)


def main():

    ion = '020807'

    dbname1 = 'atomdb313'
    dbname2 = 'chianti1102'
    dbname3 = 'spex30801'

    unit = 'eV'

    cut_big_gap = True
    max_blank_gap = 5

    luser_E = None
    luser_cfg = None
    luser_idx = None
    user_choosename = None

    ldbname = [dbname1, dbname2]
    if dbname3 is not None and str(dbname3).strip() != '':
        ldbname.append(dbname3)

    print("Current databases:", ldbname)
    if len(ldbname) < 2:
        raise ValueError("At least two databases are required")
    print("Three or more databases detected: generating only E-versus-database plots" if len(ldbname) > 2 else "Two databases detected: generating the complete pairwise comparison plots")

    user_name = None
    if isinstance(user_choosename, str):
        if user_choosename in ldbname:
            user_name = str(ldbname.index(user_choosename) + 1); print("user_name:", user_name)
        else:
            raise ValueError("user_choosename is not present in ldbname; check the database name")

    lenergy = []
    cfg_part, set_name, lsort_name = seq_ele_sort(int(ion[:2]))

    for i in range(len(ldbname)):
        file_path_atdata = os.path.join(directory, 'clean_data', ion + '.h5')
        dataset_path = '/lev_' + ldbname[i]
        print("Reading:", dataset_path)
        data = show_tbl_lev.lev_df(file_path_atdata, dataset_path)
        split_data = data['cfg'].astype(str).str.split('.', expand=True)
        need_cols = len(cfg_part)
        have_cols = split_data.shape[1]
        if have_cols < need_cols:
            for col_i in range(have_cols, need_cols):
                split_data[col_i] = None
        if have_cols > need_cols:
            split_data = split_data.iloc[:, :need_cols]
        split_data.columns = cfg_part
        data[cfg_part] = split_data
        data['cfg_c'] = data['cfg'].apply(clean_cfg_sha)
        data = data.rename(columns={'idx': 'idx' + str(i + 1), unit: unit + str(i + 1)})
        lenergy.append(data)

    directory_d = os.path.join(directory, "data", 'seq' + ion[:2], ion, ldbname[0] + '_' + ldbname[1]) if len(ldbname) == 2 else os.path.join(directory, "data", 'seq' + ion[:2], ion, '_'.join(ldbname))
    os.makedirs(directory_d, exist_ok=True)

    if len(ldbname) == 2:
        sum_table, sha_table = levcate(ion, ldbname, lenergy, cfg_part, set_name, lsort_name, unit)
    else:
        sum_table = levelnum_table(ldbname, lenergy, cfg_part, set_name, lsort_name, unit)
        try:
            sum_table = sum_table.drop('last_char', axis=1)
        except KeyError:
            pass
        sha_table = None
        mask = (sum_table['category'] == 'other') & (sum_table.drop('category', axis=1).ne(0).any(axis=1))
        if len(sum_table[mask]) != 0:
            print("An other category is present:"); print(ion); print(sum_table)

    plt_levelnum(ldbname, lenergy, sum_table, 25, directory_d, unit, cut_big_gap=cut_big_gap, max_gap=max_blank_gap)
    plt_leveloverlap(ldbname, lenergy, sum_table, 25, directory_d, unit, cut_big_gap=cut_big_gap, max_gap=max_blank_gap)

    if len(ldbname) > 2:
        print("Three-or-more-database mode completed: generated Evsdb_num.png and Evsdb_lap.png")
        return

    lev_merged_path = os.path.join(directory_d, 'cflev_map0.dat')
    if not os.path.exists(lev_merged_path):
        raise FileNotFoundError("Could not find the two-database alignment file cflev_map0.dat:\n" + lev_merged_path + "\nTo generate only E-versus-database plots, provide a third database through dbname3 or disable the pairwise comparison section.")

    lev_merged = pd.read_csv(lev_merged_path, sep='\t', comment='#')
    lev_merged['cfg_c'] = lev_merged['cfg'].apply(clean_cfg_sha)

    if isinstance(luser_E, list) and isinstance(user_name, str):
        plt_user_E_ca(ldbname[0], ldbname[1], lev_merged, luser_E, user_name, directory_d, unit, Elev=1, EdE=1)
    elif isinstance(luser_cfg, list):
        plt_user_cfg_ca(ldbname[0], ldbname[1], lev_merged, luser_cfg, directory_d, unit, Elev=1, EdE=1)
    elif isinstance(luser_idx, list) and isinstance(user_name, str):
        plt_user_idx_ca(ldbname[0], ldbname[1], lev_merged, luser_idx, user_name, directory_d, unit, Elev=1, EdE=1)
    else:
        plt_slevelnum_ca(ldbname[0], ldbname[1], lev_merged, sha_table, directory_d, unit, Elev=1, EdE=1)


if __name__ == "__main__":
    main()
