"""Figure 1: O VII level distributions from clean_data/020807.h5."""
from pathlib import Path
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import re
from core.data import ds_to_df
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
        raise ValueError(prefix_num)


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


            for n in aln_bc:

                unset_name.append(lsort_name[bc]+n)
                row_sc1 = row_sc[row_sc[cfg_part[nums_part-1]].str.startswith(n)]
                n_bc = len(row_sc1)

                chartl.append(n_bc)

                row_sc1 = row_sc1.sort_values(by=unit+str(i+1))
                data_new = pd.concat([data_new, row_sc1], ignore_index=True)
                used_index.update(row_sc1.index)

        other_data = data[~data.index.isin(used_index)]
        other_data = other_data.sort_values(by=unit+str(i+1))
        data_new = pd.concat([data_new, other_data], ignore_index=True)

        if  'level_idx'+str(i+1) not in data_new.columns:
            lenergy[i] = data_new.reset_index().rename(columns={'index': 'level_idx'+str(i+1)})


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

# Analysis and plotting parameters
ion = '020807'
dbnames = ['atomdb313', 'chianti1102', 'spex30801']
unit = 'eV'

# Boundary separating the low- and high-energy views.
energy_boundary = 1000

# Compress a shared empty interval when its physical width exceeds the threshold.
cut_big_gap = True
compress_gap_threshold = 5.0
compressed_gap_display = 3.0

# Physical energy interval between y-axis tick labels; use None for automatic ticks.
ytick_interval = 20000

# Draw break marks at compressed intervals.
draw_compress_breaks = True
break_mark_width = 0.035
break_mark_height = 0.012

# Figure dimensions.
figsize_full = (4, 7)
figsize_split = (5, 6.5)
figsize_combined = (15, 5.5)

shadow_alpha = 0.1
line_width = 0.8
n_colors = 25
dpi = 200
show_boundary_line = True
output_format = 'pdf'


# Import dependencies and load the plotting module from the project package.
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# Read the cleaned level tables and assign configuration categories.
def prepare_energy_data(ion, dbnames, unit):
    """Load level data, derive category fields, and create the output directory."""
    cfg_part, set_name, lsort_name = seq_ele_sort(int(ion[:2]))
    if cfg_part is None:
        raise ValueError(f'Sequence {ion[:2]} is not defined in seq_ele_sort')

    lenergy = []
    for i, dbname in enumerate(dbnames):
        file_path_atdata = os.path.join(str(ROOT), 'clean_data', ion + '.h5')
        dataset_path = '/lev_' + dbname

        data = ds_to_df(file_path_atdata, dataset_path)

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

    sum_table = levelnum_table(dbnames, lenergy, cfg_part, set_name, lsort_name, unit)
    try:
        sum_table = sum_table.drop('last_char', axis=1)
    except KeyError:
        pass

    output_dir = str(HERE/'photos')
    os.makedirs(output_dir, exist_ok=True)
    return lenergy, sum_table, output_dir


# Plotting functions for the three energy views.
def _get_category_data(data, category, sum_table, col):
    # Select levels directly by category name to keep categories aligned.
    if category == 'other':
        known_categories = set(sum_table.loc[sum_table['category'] != 'total', 'category'])
        known_categories.discard('other')
        cat_data = data.loc[~data['cfg_c'].isin(known_categories)].copy()
    else:
        cat_data = data.loc[data['cfg_c'] == category].copy()
    return cat_data.sort_values(by=col).reset_index(drop=True)


def _all_energy_values(dbnames, lenergy, unit):
    vals = []
    for i in range(len(dbnames)):
        col = unit + str(i + 1)
        vals.extend(pd.to_numeric(lenergy[i][col], errors='coerce').dropna().tolist())
    return np.array(vals, dtype=float)


def _collect_visible_intervals(dbnames, lenergy, sum_table, unit, view_min, view_max):
    intervals = []
    for i, dname in enumerate(dbnames):
        data = lenergy[i]
        col = unit + str(i + 1)
        for j in range(len(sum_table) - 1):
            if int(sum_table.loc[j, dname]) == 0:
                continue
            category = sum_table.loc[j, 'category']
            cat_data = _get_category_data(data, category, sum_table, col)
            ys = pd.to_numeric(cat_data[col], errors='coerce').dropna()
            if len(ys) == 0:
                continue
            y0, y1 = float(ys.min()), float(ys.max())
            y0, y1 = max(y0, view_min), min(y1, view_max)
            if y0 <= y1:
                intervals.append((y0, y1))
    return intervals


def _make_gap_compressor(dbnames, lenergy, sum_table, unit, view_min, view_max,
                         cut_big_gap=True, gap_threshold=5.0, display_gap=2.0,
                         compress_outer_gaps=True):
    """Create a coordinate transform for shared empty energy intervals.

    ``gap_threshold`` is the minimum physical width to compress, and
    ``display_gap`` is the displayed width after compression.
    """
    def identity(y):
        if hasattr(y, 'apply'):
            return y
        if isinstance(y, (list, tuple, np.ndarray)):
            return y
        return y

    if (not cut_big_gap) or gap_threshold is None or display_gap is None:
        return identity, []

    intervals = _collect_visible_intervals(dbnames, lenergy, sum_table, unit, view_min, view_max)
    if len(intervals) == 0:
        return identity, []

    intervals.sort(key=lambda x: x[0])
    merged = []
    for y0, y1 in intervals:
        if not merged or y0 > merged[-1][1]:
            merged.append([y0, y1])
        else:
            merged[-1][1] = max(merged[-1][1], y1)

    steps = []
    if compress_outer_gaps and merged[0][0] - view_min > gap_threshold:
        steps.append((view_min, merged[0][0], merged[0][0] - view_min, display_gap))

    for k in range(len(merged) - 1):
        gap_start = merged[k][1]
        gap_end = merged[k + 1][0]
        gap = gap_end - gap_start
        if gap > gap_threshold:
            steps.append((gap_start, gap_end, gap, display_gap))

    if compress_outer_gaps and view_max - merged[-1][1] > gap_threshold:
        steps.append((merged[-1][1], view_max, view_max - merged[-1][1], display_gap))

    if len(steps) == 0:
        return identity, []

    def compress_scalar(y):
        if pd.isna(y):
            return y
        y = float(y)
        shift = 0.0
        for gap_start, gap_end, gap, display_len in steps:
            if y <= gap_start:
                break
            if y < gap_end:
                # Map values linearly within a compressed interval.
                return gap_start - shift + (y - gap_start) * display_len / gap
            shift += gap - display_len
        return y - shift

    def compress(y):
        if hasattr(y, 'apply'):
            return y.apply(compress_scalar)
        if isinstance(y, np.ndarray):
            return np.array([compress_scalar(v) for v in y])
        if isinstance(y, (list, tuple)):
            return [compress_scalar(v) for v in y]
        return compress_scalar(y)

    return compress, steps


def _make_real_energy_ticks(view_min, view_max, tick_interval=None, steps=None, nticks=8):
    # Generate physical-energy ticks and omit intermediate ticks inside compressed gaps.
    steps = steps or []
    eps = max(abs(view_max - view_min), 1.0) * 1e-10

    if view_max <= view_min:
        ticks = [float(view_min)]
    elif tick_interval is None or tick_interval <= 0:
        ticks = list(np.linspace(view_min, view_max, nticks))
    else:
        start = np.ceil((view_min - eps) / tick_interval) * tick_interval
        ticks = list(np.arange(start, view_max + eps, tick_interval))
        ticks.extend([view_min, view_max])

    # Keep the endpoints of each compressed interval.
    for step in steps:
        gap_start, gap_end = step[0], step[1]
        ticks.extend([gap_start, gap_end])

    filtered = []
    for t in ticks:
        t = float(t)
        if t < view_min - eps or t > view_max + eps:
            continue
        inside_compressed_gap = any((step[0] + eps < t < step[1] - eps) for step in steps)
        if not inside_compressed_gap:
            filtered.append(t)

    filtered = sorted(filtered)
    unique = []
    for t in filtered:
        if not unique or abs(t - unique[-1]) > eps:
            unique.append(t)
    return unique


def _set_real_energy_ticks(axes, transform_y, view_min, view_max, steps=None, tick_interval=None, nticks=8):
    ticks = _make_real_energy_ticks(view_min, view_max, tick_interval=tick_interval, steps=steps, nticks=nticks)
    tick_pos = [transform_y(float(t)) for t in ticks]
    tick_lab = ['0' if abs(float(t)) < 1e-10 else f'{float(t):.6g}' for t in ticks]
    for ax in axes:
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)


def _draw_compression_breaks(ax, transform_y, steps, view_min, view_max, width=0.035, height=0.012):
    # Draw paired slash marks on the left y-axis at compressed intervals.
    if not steps:
        return

    y_min_t = transform_y(view_min)
    y_max_t = transform_y(view_max)
    y_span_t = y_max_t - y_min_t
    if y_span_t <= 0:
        return

    for step in steps:
        gap_start, gap_end = step[0], step[1]
        y0 = transform_y(gap_start)
        y1 = transform_y(gap_end)
        y_mid = (y0 + y1) / 2

        # Use two parallel slashes to mark each compressed interval.
        slash_len = y_span_t * height
        slash_sep = slash_len * 0.9
        x0, x1 = -width, width
        for yc in (y_mid - slash_sep / 2, y_mid + slash_sep / 2):
            ax.plot([x0, x1], [yc - slash_len / 2, yc + slash_len / 2],
                    transform=ax.get_yaxis_transform(),
                    color='k', linewidth=1.0, clip_on=False, solid_capstyle='butt')


def _reference_db_for_labels(dbnames, sum_table):
    nonzero_counts = (sum_table.drop('category', axis=1) != 0).sum()
    return nonzero_counts.idxmax()


def _plot_energy_view_on_axes(dbnames, lenergy, sum_table, unit, view_min, view_max, axes,
                              show_labels=True, energy_boundary=None, show_boundary_line=True,
                              cut_big_gap=True, gap_threshold=5.0, display_gap=2.0,
                              shadow_alpha=0.35, line_width=1.1, n_colors=25,
                              ytick_interval=None, draw_compress_breaks=True,
                              break_mark_width=0.035, break_mark_height=0.012,
                              show_ylabel=True, title=None):
    axes = list(axes) if hasattr(axes, '__len__') else [axes]

    palette = sns.color_palette('husl', max(n_colors, len(sum_table) - 1))
    colors = palette.as_hex()
    transform_y, steps = _make_gap_compressor(dbnames, lenergy, sum_table, unit, view_min, view_max,
                                              cut_big_gap=cut_big_gap,
                                              gap_threshold=gap_threshold,
                                              display_gap=display_gap)

    for i, dname in enumerate(dbnames):
        data = lenergy[i]
        col = unit + str(i + 1)
        ax = axes[i]

        for j in range(len(sum_table) - 1):
            category = sum_table.loc[j, 'category']
            if int(sum_table.loc[j, dname]) == 0:
                continue

            cat_data = _get_category_data(data, category, sum_table, col)
            ys = pd.to_numeric(cat_data[col], errors='coerce').dropna().sort_values()
            if len(ys) == 0:
                continue

            y0, y1 = float(ys.min()), float(ys.max())
            y0_clip, y1_clip = max(y0, view_min), min(y1, view_max)
            if y0_clip <= y1_clip:
                ax.axhspan(transform_y(y0_clip), transform_y(y1_clip),
                           color=colors[j % len(colors)], alpha=shadow_alpha, xmin=0.05, xmax=0.95)

            ys_show = ys[(ys >= view_min) & (ys <= view_max)]
            if len(ys_show) > 0:
                ax.hlines(transform_y(ys_show), 0, 1, colors=colors[j % len(colors)], linewidth=line_width)

        if show_boundary_line and energy_boundary is not None and view_min <= energy_boundary <= view_max:
            ax.axhline(transform_y(energy_boundary), color='k', linestyle=':', linewidth=1.0)

        ax.set_xticks([0.5])
        ax.set_xticklabels([dname], rotation=0)
        if i == 0 and show_ylabel:
            ax.set_ylabel('Energy (' + unit + ')')
        else:
            ax.tick_params(left=False, labelleft=False)
            ax.set_ylabel('')
            ax.spines['left'].set_visible(False)

        if title and i == len(dbnames) // 2:
            ax.set_title(title, fontsize=11)

    if show_labels:
        ref_db = _reference_db_for_labels(dbnames, sum_table)
        ref_i = dbnames.index(ref_db)
        ref_data = lenergy[ref_i]
        ref_col = unit + str(ref_i + 1)
        label_rows = sum_table[(sum_table[ref_db] != 0) & (sum_table['category'] != 'total')]

        for row_idx, row in label_rows.iterrows():
            category = row['category']
            cat_data = _get_category_data(ref_data, category, sum_table, ref_col)
            ys = pd.to_numeric(cat_data[ref_col], errors='coerce').dropna()
            if len(ys) == 0:
                continue
            y0, y1 = float(ys.min()), float(ys.max())
            y0_clip, y1_clip = max(y0, view_min), min(y1, view_max)
            if y0_clip <= y1_clip:
                y_text = (y0_clip + y1_clip) / 2
                axes[-1].text(1.1, transform_y(y_text), category, va='center', ha='left',
                              color=colors[row_idx % len(colors)], fontsize=12)

    y_min_t = transform_y(view_min)
    y_max_t = transform_y(view_max)
    for ax in axes:
        ax.set_ylim(y_min_t, y_max_t)
    _set_real_energy_ticks(axes, transform_y, view_min, view_max, steps=steps, tick_interval=ytick_interval)

    if draw_compress_breaks:
        _draw_compression_breaks(axes[0], transform_y, steps, view_min, view_max,
                                 width=break_mark_width, height=break_mark_height)

    return steps


def plot_energy_view(dbnames, lenergy, sum_table, unit, view_min, view_max, filename, title='',
                     show_labels=True, energy_boundary=None, show_boundary_line=True,
                     cut_big_gap=True, gap_threshold=5.0, display_gap=2.0,
                     figsize=None, shadow_alpha=0.35, line_width=1.1, n_colors=25, dpi=200,
                     ytick_interval=None, draw_compress_breaks=True, break_mark_width=0.035,
                     break_mark_height=0.012):
    if figsize is None:
        figsize = (len(dbnames) * 2.2, 8 if show_labels else 10)

    fig, axes = plt.subplots(1, len(dbnames), figsize=figsize, sharey=True)
    axes = list(axes) if hasattr(axes, '__len__') else [axes]

    steps = _plot_energy_view_on_axes(dbnames, lenergy, sum_table, unit, view_min, view_max, axes,
                                      show_labels=show_labels, energy_boundary=energy_boundary,
                                      show_boundary_line=show_boundary_line, cut_big_gap=cut_big_gap,
                                      gap_threshold=gap_threshold, display_gap=display_gap,
                                      shadow_alpha=shadow_alpha, line_width=line_width, n_colors=n_colors,
                                      ytick_interval=ytick_interval, draw_compress_breaks=draw_compress_breaks,
                                      break_mark_width=break_mark_width, break_mark_height=break_mark_height,
                                      show_ylabel=True, title=title)

    plt.subplots_adjust(wspace=-0.01)
    if show_labels:
        fig.subplots_adjust(left=0.18, right=0.75, top=0.94, bottom=0.06)
    else:
        fig.subplots_adjust(left=0.18, right=0.95, top=0.94, bottom=0.06)

    fig.savefig(filename, dpi=dpi, bbox_inches='tight')

    plt.show()
    return fig, axes, steps


def _get_three_view_ranges(dbnames, lenergy, unit, energy_boundary):
    all_vals = _all_energy_values(dbnames, lenergy, unit)
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) == 0:
        raise ValueError('No valid energy values were found')

    global_min = float(np.min(all_vals))
    global_max = float(np.max(all_vals))

    below = all_vals[all_vals <= energy_boundary]
    above = all_vals[all_vals >= energy_boundary]

    low_min = 0.0
    low_max = float(np.max(below)) if len(below) > 0 else float(energy_boundary)
    high_min = float(np.min(above)) if len(above) > 0 else float(energy_boundary)
    high_max = global_max

    return {
        'full': (global_min, global_max),
        'low': (low_min, low_max),
        'high': (high_min, high_max)
    }


def plot_combined_three_views(dbnames, lenergy, sum_table, unit, output_dir, energy_boundary,
                              cut_big_gap=True, gap_threshold=5.0, display_gap=2.0,
                              figsize_combined=(15, 5.5), shadow_alpha=0.35, line_width=1.1,
                              n_colors=25, dpi=200, show_boundary_line=True, ytick_interval=None,
                              draw_compress_breaks=True, break_mark_width=0.035, break_mark_height=0.012,
                              output_format='pdf'):
    ranges = _get_three_view_ranges(dbnames, lenergy, unit, energy_boundary)
    view_specs = [
        ('Full', ranges['full'][0], ranges['full'][1], False),
        ('Low', ranges['low'][0], ranges['low'][1], True),
        ('High', ranges['high'][0], ranges['high'][1], True),
    ]

    ndb = len(dbnames)
    fig, all_axes = plt.subplots(1, ndb * 3, figsize=figsize_combined, sharey=False)
    all_axes = np.array(all_axes).reshape(3, ndb)

    for view_idx, (title, view_min, view_max, show_labels) in enumerate(view_specs):
        axes = list(all_axes[view_idx])
        _plot_energy_view_on_axes(dbnames, lenergy, sum_table, unit, view_min, view_max, axes,
                                  show_labels=show_labels, energy_boundary=energy_boundary,
                                  show_boundary_line=show_boundary_line, cut_big_gap=cut_big_gap,
                                  gap_threshold=gap_threshold, display_gap=display_gap,
                                  shadow_alpha=shadow_alpha, line_width=line_width, n_colors=n_colors,
                                  ytick_interval=ytick_interval, draw_compress_breaks=draw_compress_breaks,
                                  break_mark_width=break_mark_width, break_mark_height=break_mark_height,
                                  show_ylabel=True, title=title)

        # Restore the left axis at the start of each view group.
        if view_idx > 0:
            axes[0].spines['left'].set_visible(True)
            axes[0].tick_params(left=True, labelleft=True)

    fig.subplots_adjust(left=0.055, right=0.94, top=0.92, bottom=0.09, wspace=0.10)
    combined_path = os.path.join(output_dir, f'Evsdb_0_three_views_combined.{output_format}')
    fig.savefig(combined_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return fig, all_axes


def plot_three_energy_views(dbnames, lenergy, sum_table, unit, output_dir, energy_boundary,
                            cut_big_gap=True, gap_threshold=5.0, display_gap=2.0,
                            figsize_full=None, figsize_split=None, figsize_combined=(15, 5.5),
                            shadow_alpha=0.35, line_width=1.1, n_colors=25, dpi=200,
                            show_boundary_line=True, ytick_interval=None, draw_compress_breaks=True,
                            break_mark_width=0.035, break_mark_height=0.012,
                            output_format='pdf', make_combined=True):
    ranges = _get_three_view_ranges(dbnames, lenergy, unit, energy_boundary)
    global_min, global_max = ranges['full']
    low_min, low_max = ranges['low']
    high_min, high_max = ranges['high']

    if figsize_full is None:
        figsize_full = (len(dbnames) * 2.2, 10)
    if figsize_split is None:
        figsize_split = (len(dbnames) * 2.2, 7)

    os.makedirs(output_dir, exist_ok=True)




    plot_energy_view(dbnames, lenergy, sum_table, unit, global_min, global_max,
                     os.path.join(output_dir, f'fig01a_o_vii_energy_full.{output_format}'),
                     title=None, show_labels=False, energy_boundary=energy_boundary,
                     show_boundary_line=show_boundary_line, cut_big_gap=cut_big_gap,
                     gap_threshold=gap_threshold, display_gap=display_gap, figsize=figsize_full,
                     shadow_alpha=shadow_alpha, line_width=line_width, n_colors=n_colors, dpi=dpi,
                     ytick_interval=ytick_interval, draw_compress_breaks=draw_compress_breaks,
                     break_mark_width=break_mark_width, break_mark_height=break_mark_height)

    plot_energy_view(dbnames, lenergy, sum_table, unit, low_min, low_max,
                     os.path.join(output_dir, f'fig01b_o_vii_energy_low.{output_format}'),
                     title=None, show_labels=True, energy_boundary=energy_boundary,
                     show_boundary_line=show_boundary_line, cut_big_gap=cut_big_gap,
                     gap_threshold=gap_threshold, display_gap=display_gap, figsize=figsize_split,
                     shadow_alpha=shadow_alpha, line_width=line_width, n_colors=n_colors, dpi=dpi,
                     ytick_interval=ytick_interval, draw_compress_breaks=draw_compress_breaks,
                     break_mark_width=break_mark_width, break_mark_height=break_mark_height)

    plot_energy_view(dbnames, lenergy, sum_table, unit, high_min, high_max,
                     os.path.join(output_dir, f'fig01c_o_vii_energy_high.{output_format}'),
                     title=None, show_labels=True, energy_boundary=energy_boundary,
                     show_boundary_line=show_boundary_line, cut_big_gap=cut_big_gap,
                     gap_threshold=gap_threshold, display_gap=display_gap, figsize=figsize_split,
                     shadow_alpha=shadow_alpha, line_width=line_width, n_colors=n_colors, dpi=dpi,
                     ytick_interval=ytick_interval, draw_compress_breaks=draw_compress_breaks,
                     break_mark_width=break_mark_width, break_mark_height=break_mark_height)

    if make_combined:
        plot_combined_three_views(dbnames, lenergy, sum_table, unit, output_dir, energy_boundary,
                                  cut_big_gap=cut_big_gap, gap_threshold=gap_threshold,
                                  display_gap=display_gap, figsize_combined=figsize_combined,
                                  shadow_alpha=shadow_alpha, line_width=line_width, n_colors=n_colors,
                                  dpi=dpi, show_boundary_line=show_boundary_line,
                                  ytick_interval=ytick_interval,
                                  draw_compress_breaks=draw_compress_breaks,
                                  break_mark_width=break_mark_width,
                                  break_mark_height=break_mark_height,
                                  output_format=output_format)



def main():
    lenergy, counts, output = prepare_energy_data(ion, dbnames, unit)
    plot_three_energy_views(dbnames, lenergy, counts, unit, output, energy_boundary,
        cut_big_gap=cut_big_gap, gap_threshold=compress_gap_threshold,
        display_gap=compressed_gap_display, figsize_full=figsize_full, figsize_split=figsize_split,
        shadow_alpha=shadow_alpha, line_width=line_width, n_colors=n_colors, dpi=dpi,
        show_boundary_line=show_boundary_line, ytick_interval=ytick_interval,
        draw_compress_breaks=draw_compress_breaks, break_mark_width=break_mark_width,
        break_mark_height=break_mark_height, output_format='pdf', make_combined=False)
    print(output)

if __name__ == '__main__':
    main()
