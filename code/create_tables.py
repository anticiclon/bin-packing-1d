# -*- coding: utf-8 -*-
"""
create_tables.py
Generates every LaTeX table and figure used in the bin dimensioning paper.
All values are computed from the pipeline's CSV files — no hardcoded data.

This script merges two previously separate, exploratory scripts:
  - the original create_tables.py (Tables 1-4, "full-scale" results)
  - an interactive scalability/plotting script (Tables 6-7, Figures 5-8)
into a single, reproducible entry point.

Expected folder structure (this script lives in code/):
    repo/
    ├── code/
    │   ├── create_tables.py            <- this script
    │   └── create_configurations.py    <- required (imports create_bin_types)
    ├── data/
    │   ├── {DS-X}_skus.csv             (Module 1: items, n_s, rotatable)
    │   └── {DS-X}_blocks.csv           (Module 2: unique configs / feasible pairs)
    └── results/
        ├── scalability_results_{DS-X}.csv   (directly in results/, no subfolder)
        └── {DS-X}/
            ├── {DS-X}_bfd_bins.csv
            └── {DS-X}_cg_summary.csv        (only used for DS-1)

Usage:
  python create_tables.py

Outputs (written next to this script, in code/; change OUT_DIR to relocate):
  table_datasets_skus.tex       Table 1  (tab:datasets_skus)
  table_datasets_blocks.tex     Table 2  (tab:datasets_bloques)
  table_fullscale_results.tex   Table 3  (tab:fullscale_results)
  table_bfd_packing.tex         Table 4  (tab:bfd_packing)
  table_scalability.tex         Table 6  (tab:scalability)
  table_summary.tex             Table 7  (tab:summary)
  figura_gap_skus.pdf           Figure 5 (fig:gap_skus)
  figura_variabilidad_skus.pdf  Figure 6 (fig:variability_skus)
  figura_mejora_skus.pdf        Figure 7 (fig:trivial_skus)
  figura_cgtime_skus.pdf        Figure 8 (fig:cgtime_skus)

Figure file names match the \\includegraphics{imagenes/...} paths already
used in the paper's .tex source, so the paper does not need to change.
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import functions from create_configurations.py (Module 3) so feasible_pairs
# is computed with the exact same bin catalog logic as the pipeline, instead
# of duplicating it here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_configurations import create_bin_types

# ============================================================================
# PATHS
# Repo structure (repo_dir is the parent of code/, data/ and results/):
#   repo/
#   ├── code/                              <- this script
#   ├── data/
#   │   └── {DS-X}_skus.csv                (Module 1)
#   │   └── {DS-X}_blocks.csv              (Module 2)
#   └── results/
#       ├── scalability_results_{DS-X}.csv (run_scalability.py, directly in results/)
#       └── {DS-X}/
#           ├── {DS-X}_bfd_bins.csv        (bfd_heuristic.py)
#           ├── {DS-X}_bfd_solution.csv
#           ├── {DS-X}_bfd_report.txt
#           ├── {DS-X}_cg_solution.csv     (column_generation.py)
#           └── {DS-X}_cg_summary.csv
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)

DATA_DIR    = os.path.join(REPO_DIR, 'data')
RESULTS_DIR = os.path.join(REPO_DIR, 'results')
OUT_DIR     = SCRIPT_DIR  # outputs are written next to the script; change if needed

def data_path(fname):
    """Module 1/2 files: data/{fname}"""
    return os.path.join(DATA_DIR, fname)

def results_path(label, fname):
    """Per-dataset files (BFD, CG): results/{DS-X}/{fname}"""
    return os.path.join(RESULTS_DIR, label, fname)

def scalability_path(fname):
    """scalability_results_*.csv: directly in results/, no subfolder"""
    return os.path.join(RESULTS_DIR, fname)

def out(fname):
    return os.path.join(OUT_DIR, fname)

# ============================================================================
# DATASET ORDER
# DS-1, DS-2: large products  |  DS-3, DS-4: small products
# Every file and folder name in this script uses these labels consistently.
# ============================================================================

DS_ORDER = ['DS-1', 'DS-2', 'DS-3', 'DS-4']

DS_CATALOG = {
    'DS-1': ('large products', 30),
    'DS-2': ('large products', 30),
    'DS-3': ('small products', 24),
    'DS-4': ('small products', 24),
}

# Plot color/label per dataset, used consistently across all figures
DS_COLOR = {
    'DS-1': 'steelblue',
    'DS-2': 'darkorange',
    'DS-3': 'green',
    'DS-4': 'red',
}

# ============================================================================
# DATA LOADING — per-dataset full-scale results (Tables 1-4)
# ============================================================================

def load_bfd_bins(label):
    return pd.read_csv(results_path(label, f'{label}_bfd_bins.csv'))

def load_scalability_full(label):
    """Full-instance row from the scalability CSV."""
    df = pd.read_csv(scalability_path(f'scalability_results_{label}.csv'))
    df['cg_converged'] = df['cg_converged'].astype(str).str.strip().eq('True')
    return df[df['n_skus_requested'] == df['n_skus_requested'].max()].iloc[0]

def load_sku_info(label):
    """
    Reads data/{DS-X}_skus.csv (output of Module 1, data_processing.py).
    Columns: sku_id, length, width, height, total_items, n_s, rotatable
    Returns: total items, median n_s, % rotatable.
    """
    df = pd.read_csv(data_path(f'{label}_skus.csv'))
    return {
        'items':     int(df['total_items'].sum()),
        'ns_med':    int(df['n_s'].median()),
        'rotatable': 100 * df['rotatable'].astype(bool).mean(),
    }

def load_block_info(label):
    """
    Reads data/{DS-X}_blocks.csv (output of Module 2, adding_blocks.py) and
    recomputes unique configs / feasible pairs using the same math as
    compute_bounding_box_optimal() from Module 3, but vectorized with NumPy
    (the pipeline's own "vectorized" function still uses nested .iterrows()
    and is intractable for large datasets).

    For each unique config (q_sk, l, w, h, rotatable) and each bin type, a
    pair is feasible if at least one orientation (out of the 6 permutations,
    or 2 if not rotatable) satisfies:
        a_max = floor(L_t / dim_X) >= 1
        b_max = floor(W_t / dim_Y) >= 1
        c_max = floor(H_t / dim_Z) >= 1
        a_min = ceil(q / (b_max * c_max)) <= a_max
    """
    df = pd.read_csv(data_path(f'{label}_blocks.csv'))
    config_cols = ['q_sk', 'length', 'width', 'height', 'rotatable']
    df_configs = df[config_cols].drop_duplicates().reset_index(drop=True)
    n_configs = len(df_configs)

    df_bins = create_bin_types(label)
    n_bin_types = len(df_bins)
    total_pairs = n_configs * n_bin_types

    L_t = df_bins['L_t'].to_numpy()  # (n_bins,)
    W_t = df_bins['W_t'].to_numpy()
    H_t = df_bins['H_t'].to_numpy()

    q   = df_configs['q_sk'].to_numpy().astype(np.int64)      # (n_cfg,)
    l_s = df_configs['length'].to_numpy()
    w_s = df_configs['width'].to_numpy()
    h_s = df_configs['height'].to_numpy()
    rot = df_configs['rotatable'].to_numpy().astype(bool)

    # The 6 axis permutations (indices into [l, w, h])
    perms = list(itertools.permutations([0, 1, 2]))  # 6 permutations
    dims_stack = np.stack([l_s, w_s, h_s], axis=1)   # (n_cfg, 3)

    feasible_any = np.zeros((n_configs, n_bin_types), dtype=bool)

    for k, perm in enumerate(perms):
        # Non-rotatable items only use the two horizontal orientations:
        # (l, w, h) and (w, l, h) — i.e. permutations that keep h on Z.
        is_horizontal = (perm[2] == 2)
        active = rot | is_horizontal  # (n_cfg,) — which configs use this orientation

        if not active.any():
            continue

        dim_X = dims_stack[:, perm[0]]  # (n_cfg,)
        dim_Y = dims_stack[:, perm[1]]
        dim_Z = dims_stack[:, perm[2]]

        # Broadcasting (n_cfg, 1) vs (1, n_bins) -> (n_cfg, n_bins)
        a_max = np.floor(L_t[None, :] / dim_X[:, None]).astype(np.int64)
        b_max = np.floor(W_t[None, :] / dim_Y[:, None]).astype(np.int64)
        c_max = np.floor(H_t[None, :] / dim_Z[:, None]).astype(np.int64)

        valid_axes = (a_max >= 1) & (b_max >= 1) & (c_max >= 1)

        cap_slice = np.maximum(b_max * c_max, 1)  # avoid division by zero
        a_min = np.ceil(q[:, None] / cap_slice).astype(np.int64)

        ok = valid_axes & (a_min <= a_max)
        ok &= active[:, None]  # only applies to configs using this orientation

        feasible_any |= ok

    n_feasible = int(feasible_any.sum())
    feasible_pct = 100 * n_feasible / total_pairs if total_pairs > 0 else 0.0

    return {
        'unique_configs':  n_configs,
        'feasible_pairs':  n_feasible,
        'feasible_pct':    feasible_pct,
    }

# ============================================================================
# DATA LOADING — scalability sub-instances (Tables 6-7, Figures 5-8)
# ============================================================================

def load_scalability_all():
    """
    Reads results/scalability_results_{DS-X}.csv for every dataset and
    concatenates them into a single DataFrame with a clean 'cg_converged'
    boolean and numeric dtypes for the metrics used downstream.
    """
    frames = []
    for label in DS_ORDER:
        path = scalability_path(f'scalability_results_{label}.csv')
        if not os.path.exists(path):
            print(f'  ⚠ Missing {path} — skipping {label} in scalability tables/figures')
            continue
        df_ds = pd.read_csv(path)
        df_ds['dataset'] = label
        frames.append(df_ds)

    if not frames:
        raise FileNotFoundError(
            'No scalability_results_{DS-X}.csv files found in results/. '
            'Run run_scalability.py before create_tables.py.'
        )

    df = pd.concat(frames, ignore_index=True)

    df['cg_converged'] = df['cg_converged'].astype(str).str.strip().eq('True')
    for c in ['bfd_vol', 'z_lp', 'cg_time', 'eta_bfd', 'n_blocks', 'bfd_bins', 'bfd_time']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def compute_scalability_summaries(df):
    """
    Builds the per-(dataset, size) summary tables shared by Table 6, Table 7,
    and Figures 5-8.

    Returns a dict of DataFrames:
      gap        — BFD-LP gap statistics, CG-converged rows only
      cg_time    — CG runtime statistics, CG-converged rows only
      trivial    — bin reduction vs. the trivial singleton bound, all rows
      variability— coefficient of variation of the gap, CG-converged rows only
      table      — one row per (dataset, size) with everything needed for
                   Table 6 (n_blocks, bfd_bins, eta_bfd, gap, cg_time, seeds)
    """
    df_cg = df[df['cg_converged']].copy()
    df_cg['gap_bfd_lp'] = 100 * (df_cg['bfd_vol'] - df_cg['z_lp']) / df_cg['z_lp']

    gap = df_cg.groupby(['dataset', 'n_skus_requested']).agg(
        n_blocks=('n_blocks', 'mean'),
        gap_mean=('gap_bfd_lp', 'mean'),
        gap_std=('gap_bfd_lp', 'std'),
        n_seeds_converged=('gap_bfd_lp', 'count'),
    ).round(4).reset_index()

    cg_time = df_cg.groupby(['dataset', 'n_skus_requested']).agg(
        n_blocks=('n_blocks', 'mean'),
        cg_time_mean=('cg_time', 'mean'),
        cg_time_std=('cg_time', 'std'),
    ).round(1).reset_index()

    df = df.copy()
    df['trivial_reduction'] = 100 * (df['n_blocks'] - df['bfd_bins']) / df['n_blocks']
    trivial = df.groupby(['dataset', 'n_skus_requested']).agg(
        n_blocks=('n_blocks', 'mean'),
        reduction_mean=('trivial_reduction', 'mean'),
        reduction_std=('trivial_reduction', 'std'),
    ).round(2).reset_index()

    variability = df_cg.groupby(['dataset', 'n_skus_requested']).agg(
        n_blocks=('n_blocks', 'mean'),
        gap_cv=('gap_bfd_lp', lambda x: 100 * x.std() / x.mean() if x.mean() else np.nan),
    ).round(2).reset_index()

    # Total seeds actually run per (dataset, size), used to flag partial
    # convergence ("*") and full-instance rows ("dagger", single seed) in
    # Table 6, without hardcoding the experimental design here.
    seed_counts = df.groupby(['dataset', 'n_skus_requested']).size().rename('n_seeds_total')
    max_size_per_ds = df.groupby('dataset')['n_skus_requested'].transform('max')
    is_full = (df['n_skus_requested'] == max_size_per_ds)

    table = df.groupby(['dataset', 'n_skus_requested']).agg(
        n_blocks=('n_blocks', 'mean'),
        bfd_bins=('bfd_bins', 'mean'),
        eta_bfd=('eta_bfd', 'mean'),
        cg_converged_count=('cg_converged', 'sum'),
    ).reset_index()
    table = table.merge(seed_counts, on=['dataset', 'n_skus_requested'])
    table['is_full'] = is_full.groupby([df['dataset'], df['n_skus_requested']]).first().values
    table = table.merge(
        gap[['dataset', 'n_skus_requested', 'gap_mean', 'gap_std', 'n_seeds_converged']],
        on=['dataset', 'n_skus_requested'], how='left'
    )
    table = table.merge(
        cg_time[['dataset', 'n_skus_requested', 'cg_time_mean']],
        on=['dataset', 'n_skus_requested'], how='left'
    )

    table['n_blocks'] = table['n_blocks'].round(0).astype(int)
    table['bfd_bins'] = table['bfd_bins'].round(0).astype(int)
    table['eta_bfd']  = table['eta_bfd'].round(2)
    table['cg_time_h'] = (table['cg_time_mean'] / 3600).round(2)

    return {
        'gap': gap,
        'cg_time': cg_time,
        'trivial': trivial,
        'variability': variability,
        'table': table,
    }

# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def fmt_int(n):
    """Integer with LaTeX thousands separators: 1{,}234{,}567"""
    return f'{int(n):,}'.replace(',', '{,}')

def fmt_pct(v):
    if v < 0.05:
        return r'---'
    return f'{v:.1f}\\%'

def fmt_time(seconds):
    if seconds < 90:
        return f'{seconds:.0f} s'
    elif seconds < 7200:
        return f'{seconds/60:.1f} min'
    else:
        return f'{seconds/3600:.1f} h'

def fmt_time_h(hours):
    if pd.isna(hours):
        return '---'
    return f'{hours:.2f}'

def fmt_gap(mean, std, n_seeds_converged, n_seeds_total, is_full):
    """
    Formats a mean +/- std gap entry for Table 6.
    '---'        no converged seed for this (dataset, size)
    'X.XX^dag'   single converged seed (full instance, std undefined)
    'X.XX+-Y.YY*' fewer converged seeds than seeds run (partial convergence)
    'X.XX+-Y.YY' all run seeds converged
    """
    if pd.isna(mean):
        return r'---'
    if pd.isna(std) or is_full:
        return f'{mean:.2f}$^\\dagger$'
    if n_seeds_converged < n_seeds_total:
        return f'{mean:.2f} $\\pm$ {std:.2f}$^*$'
    return f'{mean:.2f} $\\pm$ {std:.2f}'

def packing_struct(df_bins):
    """Percentage distribution of bins by number of blocks."""
    total = len(df_bins)
    counts = df_bins['n_blocks'].value_counts().sort_index()
    return {n: 100 * counts.get(n, 0) / total for n in [1, 2, 3, 4]}

# ============================================================================
# TABLE 1: tab:datasets_skus
# ============================================================================

def table_datasets_skus(sc, ds_sku_info):
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Dataset characteristics.}',
        r'\label{tab:datasets_skus}',
        r'\begin{tabular}{@{}llrrrrrr@{}}',
        r'\toprule',
        (r'\textbf{Dataset} & \textbf{Catalog} & \textbf{SKUs} & \textbf{Items} & '
         r'\textbf{$n_s$ (med.)} & \textbf{Rotatable} & \textbf{$|T|$} \\'),
        r'\midrule',
    ]
    for label in DS_ORDER:
        info      = ds_sku_info[label]
        cat, T    = DS_CATALOG[label]
        n_skus    = int(sc[label]['n_skus_requested'])
        line = (
            f'{label} & {cat} & '
            f'{fmt_int(n_skus)} & '
            f'{fmt_int(info["items"])} & '
            f'{info["ns_med"]} & '
            f'{info["rotatable"]:.1f}\\% & '
            f'{T} \\\\'
        )
        lines.append(line)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)

# ============================================================================
# TABLE 2: tab:datasets_bloques
# ============================================================================

def table_datasets_blocks(sc, ds_block_info):
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Block aggregation and configuration computation results.}',
        r'\label{tab:datasets_bloques}',
        r'\begin{tabular}{@{}lrrrr@{}}',
        r'\toprule',
        (r'\textbf{Dataset} & \textbf{$|\mathcal{B}|$} & \textbf{Blocks/SKU} & '
         r'\textbf{Unique configs.} & \textbf{Feasible pairs} \\'),
        r'\midrule',
    ]
    for label in DS_ORDER:
        n_skus   = int(sc[label]['n_skus_requested'])
        n_blocks = int(sc[label]['n_blocks'])
        bpsku    = n_blocks / n_skus
        info     = ds_block_info[label]
        fp_str   = (f'{fmt_int(info["feasible_pairs"])}'
                    f'\\phantom{{0}}({info["feasible_pct"]:.1f}\\%)')
        line = (
            f'{label} & '
            f'{fmt_int(n_blocks)} & '
            f'{bpsku:.1f} & '
            f'{fmt_int(info["unique_configs"])} & '
            f'{fp_str} \\\\'
        )
        lines.append(line)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)

# ============================================================================
# TABLE 3: tab:fullscale_results
# ============================================================================

def table_fullscale_results(sc, bfd, cg):
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Full-scale results ($m = 4$).}',
        r'\label{tab:fullscale_results}',
        r'\resizebox{\textwidth}{!}{',
        r'\begin{tabular}{@{}lrrrrrrrrr@{}}',
        r'\toprule',
        r'\textbf{Dataset}',
        r'  & \textbf{$|\mathcal{B}|$}',
        r'  & \textbf{Isolated (\%)}',
        r'  & \textbf{BFD bins}',
        r'  & \textbf{BFD vol.\ (cm$^3$)}',
        r'  & \textbf{$\eta_{\text{BFD}}$}',
        r'  & \textbf{BFD time}',
        r'  & \textbf{$z^*_{\text{LP}}$ (cm$^3$)}',
        r'  & \textbf{Gap~(\%)}',
        r'  & \textbf{CG time} \\',
        r'\midrule',
    ]
    for label in DS_ORDER:
        df_bins   = bfd[label]
        row_sc    = sc[label]
        n_blocks  = int(row_sc['n_blocks'])
        isolated  = float(row_sc['isolated_pct'])
        n_bins    = len(df_bins)
        vol_bins  = int(df_bins['volume'].sum())
        vol_items = df_bins['items_volume'].sum()
        eta       = vol_items / vol_bins * 100
        bfd_time  = float(row_sc['bfd_time'])

        if label == 'DS-1':
            # z_lp    = int(cg['lb_lp'])
            z_lp    = int(cg['lower_bound_lp'])
            gap     = float(cg['gap_pct'])
            cg_time = float(cg['runtime_s'])
            z_lp_str  = fmt_int(z_lp)
            gap_str   = f'{gap:.2f}'
            cg_str    = fmt_time(cg_time)
        else:
            z_lp_str = '---'
            gap_str  = '---'
            cg_str   = r'$>$5 days$^\dagger$'

        line = (
            f'{label}\n'
            f'  & {fmt_int(n_blocks)}\n'
            f'  & {isolated:.2f}\n'
            f'  & {fmt_int(n_bins)}\n'
            f'  & {fmt_int(vol_bins)}\n'
            f'  & {eta:.1f}\\%\n'
            f'  & {fmt_time(bfd_time)}\n'
            f'  & {z_lp_str}\n'
            f'  & {gap_str}\n'
            f'  & {cg_str} \\\\'
        )
        lines.append(line)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'}',
        r'\par\smallskip',
        (r'{\small $^\dagger$Column generation interrupted at the 5-day wall-clock '
         r'limit during the first RMP solve, before any pricing iteration completed.}'),
        r'\end{table}',
    ]
    return '\n'.join(lines)

# ============================================================================
# TABLE 4: tab:bfd_packing
# ============================================================================

def table_bfd_packing(bfd):
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{BFD packing structure: percentage of bins by number of blocks ($m = 4$).}',
        r'\label{tab:bfd_packing}',
        r'\begin{tabular}{@{}lrrrr@{}}',
        r'\toprule',
        r'& \multicolumn{4}{c}{\textbf{Percentage of bins}} \\',
        r'\cmidrule(l){2-5}',
        r'\textbf{Dataset}',
        r'  & \textbf{1 block}',
        r'  & \textbf{2 blocks}',
        r'  & \textbf{3 blocks}',
        r'  & \textbf{4 blocks} \\',
        r'\midrule',
    ]
    for label in DS_ORDER:
        p = packing_struct(bfd[label])
        line = (
            f'{label} & '
            f'{fmt_pct(p[1])} & '
            f'{fmt_pct(p[2])} & '
            f'{fmt_pct(p[3])} & '
            f'{fmt_pct(p[4])} \\\\'
        )
        lines.append(line)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)

# ============================================================================
# TABLE 6: tab:scalability
# ============================================================================

def table_scalability(summaries):
    table = summaries['table'].sort_values(['dataset', 'n_skus_requested'])
    rows_per_ds = table.groupby('dataset').size().to_dict()

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{llrrrrrrr}',
        r'\toprule',
        (r'Dataset & $|\mathcal{S}|$ & $|\mathcal{B}|$ & BFD bins & $\eta_{\text{BFD}}$ (\%) & '
         r'Gap BFD--LP (\%) & CG time (h) \\'),
        r'\midrule',
    ]

    prev_ds = None
    ds_row_count = {}

    for _, row in table.iterrows():
        ds = row['dataset']
        n_size = int(row['n_skus_requested'])

        if prev_ds is not None and ds != prev_ds:
            lines.append(r'\midrule')

        if ds not in ds_row_count:
            ds_row_count[ds] = 0
            ds_cell = f'\\multirow{{{rows_per_ds[ds]}}}{{*}}{{{ds}}}'
        else:
            ds_cell = ''
        ds_row_count[ds] += 1
        prev_ds = ds

        gap_str = fmt_gap(
            row['gap_mean'], row['gap_std'],
            row['n_seeds_converged'], row['n_seeds_total'], row['is_full']
        )

        line = (
            f'{ds_cell} & '
            f'{n_size:,} & '
            f'{row["n_blocks"]:,} & '
            f'{row["bfd_bins"]:,} & '
            f'{row["eta_bfd"]:.2f} & '
            f'{gap_str} & '
            f'{fmt_time_h(row["cg_time_h"])} \\\\'
        )
        lines.append(line)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        (r'\caption{Scalability results ($m=4$). Gap BFD--LP reported as mean $\pm$ std over '
         r'random seeds. Dashes indicate instances where CG did not converge within the time '
         r'limit. $^\dagger$~Single seed (full instance). $^*$~Fewer than 5 seeds converged.}'),
        r'\label{tab:scalability}',
        r'\end{table}',
    ]
    return '\n'.join(lines)

# ============================================================================
# TABLE 7: tab:summary
# ============================================================================

def table_summary(df, summaries):
    gap = summaries['gap']
    rows = []

    for label in DS_ORDER:
        ds_data = df[df['dataset'] == label]
        ds_gap  = gap[gap['dataset'] == label]

        max_skus_total   = ds_data['n_skus_requested'].max()
        max_blocks_total = ds_data.loc[
            ds_data['n_skus_requested'] == max_skus_total, 'n_blocks'
        ].mean()

        if len(ds_gap) > 0:
            max_skus_cg   = ds_gap['n_skus_requested'].max()
            max_blocks_cg = ds_gap.loc[
                ds_gap['n_skus_requested'] == max_skus_cg, 'n_blocks'
            ].iloc[0]
            cg_rows = ds_data[ds_data['cg_converged']].copy()
            cg_rows['gap_bfd_lp'] = 100 * (cg_rows['bfd_vol'] - cg_rows['z_lp']) / cg_rows['z_lp']
            gap_mean = cg_rows['gap_bfd_lp'].mean()
            gap_std  = cg_rows['gap_bfd_lp'].std()
        else:
            max_skus_cg = max_blocks_cg = 0
            gap_mean = gap_std = float('nan')

        eta_mean = ds_data['eta_bfd'].mean()

        rows.append({
            'dataset': label,
            'max_skus_total': max_skus_total,
            'max_blocks_total': max_blocks_total,
            'max_skus_cg': max_skus_cg,
            'max_blocks_cg': max_blocks_cg,
            'gap_mean': gap_mean,
            'gap_std': gap_std,
            'eta_mean': eta_mean,
        })

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        (r'Dataset & $|\mathcal{S}|$ & $|\mathcal{B}|$ & Max $|\mathcal{S}|$ & '
         r'Max $|\mathcal{B}|$ & Gap BFD--LP (\%) & $\eta_{\text{BFD}}$ (\%) \\'),
        r'\midrule',
    ]
    for r in rows:
        line = (
            f'{r["dataset"]} & '
            f'{fmt_int(r["max_skus_total"])} & '
            f'{fmt_int(r["max_blocks_total"])} & '
            f'{fmt_int(r["max_skus_cg"])} & '
            f'{fmt_int(r["max_blocks_cg"])} & '
            f'{r["gap_mean"]:.2f} $\\pm$ {r["gap_std"]:.2f} & '
            f'{r["eta_mean"]:.2f} \\\\'
        )
        lines.append(line)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        (r'\caption{Summary by dataset. Max $|\mathcal{S}|$ and Max $|\mathcal{B}|$ indicate the '
         r'largest instance where column generation converged. Gap BFD--LP reported as '
         r'mean $\pm$ std over all converged sub-instances.}'),
        r'\label{tab:summary}',
        r'\end{table}',
    ]
    return '\n'.join(lines)

# ============================================================================
# FIGURE 5: fig:gap_skus — BFD-LP gap vs. number of SKUs
# ============================================================================

def figure_gap_skus(summaries):
    gap = summaries['gap']
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in DS_ORDER:
        rows = gap[gap['dataset'] == label]
        if rows.empty:
            continue
        ax.errorbar(
            rows['n_skus_requested'], rows['gap_mean'],
            yerr=rows['gap_std'].fillna(0),
            marker='o', label=label, color=DS_COLOR[label], capsize=4,
        )
    ax.set_xscale('log')
    ax.set_xlabel(r'Number of SKUs $|\mathcal{S}|$')
    ax.set_ylabel('BFD--LP gap (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out('figura_gap_skus.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)

# ============================================================================
# FIGURE 6: fig:variability_skus — coefficient of variation of the gap
# ============================================================================

def figure_variability_skus(summaries):
    variability = summaries['variability']
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in DS_ORDER:
        rows = variability[variability['dataset'] == label].dropna(subset=['gap_cv'])
        if rows.empty:
            continue
        ax.plot(
            rows['n_skus_requested'], rows['gap_cv'],
            marker='o', label=label, color=DS_COLOR[label],
        )
    ax.set_xscale('log')
    ax.set_xlabel(r'Number of SKUs $|\mathcal{S}|$')
    ax.set_ylabel('Coefficient of variation (%)')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out('figura_variabilidad_skus.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)

# ============================================================================
# FIGURE 7: fig:trivial_skus — bin reduction vs. the trivial singleton bound
# ============================================================================

def figure_trivial_skus(summaries):
    trivial = summaries['trivial']
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in DS_ORDER:
        rows = trivial[trivial['dataset'] == label]
        if rows.empty:
            continue
        ax.errorbar(
            rows['n_skus_requested'], rows['reduction_mean'],
            yerr=rows['reduction_std'].fillna(0),
            marker='o', label=label, color=DS_COLOR[label], capsize=4,
        )
    ax.set_xscale('log')
    ax.set_xlabel(r'Number of SKUs $|\mathcal{S}|$')
    ax.set_ylabel('Percentage reduction in bins (%)')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out('figura_mejora_skus.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)

# ============================================================================
# FIGURE 8: fig:cgtime_skus — BFD runtime (solid) and CG runtime (dashed)
# ============================================================================

def figure_cgtime_skus(df, summaries):
    cg_time = summaries['cg_time']
    bfd_time = df.groupby(['dataset', 'n_skus_requested']).agg(
        bfd_time_mean=('bfd_time', 'mean'),
        bfd_time_std=('bfd_time', 'std'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in DS_ORDER:
        rows_bfd = bfd_time[bfd_time['dataset'] == label]
        ax.errorbar(
            rows_bfd['n_skus_requested'], rows_bfd['bfd_time_mean'] / 3600,
            yerr=rows_bfd['bfd_time_std'].fillna(0) / 3600,
            marker='o', linestyle='-', color=DS_COLOR[label], capsize=4,
            label=f'{label} BFD',
        )
        rows_cg = cg_time[cg_time['dataset'] == label]
        if rows_cg.empty:
            continue
        ax.errorbar(
            rows_cg['n_skus_requested'], rows_cg['cg_time_mean'] / 3600,
            yerr=rows_cg['cg_time_std'].fillna(0) / 3600,
            marker='s', linestyle='--', color=DS_COLOR[label], capsize=4,
            label=f'{label} CG',
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of SKUs $|\mathcal{S}|$')
    ax.set_ylabel('Runtime (hours)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out('figura_cgtime_skus.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)

# ============================================================================
# MAIN
# ============================================================================

def main():
    # ---- Per-dataset full-scale data (Tables 1-4) ----
    try:
        bfd = {label: load_bfd_bins(label) for label in DS_ORDER}
        sc  = {label: load_scalability_full(label) for label in DS_ORDER}
        cg  = pd.read_csv(results_path('DS-1', 'DS-1_cg_summary.csv')).iloc[0]
        ds_sku_info   = {label: load_sku_info(label) for label in DS_ORDER}
        ds_block_info = {label: load_block_info(label) for label in DS_ORDER}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f'{e}\n\n'
            'Tables 1-4 require the full-scale results for all four datasets '
            f'({", ".join(DS_ORDER)}): data/{{DS-X}}_skus.csv, '
            'data/{DS-X}_blocks.csv, results/{DS-X}/{DS-X}_bfd_bins.csv, '
            'results/scalability_results_{DS-X}.csv (full-instance row), and '
            'results/DS-1/DS-1_cg_summary.csv. Run the pipeline for every '
            'dataset before generating these tables.'
        ) from e

    tables = {
        'table_datasets_skus.tex':     table_datasets_skus(sc, ds_sku_info),
        'table_datasets_blocks.tex':   table_datasets_blocks(sc, ds_block_info),
        'table_fullscale_results.tex': table_fullscale_results(sc, bfd, cg),
        'table_bfd_packing.tex':       table_bfd_packing(bfd),
    }

    # ---- Scalability sub-instances (Tables 6-7, Figures 5-8) ----
    df_scalability = load_scalability_all()
    summaries = compute_scalability_summaries(df_scalability)

    tables['table_scalability.tex'] = table_scalability(summaries)
    tables['table_summary.tex']     = table_summary(df_scalability, summaries)

    for fname, content in tables.items():
        fpath = out(fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'\u2713 {fname}\n')
        print(content)
        print()

    figure_gap_skus(summaries)
    figure_variability_skus(summaries)
    figure_trivial_skus(summaries)
    figure_cgtime_skus(df_scalability, summaries)
    print('\u2713 figura_gap_skus.pdf')
    print('\u2713 figura_variabilidad_skus.pdf')
    print('\u2713 figura_mejora_skus.pdf')
    print('\u2713 figura_cgtime_skus.pdf')


if __name__ == '__main__':
    main()