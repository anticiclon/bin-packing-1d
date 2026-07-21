# -*- coding: utf-8 -*-
# run_scalability.py
import os
import sys
import time
import signal
import logging
from datetime import datetime
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

import bfd_heuristic as bfd
import column_generation as cg
import restricted_ip as rip




# ── Experimental design ──────────────────────────────────────────────────────
M = 4
SEEDS = [0, 1, 2, 3, 4]

# Sizes per dataset on a geometric scale (the last one = full instance)
SIZES = {
    'DS-1': [50, 100, 250, 500, 1000, 1556],                          
    'DS-2': [100, 500, 2000, 10000, 50000, 94556],                    
    'DS-3': [100, 500, 2000, 10000, 50000, 200000, 503931],           
    'DS-4': [100, 500, 2000, 10000, 50000, 200000, 500000, 1520442],
}

# Which datasets to run (comment out the ones you don't want)
DATASETS = ['DS-1', 'DS-2', 'DS-3', 'DS-4']

# Column generation is only attempted up to this many SKUs. Beyond this size
# CG is assumed not to converge within CG_WALL_LIMIT (confirmed empirically:
# the RMP itself can take days to solve for instances above this scale), so
# the CG/RIP steps are skipped entirely and only BFD is run, saving the
# wall-clock budget for sizes where convergence is actually possible.
CG_MAX_SKUS = 2000

# Time limits
CG_WALL_LIMIT  = 300*3600   # seconds: if CG hasn't finished in 2h, interrupt it
RIP_TIME_LIMIT = 24*3600   # seconds for the RIP

# Where to save
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR    = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_DIR, "results")



class CGTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise CGTimeout()

def run_cg_with_timeout(data_cg: dict, cg_config: 'cg.ColumnGenerationConfig',
                        timeout_s: int) -> Optional['cg.ColumnGenerationResult']:
    if not hasattr(signal, 'SIGALRM'):
        # Windows: run without a timeout
        print("  ⚠ SIGALRM not available — running CG without a timeout")
        return cg.run_column_generation(data_cg, cg_config)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        result = cg.run_column_generation(data_cg, cg_config)
        signal.alarm(0)
        return result
    except CGTimeout:
        print(f"  ⚠ CG interrupted by timeout ({timeout_s}s)")
        return None
    finally:
        signal.alarm(0)


def build_rip_data_from_cg_result(cg_result: 'cg.ColumnGenerationResult') -> dict:
    """
    Builds the dict that solve_restricted_ip expects, directly from the
    ColumnGenerationResult object (without reading a CSV).
    """
    patterns     = []
    block_to_pids = defaultdict(list)
    all_blocks   = set()

    for p in cg_result.patterns:
        pid    = p.pattern_id
        lp_val = cg_result.lp_solution.get(pid, 0.0)

        patterns.append({
            'pattern_id':  pid,
            'bin_type_id': p.bin_type_id,
            'cost':        p.cost,
            'blocks':      p.blocks,     # list of (sku_id, block_k)
            'lp_val':      lp_val,
        })
        for b in p.blocks:
            block_to_pids[b].append(pid)
            all_blocks.add(b)

    return {
        'patterns':      patterns,
        'B':             sorted(all_blocks),
        'block_to_pids': dict(block_to_pids),
    }


def run_bfd_internal(df_b, df_tipos_bin, df_x, m, theta=0.05):
    """
    Runs the full BFD pipeline (pre-filter + BFD) using DataFrames already
    loaded in memory.

    Returns: solution, analysis, data, bfd_config,
             n_isolated, n_bfd_blocks, time_prefilter, time_bfd
    """
    bfd_config = bfd.BFDConfig(m=m, theta=theta)

    # 1. Build structures
    data = bfd.prepare_bfd_data(df_b, df_tipos_bin, df_x)
    n_total_blocks = len(data['B'])

    # 2. Pre-filter: identify isolated blocks
    t0 = time.time()
    isolated_blocks  = bfd.identify_isolated_blocks(data)
    time_prefilter   = time.time() - t0
    n_isolated       = len(isolated_blocks)
    n_bfd_blocks     = n_total_blocks - n_isolated
    isolated_bins_list = bfd.assign_isolated_blocks(isolated_blocks, data)

    # 3. BFD over the non-isolated blocks
    if n_bfd_blocks > 0:
        bfd_blocks = [b for b in data['B'] if b not in isolated_blocks]
        bfd_skus   = list(set(s for s, k in bfd_blocks))

        # Save the original state of data before modifying it
        original_B     = data['B']
        original_S     = data['S']
        original_x_min = data['x_min_global']

        # BFD operates on the non-isolated subset
        data['B'] = bfd_blocks
        data['S'] = bfd_skus

        # Recompute x_min for the BFD subset
        bfd_x_vals = [
            data['X_star'][(block, t)]
            for block in bfd_blocks
            for t in data['compatible_types'].get(block, [])
        ]
        if bfd_x_vals:
            data['x_min_global'] = float(min(bfd_x_vals))

        start_id = len(isolated_bins_list)
        bfd_sol  = bfd.solve_bfd_with_search(data, bfd_config,
                                              start_bin_id=start_id)
        time_bfd = bfd_sol.runtime

        # Restore data to the full set (minus infeasible blocks)
        infeasible_set = set(data.get('infeasible_blocks', []))
        all_feasible   = [b for b in original_B if b not in infeasible_set]
        data['B']             = all_feasible
        data['S']             = list(set(s for s, k in all_feasible))
        data['x_min_global']  = original_x_min
    else:
        bfd_sol  = bfd.BFDSolution(bins=[], total_volume=0,
                                   total_item_volume=0, n_bins=0,
                                   efficiency=0, runtime=0)
        time_bfd = 0.0

    # 4. Combine and analyze
    solution = bfd.combine_solutions(isolated_bins_list, bfd_sol, data)
    analysis = bfd.analyze_solution(solution, data)

    return (solution, analysis, data, bfd_config,
            n_isolated, n_bfd_blocks, time_prefilter, time_bfd)




def main(datasets=None, sizes=None, out_suffix=""):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Allow overriding datasets/sizes from argparse
    _datasets = datasets if datasets is not None else DATASETS
    _sizes    = sizes    if sizes    is not None else SIZES

    all_results = {}  # dataset_id -> DataFrame, returned to the caller

    for dataset_id in _datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_id.upper()}")
        print(f"{'='*70}")

        # One output CSV per dataset. Each dataset is resumed independently:
        # interrupting (or re-running) DS-3 never touches DS-1's file, and
        # downstream scripts (create_tables.py) read exactly this filename.
        output_csv = os.path.join(
            RESULTS_DIR, f"scalability_results_{dataset_id}{out_suffix}.csv"
        )

        # Resume: load previous results for this dataset if they exist
        if os.path.exists(output_csv):
            df_prev  = pd.read_csv(output_csv)
            # Add missing columns from older CSV formats
            for col in ['cg_attempted', 'cg_converged', 'bfd_bins']:
                if col not in df_prev.columns:
                    df_prev[col] = None
            # Rows where CG actually converged
            cg_done = df_prev['cg_converged'] == True
            # Rows where BFD failed (no point retrying)
            bfd_failed = df_prev['bfd_bins'].isna()
            # Rows where n_skus exceeded CG_MAX_SKUS: BFD-only by design,
            # not a failure, so they should not be retried either
            cg_not_attempted = (
                df_prev['cg_attempted'] == False
                if 'cg_attempted' in df_prev.columns else False
            )

            completed = set(zip(
                df_prev[cg_done | bfd_failed | cg_not_attempted]['dataset'],
                df_prev[cg_done | bfd_failed | cg_not_attempted]['n_skus_requested'],
                df_prev[cg_done | bfd_failed | cg_not_attempted]['seed']
            ))
            results = df_prev.to_dict('records')
            print(f"  Resuming {dataset_id}: {len(completed)} experiments already completed")
        else:
            completed = set()
            results   = []

        # ── Load parquet files ONCE per dataset ──────────────────────
        data_dir = os.path.join(REPO_DIR, "final_data", dataset_id)
        df_bloques_full, df_tipos_bin, df_X_full = bfd.load_model_data(
            dataset_id, data_dir
        )
        all_skus     = df_bloques_full['sku_id'].unique()
        n_total_skus = len(all_skus)
        print(f"  Total SKUs: {n_total_skus:,}")

        for n_skus in _sizes[dataset_id]:
            # If this is the full instance, only one seed (no point replicating)
            is_full      = (n_skus >= n_total_skus)
            seeds_to_run = [0] if is_full else SEEDS

            for seed in seeds_to_run:
                if (dataset_id, n_skus, seed) in completed:  
                    print(f"  → Already completed, skipping") 
                    continue                
                print(f"\n{'─'*60}")
                print(f"  {dataset_id} | size={n_skus:,} SKUs | seed={seed}")
                print(f"{'─'*60}")

                # ── a. Sample SKUs ──────────────────────────────────────
                if is_full:
                    selected = all_skus.tolist()
                else:
                    rng      = np.random.default_rng(seed)
                    selected = rng.choice(all_skus, size=n_skus,
                                         replace=False).tolist()

                # ── b. Filter DataFrames ──────────────────────────────────
                df_b   = df_bloques_full[df_bloques_full['sku_id'].isin(selected)].copy()
                b_ids  = set(df_b['block_id'].unique())
                df_x   = df_X_full[df_X_full['block_id'].isin(b_ids)].copy()
                # df_tipos_bin is passed in full: no need to filter it

                # ── Results directory for this experiment ─────────
                exp_dir = os.path.join(
                    RESULTS_DIR, dataset_id, "scalability",
                    f"size_{n_skus}_seed_{seed}"
                )
                os.makedirs(exp_dir, exist_ok=True)

                # Base results row
                row = {
                    'dataset': dataset_id,
                    'n_skus_requested': n_skus,
                    'seed': seed,
                }

                # ══════════════════════════════════════════════════════════
                # c. BFD STEP
                # ══════════════════════════════════════════════════════════
                try:
                    t_bfd_wall = time.time()
                    (solution, analysis, data_bfd, bfd_config,
                     n_isolated, n_bfd_blocks,
                     time_pf, time_bfd) = run_bfd_internal(
                        df_b, df_tipos_bin, df_x, M
                    )
                    t_bfd_wall = time.time() - t_bfd_wall

                    n_blocks_total = n_isolated + n_bfd_blocks
                    isolated_pct   = 100 * n_isolated / max(1, n_blocks_total)

                    row.update({
                        'n_skus_actual':  len(selected),
                        'n_blocks':       n_blocks_total,
                        'isolated_pct':   round(isolated_pct, 2),
                        'bfd_bins':       solution.n_bins,
                        'bfd_vol':        round(solution.total_volume, 2),
                        'eta_bfd':        round(solution.efficiency, 4),
                        'bfd_time':       round(t_bfd_wall, 2),
                    })

                    # Save BFD in exp_dir (CG reads it next)
                    bfd.save_solution(
                        solution, analysis, data_bfd, bfd_config,
                        dataset_id, exp_dir,
                        n_isolated, n_bfd_blocks, time_pf, time_bfd
                    )
                    
                    # Persist the row with BFD already done, before attempting CG
                    results.append(row)
                    _save_intermediate(results, output_csv)

                except Exception as e:
                    print(f"  ✗ BFD failed: {e}")
                    row.update({k: None for k in [
                        'n_skus_actual', 'n_blocks', 'isolated_pct',
                        'bfd_bins', 'bfd_vol', 'eta_bfd', 'bfd_time',
                    ]})
                    results.append(row)
                    _save_intermediate(results, output_csv)
                    continue   # skip CG and RIP

                # ══════════════════════════════════════════════════════════
                # d. CG STEP — only attempted up to CG_MAX_SKUS
                # ══════════════════════════════════════════════════════════
                if n_skus > CG_MAX_SKUS:
                    print(f"  ⊘ CG skipped (n_skus={n_skus:,} > "
                          f"CG_MAX_SKUS={CG_MAX_SKUS:,}); BFD-only row")
                    row.update({
                        'cg_attempted': False,
                        'cg_iters': None, 'cg_cols': None,
                        'z_lp': None, 'cg_time': None,
                        'cg_converged': False,
                    })
                else:
                    cg_config = cg.ColumnGenerationConfig(
                        dataset_id              = dataset_id,
                        max_skus_per_bin        = M,
                        initialization_strategy = 'bfd',
                        results_dir             = exp_dir,   # reads BFD and writes CG here
                        max_iterations          = 5000,
                        pricing_time_limit      = 60.0,
                        verbose                 = 1,
                        log_interval            = 10,
                    )

                    cg_result = None
                    try:
                        data_cg   = cg.prepare_column_generation_data(df_b, df_tipos_bin, df_x)
                        cg_result = run_cg_with_timeout(data_cg, cg_config,
                                                        CG_WALL_LIMIT)

                        if cg_result is not None:
                            converged = cg_result.converged
                            row.update({
                                'cg_attempted': True,
                                'cg_iters':    cg_result.n_iterations,
                                'cg_cols':     cg_result.n_patterns,
                                'z_lp':        round(cg_result.lower_bound, 2),
                                'cg_time':     round(cg_result.runtime, 2),
                                'cg_converged': converged,
                            })
                            cg.save_column_generation_results(cg_result, data_cg, cg_config)
                        else:
                            row.update({
                                'cg_attempted': True,
                                'cg_iters': None, 'cg_cols': None,
                                'z_lp': None,
                                'cg_time': f'>{CG_WALL_LIMIT}',
                                'cg_converged': False,
                            })

                    except Exception as e:
                        print(f"  ✗ CG failed: {e}")
                        row.update({
                            'cg_attempted': True,
                            'cg_iters': None, 'cg_cols': None,
                            'z_lp': None, 'cg_time': None,
                            'cg_converged': False,
                        })

                # ══════════════════════════════════════════════════════════
                # e. RIP STEP (only if CG converged)
                # ══════════════════════════════════════════════════════════
                if False:  # RIP disabled for sub-instances
                    try:
                        rip_data   = build_rip_data_from_cg_result(cg_result)
                        rip_result = rip.solve_restricted_ip(
                            rip_data,
                            time_limit = RIP_TIME_LIMIT,
                            mip_gap    = 0.001,
                            lp_lb      = cg_result.lower_bound,
                        )
                        if not rip_result:
                            raise RuntimeError("RIP returned no solution")   # ← add
                        z_lp = cg_result.lower_bound
                        z_ip = rip_result['z_ip']
                        z_bfd = solution.total_volume

                        row.update({
                            'z_ip':       round(z_ip, 2),
                            'rip_time':   round(rip_result['runtime'], 2),
                            'gap_bfd_lp': round(100*(z_bfd - z_lp)/z_lp, 4)
                                          if z_lp > 0 else None,
                            'gap_lp_ip':  round(100*(z_ip  - z_lp)/z_lp, 4)
                                          if z_lp > 0 else None,
                        })
                        rip.save_rip_results(
                            rip_result, rip_data,
                            dataset_id, cg_result.lower_bound, exp_dir
                        )

                    except Exception as e:
                        print(f"  ✗ RIP failed: {e}")
                        row.update({k: None for k in
                                    ['z_ip', 'rip_time', 'gap_bfd_lp', 'gap_lp_ip']})
                else:
                    row.update({k: None for k in
                                ['z_ip', 'rip_time', 'gap_bfd_lp', 'gap_lp_ip']})

                # ── f. Save row ────────────────────────────────────────
                # ── f. Save row (row is already in results; just re-flush) ──
                _save_intermediate(results, output_csv)
                print(f"  → Row saved. Accumulated results: {len(results)}")

        print(f"\n✓ {dataset_id} complete. CSV: {output_csv}")
        all_results[dataset_id] = pd.DataFrame(results)

    return all_results


def _save_intermediate(results, path):
    """Saves the partial results CSV after each sub-instance."""
    df = pd.DataFrame(results)
    # Keep the last row per (dataset, n_skus_requested, seed)
    df = df.drop_duplicates(
        subset=['dataset', 'n_skus_requested', 'seed'],
        keep='last'
    )
    df.to_csv(path, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='BFD+CG+RIP scalability experiments'
    )
    parser.add_argument(
        '--datasets', nargs='+',
        choices=['DS-1', 'DS-2', 'DS-3', 'DS-4'],
        default=None,
        help='Datasets to run (default: all)'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=None,
        help='Seeds to run (default: all)'
    )
    parser.add_argument(
        '--sizes', nargs='+', type=int,
        default=None,
        help='Sizes to run (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.seeds is not None:
        SEEDS = args.seeds

    if args.datasets is not None:
        selected_datasets = args.datasets
        selected_sizes    = {d: SIZES[d] for d in selected_datasets}
    else:
        selected_datasets = DATASETS
        selected_sizes    = SIZES
        
    if args.sizes is not None:
        selected_sizes = {d: [s for s in selected_sizes[d] if s in args.sizes]
                          for d in selected_datasets}

    suffix = ""
    if args.sizes is not None:
        suffix += "_sz" + "-".join(map(str, args.sizes))
    if args.seeds is not None:
        suffix += "_sd" + "-".join(map(str, args.seeds))
    # NOTE: a non-empty suffix changes the output filename to
    # scalability_results_{DS-X}{suffix}.csv, which create_tables.py will
    # NOT pick up (it reads scalability_results_{DS-X}.csv with no suffix).
    # Use --sizes/--seeds for ad-hoc/debugging runs; for the results that
    # feed create_tables.py, run without --sizes/--seeds (or merge the
    # suffixed file back into scalability_results_{DS-X}.csv afterwards).
    main(datasets=selected_datasets, sizes=selected_sizes, out_suffix=suffix)