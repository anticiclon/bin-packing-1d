# -*- coding: utf-8 -*-
"""
Construction, Solution, and Analysis of the MILP Model
Bin Sizing Problem via Dimensional Decomposition

This module implements:
- MILP model construction (formulation from the LaTeX document)
- Configuration and solution using Gurobi
- Analysis and post-processing of the solution

Usage:
    python build_model.py <dataset_id>
    python build_model.py DS-1 --solve --time_limit 3600

Input:
    - final_data/{dataset_id}/*_blocks.parquet
    - final_data/{dataset_id}/*_bin_types.parquet
    - final_data/{dataset_id}/*_X_optimal.parquet

Output:
    - Gurobi model (.lp)
    - Solution in CSV
    - Detailed analysis (metrics, reports)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("ERROR: Gurobi is not installed.")
    print("Install with: pip install gurobipy")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """MILP model configuration."""
    dataset_id: str
    m: int = 3                    # Maximum distinct SKUs per bin
    time_limit: float = 3600.0    # Time limit in seconds
    mip_gap: float = 0.01         # Optimality gap (1%)
    threads: int = 0              # 0 = automatic
    output_flag: int = 1          # 1 = show Gurobi log
    presolve: int = -1            # -1 = automatic
    write_model: bool = True      # Save model as .lp
    solve: bool = False           # Solve the model
    max_skus: Optional[int] = None  # Limit number of SKUs
    selected_skus: Optional[List[str]] = None  # Specific SKU list


# ============================================================================
# DATA LOADING
# ============================================================================

def load_model_data(dataset_id: str, base_dir: str = None):
    """
    Load preprocessed data from Parquet files.
    """
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(script_dir)
        base_dir = os.path.join(repo_dir, "final_data", dataset_id)
    
    blocks_path = os.path.join(base_dir, f"{dataset_id}_blocks.parquet")
    bin_types_path = os.path.join(base_dir, f"{dataset_id}_bin_types.parquet")
    X_optimal_path = os.path.join(base_dir, f"{dataset_id}_X_optimal.parquet")
    
    for path in [blocks_path, bin_types_path, X_optimal_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
    
    df_blocks = pd.read_parquet(blocks_path)
    df_bin_types = pd.read_parquet(bin_types_path)
    df_X_optimal = pd.read_parquet(X_optimal_path)
    
    return df_blocks, df_bin_types, df_X_optimal


def prepare_model_data(
    df_blocks: pd.DataFrame,
    df_bin_types: pd.DataFrame,
    df_X_optimal: pd.DataFrame
) -> Dict:
    """
    Prepare data structures optimized for the model.
    """
    data = {}
    
    print("    Preparing sets...")
    
    # S: set of SKUs
    data['S'] = df_blocks['sku_id'].unique().tolist()
    data['n_skus'] = len(data['S'])
    
    # B: set of blocks
    data['B'] = list(zip(df_blocks['sku_id'], df_blocks['block_k']))
    data['n_blocks'] = len(data['B'])
    
    # T: bin types
    data['T'] = df_bin_types['bin_type_id'].tolist()
    data['n_bin_types'] = len(data['T'])
    
    print("    Creating mappings...")
    
    data['block_id_to_tuple'] = dict(zip(
        df_blocks['block_id'],
        zip(df_blocks['sku_id'], df_blocks['block_k'])
    ))
    
    data['tuple_to_block_id'] = {v: k for k, v in data['block_id_to_tuple'].items()}
    
    # blocks per SKU
    blocks_per_sku_df = df_blocks.groupby('sku_id')['block_k'].apply(list).to_dict()
    data['blocks_per_sku'] = {
        s: [(s, k) for k in ks] for s, ks in blocks_per_sku_df.items()
    }
    
    data['K_s'] = {s: len(blocks) for s, blocks in data['blocks_per_sku'].items()}
    
    print("    Extracting block parameters...")
    
    # q_sk
    data['q_sk'] = dict(zip(
        zip(df_blocks['sku_id'], df_blocks['block_k']),
        df_blocks['q_sk'].astype(int)
    ))
    
    dims_per_sku = df_blocks.groupby('sku_id')[['l_s', 'w_s', 'h_s']].first()
    data['l_s'] = dims_per_sku['l_s'].to_dict()
    data['w_s'] = dims_per_sku['w_s'].to_dict()
    data['h_s'] = dims_per_sku['h_s'].to_dict()
    
    print("    Extracting bin parameters...")
    
    data['L_t'] = dict(zip(df_bin_types['bin_type_id'], df_bin_types['L_t']))
    data['W_t'] = dict(zip(df_bin_types['bin_type_id'], df_bin_types['W_t']))
    data['H_t'] = dict(zip(df_bin_types['bin_type_id'], df_bin_types['H_t']))
    data['Vol_t'] = dict(zip(df_bin_types['bin_type_id'], df_bin_types['volume']))
    
    print("    Processing X* matrix...")
    
    valid_block_ids = set(data['block_id_to_tuple'].keys())
    df_X_valid = df_X_optimal[df_X_optimal['block_id'].isin(valid_block_ids)]
    
    block_tuples = df_X_valid['block_id'].map(data['block_id_to_tuple'])
    
    data['X_star'] = dict(zip(
        zip(block_tuples, df_X_valid['bin_type_id']),
        df_X_valid['X_opt']
    ))
    
    data['feasible_pairs'] = set(data['X_star'].keys())
    
    print("    Computing total volume...")
    
    df_blocks_vol = df_blocks.copy()
    df_blocks_vol['vol_item'] = df_blocks_vol['l_s'] * df_blocks_vol['w_s'] * df_blocks_vol['h_s']
    df_blocks_vol['vol_total'] = df_blocks_vol['q_sk'] * df_blocks_vol['vol_item']
    data['total_item_volume'] = df_blocks_vol['vol_total'].sum()
    
    data['J_max'] = data['n_blocks']
    data['J'] = list(range(data['J_max']))
    
    return data

# ============================================================================
# MODEL CONSTRUCTION
# ============================================================================

def build_model(data: Dict, config: ModelConfig) -> gp.Model:
    """Builds the MILP model in Gurobi."""
    print("\n" + "=" * 70)
    print("BUILDING MILP MODEL")
    print("=" * 70)
    
    S = data['S']
    B = data['B']
    T = data['T']
    J = data['J']
    m = config.m
    
    K_s = data['K_s']
    L_t = data['L_t']
    Vol_t = data['Vol_t']
    X_star = data['X_star']
    feasible_pairs = data['feasible_pairs']
    blocks_per_sku = data['blocks_per_sku']
    total_item_volume = data['total_item_volume']
    
    print(f"\nProblem dimensions:")
    print(f"  |S| = {len(S):,} SKUs")
    print(f"  |B| = {len(B):,} blocks")
    print(f"  |T| = {len(T)} bin types")
    print(f"  |J| = {len(J):,} bins (upper bound)")
    print(f"  m = {m} max SKUs per bin")
    print(f"  Feasible pairs (block, type): {len(feasible_pairs):,}")
    
    model = gp.Model(f"BinPacking_{config.dataset_id}")
    
    model.Params.TimeLimit = config.time_limit
    model.Params.MIPGap = config.mip_gap
    model.Params.Threads = config.threads
    model.Params.OutputFlag = config.output_flag
    model.Params.Presolve = config.presolve
    
    print("\n" + "-" * 70)
    print("CREATING VARIABLES")
    print("-" * 70)
    
    print("  Creating p_{sk,j} variables (block-bin assignment)...")
    p = model.addVars(((s, k, j) for (s, k) in B for j in J), vtype=GRB.BINARY, name="p")
    print(f"    {len(p):,} p variables")
    
    print("  Creating u_j variables (bin usage)...")
    u = model.addVars(J, vtype=GRB.BINARY, name="u")
    print(f"    {len(u):,} u variables")
    
    print("  Creating b_{j,t} variables (bin type)...")
    b = model.addVars(J, T, vtype=GRB.BINARY, name="b")
    print(f"    {len(b):,} b variables")
    
    print("  Creating ω_{s,j} variables (SKU presence)...")
    omega = model.addVars(S, J, vtype=GRB.BINARY, name="omega")
    print(f"    {len(omega):,} ω variables")
    
    print("  Creating α_{sk,j,t} variables (linearization)...")
    alpha = model.addVars(((s, k, j, t) for ((s, k), t) in feasible_pairs for j in J), 
                          vtype=GRB.BINARY, name="alpha")
    print(f"    {len(alpha):,} α variables")
    
    total_vars = len(p) + len(u) + len(b) + len(omega) + len(alpha)
    print(f"\n  Total variables: {total_vars:,}")
    
    model.update()
    
    print("\n" + "-" * 70)
    print("CREATING CONSTRAINTS")
    print("-" * 70)
    
    print("  (2) Complete block assignment...")
    model.addConstrs(
        (gp.quicksum(p[(s, k, j)] for j in J) == 1 for (s, k) in B),
        name="asig"
    )
    print(f"    {len(B):,} constraints")
    
    print("  (3) Bin activation...")
    model.addConstrs(
        (p[(s, k, j)] <= u[j] for (s, k) in B for j in J),
        name="activ"
    )
    print(f"    {len(B) * len(J):,} constraints")
    
    print("  (4) Unique type per bin...")
    model.addConstrs(
        (gp.quicksum(b[(j, t)] for t in T) == u[j] for j in J),
        name="type"
    )
    print(f"    {len(J):,} constraints")
    
    print("  (5) Definition of SKU presence...")
    model.addConstrs(
        (omega[(s, j)] >= p[(sku, k, j)] for s in S for (sku, k) in blocks_per_sku[s] for j in J),
        name="pres"
    )
    count = sum(len(blocks_per_sku[s]) for s in S) * len(J)
    print(f"    {count:,} constraints")
    
    print(f"  (6) Limit of {m} SKUs per bin...")
    model.addConstrs(
        (gp.quicksum(omega[(s, j)] for s in S) <= m for j in J),
        name="max_skus"
    )
    print(f"    {len(J):,} constraints")
    
    print("  (7) Block uniqueness per SKU in each bin...")
    model.addConstrs(
        (gp.quicksum(p[(sku, k, j)] for (sku, k) in blocks_per_sku[s]) <= 1
         for s in S for j in J if len(blocks_per_sku[s]) > 1),
        name="unic"
    )
    count = sum(1 for s in S if len(blocks_per_sku[s]) > 1) * len(J)
    print(f"    {count:,} constraints")
    
    # ========================================================================
    # BLOCK-TYPE COMPATIBILITY CONSTRAINT (CRITICAL)
    # ========================================================================
    # A block can only be assigned to a bin if the bin is of a compatible type
    # p[s,k,j] <= sum_{t: (sk,t) in feasible_pairs} b[j,t]
    
    print("  Block-type compatibility (critical constraint)...")
    
    # Preprocess: for each block, get its compatible types
    compatible_types_per_block = {}
    for ((s, k), t) in feasible_pairs:
        if (s, k) not in compatible_types_per_block:
            compatible_types_per_block[(s, k)] = []
        compatible_types_per_block[(s, k)].append(t)
    
    model.addConstrs(
        (p[(s, k, j)] <= gp.quicksum(b[(j, t)] for t in compatible_types_per_block.get((s, k), []))
         for (s, k) in B for j in J),
        name="compat"
    )
    print(f"    {len(B) * len(J):,} compatibility constraints")
    
    print("  Linearization of α = p * b...")
    model.addConstrs(
        (alpha[(s, k, j, t)] <= p[(s, k, j)] for ((s, k), t) in feasible_pairs for j in J),
        name="lin1"
    )
    model.addConstrs(
        (alpha[(s, k, j, t)] <= b[(j, t)] for ((s, k), t) in feasible_pairs for j in J),
        name="lin2"
    )
    model.addConstrs(
        (alpha[(s, k, j, t)] >= p[(s, k, j)] + b[(j, t)] - 1 for ((s, k), t) in feasible_pairs for j in J),
        name="lin3"
    )
    count = 3 * len(feasible_pairs) * len(J)
    print(f"    {count:,} linearization constraints")
    
    print("  (8) Capacity on X axis...")
    model.addConstrs(
        (gp.quicksum(X_star[(sk, t)] * alpha[(sk[0], sk[1], j, t)] for (sk, t) in feasible_pairs) <=
         gp.quicksum(L_t[t] * b[(j, t)] for t in T) for j in J),
        name="cap_X"
    )
    print(f"    {len(J):,} constraints")
    
    print("  (9) Symmetry breaking...")
    model.addConstrs(
        (u[j] >= u[j + 1] for j in J[:-1]),
        name="sim"
    )
    print(f"    {len(J) - 1:,} constraints")
    
    print("\n  Valid constraints (strengthening):")
    
    print("  - Lower volumetric bound...")
    model.addConstr(
        gp.quicksum(b[(j, t)] * Vol_t[t] for j in J for t in T) >= total_item_volume,
        name="cota_vol"
    )
    
    print("  - SKU presence bound...")
    model.addConstrs(
        (gp.quicksum(omega[(s, j)] for j in J) >= K_s[s] for s in S if K_s[s] > 1),
        name="cota_sku"
    )
    count = sum(1 for s in S if K_s[s] > 1)
    print(f"    {count:,} constraints")
    
    print("\n" + "-" * 70)
    print("OBJECTIVE FUNCTION")
    print("-" * 70)
    print("  Minimize total bin volume")
    
    obj = gp.quicksum(b[(j, t)] * Vol_t[t] for j in J for t in T)
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.update()
    
    print("\n" + "-" * 70)
    print("MODEL SUMMARY")
    print("-" * 70)
    print(f"  Variables:      {model.NumVars:,}")
    print(f"    - Binary:     {model.NumBinVars:,}")
    print(f"    - Continuous: {model.NumVars - model.NumBinVars:,}")
    print(f"  Constraints:   {model.NumConstrs:,}")
    print(f"  Non-zeros:     {model.NumNZs:,}")
    
    model._vars = {'p': p, 'u': u, 'b': b, 'omega': omega, 'alpha': alpha}
    model._data = data
    model._config = config
    
    return model


# ============================================================================
# SOLVING AND SOLUTION EXTRACTION
# ============================================================================

def solve_model(model: gp.Model) -> Optional[Dict]:
    """Solves the model and extracts the solution."""
    print("\n" + "=" * 70)
    print("SOLVING MODEL")
    print("=" * 70)
    
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        print(f"\n✓ Optimal solution found")
    elif model.Status == GRB.TIME_LIMIT:
        print(f"\n⚠ Time limit reached")
        if model.SolCount == 0:
            print("  No feasible solution found")
            return None
        print(f"  Best solution found with gap {model.MIPGap*100:.2f}%")
    elif model.Status == GRB.INFEASIBLE:
        print(f"\n✗ Infeasible model")
        return None
    else:
        print(f"\n✗ Solver status: {model.Status}")
        return None
    
    return extract_solution(model)


def extract_solution(model: gp.Model) -> Dict:
    """Extracts the solution from the solved model."""
    p = model._vars['p']
    u = model._vars['u']
    b = model._vars['b']
    data = model._data
    
    solution = {
        'objective': model.ObjVal,
        'mip_gap': model.MIPGap,
        'runtime': model.Runtime,
        'bins_used': 0,
        'assignments': [],
        'bins': {},
    }
    
    for j in data['J']:
        if u[j].X > 0.5:
            solution['bins_used'] += 1
            
            bin_type = None
            for t in data['T']:
                if b[(j, t)].X > 0.5:
                    bin_type = t
                    break
            
            solution['bins'][j] = {
                'type': bin_type,
                'L_t': data['L_t'][bin_type],
                'W_t': data['W_t'][bin_type],
                'H_t': data['H_t'][bin_type],
                'volume': data['Vol_t'][bin_type],
                'blocks': []
            }
    
    for (s, k) in data['B']:
        for j in data['J']:
            if p[(s, k, j)].X > 0.5:
                bin_info = solution['bins'].get(j, {})
                bin_type = bin_info.get('type', 'N/A')
                
                solution['assignments'].append({
                    'sku_id': s,
                    'block_k': k,
                    'bin_j': j,
                    'bin_type': bin_type,
                    'q_sk': data['q_sk'][(s, k)]
                })
                
                if j in solution['bins']:
                    solution['bins'][j]['blocks'].append((s, k))
                
                break
    
    return solution

# ============================================================================
# ANALYSIS AND POST-PROCESSING
# ============================================================================

def analyze_solution(solution: Dict, data: Dict) -> Dict:
    """Complete analysis of the model solution."""
    analysis = {}
    
    analysis['objective'] = solution['objective']
    analysis['bins_used'] = solution['bins_used']
    analysis['mip_gap'] = solution['mip_gap']
    analysis['runtime'] = solution['runtime']
    
    total_item_volume = data['total_item_volume']
    max_bin_volume = max(data['Vol_t'].values())
    min_bin_volume = min(data['Vol_t'].values())
    
    volume_lower_bound = int(np.ceil(total_item_volume / max_bin_volume))
    volume_upper_bound = int(np.ceil(total_item_volume / min_bin_volume))
    
    analysis['bounds'] = {
        'volume_lower': volume_lower_bound,
        'volume_upper': volume_upper_bound,
        'total_item_volume': total_item_volume,
        'max_bin_volume': max_bin_volume,
        'min_bin_volume': min_bin_volume
    }
    
    if volume_lower_bound > 0:
        analysis['gap_vs_bound'] = (solution['bins_used'] - volume_lower_bound) / volume_lower_bound * 100
    else:
        analysis['gap_vs_bound'] = 0
    
    type_stats = {}
    for j, info in solution['bins'].items():
        bin_type = info['type']
        if bin_type not in type_stats:
            type_stats[bin_type] = {
                'count': 0,
                'unit_volume': info['volume'],
                'dims': (info['L_t'], info['W_t'], info['H_t']),
                'total_blocks': 0,
                'total_items': 0
            }
        type_stats[bin_type]['count'] += 1
        type_stats[bin_type]['total_blocks'] += len(info['blocks'])
        
        for (s, k) in info['blocks']:
            type_stats[bin_type]['total_items'] += data['q_sk'][(s, k)]
    
    for bin_type in type_stats:
        type_stats[bin_type]['total_volume'] = type_stats[bin_type]['count'] * type_stats[bin_type]['unit_volume']
    
    analysis['by_type'] = type_stats
    
    utilizations = []
    for j, info in solution['bins'].items():
        bin_volume = info['volume']
        content_volume = 0
        
        for (s, k) in info['blocks']:
            q = data['q_sk'][(s, k)]
            item_volume = data['l_s'][s] * data['w_s'][s] * data['h_s'][s]
            content_volume += q * item_volume
        
        utilization = content_volume / bin_volume * 100 if bin_volume > 0 else 0
        utilizations.append({
            'bin_j': j,
            'type': info['type'],
            'bin_volume': bin_volume,
            'content_volume': content_volume,
            'utilization_pct': utilization,
            'n_blocks': len(info['blocks']),
            'n_skus': len(set(s for (s, k) in info['blocks']))
        })
    
    df_util = pd.DataFrame(utilizations)
    
    analysis['utilization'] = {
        'average': df_util['utilization_pct'].mean() if len(df_util) > 0 else 0,
        'median': df_util['utilization_pct'].median() if len(df_util) > 0 else 0,
        'min': df_util['utilization_pct'].min() if len(df_util) > 0 else 0,
        'max': df_util['utilization_pct'].max() if len(df_util) > 0 else 0,
        'std': df_util['utilization_pct'].std() if len(df_util) > 0 else 0,
        'detail': df_util
    }
    
    skus_per_bin = df_util['n_skus'].value_counts().sort_index().to_dict() if len(df_util) > 0 else {}
    blocks_per_bin = df_util['n_blocks'].value_counts().sort_index().to_dict() if len(df_util) > 0 else {}
    
    analysis['distribution'] = {
        'skus_per_bin': skus_per_bin,
        'blocks_per_bin_hist': blocks_per_bin,
        'average_skus_per_bin': df_util['n_skus'].mean() if len(df_util) > 0 else 0,
        'average_blocks_per_bin': df_util['n_blocks'].mean() if len(df_util) > 0 else 0
    }
    
    total_bin_volume = sum(info['volume'] for info in solution['bins'].values())
    global_efficiency = total_item_volume / total_bin_volume * 100 if total_bin_volume > 0 else 0
    
    analysis['efficiency'] = {
        'global_pct': global_efficiency,
        'total_item_volume': total_item_volume,
        'total_bin_volume': total_bin_volume,
        'waste_pct': 100 - global_efficiency,
        'waste_cm3': total_bin_volume - total_item_volume
    }
    
    return analysis


def print_analysis_report(analysis: Dict):
    """Prints a detailed analysis report."""
    print("\n" + "=" * 70)
    print("DETAILED SOLUTION ANALYSIS")
    print("=" * 70)
    
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " EXECUTIVE SUMMARY".center(68) + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Total bin volume:          {analysis['objective']:>15,.0f} cm³" + " " * 20 + "│")
    print(f"│  Bins used:                 {analysis['bins_used']:>15,}" + " " * 20 + "│")
    print(f"│  Volumetric efficiency:     {analysis['efficiency']['global_pct']:>14.1f}%" + " " * 20 + "│")
    print(f"│  MIP Gap:                   {analysis['mip_gap']*100:>14.2f}%" + " " * 20 + "│")
    print(f"│  Solution time:             {analysis['runtime']:>14.1f} s" + " " * 20 + "│")
    print("└" + "─" * 68 + "┘")
    
    print("\n" + "-" * 70)
    print("COMPARISON WITH LOWER BOUNDS")
    print("-" * 70)
    bounds = analysis['bounds']
    print(f"  Total item volume:            {bounds['total_item_volume']:>15,.0f} cm³")
    print(f"  Largest bin volume:           {bounds['max_bin_volume']:>15,.0f} cm³")
    print(f"  Lower bound (volumetric):     {bounds['volume_lower']:>15,} bins")
    print(f"  Bins used:                    {analysis['bins_used']:>15,} bins")
    print(f"  Gap vs lower bound:           {analysis['gap_vs_bound']:>14.1f}%")
    
    print("\n" + "-" * 70)
    print("DISTRIBUTION BY BIN TYPE")
    print("-" * 70)
    print(f"  {'Type':<12} {'Count':>6} {'Unit Vol.':>12} {'Total Vol.':>14} {'Blocks':>8} {'Items':>10}")
    print("  " + "-" * 64)
    
    total_bins = 0
    total_vol = 0
    total_blocks = 0
    total_items = 0
    
    for bin_type, stats in sorted(analysis['by_type'].items()):
        print(f"  {bin_type:<12} {stats['count']:>6,} {stats['unit_volume']:>12,.0f} "
              f"{stats['total_volume']:>14,.0f} {stats['total_blocks']:>8,} {stats['total_items']:>10,}")
        total_bins += stats['count']
        total_vol += stats['total_volume']
        total_blocks += stats['total_blocks']
        total_items += stats['total_items']
    
    print("  " + "-" * 64)
    print(f"  {'TOTAL':<12} {total_bins:>6,} {'-':>12} {total_vol:>14,.0f} {total_blocks:>8,} {total_items:>10,}")
    
    print("\n" + "-" * 70)
    print("VOLUMETRIC UTILIZATION")
    print("-" * 70)
    util = analysis['utilization']
    print(f"  Average:     {util['average']:>6.1f}%")
    print(f"  Median:      {util['median']:>6.1f}%")
    print(f"  Minimum:     {util['min']:>6.1f}%")
    print(f"  Maximum:     {util['max']:>6.1f}%")
    print(f"  Std. Dev.:   {util['std']:>6.1f}%")
    
    print("\n" + "-" * 70)
    print("SKU DISTRIBUTION PER BIN")
    print("-" * 70)
    dist = analysis['distribution']
    print(f"  Average SKUs/bin:    {dist['average_skus_per_bin']:.2f}")
    print(f"  Average blocks/bin:  {dist['average_blocks_per_bin']:.2f}")
    print("\n  Histogram of SKUs per bin:")
    for n_skus, count in sorted(dist['skus_per_bin'].items()):
        bar = "█" * min(count, 50)
        print(f"    {n_skus} SKUs: {count:>5} bins  {bar}")
    
    print("\n" + "-" * 70)
    print("WASTE ANALYSIS")
    print("-" * 70)
    eff = analysis['efficiency']
    print(f"  Item volume:         {eff['total_item_volume']:>15,.0f} cm³")
    print(f"  Bin volume:          {eff['total_bin_volume']:>15,.0f} cm³")
    print(f"  Wasted space:        {eff['waste_cm3']:>15,.0f} cm³ ({eff['waste_pct']:.1f}%)")
    
    print("\n" + "=" * 70)

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(model: gp.Model, solution: Optional[Dict], analysis: Optional[Dict], 
                 output_dir: str, config: ModelConfig):
    """Saves the model, solution, and analysis."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_id = config.dataset_id
    
    if config.write_model:
        lp_path = os.path.join(output_dir, f"{dataset_id}_model.lp")
        model.write(lp_path)
        print(f"\n  ✓ Model saved: {lp_path}")
    
    if solution:
        df_assign = pd.DataFrame(solution['assignments'])
        csv_path = os.path.join(output_dir, f"{dataset_id}_solution.csv")
        df_assign.to_csv(csv_path, index=False)
        print(f"  ✓ Solution saved: {csv_path}")
        
        if analysis:
            csv_util_path = os.path.join(output_dir, f"{dataset_id}_bin_utilization.csv")
            analysis['utilization']['detail'].to_csv(csv_util_path, index=False)
            print(f"  ✓ Utilization detail: {csv_util_path}")
            
            types_data = []
            for bin_type, stats in analysis['by_type'].items():
                types_data.append({
                    'bin_type': bin_type,
                    'count': stats['count'],
                    'unit_volume': stats['unit_volume'],
                    'total_volume': stats['total_volume'],
                    'total_blocks': stats['total_blocks'],
                    'total_items': stats['total_items'],
                    'dims_LxWxH': f"{stats['dims'][0]}x{stats['dims'][1]}x{stats['dims'][2]}"
                })
            df_types = pd.DataFrame(types_data)
            csv_types_path = os.path.join(output_dir, f"{dataset_id}_type_summary.csv")
            df_types.to_csv(csv_types_path, index=False)
            print(f"  ✓ Summary by type: {csv_types_path}")
            
            txt_path = os.path.join(output_dir, f"{dataset_id}_analysis_report.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"SOLUTION ANALYSIS - {dataset_id.upper()}\n")
                f.write("=" * 60 + "\n\n")
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total bin volume:         {analysis['objective']:,.0f} cm³\n")
                f.write(f"Bins used:                {analysis['bins_used']:,}\n")
                f.write(f"Volumetric efficiency:    {analysis['efficiency']['global_pct']:.1f}%\n")
                f.write(f"MIP Gap:                  {analysis['mip_gap']*100:.4f}%\n")
                f.write(f"Solution time:            {analysis['runtime']:.1f} s\n\n")
                f.write("BOUNDS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Lower bound (volume): {analysis['bounds']['volume_lower']:,} bins\n")
                f.write(f"Gap vs lower bound:       {analysis['gap_vs_bound']:.1f}%\n\n")
                f.write("VOLUMETRIC UTILIZATION\n")
                f.write("-" * 40 + "\n")
                util = analysis['utilization']
                f.write(f"Average: {util['average']:.1f}%\n")
                f.write(f"Median:  {util['median']:.1f}%\n")
                f.write(f"Minimum: {util['min']:.1f}%\n")
                f.write(f"Maximum: {util['max']:.1f}%\n\n")
                f.write("DISTRIBUTION BY BIN TYPE\n")
                f.write("-" * 40 + "\n")
                for bin_type, stats in sorted(analysis['by_type'].items()):
                    f.write(f"{bin_type}: {stats['count']} bins, {stats['total_blocks']} blocks, "
                           f"{stats['total_items']} items\n")
            print(f"  ✓ Full report: {txt_path}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(dataset_id: str, solve: bool = False, time_limit: float = 3600.0, 
         m: int = 3, max_skus: Optional[int] = None, 
         selected_skus: Optional[List[str]] = None):
    """
    Main function: loads data, builds, solves, and analyzes the model.
    
    Args:
        dataset_id: Dataset ID ('brmg01', 'mxjc01', etc.)
        solve: Whether to solve the model
        time_limit: Time limit in seconds
        m: Maximum number of SKUs per bin
        max_skus: Number of SKUs to select (the first N)
        selected_skus: Specific list of SKUs to use (takes priority over max_skus)
    """
    if not GUROBI_AVAILABLE:
        print("ERROR: Gurobi is not available")
        return None, None, None
    
    start_time = datetime.now()
    
    print("=" * 70)
    print(f"BIN PACKING MILP MODEL - {dataset_id.upper()}")
    print("=" * 70)
    
    config = ModelConfig(
        dataset_id=dataset_id,
        m=m,
        time_limit=time_limit,
        solve=solve,
        max_skus=max_skus,
        selected_skus=selected_skus
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(repo_dir, "final_data", dataset_id)
    output_dir = os.path.join(repo_dir, "results", dataset_id)
    
    print("\n" + "-" * 70)
    print("STEP 1: Loading preprocessed data")
    print("-" * 70)
    
    df_blocks, df_bin_types, df_optimal_X = load_model_data(dataset_id, data_dir)
    
    print(f"  Complete data:")
    print(f"    Blocks: {len(df_blocks):,}")
    print(f"    SKUs: {df_blocks['sku_id'].nunique():,}")
    print(f"    Pairs (block, type): {len(df_optimal_X):,}")
    
    # SKU filtering: selected_skus takes priority over max_skus
    if selected_skus is not None:
        print(f"\n  Filtering to specific SKUs: {selected_skus}")
        
        # Verify that the SKUs exist
        available_skus = set(df_blocks['sku_id'].unique())
        missing = [s for s in selected_skus if s not in available_skus]
        if missing:
            print(f"  ⚠ SKUs not found: {missing}")
            selected_skus = [s for s in selected_skus if s in available_skus]
        
        df_blocks = df_blocks[df_blocks['sku_id'].isin(selected_skus)]
        
        selected_block_ids = set(df_blocks['block_id'].unique())
        df_optimal_X = df_optimal_X[df_optimal_X['block_id'].isin(selected_block_ids)]
        
        print(f"  Filtered data:")
        print(f"    Blocks: {len(df_blocks):,}")
        print(f"    SKUs: {len(selected_skus):,}")
        print(f"    Pairs (block, type): {len(df_optimal_X):,}")
        
    elif max_skus is not None:
        print(f"\n  Filtering to {max_skus} SKUs...")
        
        unique_skus = df_blocks['sku_id'].unique()
        selected_skus = unique_skus[:max_skus]
        
        df_blocks = df_blocks[df_blocks['sku_id'].isin(selected_skus)]
        
        selected_block_ids = set(df_blocks['block_id'].unique())
        df_optimal_X = df_optimal_X[df_optimal_X['block_id'].isin(selected_block_ids)]
        
        print(f"  Filtered data:")
        print(f"    Blocks: {len(df_blocks):,}")
        print(f"    SKUs: {len(selected_skus):,}")
        print(f"    Pairs (block, type): {len(df_optimal_X):,}")
    
    print(f"\n  Bin types: {len(df_bin_types)}")
    
    print("\n" + "-" * 70)
    print("STEP 2: Preparing data structures")
    print("-" * 70)
    
    data = prepare_model_data(df_blocks, df_bin_types, df_optimal_X)
    
    print(f"  SKUs: {data['n_skus']:,}")
    print(f"  Blocks: {data['n_blocks']:,}")
    print(f"  Bin upper bound |J|: {data['J_max']:,}")
    print(f"  Total item volume: {data['total_item_volume']:,.0f} cm³")
    
    model = build_model(data, config)
    
    solution = None
    analysis = None
    
    if solve:
        solution = solve_model(model)
        
        if solution:
            print("\n" + "-" * 70)
            print("ANALYZING SOLUTION")
            print("-" * 70)
            
            analysis = analyze_solution(solution, data)
            print_analysis_report(analysis)
    
    print("\n" + "-" * 70)
    print("SAVING RESULTS")
    print("-" * 70)
    
    save_results(model, solution, analysis, output_dir, config)
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETED")
    print("=" * 70)
    print(f"""
  Dataset: {dataset_id.upper()}
  Total time: {total_time}
  
  Model built:
    • Variables: {model.NumVars:,}
    • Constraints: {model.NumConstrs:,}
  
  Files in: {output_dir}
""")
    
    if solution:
        print(f"""  Solution found:
    • Total volume: {solution['objective']:,.0f} cm³
    • Bins used: {solution['bins_used']:,}
    • Efficiency: {analysis['efficiency']['global_pct']:.1f}%
""")
    
    print("=" * 70)
    
    return model, solution, analysis



# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    DEFAULT_DATASET = 'DS-1'
    DEFAULT_SOLVE = True
    DEFAULT_TIME_LIMIT = 3600.0
    DEFAULT_M = 4
    DEFAULT_MAX_SKUS = None  # None when SELECTED_SKUS is used
    
    # ========================================================================
    # SPECIFIC SKU SELECTION
    # ========================================================================
    # Option 1: Use specific SKUs (they take priority over max_skus)
    # Run find_small_skus.py to find the best candidates
    
    # skus = [
    #     "1122", "1149", "1111", "1026", "398", "395", "1309", "1150",
    #     "1171", "746", "130", "370", "384", "693", "617", "397",
    #     "114", "139", "133", "125", "140", "737", "597", "587",
    #     "392", "42", "IN425", "47", "430", "373", "431", "97", "394",
    #     "400", "109", "428", "1061", "1419", "590"
    # ]
    
    skus = [
        "1122", "1149"]

    skus = list(set(skus))
    
    SELECTED_SKUS = skus
    
    
    
    # Option 2: Use the first N SKUs (if SELECTED_SKUS is None)
    # DEFAULT_MAX_SKUS = 2
    # ========================================================================
    
    import argparse
    
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Bin packing MILP model')
        parser.add_argument('dataset_id', type=str, help='Dataset ID')
        parser.add_argument('--solve', action='store_true', help='Solve the model')
        parser.add_argument('--time_limit', type=float, default=3600.0, help='Time limit (s)')
        parser.add_argument('--m', type=int, default=4, help='Maximum SKUs per bin (paper: m=4)')
        parser.add_argument('--max_skus', type=int, default=None, help='Limit to N SKUs')
        parser.add_argument('--skus', type=str, default=None, help='Specific SKUs separated by commas')
        
        args = parser.parse_args()
        
        # Parse SKUs if provided
        selected_skus = None
        if args.skus:
            selected_skus = [s.strip() for s in args.skus.split(',')]
        
        model, solution, analysis = main(
            dataset_id=args.dataset_id,
            solve=args.solve,
            time_limit=args.time_limit,
            m=args.m,
            max_skus=args.max_skus,
            selected_skus=selected_skus
        )
    else:
        print(f"Using default configuration:")
        print(f"  dataset_id = {DEFAULT_DATASET}")
        print(f"  solve = {DEFAULT_SOLVE}")
        print(f"  time_limit = {DEFAULT_TIME_LIMIT}")
        print(f"  m = {DEFAULT_M}")
        if SELECTED_SKUS:
            print(f"  selected_skus = {SELECTED_SKUS}")
        else:
            print(f"  max_skus = {DEFAULT_MAX_SKUS}")
        print()
        
        model, solution, analysis = main(
            dataset_id=DEFAULT_DATASET,
            solve=DEFAULT_SOLVE,
            time_limit=DEFAULT_TIME_LIMIT,
            m=DEFAULT_M,
            max_skus=DEFAULT_MAX_SKUS,
            selected_skus=SELECTED_SKUS
        )