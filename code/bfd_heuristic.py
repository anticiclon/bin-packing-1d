"""
Best-Fit-Decreasing Heuristic for Bin Sizing

Pre-filter + BFD:

  PRE-FILTER: Identify "isolated" blocks that cannot share a bin
    with any other block of a different SKU in any compatible type.
    These are assigned directly to the compatible bin with the smallest volume.

  BFD: The remaining blocks are processed using true Best-Fit-Decreasing:
    - Round-robin interleaved ordering among SKUs
    - "Minimum" type selection (smallest volume) for new bins
    - Indexed search by bin type O(|T|·k)
    - Finalization with x_min_global (keep bins open)

  RESULT: Both parts are combined for the complete solution.

Usage:
    python bfd_heuristic.py <dataset_id>
    python bfd_heuristic.py DS-1 --m 4

Input:
    - final_data/{dataset_id}/*_blocks.parquet
    - final_data/{dataset_id}/*_bin_types.parquet
    - final_data/{dataset_id}/*_X_optimal.parquet

Output:
    - results/{dataset_id}/bfd_solution.csv
    - results/{dataset_id}/bfd_bins.csv
    - results/{dataset_id}/bfd_report.txt
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OpenBin:
    """Open bin: tuple (t, B_j, S_j, L_j^res)."""
    bin_id: int
    bin_type_id: str
    L_t: float
    W_t: float
    H_t: float
    volume: float
    residual_x: float
    blocks: List[Tuple[str, int]] = field(default_factory=list)
    skus: Set[str] = field(default_factory=set)
    items_volume: float = 0.0

@dataclass
class BFDConfig:
    """
    Configuration of the BFD heuristic described in the document.

    The fit criterion is fixed to "minimize residual X-length after
    insertion" (Section 7: "selects the compatible open bin that minimizes
    the residual length after insertion"). This is the only fit metric
    described in the paper; see compute_fit() below.
    """
    m: int = 4
    theta: float = 0.05

@dataclass
class BFDSolution:
    """Complete solution of the BFD algorithm."""
    bins: List[Dict]
    total_volume: float
    total_item_volume: float
    n_bins: int
    efficiency: float
    runtime: float

# ============================================================================
# OPEN BIN INDEX
# ============================================================================

class OpenBinIndex:
    """
    Index of open bins by bin type + reverse SKU→bins index.
    """

    def __init__(self):
        self.by_type: Dict[str, List[OpenBin]] = defaultdict(list)
        self.bins_with_sku: Dict[str, Set[int]] = defaultdict(set)
        self.by_id: Dict[int, OpenBin] = {}
        self.total_count: int = 0

    def add(self, ob: OpenBin):
        self.by_type[ob.bin_type_id].append(ob)
        self.by_id[ob.bin_id] = ob
        for sku in ob.skus:
            self.bins_with_sku[sku].add(ob.bin_id)
        self.total_count += 1

    def update_sku(self, ob: OpenBin, new_sku: str):
        self.bins_with_sku[new_sku].add(ob.bin_id)

    def remove(self, ob: OpenBin):
        type_list = self.by_type[ob.bin_type_id]
        for i, b in enumerate(type_list):
            if b.bin_id == ob.bin_id:
                type_list.pop(i)
                break
        if not type_list:
            del self.by_type[ob.bin_type_id]
        for sku in ob.skus:
            self.bins_with_sku[sku].discard(ob.bin_id)
        del self.by_id[ob.bin_id]
        self.total_count -= 1

    def find_compatible(self, block, data, config):
        """Indexed search for compatible bins."""
        sku, k = block
        compatible = []
        excluded_bins = self.bins_with_sku.get(sku, set())
        compatible_types = data['compatible_types'].get(block, [])

        for bin_type in compatible_types:
            bins_of_type = self.by_type.get(bin_type, [])
            if not bins_of_type:
                continue
            x_opt = data['X_star'][(block, bin_type)]
            for ob in bins_of_type:
                if ob.bin_id in excluded_bins:
                    continue
                if len(ob.skus) >= config.m:
                    continue
                if x_opt > ob.residual_x + 1e-9:
                    continue
                compatible.append(ob)

        return compatible

# ============================================================================
# DATA LOADING
# ============================================================================

def load_model_data(dataset_id: str, base_dir: str = None):
    """Load preprocessed data from Parquet files."""
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(script_dir)
        base_dir = os.path.join(repo_dir, "final_data", dataset_id)

    blocks_path = os.path.join(base_dir, f"{dataset_id}_blocks.parquet")
    bin_types_path = os.path.join(base_dir, f"{dataset_id}_bin_types.parquet")
    optimal_X_path = os.path.join(base_dir, f"{dataset_id}_X_optimal.parquet")

    for path in [blocks_path, bin_types_path, optimal_X_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

    df_blocks = pd.read_parquet(blocks_path)
    df_bin_types = pd.read_parquet(bin_types_path)
    df_optimal_X = pd.read_parquet(optimal_X_path)

    return df_blocks, df_bin_types, df_optimal_X

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_bfd_data(df_blocks, df_bin_types, df_optimal_X) -> Dict:
    """Prepare optimized data structures for BFD."""
    data = {}

    # Basic sets
    data['S'] = df_blocks['sku_id'].unique().tolist()
    data['B'] = list(zip(df_blocks['sku_id'], df_blocks['block_k']))
    data['T'] = df_bin_types['bin_type_id'].tolist()

    # Identification mappings
    data['block_id_to_tuple'] = dict(zip(
        df_blocks['block_id'],
        zip(df_blocks['sku_id'], df_blocks['block_k'])
    ))
    data['tuple_to_block_id'] = {
        v: k for k, v in data['block_id_to_tuple'].items()
    }

    # Block parameters
    data['q_sk'] = dict(zip(
        zip(df_blocks['sku_id'], df_blocks['block_k']),
        df_blocks['q_sk'].astype(int)
    ))
    dims_by_sku = df_blocks.groupby('sku_id')[['l_s', 'w_s', 'h_s']].first()
    data['l_s'] = dims_by_sku['l_s'].to_dict()
    data['w_s'] = dims_by_sku['w_s'].to_dict()
    data['h_s'] = dims_by_sku['h_s'].to_dict()

    # Blocks by SKU
    blocks_by_sku = defaultdict(list)
    for sku, k in data['B']:
        blocks_by_sku[sku].append((sku, k))
    data['blocks_by_sku'] = dict(blocks_by_sku)

    # Bin type parameters
    data['L_t'] = dict(zip(df_bin_types['bin_type_id'], df_bin_types['L_t']))
    data['W_t'] = dict(zip(df_bin_types['bin_type_id'], df_bin_types['W_t']))
    data['H_t'] = dict(zip(df_bin_types['bin_type_id'], df_bin_types['H_t']))
    data['Vol_t'] = dict(zip(df_bin_types['bin_type_id'], df_bin_types['volume']))
    data['T_sorted_by_volume'] = (
        df_bin_types.sort_values('volume')['bin_type_id'].tolist()
    )
    data['V_max'] = df_bin_types['volume'].max()

    # X* matrix
    valid_block_ids = set(data['block_id_to_tuple'].keys())
    df_X_valid = df_optimal_X[df_optimal_X['block_id'].isin(valid_block_ids)]
    block_tuples = df_X_valid['block_id'].map(data['block_id_to_tuple'])
    data['X_star'] = dict(zip(
        zip(block_tuples, df_X_valid['bin_type_id']),
        df_X_valid['X_opt']
    ))
    data['feasible_pairs'] = set(data['X_star'].keys())

    # Compatible types by block (sorted by increasing volume)
    types_by_block = defaultdict(list)
    for (block_tuple, bin_type_id) in data['X_star'].keys():
        types_by_block[block_tuple].append(bin_type_id)
    for block_tuple in types_by_block:
        types_by_block[block_tuple].sort(key=lambda t: data['Vol_t'][t])
    data['compatible_types'] = dict(types_by_block)

    # X* statistics
    all_x_values = np.array(list(data['X_star'].values()))
    data['x_min_global'] = float(np.min(all_x_values)) if len(all_x_values) > 0 else 0
    data['x_p10_global'] = float(np.percentile(all_x_values, 10)) if len(all_x_values) > 0 else 0
    data['x_p25_global'] = float(np.percentile(all_x_values, 25)) if len(all_x_values) > 0 else 0
    data['x_median_global'] = float(np.median(all_x_values)) if len(all_x_values) > 0 else 0
    data['x_p75_global'] = float(np.percentile(all_x_values, 75)) if len(all_x_values) > 0 else 0

    # Minimum X* by block (for closing)
    x_min_by_block = {}
    for (block_tuple, bin_type), x in data['X_star'].items():
        if block_tuple not in x_min_by_block or x < x_min_by_block[block_tuple]:
            x_min_by_block[block_tuple] = x
    data['x_min_by_block'] = x_min_by_block

    # L_t values of each type (for diagnostics)
    all_L = set(data['L_t'].values())
    data['all_L_values'] = sorted(all_L)

    # Volume of each block
    data['block_volume'] = {}
    for (sku, k) in data['B']:
        q = data['q_sk'][(sku, k)]
        data['block_volume'][(sku, k)] = (
            q * data['l_s'][sku] * data['w_s'][sku] * data['h_s'][sku]
        )
    data['total_items_volume'] = sum(data['block_volume'].values())

    return data

# ============================================================================
# PRE-FILTER: IDENTIFICATION OF ISOLATED BLOCKS
# ============================================================================

def identify_isolated_blocks(data: Dict) -> Set[Tuple[str, int]]:
    """
    Identifies blocks that cannot share a bin with any other block
    from a different SKU, in any of their compatible types.

    For each type t, the 2 smallest X* values are precomputed
    (from different SKUs).
    A block (s, k) is isolated if, for EVERY compatible type t:
        min X*_{(s',k'),t} for s'≠s  >  L_t - X*_{(s,k),t}

    That is, the residual after placing the block is smaller than the
    minimum X* of any block from another SKU compatible with that type.

    Complexity: O(|feasible_pairs|) for preprocessing + O(|B|·|T_compat|) for testing.
    """
    # --- Step 1: For each type t, compute min X* by SKU ---
    min_x_per_type_sku = defaultdict(dict)
    for (block, t), x in data['X_star'].items():
        sku = block[0]
        if sku not in min_x_per_type_sku[t] or x < min_x_per_type_sku[t][sku]:
            min_x_per_type_sku[t][sku] = x

    # --- Step 2: For each type t, compute the 2 smallest min-X* values from different SKUs ---
    top2_by_type = {}
    for t in data['T']:
        sku_mins = min_x_per_type_sku.get(t, {})
        if len(sku_mins) == 0:
            top2_by_type[t] = (float('inf'), None, float('inf'), None)
        elif len(sku_mins) == 1:
            sku0, value0 = next(iter(sku_mins.items()))
            top2_by_type[t] = (value0, sku0, float('inf'), None)
        else:
            best1_val, best1_sku = float('inf'), None
            best2_val, best2_sku = float('inf'), None
            for sku, value in sku_mins.items():
                if value < best1_val:
                    best2_val, best2_sku = best1_val, best1_sku
                    best1_val, best1_sku = value, sku
                elif value < best2_val:
                    best2_val, best2_sku = value, sku
            top2_by_type[t] = (best1_val, best1_sku, best2_val, best2_sku)

    # --- Step 3: For each block, verify whether it is isolated ---
    isolated = set()
    for block in data['B']:
        sku, k = block
        compatible_types = data['compatible_types'].get(block, [])
        if not compatible_types:
            continue  # infeasible, handled separately

        is_isolated = True
        for t in compatible_types:
            x_block = data['X_star'][(block, t)]
            residual = data['L_t'][t] - x_block

            val1, sku1, val2, sku2 = top2_by_type[t]
            if sku1 != sku:
                x_min_other = val1
            else:
                x_min_other = val2

            if x_min_other <= residual + 1e-9:
                is_isolated = False
                break

        if is_isolated:
            isolated.add(block)

    return isolated


def assign_isolated_blocks(isolated_blocks: Set[Tuple[str, int]],
                           data: Dict) -> List[Dict]:
    """
    Assigns each isolated block to its own bin using the compatible bin type
    with the smallest volume. Returns the list of bin dictionaries.
    """
    bins_list = []
    for i, block in enumerate(sorted(isolated_blocks)):
        sku, k = block
        compatible_types = data['compatible_types'].get(block, [])
        if not compatible_types:
            continue
        bin_type = compatible_types[0]  # already sorted by increasing volume
        x_opt = data['X_star'][(block, bin_type)]
        block_volume = data['block_volume'][block]
        bin_volume = data['Vol_t'][bin_type]

        bins_list.append({
            'bin_id': i,
            'bin_type_id': bin_type,
            'L_t': data['L_t'][bin_type],
            'W_t': data['W_t'][bin_type],
            'H_t': data['H_t'][bin_type],
            'volume': bin_volume,
            'residual_x': data['L_t'][bin_type] - x_opt,
            'n_blocks': 1,
            'n_skus': 1,
            'blocks': [block],
            'skus': [sku],
            'items_volume': block_volume,
            'utilization_pct': 100 * block_volume / bin_volume if bin_volume > 0 else 0,
        })

    return bins_list


# ============================================================================
# BLOCK ORDERING (ROUND-ROBIN)
# ============================================================================

def sort_blocks_interleaved(data: Dict) -> List[Tuple[str, int]]:
    """
    Interleaves blocks among SKUs in round-robin order.
    """
    blocks_by_sku = defaultdict(list)
    for block in data['B']:
        blocks_by_sku[block[0]].append(block)

    for sku in blocks_by_sku:
        blocks_by_sku[sku].sort(
            key=lambda block: data['block_volume'][block],
            reverse=True
        )

    sku_order = sorted(
        blocks_by_sku.keys(),
        key=lambda sku: data['block_volume'][blocks_by_sku[sku][0]],
        reverse=True
    )

    result = []
    max_blocks = max(len(blocks_by_sku[sku]) for sku in sku_order) if sku_order else 0
    for round_k in range(max_blocks):
        for sku in sku_order:
            blocks = blocks_by_sku[sku]
            if round_k < len(blocks):
                result.append(blocks[round_k])

    return result


def compute_fit(block, open_bin, data) -> float:
    """
    Best-fit criterion (smaller is better): residual X-length remaining in
    the bin after inserting the block. This is the sole fit metric used in
    the paper (Section 7, Best-Fit-Decreasing Heuristic): "selects the
    compatible open bin that minimizes the residual length after insertion".
    """
    x_opt = data['X_star'][(block, open_bin.bin_type_id)]
    return open_bin.residual_x - x_opt


# ============================================================================
# TYPE SELECTION FOR A NEW BIN
# ============================================================================

def select_bin_type(block, data) -> Optional[str]:
    compatible_types = data['compatible_types'].get(block, [])
    if not compatible_types:
        return None
    return compatible_types[0]  # smallest compatible volume


# ============================================================================
# CLOSING CRITERION
# ============================================================================

def should_close_bin(open_bin, config, global_min_x) -> bool:
    """Bin closing rule."""
    if open_bin.residual_x < config.theta * open_bin.L_t:
        return True
    if len(open_bin.skus) >= config.m:
        return True
    if open_bin.residual_x < global_min_x - 1e-9:
        return True
    return False

# ============================================================================
# BFD WITH INDEXED SEARCH
# ============================================================================

def solve_bfd_with_search(data: Dict, config: BFDConfig,
                          start_bin_id: int = 0) -> BFDSolution:
    """
    BFD with indexed search among open bins.
    Processes data['B'] in round-robin order and assigns each block to the open bin
    where it fits best (best-fit). If there is no compatible bin, opens a new one.
    """
    start_time = datetime.now()
    global_min_x = data['x_min_global']

    print("  Sorting blocks (round-robin)...")
    sorted_blocks = sort_blocks_interleaved(data)
    total_blocks = len(sorted_blocks)
    print(f"    {total_blocks:,} blocks from {len(data['S']):,} SKUs")

    if total_blocks == 0:
        print("  No blocks to process.")
        return BFDSolution(
            bins=[],
            total_volume=0,
            total_item_volume=0,
            n_bins=0,
            efficiency=0,
            runtime=0
        )

    index = OpenBinIndex()
    closed_bins: List[OpenBin] = []
    next_bin_id = start_bin_id
    infeasible_blocks = []
    stats = {
        'blocks_to_existing': 0,
        'blocks_new_bin': 0,
        'bins_closed_theta': 0,
        'bins_closed_skus': 0,
        'bins_closed_space': 0,
        'max_open_bins': 0,
        'fit_candidates_total': 0,
    }

    progress_interval = max(1, total_blocks // 20)
    print("  Processing blocks...")

    for i, block in enumerate(sorted_blocks):
        if (i + 1) % progress_interval == 0 or i == total_blocks - 1:
            pct = 100 * (i + 1) / total_blocks
            n_open = index.total_count
            n_closed = len(closed_bins)
            reuse_ratio = 100 * stats['blocks_to_existing'] / (i + 1)
            print(
                f"    [{pct:5.1f}%] Block {i+1:,}/{total_blocks:,} | "
                f"Open: {n_open:,} | Closed: {n_closed:,} | "
                f"Reuse: {reuse_ratio:.1f}%"
            )

        bin_type = select_bin_type(block, data)
        if bin_type is None:
            infeasible_blocks.append(block)
            continue

        compatible_bins = index.find_compatible(block, data, config)
        stats['fit_candidates_total'] += len(compatible_bins)

        if compatible_bins:
            best_bin = min(
                compatible_bins,
                key=lambda ob: compute_fit(block, ob, data)
            )
            sku, k = block
            x_opt = data['X_star'][(block, best_bin.bin_type_id)]
            best_bin.blocks.append(block)
            if sku not in best_bin.skus:
                best_bin.skus.add(sku)
                index.update_sku(best_bin, sku)
            best_bin.residual_x -= x_opt
            best_bin.items_volume += data['block_volume'][block]
            stats['blocks_to_existing'] += 1

            if should_close_bin(best_bin, config, global_min_x):
                index.remove(best_bin)
                closed_bins.append(best_bin)
                if best_bin.residual_x < config.theta * best_bin.L_t:
                    stats['bins_closed_theta'] += 1
                elif len(best_bin.skus) >= config.m:
                    stats['bins_closed_skus'] += 1
                else:
                    stats['bins_closed_space'] += 1

        else:
            new_bin = OpenBin(
                bin_id=next_bin_id,
                bin_type_id=bin_type,
                L_t=data['L_t'][bin_type],
                W_t=data['W_t'][bin_type],
                H_t=data['H_t'][bin_type],
                volume=data['Vol_t'][bin_type],
                residual_x=data['L_t'][bin_type],
            )
            next_bin_id += 1
            sku, k = block
            x_opt = data['X_star'][(block, bin_type)]
            new_bin.blocks.append(block)
            new_bin.skus.add(sku)
            new_bin.residual_x -= x_opt
            new_bin.items_volume = data['block_volume'][block]
            index.add(new_bin)
            stats['blocks_new_bin'] += 1

            if should_close_bin(new_bin, config, global_min_x):
                index.remove(new_bin)
                closed_bins.append(new_bin)
                if new_bin.residual_x < config.theta * new_bin.L_t:
                    stats['bins_closed_theta'] += 1
                elif len(new_bin.skus) >= config.m:
                    stats['bins_closed_skus'] += 1
                else:
                    stats['bins_closed_space'] += 1

        stats['max_open_bins'] = max(stats['max_open_bins'], index.total_count)

    # Close remaining bins
    for ob in list(index.by_id.values()):
        closed_bins.append(ob)

    if infeasible_blocks:
        print(f"\n  ⚠ {len(infeasible_blocks):,} infeasible blocks — excluded")
        data['infeasible_blocks'] = data.get('infeasible_blocks', []) + infeasible_blocks
        infeasible_volume = sum(data['block_volume'][b] for b in infeasible_blocks)
        data['total_items_volume'] -= infeasible_volume
        feasible_set = set(data['B']) - set(infeasible_blocks)
        data['B'] = [b for b in data['B'] if b in feasible_set]

    runtime = (datetime.now() - start_time).total_seconds()
    total_volume = sum(b.volume for b in closed_bins)
    total_item_volume = sum(b.items_volume for b in closed_bins)
    n_bins = len(closed_bins)
    efficiency = 100 * total_item_volume / total_volume if total_volume > 0 else 0

    bins_list = []
    for ob in closed_bins:
        bins_list.append({
            'bin_id': ob.bin_id,
            'bin_type_id': ob.bin_type_id,
            'L_t': ob.L_t,
            'W_t': ob.W_t,
            'H_t': ob.H_t,
            'volume': ob.volume,
            'residual_x': ob.residual_x,
            'n_blocks': len(ob.blocks),
            'n_skus': len(ob.skus),
            'blocks': ob.blocks.copy(),
            'skus': list(ob.skus),
            'items_volume': ob.items_volume,
            'utilization_pct': 100 * ob.items_volume / ob.volume if ob.volume > 0 else 0,
        })

    solution = BFDSolution(
        bins=bins_list,
        total_volume=total_volume,
        total_item_volume=total_item_volume,
        n_bins=n_bins,
        efficiency=efficiency,
        runtime=runtime,
    )

    avg_candidates = stats['fit_candidates_total'] / total_blocks if total_blocks > 0 else 0
    print(f"\n  BFD Statistics:")
    print(f"    Blocks → existing bin:          {stats['blocks_to_existing']:,}")
    print(f"    Blocks → new bin:               {stats['blocks_new_bin']:,}")
    print(f"    Reuse ratio:                    {100 * stats['blocks_to_existing'] / max(1, total_blocks):.1f}%")
    print(f"    Max simultaneous open bins:     {stats['max_open_bins']:,}")
    print(f"    Candidates/search (average):    {avg_candidates:.1f}")
    print(f"    Bins closed by threshold θ:     {stats['bins_closed_theta']:,}")
    print(f"    Bins closed by SKUs:            {stats['bins_closed_skus']:,}")
    print(f"    Bins closed by space:           {stats['bins_closed_space']:,}")

    return solution
# ============================================================================
# COMBINING SOLUTIONS
# ============================================================================

def combine_solutions(isolated_bins: List[Dict], bfd_solution: BFDSolution,
                      data: Dict) -> BFDSolution:
    """Combines bins from isolated blocks with bins from the BFD solution."""
    all_bins = isolated_bins + bfd_solution.bins
    total_volume = sum(b['volume'] for b in all_bins)
    total_item_volume = data['total_items_volume']
    n_bins = len(all_bins)
    efficiency = 100 * total_item_volume / total_volume if total_volume > 0 else 0
    runtime = bfd_solution.runtime

    return BFDSolution(
        bins=all_bins,
        total_volume=total_volume,
        total_item_volume=total_item_volume,
        n_bins=n_bins,
        efficiency=efficiency,
        runtime=runtime,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_solution(solution: BFDSolution, data: Dict, config: BFDConfig) -> bool:
    """Validates the feasibility of the solution."""
    print("\n  Validating solution...")
    all_ok = True

    all_assigned = []
    for b in solution.bins:
        all_assigned.extend(b['blocks'])

    n_infeasible = len(data.get('infeasible_blocks', []))
    expected = len(data['B'])

    if len(all_assigned) != expected:
        print(f"    ✗ Assigned blocks ({len(all_assigned)}) ≠ Expected ({expected})")
        all_ok = False
    else:
        msg = f"    ✓ All {expected:,} feasible blocks assigned"
        if n_infeasible > 0:
            msg += f" ({n_infeasible:,} infeasible excluded)"
        print(msg)

    if len(all_assigned) != len(set(all_assigned)):
        print(f"    ✗ There are duplicate blocks")
        all_ok = False
    else:
        print(f"    ✓ No duplicates")

    if set(all_assigned) != set(data['B']):
        missing = set(data['B']) - set(all_assigned)
        extra = set(all_assigned) - set(data['B'])
        if missing:
            print(f"    ✗ {len(missing)} missing blocks")
        if extra:
            print(f"    ✗ {len(extra)} extra blocks")
        all_ok = False
    else:
        print(f"    ✓ Correct set")

    violations = {'skus': 0, 'uniqueness': 0, 'capacity': 0, 'compatibility': 0}
    for b in solution.bins:
        bt = b['bin_type_id']
        blocks = b['blocks']
        skus = set(s for s, k in blocks)

        if len(skus) > config.m:
            violations['skus'] += 1

        sku_counts = defaultdict(int)
        for s, k in blocks:
            sku_counts[s] += 1
        if any(c > 1 for c in sku_counts.values()):
            violations['uniqueness'] += 1

        total_x = sum(data['X_star'].get((block, bt), float('inf')) for block in blocks)
        if total_x > data['L_t'][bt] + 1e-6:
            violations['capacity'] += 1

        for block in blocks:
            if (block, bt) not in data['X_star']:
                violations['compatibility'] += 1

    for name, count in violations.items():
        if count > 0:
            print(f"    ✗ {count} {name} violations")
            all_ok = False
        else:
            print(f"    ✓ No {name} violations")

    return all_ok


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_solution(solution: BFDSolution, data: Dict) -> Dict:
    """Detailed analysis of the solution."""
    analysis = {}

    utilizations = [b['utilization_pct'] for b in solution.bins]
    analysis['util_mean'] = np.mean(utilizations)
    analysis['util_median'] = np.median(utilizations)
    analysis['util_min'] = np.min(utilizations)
    analysis['util_max'] = np.max(utilizations)
    analysis['util_std'] = np.std(utilizations)

    blocks_per_bin = [b['n_blocks'] for b in solution.bins]
    analysis['blocks_per_bin_mean'] = np.mean(blocks_per_bin)
    analysis['blocks_per_bin_max'] = np.max(blocks_per_bin)
    analysis['blocks_per_bin_min'] = np.min(blocks_per_bin)

    skus_per_bin = [b['n_skus'] for b in solution.bins]
    analysis['skus_per_bin_mean'] = np.mean(skus_per_bin)
    analysis['skus_per_bin_max'] = np.max(skus_per_bin)

    from collections import Counter
    blocks_hist = Counter(blocks_per_bin)
    analysis['blocks_histogram'] = dict(sorted(blocks_hist.items()))

    type_counts = defaultdict(int)
    type_volumes = defaultdict(float)
    for b in solution.bins:
        bin_type = b['bin_type_id']
        type_counts[bin_type] += 1
        type_volumes[bin_type] += b['volume']
    analysis['type_counts'] = dict(type_counts)
    analysis['type_volumes'] = dict(type_volumes)
    analysis['n_types_used'] = len(type_counts)

    analysis['bins_1_block'] = sum(1 for n in blocks_per_bin if n == 1)
    analysis['bins_multi_block'] = sum(1 for n in blocks_per_bin if n > 1)
    analysis['pct_multi_block'] = (
        100 * analysis['bins_multi_block'] / len(blocks_per_bin)
        if blocks_per_bin else 0
    )

    max_bin_volume = max(data['Vol_t'].values())
    analysis['lower_bound'] = int(np.ceil(data['total_items_volume'] / max_bin_volume))

    return analysis

# ============================================================================
# REPORT
# ============================================================================

def generate_report(solution, analysis, data, config, dataset_id,
                    n_isolated, n_bfd, time_prefilter, time_bfd):
    """Generates a text report."""
    report = f"""{'='*70}
BFD HEURISTIC - SOLUTION REPORT
Dataset: {dataset_id.upper()}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

CONFIGURATION:
  m (max SKUs/bin):      {config.m}
  θ (closing threshold): {config.theta}

{'='*70}
PRE-FILTER
{'='*70}

  Isolated blocks:       {n_isolated:>12,}
  Blocks for BFD:        {n_bfd:>12,}
  Pre-filter time:       {time_prefilter:>11.2f} s

{'='*70}
SOLUTION SUMMARY (Pre-filter + BFD)
{'='*70}

  Bins used:                 {solution.n_bins:>12,}
  Total bin volume:          {solution.total_volume:>12,.0f} cm³
  Total item volume:         {solution.total_item_volume:>12,.0f} cm³
  Overall efficiency:        {solution.efficiency:>11.1f}%
  BFD time:                  {time_bfd:>11.2f} s

  Lower bound (volumetric):  {analysis.get('lower_bound', 'N/A'):>8,}

{'='*70}
UTILIZATION DISTRIBUTION
{'='*70}

  Mean:      {analysis['util_mean']:>8.1f}%
  Median:    {analysis['util_median']:>8.1f}%
  Minimum:   {analysis['util_min']:>8.1f}%
  Maximum:   {analysis['util_max']:>8.1f}%
  Std. dev.: {analysis['util_std']:>8.1f}%

{'='*70}
PACKING
{'='*70}

  Bins with 1 block:      {analysis['bins_1_block']:>8,} ({100 * analysis['bins_1_block'] / solution.n_bins:.1f}%)
  Bins with 2+ blocks:    {analysis['bins_multi_block']:>8,} ({analysis['pct_multi_block']:.1f}%)
  Blocks/bin (mean):      {analysis['blocks_per_bin_mean']:>8.2f}
  Blocks/bin (max):       {analysis['blocks_per_bin_max']:>8}
  SKUs/bin (mean):        {analysis['skus_per_bin_mean']:>8.2f}

Histogram of blocks per bin:
"""
    for n_blocks, count in sorted(analysis['blocks_histogram'].items()):
        bar = '█' * min(50, max(1, int(50 * count / solution.n_bins)))
        pct = 100 * count / solution.n_bins
        report += f"  {n_blocks:>3} blocks: {count:>8,} bins ({pct:5.1f}%) {bar}\n"

    report += f"""
{'='*70}
DISTRIBUTION BY BIN TYPE
{'='*70}
"""
    sorted_types = sorted(analysis['type_counts'].items(), key=lambda x: x[1], reverse=True)
    report += f"\n  {'Type':<25} {'Count':>8} {'Total Vol. (cm³)':>16} {'L×W×H':>20}\n"
    report += f"  {'-'*25} {'-'*8} {'-'*16} {'-'*20}\n"
    for t, count in sorted_types:
        vol = analysis['type_volumes'][t]
        L = data['L_t'].get(t, 0)
        W = data['W_t'].get(t, 0)
        H = data['H_t'].get(t, 0)
        report += f"  {t:<25} {count:>8,} {vol:>16,.0f} {L:.0f}×{W:.0f}×{H:.0f}\n"

    report += f"""
  Distinct types used: {analysis['n_types_used']}

{'='*70}
END OF REPORT
{'='*70}
"""
    return report


# ============================================================================
# SAVE
# ============================================================================

def save_solution(solution, analysis, data, config, dataset_id, output_dir,
                  n_isolated=0, n_bfd=0, time_prefilter=0, time_bfd=0):
    """Saves the solution as CSV and the report as TXT."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for bin_info in solution.bins:
        for block in bin_info['blocks']:
            sku, k = block
            block_id = data['tuple_to_block_id'].get(block, f"{sku}_{k}")
            rows.append({
                'block_id': block_id,
                'sku_id': sku,
                'block_k': k,
                'bin_id': bin_info['bin_id'],
                'bin_type_id': bin_info['bin_type_id'],
                'bin_volume': bin_info['volume'],
                'X_opt': data['X_star'].get((block, bin_info['bin_type_id']), None),
                'block_volume': data['block_volume'][block],
            })
    df_solution = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"{dataset_id}_bfd_solution.csv")
    df_solution.to_csv(csv_path, index=False)
    print(f"  ✓ Solution: {csv_path}")

    bin_rows = []
    for bin_info in solution.bins:
        bin_rows.append({
            'bin_id': bin_info['bin_id'],
            'bin_type_id': bin_info['bin_type_id'],
            'volume': bin_info['volume'],
            'n_blocks': bin_info['n_blocks'],
            'n_skus': bin_info['n_skus'],
            'items_volume': bin_info['items_volume'],
            'utilization_pct': bin_info['utilization_pct'],
            'residual_x': bin_info['residual_x'],
        })
    df_bins = pd.DataFrame(bin_rows)
    bins_path = os.path.join(output_dir, f"{dataset_id}_bfd_bins.csv")
    df_bins.to_csv(bins_path, index=False)
    print(f"  ✓ Bins: {bins_path}")

    report = generate_report(
        solution, analysis, data, config, dataset_id,
        n_isolated, n_bfd, time_prefilter, time_bfd
    )
    report_path = os.path.join(output_dir, f"{dataset_id}_bfd_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Report: {report_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(
    dataset_id: str,
    m: int = 4,
    theta: float = 0.05,
):
    """Runs isolated-block pre-filtering + BFD for the remaining blocks."""
    start_time = datetime.now()

    bfd_config = BFDConfig(m=m, theta=theta)

    print("=" * 70)
    print(f"BFD HEURISTIC (PRE-FILTER + BFD) - {dataset_id.upper()}")
    print("=" * 70)
    print(f"\nConfiguration: m={m}, θ={theta}")

    # --- Load ---
    print("\n" + "-" * 70)
    print("STEP 1: Loading preprocessed data")
    print("-" * 70)
    df_blocks, df_bin_types, df_optimal_X = load_model_data(dataset_id)
    print(f"  Blocks: {len(df_blocks):,}")
    print(f"  SKUs: {df_blocks['sku_id'].nunique():,}")
    print(f"  Bin types: {len(df_bin_types)}")
    print(f"  Pairs (block, type): {len(df_optimal_X):,}")

    # --- Prepare ---
    print("\n" + "-" * 70)
    print("STEP 2: Preparing data structures")
    print("-" * 70)
    data = prepare_bfd_data(df_blocks, df_bin_types, df_optimal_X)
    total_blocks = len(data['B'])
    print(f"  SKUs: {len(data['S']):,}")
    print(f"  Blocks: {total_blocks:,}")
    print(f"  Feasible pairs: {len(data['X_star']):,}")
    print(f"  Total item volume: {data['total_items_volume']:,.0f} cm³")
    print(
        f"  X* distribution: min={data['x_min_global']:.1f} | "
        f"p10={data['x_p10_global']:.1f} | p25={data['x_p25_global']:.1f} | "
        f"p50={data['x_median_global']:.1f} | p75={data['x_p75_global']:.1f} cm"
    )
    print(f"  Distinct L_t values: {data['all_L_values']}")

    # --- Pre-filter: Identify isolated blocks ---
    print("\n" + "-" * 70)
    print("STEP 3: Pre-filter — Identifying isolated blocks")
    print("-" * 70)
    prefilter_start = datetime.now()
    isolated_blocks = identify_isolated_blocks(data)
    n_isolated = len(isolated_blocks)
    n_bfd = total_blocks - n_isolated
    time_prefilter = (datetime.now() - prefilter_start).total_seconds()

    print(f"  Isolated blocks: {n_isolated:,} ({100 * n_isolated / total_blocks:.1f}%)")
    print(f"  Blocks for BFD: {n_bfd:,} ({100 * n_bfd / total_blocks:.1f}%)")
    print(f"  Time: {time_prefilter:.2f} s")

    # Assign isolated blocks directly
    print("\n  Assigning isolated blocks → bin with minimum type...")
    isolated_bins_list = assign_isolated_blocks(isolated_blocks, data)
    if isolated_bins_list:
        isolated_volume = sum(b['volume'] for b in isolated_bins_list)
        isolated_items_volume = sum(b['items_volume'] for b in isolated_bins_list)
        isolated_efficiency = 100 * isolated_items_volume / isolated_volume if isolated_volume > 0 else 0
        print(
            f"  → {len(isolated_bins_list):,} bins created, "
            f"volume {isolated_volume:,.0f} cm³, efficiency {isolated_efficiency:.1f}%"
        )
    else:
        print("  → 0 isolated blocks")

    # --- BFD on non-isolated blocks ---
    print("\n" + "-" * 70)
    print("STEP 4: BFD — Best-Fit-Decreasing on non-isolated blocks")
    print("-" * 70)

    if n_bfd > 0:
        # Filter data for BFD
        bfd_blocks = [b for b in data['B'] if b not in isolated_blocks]
        bfd_skus = list(set(sku for sku, k in bfd_blocks))

        original_B = data['B']
        original_S = data['S']
        original_x_min = data['x_min_global']

        data['B'] = bfd_blocks
        data['S'] = bfd_skus

        # Recompute x_min_global for BFD blocks (for closing criterion)
        bfd_x_values = []
        for block in bfd_blocks:
            for bin_type in data['compatible_types'].get(block, []):
                bfd_x_values.append(data['X_star'][(block, bin_type)])
        if bfd_x_values:
            data['x_min_global'] = float(min(bfd_x_values))

        start_id = len(isolated_bins_list)
        bfd_solution = solve_bfd_with_search(data, bfd_config, start_bin_id=start_id)
        time_bfd = bfd_solution.runtime

        # Restore data to the full set (excluding infeasible blocks)
        infeasible_set = set(data.get('infeasible_blocks', []))
        all_feasible = [b for b in original_B if b not in infeasible_set]
        data['B'] = all_feasible
        data['S'] = list(set(sku for sku, k in all_feasible))
        data['x_min_global'] = original_x_min
    else:
        print("  All blocks are isolated — no BFD needed.")
        bfd_solution = BFDSolution(
            bins=[],
            total_volume=0,
            total_item_volume=0,
            n_bins=0,
            efficiency=0,
            runtime=0
        )
        time_bfd = 0

    # --- Combine ---
    print("\n" + "-" * 70)
    print("STEP 5: Combining solutions")
    print("-" * 70)
    final_solution = combine_solutions(isolated_bins_list, bfd_solution, data)
    print(f"  Isolated bins:  {len(isolated_bins_list):,}")
    print(f"  BFD bins:       {bfd_solution.n_bins:,}")
    print(f"  Total bins:     {final_solution.n_bins:,}")
    print(f"  Total volume:   {final_solution.total_volume:,.0f} cm³")
    print(f"  Efficiency:     {final_solution.efficiency:.1f}%")

    # --- Validate ---
    print("\n" + "-" * 70)
    print("STEP 6: Validation")
    print("-" * 70)
    is_valid = validate_solution(final_solution, data, bfd_config)

    # --- Analyze ---
    print("\n" + "-" * 70)
    print("STEP 7: Analysis")
    print("-" * 70)
    analysis = analyze_solution(final_solution, data)
    print(f"  Average utilization:   {analysis['util_mean']:.1f}%")
    print(f"  Bin types used:        {analysis['n_types_used']}")
    print(f"  Blocks/bin (mean):     {analysis['blocks_per_bin_mean']:.2f}")
    print(f"  Bins with 2+ blocks:   {analysis['bins_multi_block']:,} ({analysis['pct_multi_block']:.1f}%)")
    print(f"  Lower bound:           {analysis['lower_bound']:,}")

    # --- Save ---
    print("\n" + "-" * 70)
    print("STEP 8: Saving results")
    print("-" * 70)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(repo_dir, "results", dataset_id)
    save_solution(
        final_solution, analysis, data, bfd_config, dataset_id, output_dir,
        n_isolated, n_bfd, time_prefilter, time_bfd
    )

    # --- Summary ---
    total_time = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("BFD COMPLETED")
    print("=" * 70)
    print(f"""
  Dataset: {dataset_id.upper()}
  Total time: {total_time}

  Configuration: m={m}, θ={theta}

  Pre-filter:
    Isolated blocks:     {n_isolated:,} ({100 * n_isolated / total_blocks:.1f}%)
    BFD blocks:          {n_bfd:,} ({100 * n_bfd / total_blocks:.1f}%)
    Time:                {time_prefilter:.2f} s

  BFD:
    Bins created:        {bfd_solution.n_bins:,}
    Reuse:               {100 * (n_bfd - bfd_solution.n_bins) / max(1, n_bfd):.1f}%
    Time:                {time_bfd:.2f} s

  Final solution:
    Bins:                {final_solution.n_bins:,}
    Volume:              {final_solution.total_volume:,.0f} cm³
    Efficiency:          {final_solution.efficiency:.1f}%
    Blocks/bin:          {analysis['blocks_per_bin_mean']:.2f}
    Multi-block bins:    {analysis['pct_multi_block']:.1f}%
    Lower bound:         {analysis['lower_bound']:,}
    Valid:               {'Yes' if is_valid else 'NO'}
""")
    print("=" * 70)

    return final_solution, analysis, data


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='BFD Heuristic')
        parser.add_argument('dataset_id', type=str)
        parser.add_argument('--m', type=int, default=4,
                             help='Maximum number of distinct SKUs per bin (paper: m=4)')
        parser.add_argument('--theta', type=float, default=0.05)
        args = parser.parse_args()
        main(args.dataset_id, args.m, args.theta)
    else:
        main(dataset_id='DS-1', m=4, theta=0.05)
