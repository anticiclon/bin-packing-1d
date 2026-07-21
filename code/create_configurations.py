# -*- coding: utf-8 -*-
"""
MÓDULO 2: Cálculo de Configuraciones Óptimas X*, Y*, Z* y Volúmenes

Uso:
    python create_configurations.py <ruta_archivo_blocks.csv>
    
Ejemplos:
    python create_configurations.py data/DS-1_blocks.csv
    python create_configurations.py data/DS-2_blocks.csv

Input:  *_blocks.csv (output del Módulo 1)
Output: *_configurations.csv, *_module2_stats.txt
"""

import math
import itertools
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# ============================================================================
# DATA STRUCTURE FOR CONFIGURATION RESULT
# ============================================================================

@dataclass
class BoundingBoxResult:
    """Complete result of the optimal bounding box calculation."""
    X_opt: float          # X dimension of the bounding box (cm)
    Y_opt: float          # Y dimension of the bounding box (cm)
    Z_opt: float          # Z dimension of the bounding box (cm)
    a: int                # Number of positions along X axis
    b: int                # Number of positions along Y axis
    c: int                # Number of positions along Z axis
    block_volume: float   # Total volume of the items (cm³)
    bbox_volume: float    # Bounding box volume (cm³)
    volume_match: bool    # True if block_volume == bbox_volume
    orientation: Tuple[float, float, float]  # (dim_X, dim_Y, dim_Z) of the item


# ============================================================================
# BIN TYPES CONFIGURATION
# ============================================================================

def create_bin_types(dataset_id: str) -> pd.DataFrame:
    """
    Creates the bin types table based on the dataset.
    
    Args:
        dataset_id: 'DS-3', 'DS-4' (small) or 'DS-1', 'DS-2' (large)
    
    Returns:
        DataFrame with columns: bin_type_id, L_t, W_t, H_t, volume
    """
    if dataset_id.lower() in ['ds-3', 'ds-4']:
        depths = [30, 60]       # L_t (X axis)
        widths = [38, 46, 57]   # W_t (Y axis)
        heights = [25, 35, 45, 55]  # H_t (Z axis)
    elif dataset_id.lower() in ['ds-1', 'ds-2']:
        depths = [70, 100]
        widths = [57, 76, 115]
        heights = [50, 70, 100, 150, 200]
    else:
        raise ValueError(f"Unrecognized dataset: {dataset_id}")
    
    bins_list = []
    # L = width, W = depth, H = height
    for L, W, H in itertools.product(widths, depths, heights): 
        bin_type_id = f"L{L:03d}_W{W:03d}_H{H:03d}"
        volume = L * W * H
        bins_list.append({
            'bin_type_id': bin_type_id,
            'L_t': float(L),
            'W_t': float(W),
            'H_t': float(H),
            'volume': float(volume)
        })
    
    df_bins = pd.DataFrame(bins_list)
    df_bins = df_bins.sort_values('volume').reset_index(drop=True)
    
    return df_bins


# ============================================================================
# DATA LOADING
# ============================================================================

def detect_dataset(file_path: str) -> str:
    """Detects the dataset based on the file name."""
    name = os.path.basename(file_path).lower()
    for ds in ['DS-1', 'DS-2', 'DS-3', 'DS-4']:
        if ds.lower() in name:
            return ds
    raise ValueError(f"Unrecognized dataset in: {file_path}")


def load_blocks(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Loads the blocks file and detects the dataset.
    
    Returns:
        (blocks DataFrame, dataset_id)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Not found: {file_path}")
    
    dataset_id = detect_dataset(file_path)
    df = pd.read_csv(file_path)
    
    required = ['sku_id', 'block_k', 'q_sk', 'length', 'width', 'height', 'rotatable', 'block_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    if df[required].isnull().any().any():
        raise ValueError("There are null values in required columns")
    
    print(f"Detected dataset: {dataset_id.upper()}")
    print(f"Blocks loaded: {len(df):,}")
    print(f"Unique SKUs: {df['sku_id'].nunique():,}")
    print(f"q_sk range: [{df['q_sk'].min()}, {df['q_sk'].max()}]")
    
    return df, dataset_id


# ============================================================================
# MAIN ALGORITHM: OPTIMAL BOUNDING BOX CALCULATION
# ============================================================================

def get_orientations(l_s: float, w_s: float, h_s: float, rotatable: bool) -> List[Tuple[float, float, float]]:
    """
    Generates valid orientations for an item.
    
    For rotatable items: 6 permutations of (l_s, w_s, h_s) assigned to (X, Y, Z)
    For non-rotatable items: only the original orientation
    
    Returns:
        List of tuples (dim_X, dim_Y, dim_Z)
    """
    if rotatable:
        # All permutations of the 3 dimensions
        return list(itertools.permutations([l_s, w_s, h_s]))
    else:
        # Only the original orientation
        return [(l_s, w_s, h_s), (w_s, l_s, h_s)]


def compute_bounding_box_optimal(
    q: int,
    l_s: float, w_s: float, h_s: float,
    L_t: float, W_t: float, H_t: float,
    rotatable: bool
) -> Optional[BoundingBoxResult]:
    """
    Computes the optimal bounding box for a block within a bin type.
    
    Implements an O(1) algorithm based on bounding boxes with 3D rotation:
    1. For each orientation (6 for rotatable items, 1 for non-rotatable)
    2. Maximizes capacity along Y and Z
    3. Computes the minimum number of "slices" along X required
    4. Returns the result with minimum X among all valid orientations
    
    Args:
        q: Number of items in the block (q_sk)
        l_s, w_s, h_s: Item dimensions (length, width, height)
        L_t, W_t, H_t: Bin dimensions (Length, Width, Height)
        rotatable: Whether the item can be rotated
    
    Returns:
        BoundingBoxResult with all dimensions and volumes, or None if it doesn't fit
    """
    orientations = get_orientations(l_s, w_s, h_s, rotatable)
    
    best_result: Optional[BoundingBoxResult] = None
    best_X = float('inf')
    
    # Block volume (sum of item volumes) - constant
    item_volume = l_s * w_s * h_s
    block_volume = q * item_volume
    
    for dim_X, dim_Y, dim_Z in orientations:
        # Compute maximum capacity along each axis
        a_max = int(L_t // dim_X)  # Max along X axis
        b_max = int(W_t // dim_Y)  # Max along Y axis
        c_max = int(H_t // dim_Z)  # Max along Z axis
        
        # Check that at least 1 item fits in each dimension
        if a_max < 1 or b_max < 1 or c_max < 1:
            continue  # This orientation is not valid
        
        # Capacity per "slice" perpendicular to X axis
        capacity_per_slice = b_max * c_max
        
        # Minimum number of slices required for q items
        a_min = math.ceil(q / capacity_per_slice)
        
        # Check if it fits within the bin
        if a_min <= a_max:
            X_opt = a_min * dim_X
            
            if X_opt < best_X:
                best_X = X_opt
                
                # Compute Y and Z dimensions of the bounding box
                # We use b_max and c_max because we maximize capacity in those directions
                Y_opt = b_max * dim_Y
                Z_opt = c_max * dim_Z
                
                # Bounding box volume
                bbox_volume = X_opt * Y_opt * Z_opt
                
                # Check if the packing is perfect
                bbox_positions = a_min * b_max * c_max
                volume_match = (bbox_positions == q)
                
                best_result = BoundingBoxResult(
                    X_opt=X_opt,
                    Y_opt=Y_opt,
                    Z_opt=Z_opt,
                    a=a_min,
                    b=b_max,
                    c=c_max,
                    block_volume=block_volume,
                    bbox_volume=bbox_volume,
                    volume_match=volume_match,
                    orientation=(dim_X, dim_Y, dim_Z)
                )
    
    return best_result


def compute_bbox_optimal_vectorized(
    df_configs: pd.DataFrame,
    df_bins: pd.DataFrame
) -> Tuple[pd.DataFrame, int, int]:
    """
    Computes the optimal bounding box for all combinations (unique configuration, bin).
    
    Optimization: groups blocks by unique configuration (q_sk, dims, rotatable)
    to avoid redundant computations.
    
    Args:
        df_configs: DataFrame with unique configurations
        df_bins: DataFrame with bin types
    
    Returns:
        (DataFrame with results, feasible pairs, infeasible pairs)
    """
    results = []
    
    total_pairs = len(df_configs) * len(df_bins)
    feasible_pairs = 0
    infeasible_pairs = 0
    
    for _, config in df_configs.iterrows():
        q = int(config['q_sk'])
        l_s = config['length']
        w_s = config['width']
        h_s = config['height']
        rotatable = config['rotatable']
        config_key = (q, l_s, w_s, h_s, rotatable)
        
        for _, bin_row in df_bins.iterrows():
            bin_type_id = bin_row['bin_type_id']
            L_t = bin_row['L_t']
            W_t = bin_row['W_t']
            H_t = bin_row['H_t']
            
            bbox_result = compute_bounding_box_optimal(
                q, l_s, w_s, h_s, L_t, W_t, H_t, rotatable
            )
            
            if bbox_result is not None:
                results.append({
                    'config_key': config_key,
                    'bin_type_id': bin_type_id,
                    'X_opt': bbox_result.X_opt,
                    'Y_opt': bbox_result.Y_opt,
                    'Z_opt': bbox_result.Z_opt,
                    'a': bbox_result.a,
                    'b': bbox_result.b,
                    'c': bbox_result.c,
                    'block_volume': bbox_result.block_volume,
                    'bbox_volume': bbox_result.bbox_volume,
                    'volume_match': bbox_result.volume_match,
                })
                feasible_pairs += 1
            else:
                infeasible_pairs += 1
    
    print(f"\nOptimal bounding box statistics:")
    print(f"  Total pairs (config, bin): {total_pairs:,}")
    print(f"  Feasible pairs: {feasible_pairs:,} ({100*feasible_pairs/total_pairs:.1f}%)")
    print(f"  Infeasible pairs: {infeasible_pairs:,} ({100*infeasible_pairs/total_pairs:.1f}%)")
    
    return pd.DataFrame(results), feasible_pairs, infeasible_pairs


def expand_to_blocks(
    df_blocks: pd.DataFrame,
    df_bbox_optimal: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """
    Expands the optimal bounding box results to all individual blocks.
    
    Args:
        df_blocks: DataFrame of blocks
        df_bbox_optimal: DataFrame with bounding box results
    
    Returns:
        (Final DataFrame, number of blocks without feasible bins)
    """
    # Create configuration key for each block
    df_blocks = df_blocks.copy()
    df_blocks['config_key'] = list(zip(
        df_blocks['q_sk'].astype(int),
        df_blocks['length'],
        df_blocks['width'],
        df_blocks['height'],
        df_blocks['rotatable']
    ))
    
    # Create dictionary of results by configuration
    bbox_by_config = {}
    for _, row in df_bbox_optimal.iterrows():
        config_key = row['config_key']
        if config_key not in bbox_by_config:
            bbox_by_config[config_key] = {}
        bbox_by_config[config_key][row['bin_type_id']] = {
            'X_opt': row['X_opt'],
            'Y_opt': row['Y_opt'],
            'Z_opt': row['Z_opt'],
            'a': row['a'],
            'b': row['b'],
            'c': row['c'],
            'block_volume': row['block_volume'],
            'bbox_volume': row['bbox_volume'],
            'volume_match': row['volume_match'],
        }
    
    # Expand to blocks
    results = []
    blocks_without_bins = 0
    
    for _, block in df_blocks.iterrows():
        config_key = block['config_key']
        
        if config_key not in bbox_by_config or len(bbox_by_config[config_key]) == 0:
            blocks_without_bins += 1
            continue
        
        for bin_type_id, bbox_data in bbox_by_config[config_key].items():
            results.append({
                'block_id': block['block_id'],
                'sku_id': block['sku_id'],
                'block_k': block['block_k'],
                'q_sk': block['q_sk'],
                'bin_type_id': bin_type_id,
                'X_opt': bbox_data['X_opt'],
                'Y_opt': bbox_data['Y_opt'],
                'Z_opt': bbox_data['Z_opt'],
                'a': bbox_data['a'],
                'b': bbox_data['b'],
                'c': bbox_data['c'],
                'block_volume': bbox_data['block_volume'],
                'bbox_volume': bbox_data['bbox_volume'],
                'volume_match': bbox_data['volume_match'],
            })
    
    df_final = pd.DataFrame(results)
    
    print(f"\nExpansion to blocks:")
    print(f"  Blocks with feasible bins: {len(df_blocks) - blocks_without_bins:,}")
    print(f"  Blocks without feasible bins: {blocks_without_bins:,}")
    print(f"  Rows generated: {len(df_final):,}")
    
    return df_final, blocks_without_bins


# ============================================================================
# SAVE THE RESULTS
# ============================================================================

def save_results(
    df_final: pd.DataFrame,
    dataset_id: str,
    output_dir: str,
    stats: Dict
) -> None:
    """Saves the configurations CSV and the statistics report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{dataset_id}_configurations.csv")
    df_final.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved: {csv_path}")
    
    # Generate report
    stats_path = os.path.join(output_dir, f"{dataset_id}_module3_stats.txt")
    
    report = f"""{'='*70}
{dataset_id.upper()} - MODULE 3: OPTIMAL CONFIGURATIONS REPORT
Version 3.0 - Full Bounding Box with volume analysis
{'='*70}

Processing date: {stats['timestamp']}
Execution time: {stats['execution_time']}

Input file:   {stats['input_file']}
Output file:  {csv_path}

{'='*70}
1. INPUT DATA
{'='*70}

Total blocks:                  {stats['total_blocks']:>12,}
Total SKUs:                    {stats['total_skus']:>12,}
Available bin types:           {stats['total_bins']:>12}
q_sk range:                    [{stats['q_sk_min']}, {stats['q_sk_max']}]

{'='*70}
2. OPTIMIZATION BY UNIQUE CONFIGURATIONS
{'='*70}

Unique configurations:         {stats['unique_configs']:>12,}
Reduction vs blocks:           {stats['config_reduction']:.1f}x

{'='*70}
3. OPTIMAL BOUNDING BOX CALCULATION
{'='*70}

Total pairs (config, bin):     {stats['total_pairs']:>12,}
Feasible pairs:                {stats['feasible_pairs']:>12,} ({stats['pct_feasible']:.1f}%)
Infeasible pairs:              {stats['infeasible_pairs']:>12,} ({stats['pct_infeasible']:.1f}%)

{'='*70}
4. EXPANSION TO BLOCKS
{'='*70}

Blocks with feasible bins:     {stats['blocks_with_bins']:>12,} ({stats['pct_blocks_with_bins']:.1f}%)
Blocks without feasible bins:  {stats['blocks_without_feasible_bins']:>12,} ({stats['pct_blocks_without_bins']:.1f}%)
Generated rows:                {stats['output_rows']:>12,}
Average bins per block:        {stats['avg_bins_per_block']:.1f}

{'='*70}
5. BOUNDING BOX DIMENSIONS
{'='*70}

X (optimal):  {stats['X_min']:.2f} → {stats['X_max']:.2f} cm
Y:            {stats['Y_min']:.2f} → {stats['Y_max']:.2f} cm
Z:            {stats['Z_min']:.2f} → {stats['Z_max']:.2f} cm

{'='*70}
6. VOLUME ANALYSIS
{'='*70}

Block volume avg:              {stats['block_volume_avg']:.2f} cm³
Bounding box avg:              {stats['bbox_volume_avg']:.2f} cm³
Efficiency avg:                {stats['efficiency_avg']:.1f}%

Perfect packing cases:         {stats['perfect_packing']:,} ({stats['pct_perfect']:.1f}%)

{'='*70}
END OF REPORT
{'='*70}
"""
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Statistics saved: {stats_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def process_configurations(file_path: str) -> pd.DataFrame:
    """Processes a blocks file and generates the configurations table."""
    
    start_time = datetime.now()
    
    print("=" * 70)
    print("MODULE 3: OPTIMAL BOUNDING BOX CALCULATION")
    print("Version 3.0 - Full dimensions and volume analysis")
    print("=" * 70)
    
    # Step 1: Load blocks
    print("\n" + "-" * 70)
    print("STEP 1: Loading blocks")
    print("-" * 70)
    df_blocks, dataset_id = load_blocks(file_path)
    
    # Step 2: Create bin types
    print("\n" + "-" * 70)
    print("STEP 2: Creating bin types")
    print("-" * 70)
    df_bins = create_bin_types(dataset_id)
    print(f"Bin types: {len(df_bins)}")
    print(f"Minimum volume: {df_bins['volume'].min():,.0f} cm³")
    print(f"Maximum volume: {df_bins['volume'].max():,.0f} cm³")
    
    # Step 3: Extract unique configurations
    print("\n" + "-" * 70)
    print("STEP 3: Extracting unique configurations")
    print("-" * 70)
    config_cols = ['q_sk', 'length', 'width', 'height', 'rotatable']
    df_configs = df_blocks[config_cols].drop_duplicates().reset_index(drop=True)
    print(f"Unique configurations: {len(df_configs):,}")
    print(f"Reduction: {len(df_blocks) / len(df_configs):.1f}x")
    
    # Step 4: Compute optimal bounding box
    print("\n" + "-" * 70)
    print("STEP 4: Computing optimal bounding box (3D rotation, 6 orientations)")
    print("-" * 70)
    df_bbox_optimal, feasible_pairs, infeasible_pairs = compute_bbox_optimal_vectorized(
        df_configs, df_bins
    )
    
    # Step 5: Expand to blocks
    print("\n" + "-" * 70)
    print("STEP 5: Expanding to individual blocks")
    print("-" * 70)
    df_final, blocks_without_bins = expand_to_blocks(df_blocks, df_bbox_optimal)
    
    # Compute statistics
    end_time = datetime.now()
    total_time = end_time - start_time
    
    total_pairs = len(df_configs) * len(df_bins)
    blocks_with_bins = len(df_blocks) - blocks_without_bins
    
    if len(df_final) > 0:
        # Dimension statistics
        X_values = df_final['X_opt'].values
        Y_values = df_final['Y_opt'].values
        Z_values = df_final['Z_opt'].values
        
        X_min, X_max = X_values.min(), X_values.max()
        X_avg, X_median = X_values.mean(), np.median(X_values)
        
        Y_min, Y_max = Y_values.min(), Y_values.max()
        Y_avg, Y_median = Y_values.mean(), np.median(Y_values)
        
        Z_min, Z_max = Z_values.min(), Z_values.max()
        Z_avg, Z_median = Z_values.mean(), np.median(Z_values)
        
        # Volume statistics
        block_vol = df_final['block_volume'].values
        bbox_vol = df_final['bbox_volume'].values
        
        block_volume_min, block_volume_max = block_vol.min(), block_vol.max()
        block_volume_avg = block_vol.mean()
        
        bbox_volume_min, bbox_volume_max = bbox_vol.min(), bbox_vol.max()
        bbox_volume_avg = bbox_vol.mean()
        
        # Volume efficiency
        efficiency = 100 * block_vol / bbox_vol
        efficiency_min, efficiency_max = efficiency.min(), efficiency.max()
        efficiency_avg = efficiency.mean()
        
        # Perfect packing
        perfect_packing = df_final['volume_match'].sum()
        pct_perfect = 100 * perfect_packing / len(df_final)
    else:
        X_min = X_max = X_avg = X_median = 0
        Y_min = Y_max = Y_avg = Y_median = 0
        Z_min = Z_max = Z_avg = Z_median = 0
        block_volume_min = block_volume_max = block_volume_avg = 0
        bbox_volume_min = bbox_volume_max = bbox_volume_avg = 0
        efficiency_min = efficiency_max = efficiency_avg = 0
        perfect_packing = pct_perfect = 0
    
    stats = {
        'timestamp': end_time.strftime("%Y-%m-%d %H:%M:%S"),
        'execution_time': total_time,
        'input_file': file_path,
        'total_blocks': len(df_blocks),
        'total_skus': df_blocks['sku_id'].nunique(),
        'total_bins': len(df_bins),
        'q_sk_min': df_blocks['q_sk'].min(),
        'q_sk_max': df_blocks['q_sk'].max(),
        'unique_configs': len(df_configs),
        'config_reduction': len(df_blocks) / len(df_configs),
        'total_pairs': total_pairs,
        'feasible_pairs': feasible_pairs,
        'pct_feasible': 100 * feasible_pairs / total_pairs if total_pairs > 0 else 0,
        'infeasible_pairs': infeasible_pairs,
        'pct_infeasible': 100 * infeasible_pairs / total_pairs if total_pairs > 0 else 0,
        'blocks_with_bins': blocks_with_bins,
        'pct_blocks_with_bins': 100 * blocks_with_bins / len(df_blocks) if len(df_blocks) > 0 else 0,
        'blocks_without_feasible_bins': blocks_without_bins,
        'pct_blocks_without_bins': 100 * blocks_without_bins / len(df_blocks) if len(df_blocks) > 0 else 0,
        'output_rows': len(df_final),
        'avg_bins_per_block': len(df_final) / blocks_with_bins if blocks_with_bins > 0 else 0,
        
        # Dimensions
        'X_min': X_min, 'X_max': X_max, 'X_avg': X_avg, 'X_median': X_median,
        'Y_min': Y_min, 'Y_max': Y_max, 'Y_avg': Y_avg, 'Y_median': Y_median,
        'Z_min': Z_min, 'Z_max': Z_max, 'Z_avg': Z_avg, 'Z_median': Z_median,
        
        # Volumes
        'block_volume_min': block_volume_min,
        'block_volume_max': block_volume_max,
        'block_volume_avg': block_volume_avg,
        'bbox_volume_min': bbox_volume_min,
        'bbox_volume_max': bbox_volume_max,
        'bbox_volume_avg': bbox_volume_avg,
        
        # Efficiency
        'efficiency_min': efficiency_min,
        'efficiency_max': efficiency_max,
        'efficiency_avg': efficiency_avg,
        
        # Packing
        'perfect_packing': perfect_packing,
        'pct_perfect': pct_perfect,
    }
    
    # Step 6: Save results
    print("\n" + "-" * 70)
    print("STEP 6: Saving results")
    print("-" * 70)
    output_dir = os.path.dirname(file_path)
    if not output_dir:
        output_dir = "."
    save_results(df_final, dataset_id, output_dir, stats)
    
    # Final summary
    print("\n" + "=" * 70)
    print("MODULE 3 COMPLETED")
    print("=" * 70)
    print(f"""
  Dataset: {dataset_id.upper()}
  Execution time: {total_time}
  
  Files generated in: {output_dir}
    • {dataset_id}_configurations.csv
    • {dataset_id}_module3_stats.txt

  Summary:
    • Blocks processed:     {len(df_blocks):,}
    • Unique configs:       {len(df_configs):,} (reduction {len(df_blocks)/len(df_configs):.1f}x)
    • Feasible blocks:      {blocks_with_bins:,} ({100*blocks_with_bins/len(df_blocks):.1f}%)
    • Rows generated:       {len(df_final):,}
    
  Average bounding box dimensions:
    • X_opt: {X_avg:.2f} cm
    • Y_opt: {Y_avg:.2f} cm
    • Z_opt: {Z_avg:.2f} cm
    
  Volume analysis:
    • Average volume efficiency: {efficiency_avg:.1f}%
    • Perfect packing: {perfect_packing:,} cases ({pct_perfect:.1f}%)
""")
    print("=" * 70)
    
    return df_final


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        default_paths = [
            "data/DS-1_blocks.csv",
            "../data/DS-1_blocks.csv"
        ]
        # default_paths = [
        #     "data/DS-2_blocks.csv",
        #     "../data/DS-2_blocks.csv"
        # ]
        # default_paths = [
        #     "data/DS-3_blocks.csv",
        #     "../data/DS-3_blocks.csv"
        # ]
        # default_paths = [
        #     "data/DS-4_blocks.csv",
        #     "../data/DS-4_blocks.csv"
        # ]
        file_path = None
        for path in default_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            print("ERROR: Blocks file not found.")
            print("\nUsage: python create_configurations.py <blocks_file_path.csv>")
            print("\nExamples:")
            print("  python create_configurations.py data/DS-1_blocks.csv")
            print("  python create_configurations.py data/DS-2_blocks.csv")
            sys.exit(1)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    
    df_result = process_configurations(file_path)