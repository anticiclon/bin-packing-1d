# -*- coding: utf-8 -*-
"""
MODULE 1: Block Generation
Generic version for datasets DS-1, DS-2, DS-3, DS-4

Usage:
    python adding_blocks.py <datasets_path.csv>
    
Examples:
    python adding_blocks.py data/DS-1_skus.csv
    python adding_blocks.py data/DS-2_skus.csv

Input:  *_skus.csv 
Output: *_blocks.csv, *_module1_stats.txt
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from create_configurations import create_bin_types, get_orientations, detect_dataset



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def geometric_capacity(l, w, h, rotatable, bins_lwh):
    """Maximum number of SKU units that fit into a single bin 
    (the largest one), based on possible orientations."""
    best = 0
    for (dx, dy, dz) in get_orientations(l, w, h, rotatable):
        for (L, W, H) in bins_lwh:
            cap = int(L // dx) * int(W // dy) * int(H // dz)
            if cap > best:
                best = cap
    return best

def cap_ns_to_capacity(df_skus, file_path):
    """Returns df_skus with n_s reduced to min(n_s, geometric capacity)."""
    dataset_id = detect_dataset(file_path)
    bins = create_bin_types(dataset_id)
    bins_lwh = list(zip(bins["L_t"].tolist(), bins["W_t"].tolist(), bins["H_t"].tolist()))
 
    df = df_skus.copy()
    rot = df["rotatable"].astype(str).str.lower().isin(["true", "1", "1.0"])
    uniq = pd.DataFrame({"length": df["length"], "width": df["width"],
                         "height": df["height"], "rot": rot}).drop_duplicates()
    capmap = {}
    for r in uniq.itertuples(index=False):
        capmap[(r.length, r.width, r.height, bool(r.rot))] = \
            geometric_capacity(r.length, r.width, r.height, bool(r.rot), bins_lwh)
 
    cap = pd.Series(
        [capmap[(L, W, H, bool(R))] for L, W, H, R in
         zip(df["length"], df["width"], df["height"], rot)],
        index=df.index)
 
    n_orig = df["n_s"].astype(int)
    df["n_s"] = pd.concat([n_orig, cap], axis=1).min(axis=1).clip(lower=1).astype(int)
 
    n_capped = int(((cap < n_orig) & (cap >= 1)).sum())
    n_impossible = int((cap == 0).sum())   # the item itself does not fit in ANY bin (it's not an n_s problem)
    if n_capped:
        print(f"  [reduce n_s] {n_capped:,} SKUs with n_s > bin capacity -> capped at the maximum block size that fits")
    else:
        print(f"  [reduce n_s] no SKU exceeds capacity; n_s unchanged")
    if n_impossible:
        print(f"  [warning] {n_impossible:,} SKUs whose item does not fit in any catalog bin (data issue, not n_s issue)")
    return df
 



def detect_dataset(file_path):
    """
    Detects the dataset based on the file name."""
    file_name = os.path.basename(file_path).lower()
    
    datasets = ["ds-1", "ds-2", "ds-3", "ds-4"]
    for ds in datasets:
        if ds in file_name:
            return ds.upper()
    
    raise ValueError(f"Could not detect the dataset from file: {file_path}\n"
                     f"Supported datasets: {datasets}")



def generate_blocks(df_skus):
    """
    Generates the blocks table from the SKUs table.
    
    For each SKU s with |I_s| items and parameter n_s:
    - Computes K_s = ceil(|I_s| / n_s) blocks
    - Assigns q_sk items to each block according to equation (2)
    
    Returns:
        DataFrame with columns: sku_id, block_k, q_sk, length, width, height, rotatable, block_id
    """
    # Compute number of blocks per SKU
    df_skus = df_skus.copy()
    df_skus['K_s'] = np.ceil(df_skus['total_items'] / df_skus['n_s']).astype(int)
    
    # Expand: each SKU is repeated K_s times
    df_blocks = df_skus.loc[df_skus.index.repeat(df_skus['K_s'])].reset_index(drop=True)
    
    # Create block index k = 1, 2, ..., K_s within each SKU
    df_blocks['block_k'] = df_blocks.groupby('sku_id').cumcount() + 1
    
    # Compute q_sk according to equation (2):
    # - Blocks 1 to K_s-1: q_sk = n_s (full blocks)
    # - Block K_s: q_sk = |I_s| - (K_s - 1) * n_s (remainder)
    df_blocks['q_sk'] = np.where(
        df_blocks['block_k'] < df_blocks['K_s'],
        df_blocks['n_s'],
        df_blocks['total_items'] - (df_blocks['K_s'] - 1) * df_blocks['n_s']
    ).astype(int)
    
    # Create unique block identifier
    df_blocks['block_id'] = df_blocks['sku_id'].astype(str) + '_' + df_blocks['block_k'].astype(str)
    
    # Store auxiliary columns for validation
    df_blocks['_n_s'] = df_blocks['n_s']
    df_blocks['_total_items'] = df_blocks['total_items']
    df_blocks['_K_s'] = df_blocks['K_s']
    
    return df_blocks


def validate_blocks(df_blocks):
    """
    Runs integrity validations on the blocks table.
    
    Returns:
        Tuple (bool, list): (all_validations_ok, list_of_messages)
    """
    messages = []
    all_ok = True
    
    # 1. No null values in main columns
    main_cols = ['sku_id', 'block_k', 'q_sk', 'length', 'width', 'height', 'rotatable', 'block_id']
    nulls = df_blocks[main_cols].isnull().sum().sum()
    if nulls == 0:
        messages.append("✓ No null values in main columns")
    else:
        messages.append(f"✗ There are {nulls} null values")
        all_ok = False
    
    # 2. q_sk ≥ 1 (every block has at least 1 item)
    if (df_blocks['q_sk'] >= 1).all():
        messages.append("✓ All blocks have q_sk ≥ 1")
    else:
        n_invalid = (df_blocks['q_sk'] < 1).sum()
        messages.append(f"✗ There are {n_invalid} blocks with q_sk < 1")
        all_ok = False
    
    # 3. q_sk ≤ n_s (no block exceeds capacity)
    if (df_blocks['q_sk'] <= df_blocks['_n_s']).all():
        messages.append("✓ All blocks have q_sk ≤ n_s")
    else:
        n_exceeded = (df_blocks['q_sk'] > df_blocks['_n_s']).sum()
        messages.append(f"✗ There are {n_exceeded} blocks with q_sk > n_s")
        all_ok = False
    
    # 4. Correct sum: ∑ q_sk per SKU = total_items
    sum_per_sku = df_blocks.groupby('sku_id')['q_sk'].sum()
    total_per_sku = df_blocks.groupby('sku_id')['_total_items'].first()
    differences = (sum_per_sku != total_per_sku).sum()
    if differences == 0:
        messages.append("✓ Sum of q_sk matches total_items for all SKUs")
    else:
        messages.append(f"✗ There are {differences} SKUs where ∑q_sk ≠ total_items")
        all_ok = False
    
    # 5. Block count: rows = ∑ K_s
    total_rows = len(df_blocks)
    sum_Ks = df_blocks.groupby('sku_id')['_K_s'].first().sum()
    if total_rows == sum_Ks:
        messages.append(f"✓ Total blocks ({total_rows:,}) matches ∑K_s")
    else:
        messages.append(f"✗ Inconsistency: {total_rows:,} rows vs ∑K_s = {sum_Ks:,}")
        all_ok = False
    
    # 6. Unique block_id
    if df_blocks['block_id'].is_unique:
        messages.append("✓ All block_id values are unique")
    else:
        n_duplicates = df_blocks['block_id'].duplicated().sum()
        messages.append(f"✗ There are {n_duplicates} duplicated block_id values")
        all_ok = False
    
    return all_ok, messages



def compute_statistics(df_blocks):
    """Computes statistics for the report."""
    stats = {}
    
    # Basic counts
    stats['total_skus'] = df_blocks['sku_id'].nunique()
    stats['total_blocks'] = len(df_blocks)
    stats['total_items'] = df_blocks['q_sk'].sum()
    
    # Blocks per SKU
    blocks_per_sku = df_blocks.groupby('sku_id').size()
    stats['avg_blocks_per_sku'] = blocks_per_sku.mean()
    stats['median_blocks_per_sku'] = blocks_per_sku.median()
    stats['max_blocks_per_sku'] = blocks_per_sku.max()
    stats['min_blocks_per_sku'] = blocks_per_sku.min()
    
    # Distribution of q_sk
    stats['q_sk_avg'] = df_blocks['q_sk'].mean()
    stats['q_sk_median'] = df_blocks['q_sk'].median()
    stats['q_sk_min'] = df_blocks['q_sk'].min()
    stats['q_sk_max'] = df_blocks['q_sk'].max()
    stats['q_sk_p25'] = df_blocks['q_sk'].quantile(0.25)
    stats['q_sk_p75'] = df_blocks['q_sk'].quantile(0.75)
    
    # Full vs partial blocks
    full_blocks = (df_blocks['q_sk'] == df_blocks['_n_s']).sum()
    partial_blocks = (df_blocks['q_sk'] < df_blocks['_n_s']).sum()
    stats['full_blocks'] = full_blocks
    stats['partial_blocks'] = partial_blocks
    stats['pct_full'] = full_blocks / stats['total_blocks'] * 100
    stats['pct_partial'] = partial_blocks / stats['total_blocks'] * 100
    
    # SKUs with only 1 block
    skus_1_block = (blocks_per_sku == 1).sum()
    stats['skus_1_block'] = skus_1_block
    stats['pct_skus_1_block'] = skus_1_block / stats['total_skus'] * 100
    
    return stats


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def process_blocks(file_path):
    """Processes a SKUs file and generates the blocks table."""
    
    start_time = datetime.now()
    
    # ========================================================================
    # STEP 0: CONFIGURATION
    # ========================================================================
    
    dataset_id = detect_dataset(file_path)
    
    # Output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(repo_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"MODULE 1: BLOCK GENERATION - {dataset_id.upper()}")
    print("=" * 70)
    print(f"\nInput file: {file_path}")
    
    # ========================================================================
    # STEP 1: LOAD DATA FROM MODULE 1
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 1: Loading data from Module 1")
    print("-" * 70)
    
    required_columns = ['sku_id', 'length', 'width', 'height', 'total_items', 'n_s', 'rotatable']
    
    df_skus = pd.read_csv(file_path)
    
    # Check columns
    missing = [c for c in required_columns if c not in df_skus.columns]
    if missing:
        raise ValueError(f"Missing columns in file: {missing}")
    
    # Check no nulls
    if df_skus[required_columns].isnull().any().any():
        raise ValueError("The file contains null values in required columns")
    
    print(f"  Loaded SKUs: {len(df_skus):,}")
    print(f"  Total items: {df_skus['total_items'].sum():,}")
    
    # ========================================================================
    # STEP 2: GENERATE BLOCKS
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 2: Generating blocks")
    print("-" * 70)
    
    df_skus = cap_ns_to_capacity(df_skus, file_path)   # trim n_s to the maximum block that fits
    
    df_blocks = generate_blocks(df_skus)
    
    print(f"  Blocks generated: {len(df_blocks):,}")
    print(f"  Blocks per SKU (average): {len(df_blocks)/len(df_skus):.2f}")
    
    # ========================================================================
    # STEP 3: VALIDATIONS
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 3: Validations")
    print("-" * 70)
    
    validations_ok, messages = validate_blocks(df_blocks)
    
    for msg in messages:
        print(f"  {msg}")
    
    if not validations_ok:
        raise RuntimeError("Validations failed. Please review the input data.")
    
    # ========================================================================
    # STEP 4: PREPARE FINAL TABLE
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 4: Preparing final table")
    print("-" * 70)
    
    # Select final columns (excluding auxiliary _*)
    final_columns = [
        'sku_id',
        'block_k',
        'q_sk',
        'length',
        'width',
        'height',
        'rotatable',
        'block_id'
    ]
    
    df_final = df_blocks[final_columns].copy()
    
    # Ensure correct data types
    df_final = df_final.astype({
        'sku_id': 'string',
        'block_k': 'int32',
        'q_sk': 'int32',
        'length': 'float64',
        'width': 'float64',
        'height': 'float64',
        'rotatable': 'bool',
        'block_id': 'string'
    })
    
    print(f"  Columns: {final_columns}")
    print(f"  Shape: {df_final.shape}")
    
    # ========================================================================
    # STEP 5: SAVE RESULTS
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 5: Saving results")
    print("-" * 70)
    
    # File names
    output_csv = os.path.join(output_dir, f"{dataset_id}_blocks.csv")
    output_stats = os.path.join(output_dir, f"{dataset_id}_blocks_stats.txt")
    
    # Save CSV
    df_final.to_csv(output_csv, index=False)
    print(f"\n  ✓ CSV saved: {output_csv}")
    
    # Compute statistics
    stats = compute_statistics(df_blocks)
    
    # Execution time
    end_time = datetime.now()
    total_time = end_time - start_time
    
    # Generate report
    content = f"""{'='*70}
{dataset_id.upper()} - MODULE 1: BLOCK GENERATION REPORT
{'='*70}

Processing date: {end_time.strftime("%Y-%m-%d %H:%M:%S")}
Execution time: {total_time}

Input file:   {file_path}
Output file:  {output_csv}

{'='*70}
1. GENERAL SUMMARY
{'='*70}

Total SKUs processed:        {stats['total_skus']:>12,}
Total blocks generated:      {stats['total_blocks']:>12,}
Total items (∑ q_sk):        {stats['total_items']:>12,}

{'='*70}
2. BLOCKS PER SKU
{'='*70}

  Mean:                      {stats['avg_blocks_per_sku']:>12.2f}
  Median:                    {stats['median_blocks_per_sku']:>12.0f}
  Minimum:                   {stats['min_blocks_per_sku']:>12,}
  Maximum:                   {stats['max_blocks_per_sku']:>12,}

  SKUs with 1 block:         {stats['skus_1_block']:>12,} ({stats['pct_skus_1_block']:.1f}%)

{'='*70}
3. q_sk DISTRIBUTION (items per block)
{'='*70}

  Mean:                      {stats['q_sk_avg']:>12.2f}
  Median:                    {stats['q_sk_median']:>12.0f}
  Minimum:                   {stats['q_sk_min']:>12,}
  Maximum:                   {stats['q_sk_max']:>12,}
  P25:                       {stats['q_sk_p25']:>12.0f}
  P75:                       {stats['q_sk_p75']:>12.0f}

{'='*70}
4. FULL VS PARTIAL BLOCKS
{'='*70}

  Full blocks (q_sk = n_s):   {stats['full_blocks']:>10,} ({stats['pct_full']:.1f}%)
  Partial blocks (q_sk < n_s):{stats['partial_blocks']:>10,} ({stats['pct_partial']:.1f}%)

  Note: Partial blocks are typically the last block of each SKU,
        containing the remaining items.

{'='*70}
5. OUTPUT FILE STRUCTURE
{'='*70}

  Columns: {final_columns}
  
  Description:
    - sku_id:    SKU identifier
    - block_k:   Block index (1 to K_s)
    - q_sk:      Number of items in the block
    - length:    Item depth (cm)
    - width:     Item width (cm)
    - height:    Item height (cm)
    - rotatable: Whether rotation is allowed (True/False)
    - block_id:  Unique block identifier (sku_id + "_" + block_k)

{'='*70}
END OF REPORT
{'='*70}
"""
    
    with open(output_stats, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"  ✓ Statistics saved: {output_stats}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("MODULE 1 COMPLETED")
    print("=" * 70)
    
    print(f"""
  Dataset: {dataset_id.upper()}
  Execution time: {total_time}
  
  Files generated in: {output_dir}
    • {dataset_id}_blocks.csv
    • {dataset_id}_module1_stats.txt

  Summary:
    • SKUs processed:       {stats['total_skus']:,}
    • Blocks generated:     {stats['total_blocks']:,}
    • Total items:          {stats['total_items']:,}
    • Blocks/SKU (avg):     {stats['avg_blocks_per_sku']:.2f}
    • SKUs with 1 block:    {stats['pct_skus_1_block']:.1f}%
""")
    
    print("=" * 70)
    
    return df_final


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Get file path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path: relative to the repository
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(script_dir)
        file_path = os.path.join(repo_dir, "data", "DS-1_skus.csv")
        # DS-1_skus.csv
        # DS-2_skus.csv
        # DS-3_skus.csv
        # DS-4_skus.csv
    
    # Check that it exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    
    # Process
    df_result = process_blocks(file_path)
