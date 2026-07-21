# -*- coding: utf-8 -*-
"""
MODULE 3: Consolidation of Preprocessed Data
Saves the information in efficient formats (Parquet and SQLite)

This module takes the outputs from previous modules and structures them into
three relational tables optimized for the optimization model:

    1. blocks:    Information for each block (sku_id, block_k, q_sk, dimensions)
    2. bin_types: Catalog of available bin types
    3. X_optimal: Matrix of optimal dimensions X* for each pair (block, bin_type)

Output formats:
    - Parquet: Efficient compression, fast columnar reading
    - SQLite:  Ad-hoc queries, joins, interactive exploration

Usage:
    python save_preprocessed_data.py <dataset_id>
    python save_preprocessed_data.py brmg01
    python save_preprocessed_data.py --all  # Process all datasets

Input:
    - *_blocks.csv (output from Module 1)
    - *_configurations.csv (output from Module 2)

Output:
    - *_model_data.parquet (3 tables in a single file)
    - *_model_data.db (SQLite with 3 tables)
"""

import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Bin catalog: imported from create_configurations.py so there is a single
# source of truth for catalog dimensions and the small/large category rule.
# (A previous local copy of this logic had a case-mismatch bug:
# `dataset_id.lower() in ['DS-3', 'DS-4']` compares a lowercased string
# against uppercase literals, so it was never True and every dataset
# silently received the 'large' catalog. Importing avoids re-introducing
# that class of bug and keeps both modules in sync if the catalog changes.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_configurations import create_bin_types

# Parquet requires pyarrow
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("WARNING: pyarrow not available. Only SQLite will be generated.")


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = ['DS-1', 'DS-2', 'DS-3', 'DS-4']


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def find_input_files(dataset_id: str, base_dir: str = None) -> Dict[str, Optional[str]]:
    """
    Searches for input files for a dataset.
    
    Returns:
        Dict with paths to 'blocks' and 'configurations' (None if not found)
    """
    if base_dir is None:
        possible_dirs = ['data', '../data', '.']
        for d in possible_dirs:
            if os.path.exists(d):
                base_dir = d
                break
        else:
            base_dir = '.'
    
    files = {'blocks': None, 'configurations': None}
    
    blocks_path = os.path.join(base_dir, f"{dataset_id}_blocks.csv")
    config_path = os.path.join(base_dir, f"{dataset_id}_configurations.csv")
    
    if os.path.exists(blocks_path):
        files['blocks'] = blocks_path
    if os.path.exists(config_path):
        files['configurations'] = config_path
    
    return files


def load_blocks(filepath: str) -> pd.DataFrame:
    """
    Loads and validates the blocks file.
    
    Returns:
        DataFrame with columns: sku_id, block_k, q_sk, length, width, height, rotatable, block_id
    """
    df = pd.read_csv(filepath)
    
    required = ['sku_id', 'block_k', 'q_sk', 'length', 'width', 'height', 'rotatable', 'block_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in blocks: {missing}")
    
    return df[required].copy()


def load_configurations(filepath: str) -> pd.DataFrame:
    """
    Loads and validates the configurations file.
    
    Returns:
        DataFrame with columns: block_id, bin_type_id, X_opt, Y_opt, Z_opt, etc.
    """
    df = pd.read_csv(filepath)
    
    required = ['block_id', 'bin_type_id', 'X_opt']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in configurations: {missing}")
    
    return df.copy()


# ============================================================================
# CREATION OF RELATIONAL TABLES
# ============================================================================

def create_blocks_table(df_blocks: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the 'blocks' table.
    """
    df = df_blocks.rename(columns={
        'length': 'l_s',
        'width': 'w_s',
        'height': 'h_s'
    }).copy()
    
    df = df.astype({
        'sku_id': 'string',
        'block_k': 'int32',
        'q_sk': 'int32',
        'l_s': 'float32',
        'w_s': 'float32',
        'h_s': 'float32',
        'rotatable': 'bool',
        'block_id': 'string'
    })
    
    df = df.sort_values(['sku_id', 'block_k']).reset_index(drop=True)
    
    return df


def create_bin_types_table(dataset_id: str) -> pd.DataFrame:
    """
    Creates the 'bin_types' table by delegating to create_bin_types()
    in create_configurations.py — the single source of truth for the
    catalog dimensions and the small/large category rule per dataset.
    """
    df = create_bin_types(dataset_id).astype({
        'bin_type_id': 'string',
        'L_t': 'float32',
        'W_t': 'float32',
        'H_t': 'float32',
        'volume': 'float32'
    }).sort_values('volume').reset_index(drop=True)

    return df


def create_X_optimal_table(df_configs: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the 'X_optimal' table with bounding box results.
    """
    cols_to_keep = ['block_id', 'bin_type_id', 'X_opt']
    optional_cols = ['Y_opt', 'Z_opt', 'a', 'b', 'c', 'block_volume', 'bbox_volume', 'volume_match']
    
    for col in optional_cols:
        if col in df_configs.columns:
            cols_to_keep.append(col)
    
    df = df_configs[cols_to_keep].copy()
    
    type_mapping = {
        'block_id': 'string',
        'bin_type_id': 'string',
        'X_opt': 'float32'
    }
    
    for col, dtype in type_mapping.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    
    df = df.sort_values(['block_id', 'bin_type_id']).reset_index(drop=True)
    
    return df

# ============================================================================
# SAVE TO PARQUET
# ============================================================================

def save_to_parquet(
    df_blocks: pd.DataFrame,
    df_bin_types: pd.DataFrame,
    df_X_optimal: pd.DataFrame,
    output_path: str,
    dataset_id: str
) -> None:
    """
    Saves the three tables in separate Parquet files.
    
    Three files are created:
        - {dataset_id}_blocks.parquet
        - {dataset_id}_bin_types.parquet
        - {dataset_id}_X_optimal.parquet
    """
    if not PARQUET_AVAILABLE:
        print("  ⚠ Parquet not available (install pyarrow)")
        return
    
    output_dir = os.path.dirname(output_path) or '.'
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    # Save each table
    tables = [
        ('blocks', df_blocks),
        ('bin_types', df_bin_types),
        ('X_optimal', df_X_optimal)
    ]
    
    for table_name, df in tables:
        parquet_path = os.path.join(output_dir, f"{dataset_id}_{table_name}.parquet")
        
        # Convert to PyArrow table and save with compression
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            parquet_path,
            compression='snappy',  # Good balance between compression and speed
            use_dictionary=True,   # Efficient for columns with repeated values
        )
        
        # Compute size
        size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"  ✓ {parquet_path} ({len(df):,} rows, {size_mb:.2f} MB)")


# ============================================================================
# SAVE TO SQLITE
# ============================================================================

def save_to_sqlite(
    df_blocks: pd.DataFrame,
    df_bin_types: pd.DataFrame,
    df_X_optimal: pd.DataFrame,
    output_path: str
) -> None:
    """
    Saves the three tables into a SQLite database.
    
    Creates indexes to optimize common queries:
        - blocks: PRIMARY KEY (block_id), INDEX (sku_id)
        - bin_types: PRIMARY KEY (bin_type_id)
        - X_optimal: PRIMARY KEY (block_id, bin_type_id), INDEX (bin_type_id)
    """
    # Remove existing file
    if os.path.exists(output_path):
        os.remove(output_path)
    
    conn = sqlite3.connect(output_path)
    
    try:
        # ====================================================================
        # Table: blocks
        # ====================================================================
        df_blocks.to_sql('blocks', conn, index=False, if_exists='replace')
        
        # Create indexes
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_blocks_pk ON blocks(block_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_sku ON blocks(sku_id)")
        
        # ====================================================================
        # Table: bin_types
        # ====================================================================
        df_bin_types.to_sql('bin_types', conn, index=False, if_exists='replace')
        
        # Create index
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_bin_types_pk ON bin_types(bin_type_id)")
        
        # ====================================================================
        # Table: X_optimal
        # ====================================================================
        df_X_optimal.to_sql('X_optimal', conn, index=False, if_exists='replace')
        
        # Create indexes (composite key + index by bin_type_id for joins)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_X_optimal_pk ON X_optimal(block_id, bin_type_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_X_optimal_bin ON X_optimal(bin_type_id)")
        
        # ====================================================================
        # Metadata
        # ====================================================================
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        metadata = [
            ('created_at', datetime.now().isoformat()),
            ('n_blocks', str(len(df_blocks))),
            ('n_bin_types', str(len(df_bin_types))),
            ('n_X_optimal', str(len(df_X_optimal))),
            ('n_skus', str(df_blocks['sku_id'].nunique())),
        ]
        
        conn.executemany(
            "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
            metadata
        )
        
        conn.commit()
        
        # Compute size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ {output_path} ({size_mb:.2f} MB)")
        
    finally:
        conn.close()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def consolidate_data(dataset_id: str, base_dir: str = None, output_dir: str = None) -> Dict:
    """
    Consolidates the preprocessed data for a dataset.
    
    Args:
        dataset_id: Dataset identifier ('DS-1', 'DS-2', etc.)
        base_dir: Directory where input files should be searched
        output_dir: Output directory (defaults to base_dir)
    
    Returns:
        Dict with statistics and generated file paths
    """
    start_time = datetime.now()
    
    print("=" * 70)
    print(f"MODULE 4: DATA CONSOLIDATION - {dataset_id.upper()}")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: FIND INPUT FILES
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 1: Searching for input files")
    print("-" * 70)
    
    files = find_input_files(dataset_id, base_dir)
    
    if files['blocks'] is None:
        raise FileNotFoundError(f"Blocks file not found for {dataset_id}")
    if files['configurations'] is None:
        raise FileNotFoundError(f"Configurations file not found for {dataset_id}")
    
    print(f"  Blocks: {files['blocks']}")
    print(f"  Configurations: {files['configurations']}")
    
    # ========================================================================
    # STEP 2: LOAD DATA
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 2: Loading data")
    print("-" * 70)
    
    df_blocks_raw = load_blocks(files['blocks'])
    df_configs_raw = load_configurations(files['configurations'])
    
    print(f"  Blocks loaded: {len(df_blocks_raw):,}")
    print(f"  Configurations loaded: {len(df_configs_raw):,}")
    
    # ========================================================================
    # STEP 3: CREATE RELATIONAL TABLES
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 3: Creating relational tables")
    print("-" * 70)
    
    # Table: blocks
    df_blocks = create_blocks_table(df_blocks_raw)
    print(f"  blocks: {len(df_blocks):,} rows, {len(df_blocks.columns)} columns")
    print(f"    Columns: {list(df_blocks.columns)}")
    
    # Table: bin_types
    df_bin_types = create_bin_types_table(dataset_id)
    print(f"  bin_types: {len(df_bin_types):,} rows, {len(df_bin_types.columns)} columns")
    print(f"    Columns: {list(df_bin_types.columns)}")
    
    # Table: X_optimal
    df_X_optimal = create_X_optimal_table(df_configs_raw)
    print(f"  X_optimal: {len(df_X_optimal):,} rows, {len(df_X_optimal.columns)} columns")
    print(f"    Columns: {list(df_X_optimal.columns)}")
    
    # ========================================================================
    # STEP 4: VALIDATIONS
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 4: Integrity validations")
    print("-" * 70)
    
    # Validate that all block_id values in X_optimal exist in blocks
    block_ids_blocks = set(df_blocks['block_id'])
    block_ids_X_optimal = set(df_X_optimal['block_id'])
    
    missing_blocks = block_ids_X_optimal - block_ids_blocks
    if missing_blocks:
        print(f"  ⚠ {len(missing_blocks)} block_id values in X_optimal are not in blocks")
    else:
        print(f"  ✓ All block_id values in X_optimal are in blocks")
    
    # Validate that all bin_type_id values in X_optimal exist in bin_types
    bin_type_ids = set(df_bin_types['bin_type_id'])
    bin_types_X_optimal = set(df_X_optimal['bin_type_id'])
    
    missing_bins = bin_types_X_optimal - bin_type_ids
    if missing_bins:
        print(f"  ⚠ {len(missing_bins)} bin_type_id values in X_optimal are not in bin_types")
    else:
        print(f"  ✓ All bin_type_id values in X_optimal are in bin_types")
    
    # Coverage statistics
    blocks_with_configs = df_blocks[df_blocks['block_id'].isin(block_ids_X_optimal)]
    pct_coverage = 100 * len(blocks_with_configs) / len(df_blocks)
    print(f"  ✓ Coverage: {len(blocks_with_configs):,}/{len(df_blocks):,} blocks ({pct_coverage:.1f}%)")
    
    # Average feasible bins per block
    bins_per_block = df_X_optimal.groupby('block_id').size().mean()
    print(f"  ✓ Feasible bins per block (average): {bins_per_block:.1f}")
    
    # ========================================================================
    # STEP 5: SAVE RESULTS
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("STEP 5: Saving results")
    print("-" * 70)
    
    if output_dir is None:
        output_dir = os.path.dirname(files['blocks']) or '.'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Parquet
    print("\n  Parquet:")
    parquet_base = os.path.join(output_dir, f"{dataset_id}_model_data.parquet")
    save_to_parquet(df_blocks, df_bin_types, df_X_optimal, parquet_base, dataset_id)
    
    # Save to SQLite
    print("\n  SQLite:")
    sqlite_path = os.path.join(output_dir, f"{dataset_id}_model_data.db")
    save_to_sqlite(df_blocks, df_bin_types, df_X_optimal, sqlite_path)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    stats = {
        'dataset_id': dataset_id,
        'n_blocks': len(df_blocks),
        'n_skus': df_blocks['sku_id'].nunique(),
        'n_bin_types': len(df_bin_types),
        'n_X_optimal': len(df_X_optimal),
        'coverage_pct': pct_coverage,
        'bins_per_block_avg': bins_per_block,
        'execution_time': total_time,
        'output_dir': output_dir,
    }
    
    print("\n" + "=" * 70)
    print("MODULE 4 COMPLETED")
    print("=" * 70)
    print(f"""
  Dataset: {dataset_id.upper()}
  Execution time: {total_time}
  
  Generated tables:
    • blocks:    {stats['n_blocks']:,} rows ({stats['n_skus']:,} SKUs)
    • bin_types: {stats['n_bin_types']:,} rows
    • X_optimal: {stats['n_X_optimal']:,} rows
  
  Files generated in: {output_dir}
    • {dataset_id}_blocks.parquet
    • {dataset_id}_bin_types.parquet
    • {dataset_id}_X_optimal.parquet
    • {dataset_id}_model_data.db (SQLite)
  
  Coverage: {stats['coverage_pct']:.1f}% of blocks with at least 1 feasible bin
  Feasible bins/block: {stats['bins_per_block_avg']:.1f} (average)
""")
    print("=" * 70)
    
    return stats


def print_usage_examples(dataset_id: str, output_dir: str):
    """Prints examples of how to use the saved data."""
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    
    print("""
  # ======================================================================
  # OPTION 1: Load from Parquet (recommended for analysis)
  # ======================================================================
  
  import pandas as pd
  
  # Load individual tables
  df_blocks = pd.read_parquet('{dir}/{ds}_blocks.parquet')
  df_bin_types = pd.read_parquet('{dir}/{ds}_bin_types.parquet')
  df_X_optimal = pd.read_parquet('{dir}/{ds}_X_optimal.parquet')
  
  # Example: average X* dimension by bin type
  df_X_optimal.groupby('bin_type_id')['X_opt'].mean()
  
  # ======================================================================
  # OPTION 2: Query from SQLite (recommended for exploration)
  # ======================================================================
  
  import sqlite3
  import pandas as pd
  
  conn = sqlite3.connect('{dir}/{ds}_model_data.db')
  
  # Query: blocks with their optimal dimensions for a bin type
  query = '''
      SELECT b.sku_id, b.block_k, b.q_sk, x.X_opt, x.Y_opt, x.Z_opt
      FROM blocks b
      JOIN X_optimal x ON b.block_id = x.block_id
      WHERE x.bin_type_id = 'L030_W046_H035'
      ORDER BY x.X_opt DESC
      LIMIT 10
  '''
  df = pd.read_sql(query, conn)
  
  # Query: bin types ordered by volume
  df_bins = pd.read_sql('SELECT * FROM bin_types ORDER BY volume', conn)
  
  conn.close()
""".format(ds=dataset_id, dir=output_dir))


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION FOR INTERACTIVE EXECUTION (Spyder, Jupyter, etc.)
    # ========================================================================
    # If you run from Spyder without arguments, modify this variable:
    DEFAULT_DATASET = 'DS-1'  # Change as needed
    # DS-1
    # DS-2
    # DS-3
    # DS-4
    
    # Detect script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)  # Go up one level (from code/ to repo/)
    default_data_dir = os.path.join(repo_dir, "data")
    
    # ========================================================================
    # PARSE ARGUMENTS
    # ========================================================================
    if len(sys.argv) < 2:
        # No arguments: use default value (useful for Spyder)
        print(f"No command-line arguments provided.")
        print(f"Using default dataset: {DEFAULT_DATASET}")
        print(f"Searching for data in: {default_data_dir}")
        print()
        arg = DEFAULT_DATASET
    else:
        arg = sys.argv[1]
    
    # ========================================================================
    # RUN
    # ========================================================================
    if arg.lower() == '--all':
        # Process all datasets
        for ds in DATASETS:
            # Output directory: final_data/{dataset_id}/
            output_dir_ds = os.path.join(repo_dir, "final_data", ds)
            try:
                consolidate_data(ds, base_dir=default_data_dir, output_dir=output_dir_ds)
            except FileNotFoundError as e:
                print(f"\n⚠ Skipping {ds}: {e}")
                continue
    elif arg.upper() in DATASETS:
        arg = arg.upper()
        # Output directory: final_data/{dataset_id}/
        output_dir_ds = os.path.join(repo_dir, "final_data", arg)
        try:
            stats = consolidate_data(arg, base_dir=default_data_dir, output_dir=output_dir_ds)
            print_usage_examples(arg, stats['output_dir'])
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print(f"\nMake sure the following files exist:")
            print(f"  - {default_data_dir}/{arg}_blocks.csv")
            print(f"  - {default_data_dir}/{arg}_configurations.csv")
            print(f"\nThese files are generated by running first:")
            print(f"  1. python adding_blocks.py")
            print(f"  2. python create_configurations.py")
    else:
        print(f"Unrecognized dataset: {arg}")
        print("Available datasets:", DATASETS)
        print()
        print("Usage from terminal:")
        print("  python save_preprocessed_data.py <dataset_id>")
        print("  python save_preprocessed_data.py --all")
        print()
        print("Usage from Spyder:")
        print("  Modify DEFAULT_DATASET at the beginning of the __main__ block")