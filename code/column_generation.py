# -*- coding: utf-8 -*-
"""
Column Generation for Bin Sizing

Components:
  1. Restricted Master Problem (RMP): LP relaxation of set partitioning
  2. Pricing Subproblem (SP): MILP solved by bin type
  3. Initialization from BFD heuristic or singleton patterns
  4. Main column generation loop

Usage:
    python column_generation.py <dataset_id>
    python column_generation.py DS-1 --m 4 --init bfd
    python column_generation.py DS-1 --m 4 --init bfd --max_skus 200

Input:
    - final_data/{dataset_id}/*_blocks.parquet
    - final_data/{dataset_id}/*_bin_types.parquet
    - final_data/{dataset_id}/*_X_optimal.parquet
    - results/{dataset_id}/{dataset_id}_bfd_solution.csv  (if init=bfd)

Output:
    - results/{dataset_id}/{dataset_id}_cg_solution.csv
    - results/{dataset_id}/{dataset_id}_cg_summary.csv
    - results/{dataset_id}/{dataset_id}_cg_duals.csv
"""

import os
import sys
import time
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("ERROR: Gurobi is not installed. Run: pip install gurobipy")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Pattern:
    """
    A packing pattern, i.e., a column of the Restricted Master Problem.

    Represents a complete bin configuration:
    - Bin type used
    - Set of contained blocks
    - Cost = volume of the bin type
    """
    pattern_id: int
    bin_type_id: str
    blocks: List[Tuple[str, int]]      # List of (sku_id, block_k)
    cost: float                        # Bin volume = L_t * W_t * H_t
    skus: Set[str] = field(default_factory=set)
    total_x: float = 0.0               # Sum of X* values of the blocks in this bin type

    def __post_init__(self):
        if not self.skus:
            self.skus = {sku for sku, _ in self.blocks}

    @property
    def fingerprint(self) -> int:
        """Unique hash based on bin type and contained blocks."""
        return hash((self.bin_type_id, frozenset(self.blocks)))


@dataclass
class ColumnGenerationConfig:
    """Configuration for the column generation algorithm."""
    dataset_id: str
    max_skus_per_bin: int = 3
    epsilon: float = 1e-6
    max_iterations: int = 5000
    pricing_time_limit: float = 60.0
    pricing_mip_gap: float = 0.01
    pricing_strategy: str = "per_type"       # Currently supported: "per_type"
    initialization_strategy: str = "singleton"  # "singleton" or "bfd"
    verbose: int = 1                         # 0=silent, 1=summary, 2=detailed
    max_skus: Optional[int] = None
    selected_skus: Optional[List[str]] = None

    # Pricing parameters
    pricing_pool_size: int = 5
    max_patterns_per_iteration: int = 50

    # Pricing pre-filtering
    pricing_max_blocks: int = 0              # 0 = no limit; >0 = top-K blocks by dual value

    # Logging
    log_interval: int = 5

    # Output directory
    results_dir: Optional[str] = None        # If None, uses results/{dataset_id}


@dataclass
class ColumnGenerationResult:
    """Result of the column generation algorithm."""
    lower_bound: float
    upper_bound: float
    gap_pct: float
    n_iterations: int
    n_patterns: int
    runtime: float
    patterns: List[Pattern]
    dual_values: Dict[Tuple[str, int], float]
    lp_solution: Dict[int, float]            # pattern_id -> y_p*
    converged: bool = False


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_model_data(dataset_id: str, base_dir: str = None):
    """Load preprocessed model data from Parquet files."""
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(script_dir)
        base_dir = os.path.join(repo_dir, "final_data", dataset_id)

    paths = {
        "blocks": os.path.join(base_dir, f"{dataset_id}_blocks.parquet"),
        "bin_types": os.path.join(base_dir, f"{dataset_id}_bin_types.parquet"),
        "x_optimal": os.path.join(base_dir, f"{dataset_id}_X_optimal.parquet"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file was not found: {path}")

    blocks_df = pd.read_parquet(paths["blocks"])
    bin_types_df = pd.read_parquet(paths["bin_types"])
    x_optimal_df = pd.read_parquet(paths["x_optimal"])

    return blocks_df, bin_types_df, x_optimal_df


def prepare_column_generation_data(
    blocks_df: pd.DataFrame,
    bin_types_df: pd.DataFrame,
    x_optimal_df: pd.DataFrame,
) -> Dict:
    """
    Prepare optimized data structures for column generation.
    """
    data = {}

    # --- Basic sets ---
    data["skus"] = blocks_df["sku_id"].unique().tolist()
    data["blocks"] = list(zip(blocks_df["sku_id"], blocks_df["block_k"]))
    data["bin_types"] = bin_types_df["bin_type_id"].tolist()

    data["n_skus"] = len(data["skus"])
    data["n_blocks"] = len(data["blocks"])
    data["n_bin_types"] = len(data["bin_types"])

    # --- Identification mappings ---
    data["block_id_to_tuple"] = dict(zip(
        blocks_df["block_id"],
        zip(blocks_df["sku_id"], blocks_df["block_k"])
    ))
    data["tuple_to_block_id"] = {
        block_tuple: block_id
        for block_id, block_tuple in data["block_id_to_tuple"].items()
    }

    # --- Block parameters ---
    data["quantity_by_block"] = dict(zip(
        zip(blocks_df["sku_id"], blocks_df["block_k"]),
        blocks_df["q_sk"].astype(int)
    ))

    dimensions_by_sku = blocks_df.groupby("sku_id")[["l_s", "w_s", "h_s"]].first()
    data["length_by_sku"] = dimensions_by_sku["l_s"].to_dict()
    data["width_by_sku"] = dimensions_by_sku["w_s"].to_dict()
    data["height_by_sku"] = dimensions_by_sku["h_s"].to_dict()

    # Blocks by SKU
    blocks_by_sku = defaultdict(list)
    for sku, block_k in data["blocks"]:
        blocks_by_sku[sku].append((sku, block_k))

    data["blocks_by_sku"] = dict(blocks_by_sku)
    data["n_blocks_by_sku"] = {
        sku: len(blocks)
        for sku, blocks in data["blocks_by_sku"].items()
    }

    # --- Bin type parameters ---
    data["bin_length"] = dict(zip(bin_types_df["bin_type_id"], bin_types_df["L_t"]))
    data["bin_width"] = dict(zip(bin_types_df["bin_type_id"], bin_types_df["W_t"]))
    data["bin_height"] = dict(zip(bin_types_df["bin_type_id"], bin_types_df["H_t"]))
    data["bin_volume"] = dict(zip(bin_types_df["bin_type_id"], bin_types_df["volume"]))

    data["bin_types_sorted_by_volume"] = (
        bin_types_df.sort_values("volume")["bin_type_id"].tolist()
    )

    # --- X* matrix ---
    valid_block_ids = set(data["block_id_to_tuple"].keys())
    valid_x_df = x_optimal_df[x_optimal_df["block_id"].isin(valid_block_ids)]

    block_tuples = valid_x_df["block_id"].map(data["block_id_to_tuple"])

    data["x_star"] = dict(zip(
        zip(block_tuples, valid_x_df["bin_type_id"]),
        valid_x_df["X_opt"]
    ))

    data["feasible_pairs"] = set(data["x_star"].keys())

    # Compatible bin types by block, sorted by increasing volume
    compatible_types_by_block = defaultdict(list)
    for block_tuple, bin_type_id in data["x_star"].keys():
        compatible_types_by_block[block_tuple].append(bin_type_id)

    for block_tuple in compatible_types_by_block:
        compatible_types_by_block[block_tuple].sort(
            key=lambda bin_type: data["bin_volume"][bin_type]
        )

    data["compatible_types_by_block"] = dict(compatible_types_by_block)

    # Compatible blocks by bin type
    blocks_by_type = defaultdict(list)
    for block_tuple, bin_type_id in data["x_star"].keys():
        blocks_by_type[bin_type_id].append(block_tuple)

    data["blocks_by_type"] = dict(blocks_by_type)

    # Compatible SKUs by bin type
    skus_by_type = defaultdict(set)
    for block_tuple, bin_type_id in data["x_star"].keys():
        skus_by_type[bin_type_id].add(block_tuple[0])

    data["skus_by_type"] = {
        bin_type: list(skus)
        for bin_type, skus in skus_by_type.items()
    }

    # --- Block volumes ---
    data["block_volume"] = {}
    for sku, block_k in data["blocks"]:
        quantity = data["quantity_by_block"][(sku, block_k)]
        data["block_volume"][(sku, block_k)] = (
            quantity
            * data["length_by_sku"][sku]
            * data["width_by_sku"][sku]
            * data["height_by_sku"][sku]
        )

    data["total_item_volume"] = sum(data["block_volume"].values())

    # --- Numeric block indices, useful for the RMP ---
    data["block_to_index"] = {
        block: index
        for index, block in enumerate(data["blocks"])
    }
    data["index_to_block"] = {
        index: block
        for block, index in data["block_to_index"].items()
    }

    return data


# ============================================================================
# STEP 1: INITIAL PATTERN GENERATION
# ============================================================================

def generate_singleton_patterns(data: Dict, starting_pattern_id: int = 0) -> List[Pattern]:
    """
    Generate singleton patterns: one pattern per block using the smallest
    compatible bin type.

    This guarantees a trivial feasible starting point when every block has
    at least one compatible bin type.
    """
    patterns = []
    pattern_id = starting_pattern_id

    for block in data["blocks"]:
        compatible_types = data["compatible_types_by_block"].get(block, [])

        if not compatible_types:
            print(f"  ⚠ Block {block} has no compatible bin type — skipped")
            continue

        best_bin_type = compatible_types[0]
        x_value = data["x_star"][(block, best_bin_type)]
        cost = data["bin_volume"][best_bin_type]

        pattern = Pattern(
            pattern_id=pattern_id,
            bin_type_id=best_bin_type,
            blocks=[block],
            cost=cost,
            total_x=x_value,
        )

        patterns.append(pattern)
        pattern_id += 1

    print(f"  Singleton patterns generated: {len(patterns):,}")
    return patterns


def load_bfd_solution(dataset_id: str, base_dir: str = None) -> pd.DataFrame:
    """
    Load a previously generated Best Fit Decreasing solution from CSV.

    Expected columns include:
        block_id, sku_id, block_k, bin_id, bin_type_id, ...
    """
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(script_dir)
        base_dir = os.path.join(repo_dir, "results", dataset_id)

    csv_path = os.path.join(base_dir, f"{dataset_id}_bfd_solution.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"BFD solution was not found: {csv_path}\n"
            f"Run first: python bfd_heuristic.py {dataset_id}"
        )

    bfd_df = pd.read_csv(csv_path)

    print(f"  BFD solution loaded: {csv_path}")
    print(f"    Rows: {len(bfd_df):,}")
    print(f"    Bins: {bfd_df['bin_id'].nunique():,}")
    print(f"    SKUs: {bfd_df['sku_id'].nunique():,}")

    return bfd_df


def generate_patterns_from_bfd(
    bfd_df: pd.DataFrame,
    data: Dict,
    starting_pattern_id: int = 0,
) -> List[Pattern]:
    """
    Generate initial patterns from a saved BFD solution.

    Each bin in the BFD solution becomes one pattern.
    Only blocks present in data["blocks"] are included, so this respects SKU
    filtering if such filtering was applied.
    """
    valid_blocks = set(data["blocks"])
    patterns = []
    pattern_id = starting_pattern_id
    n_skipped = 0

    grouped_bins = bfd_df.groupby("bin_id").agg({
        "bin_type_id": "first",
        "sku_id": list,
        "block_k": list,
    }).reset_index()

    for _, row in grouped_bins.iterrows():
        bin_type = row["bin_type_id"]

        blocks = []
        for sku, block_k in zip(row["sku_id"], row["block_k"]):
            # block = (sku, int(block_k))
            block = (str(sku), int(block_k))
            if block in valid_blocks:
                blocks.append(block)

        if not blocks:
            n_skipped += 1
            continue

        if bin_type not in data["bin_volume"]:
            n_skipped += 1
            continue

        cost = data["bin_volume"][bin_type]
        total_x = sum(
            data["x_star"].get((block, bin_type), 0)
            for block in blocks
        )

        pattern = Pattern(
            pattern_id=pattern_id,
            bin_type_id=bin_type,
            blocks=blocks,
            cost=cost,
            total_x=total_x,
        )

        patterns.append(pattern)
        pattern_id += 1

    if n_skipped > 0:
        print(f"  ⚠ {n_skipped:,} BFD bins skipped because their blocks were outside the active set")

    print(f"  Patterns generated from BFD: {len(patterns):,}")

    if patterns:
        blocks_per_pattern = [len(pattern.blocks) for pattern in patterns]
        skus_per_pattern = [len(pattern.skus) for pattern in patterns]
        total_bfd_volume = sum(pattern.cost for pattern in patterns)

        print(
            f"    Blocks/pattern: mean={np.mean(blocks_per_pattern):.2f}, "
            f"max={max(blocks_per_pattern)}"
        )
        print(
            f"    SKUs/pattern:   mean={np.mean(skus_per_pattern):.2f}, "
            f"max={max(skus_per_pattern)}"
        )
        print(f"    Total BFD volume: {total_bfd_volume:,.0f} cm³")

    return patterns


def ensure_initial_coverage(patterns: List[Pattern], data: Dict) -> List[Pattern]:
    """
    Ensure that every block is covered by at least one initial pattern.

    This is especially useful when initializing from a filtered BFD solution.
    Missing blocks are covered with singleton patterns.
    """
    covered_blocks = set()
    for pattern in patterns:
        covered_blocks.update(pattern.blocks)

    missing_blocks = [
        block
        for block in data["blocks"]
        if block not in covered_blocks
    ]

    if not missing_blocks:
        return patterns

    print(f"  ⚠ Initial solution misses {len(missing_blocks):,} blocks")
    print("  Adding singleton fallback patterns for missing blocks")

    next_pattern_id = max((pattern.pattern_id for pattern in patterns), default=-1) + 1

    fallback_patterns = []
    for block in missing_blocks:
        compatible_types = data["compatible_types_by_block"].get(block, [])

        if not compatible_types:
            print(f"    ⚠ Block {block} has no compatible bin type — cannot add fallback pattern")
            continue

        best_bin_type = compatible_types[0]
        fallback_patterns.append(Pattern(
            pattern_id=next_pattern_id,
            bin_type_id=best_bin_type,
            blocks=[block],
            cost=data["bin_volume"][best_bin_type],
            total_x=data["x_star"][(block, best_bin_type)],
        ))
        next_pattern_id += 1

    print(f"  Fallback singleton patterns added: {len(fallback_patterns):,}")

    return patterns + fallback_patterns


# ============================================================================
# STEP 2: RESTRICTED MASTER PROBLEM
# ============================================================================

class RestrictedMasterProblem:
    """
    Restricted Master Problem: LP relaxation of set partitioning.

    min  sum_{p in P'} c_p * y_p

    s.t. sum_{p in P': block in p} y_p = 1   for every block
         y_p >= 0

    Optimized for large instances:
    - Efficient build using block -> patterns inverse index
    - O(1) hash-based pattern deduplication
    """

    def __init__(self, data: Dict, config: ColumnGenerationConfig):
        self.data = data
        self.config = config
        self.model = None
        self.y_vars = {}              # pattern_id -> Gurobi variable y_p
        self.cover_constraints = {}   # block tuple -> Gurobi constraint
        self.patterns = []            # List[Pattern]
        self.fingerprints = set()     # Hash-based deduplication

    def build(self, initial_patterns: List[Pattern]):
        """Build the RMP using the initial pattern set."""
        start_time = time.time()

        self.model = gp.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = -1  # Auto

        blocks = self.data["blocks"]

        # --- Step 1: Build inverse index block -> pattern_ids ---
        block_to_pattern_ids = defaultdict(list)
        for pattern in initial_patterns:
            for block in pattern.blocks:
                block_to_pattern_ids[block].append(pattern.pattern_id)

        # --- Step 2: Add y_p variables ---
        for pattern in initial_patterns:
            var = self.model.addVar(
                lb=0.0,
                ub=GRB.INFINITY,
                obj=pattern.cost,
                vtype=GRB.CONTINUOUS,
                name=f"y_{pattern.pattern_id}",
            )
            self.y_vars[pattern.pattern_id] = var
            self.fingerprints.add(pattern.fingerprint)

        self.patterns = list(initial_patterns)
        self.model.update()

        # --- Step 3: Add coverage constraints ---
        n_uncovered = 0

        for block in blocks:
            pattern_ids = block_to_pattern_ids.get(block, [])

            if pattern_ids:
                constraint = self.model.addConstr(
                    gp.quicksum(self.y_vars[pid] for pid in pattern_ids) == 1,
                    name=f"cover_{block[0]}_{block[1]}",
                )
            else:
                constraint = self.model.addConstr(
                    0 == 1,
                    name=f"cover_{block[0]}_{block[1]}",
                )
                n_uncovered += 1

            self.cover_constraints[block] = constraint

        self.model.update()

        elapsed = time.time() - start_time

        if n_uncovered > 0:
            print(f"  ⚠ {n_uncovered} blocks have no initial coverage")

        print(f"  Variables: {self.model.NumVars:,}")
        print(f"  Constraints: {self.model.NumConstrs:,}")
        print(f"  Build time: {elapsed:.2f}s")

    def solve(self) -> Tuple[float, Dict[Tuple[str, int], float]]:
        """Solve the RMP and return objective value and dual values."""
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"RMP did not solve to optimality. Status={self.model.Status}")

        objective_value = self.model.ObjVal

        dual_values = {}
        for block, constraint in self.cover_constraints.items():
            dual_values[block] = constraint.Pi

        return objective_value, dual_values

    def add_pattern(self, pattern: Pattern) -> bool:
        """
        Add a new pattern as a column in the RMP.

        Returns True if the pattern was added, False if it was a duplicate.
        """
        fingerprint = pattern.fingerprint

        if fingerprint in self.fingerprints:
            return False

        column = gp.Column()
        for block in pattern.blocks:
            if block in self.cover_constraints:
                column.addTerms(1.0, self.cover_constraints[block])

        var = self.model.addVar(
            lb=0.0,
            ub=GRB.INFINITY,
            obj=pattern.cost,
            vtype=GRB.CONTINUOUS,
            column=column,
            name=f"y_{pattern.pattern_id}",
        )

        self.y_vars[pattern.pattern_id] = var
        self.fingerprints.add(fingerprint)
        self.patterns.append(pattern)
        self.model.update()

        return True

    def get_lp_solution(self) -> Dict[int, float]:
        """Return the positive LP variable values: {pattern_id: y_p*}."""
        solution = {}

        for pattern_id, var in self.y_vars.items():
            value = var.X
            if value > 1e-8:
                solution[pattern_id] = value

        return solution


# ============================================================================
# STEP 3: PRICING SUBPROBLEM
# ============================================================================

def solve_pricing_for_bin_type(
    bin_type: str,
    dual_values: Dict[Tuple[str, int], float],
    data: Dict,
    config: ColumnGenerationConfig,
    next_pattern_id: int,
    pool_size: int = 1,
) -> List[Pattern]:
    """
    Solve the pricing subproblem for a fixed bin type.

    max  sum_{block} pi_block * p_block - V_t

    Subject to:
         sum_s omega_s <= m
         omega_s >= p_{s,k}                     for every block (s,k)
         sum_k p_{s,k} <= 1                     for every SKU s
         sum_{block} X*_{block,t} p_block <= L_t
         p_block, omega_s in {0,1}

    Optional pre-filtering:
        if config.pricing_max_blocks > 0, only the top-K compatible blocks
        by dual value are included in the subproblem.
    """
    bin_length = data["bin_length"][bin_type]
    bin_volume = data["bin_volume"][bin_type]
    max_skus_per_bin = config.max_skus_per_bin

    compatible_blocks = data["blocks_by_type"].get(bin_type, [])
    if not compatible_blocks:
        return []

    # --- Optional pre-filtering by dual value ---
    if (
        config.pricing_max_blocks > 0
        and len(compatible_blocks) > config.pricing_max_blocks
    ):
        scored_blocks = [
            (block, dual_values.get(block, 0.0))
            for block in compatible_blocks
        ]
        scored_blocks.sort(key=lambda item: -item[1])
        compatible_blocks = [
            block
            for block, _ in scored_blocks[:config.pricing_max_blocks]
        ]

    # SKUs and blocks available for this bin type
    skus_in_type = set()
    blocks_by_sku_in_type = defaultdict(list)

    for block in compatible_blocks:
        sku, _ = block
        skus_in_type.add(sku)
        blocks_by_sku_in_type[sku].append(block)

    sku_list = sorted(skus_in_type)

    # --- Quick upper-bound check ---
    max_dual_value = max(dual_values.get(block, 0.0) for block in compatible_blocks)
    if max_dual_value <= 0:
        return []

    best_dual_by_sku = {}
    for sku in sku_list:
        best_dual_by_sku[sku] = max(
            dual_values.get(block, 0.0)
            for block in blocks_by_sku_in_type[sku]
        )

    top_m_duals = sorted(best_dual_by_sku.values(), reverse=True)[:max_skus_per_bin]

    if sum(top_m_duals) <= bin_volume + config.epsilon:
        return []

    # --- Build pricing model ---
    pricing_model = gp.Model(f"SP_{bin_type}")
    pricing_model.Params.OutputFlag = 0
    pricing_model.Params.TimeLimit = config.pricing_time_limit
    pricing_model.Params.MIPGap = config.pricing_mip_gap
    # pricing_model.Params.Threads = 1  # each bin type solved sequentially; avoid thread explosion"""

    if pool_size > 1:
        pricing_model.Params.PoolSolutions = pool_size
        pricing_model.Params.PoolSearchMode = 2

    # Block-selection variables p_{s,k}
    block_vars = {}
    for block in compatible_blocks:
        block_vars[block] = pricing_model.addVar(
            vtype=GRB.BINARY,
            obj=dual_values.get(block, 0.0),
            name=f"p_{block[0]}_{block[1]}",
        )

    # SKU-presence variables omega_s
    sku_presence_vars = {}
    for sku in sku_list:
        sku_presence_vars[sku] = pricing_model.addVar(
            vtype=GRB.BINARY,
            obj=0.0,
            name=f"omega_{sku}",
        )

    pricing_model.ModelSense = GRB.MAXIMIZE
    pricing_model.update()

    # --- Constraints ---

    # 1. Maximum number of SKUs per bin
    pricing_model.addConstr(
        gp.quicksum(sku_presence_vars[sku] for sku in sku_list)
        <= max_skus_per_bin,
        name="max_skus",
    )

    # 2. SKU presence: omega_s >= p_{s,k}
    for sku in sku_list:
        for block in blocks_by_sku_in_type[sku]:
            pricing_model.addConstr(
                sku_presence_vars[sku] >= block_vars[block],
                name=f"presence_{block[0]}_{block[1]}",
            )

    # 3. At most one block per SKU
    for sku in sku_list:
        sku_blocks = blocks_by_sku_in_type[sku]
        if len(sku_blocks) > 1:
            pricing_model.addConstr(
                gp.quicksum(block_vars[block] for block in sku_blocks) <= 1,
                name=f"unique_block_{sku}",
            )

    # 4. Capacity constraint using X*
    pricing_model.addConstr(
        gp.quicksum(
            data["x_star"][(block, bin_type)] * block_vars[block]
            for block in compatible_blocks
        )
        <= bin_length,
        name="capacity_x",
    )

    # --- Solve ---
    pricing_model.optimize()

    if pricing_model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        pricing_model.dispose()
        return []

    if pricing_model.SolCount == 0:
        pricing_model.dispose()
        return []

    # --- Extract pool solutions ---
    new_patterns = []
    n_solutions = min(pricing_model.SolCount, pool_size)

    for solution_index in range(n_solutions):
        pricing_model.Params.SolutionNumber = solution_index
        objective_value = pricing_model.PoolObjVal

        reduced_cost = bin_volume - objective_value

        if reduced_cost >= -config.epsilon:
            continue

        selected_blocks = []
        total_x = 0.0

        for block in compatible_blocks:
            if block_vars[block].Xn > 0.5:
                selected_blocks.append(block)
                total_x += data["x_star"][(block, bin_type)]

        if not selected_blocks:
            continue

        pattern = Pattern(
            pattern_id=next_pattern_id,
            bin_type_id=bin_type,
            blocks=selected_blocks,
            cost=bin_volume,
            total_x=total_x,
        )

        new_patterns.append(pattern)
        next_pattern_id += 1

    pricing_model.dispose()

    return new_patterns


# ============================================================================
# STEP 4: MAIN COLUMN GENERATION LOOP
# ============================================================================

def run_column_generation(
    data: Dict,
    config: ColumnGenerationConfig,
) -> ColumnGenerationResult:
    """
    Main column generation algorithm.

    1. Generate initial patterns
    2. Build Restricted Master Problem
    3. Iterate:
       - Solve RMP
       - Get dual values
       - Solve pricing subproblem
       - Add improving columns
    4. Return lower bound and generated patterns
    """
    start_time = time.time()

    print("\n" + "=" * 70)
    print("COLUMN GENERATION")
    print("=" * 70)

    print("\n  Configuration:")
    print(f"    max_skus_per_bin = {config.max_skus_per_bin}")
    print(f"    pricing_strategy = {config.pricing_strategy}")
    print(f"    initialization_strategy = {config.initialization_strategy}")
    print(f"    epsilon = {config.epsilon}")
    print(f"    max_iterations = {config.max_iterations}")

    if config.pricing_max_blocks > 0:
        print(f"    pricing_max_blocks = {config.pricing_max_blocks}")

    if config.pricing_strategy != "per_type":
        raise NotImplementedError(
            f"Pricing strategy not implemented: {config.pricing_strategy}"
        )

    # ======================================================================
    # STEP 1: Initial patterns
    # ======================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Generating initial patterns")
    print("-" * 70)

    if config.initialization_strategy == "singleton":
        initial_patterns = generate_singleton_patterns(data)

    elif config.initialization_strategy == "bfd":
        bfd_df = load_bfd_solution(
            config.dataset_id,
            base_dir=config.results_dir,
        )
        initial_patterns = generate_patterns_from_bfd(bfd_df, data)
        initial_patterns = ensure_initial_coverage(initial_patterns, data)

        del bfd_df
        gc.collect()

    else:
        raise ValueError(
            f"Initialization strategy not implemented: "
            f"{config.initialization_strategy}"
        )

    if not initial_patterns:
        raise RuntimeError("No initial patterns were generated. RMP cannot be built.")

    next_pattern_id = max(pattern.pattern_id for pattern in initial_patterns) + 1

    # Upper bound: cost of the initial integer-like solution
    upper_bound = sum(pattern.cost for pattern in initial_patterns)
    print(f"  Initial upper bound: {upper_bound:,.0f} cm³")

    # ======================================================================
    # STEP 2: Build RMP
    # ======================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Building Restricted Master Problem")
    print("-" * 70)

    rmp = RestrictedMasterProblem(data, config)
    rmp.build(initial_patterns)

    # ======================================================================
    # STEP 3: Column generation loop
    # ======================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Column generation loop")
    print("-" * 70)

    history = {
        "rmp_objective": [],
        "n_new_patterns": [],
        "best_reduced_cost": [],
        "iteration_time": [],
    }

    total_patterns_added = 0
    stall_count = 0
    previous_objective = float("inf")
    converged = False
    iteration = 0

    for iteration in range(1, config.max_iterations + 1):
        iteration_start = time.time()

        # # --- 3a. Solve RMP ---
        # rmp_objective, dual_values = rmp.solve()
        
        # --- 3a. Solve RMP ---
        try:
            rmp_objective, dual_values = rmp.solve()
        except RuntimeError as e:
            print(f"\\n  ✗ RMP solve failed at iteration {iteration}: {e}")
            print(f"    Gurobi status: {rmp.model.Status}")
            if rmp.model.Status == 12:  # NUMERIC
                print("    Retrying with NumericFocus=3...")
                rmp.model.Params.NumericFocus = 3
                rmp.model.reset()
                try:
                    rmp_objective, dual_values = rmp.solve()
                    print("    Retry succeeded.")
                except RuntimeError as e2:
                    print(f"    Retry also failed: {e2}")
                    break
            else:
                break

        history["rmp_objective"].append(rmp_objective)

        # Detect stalling
        if abs(rmp_objective - previous_objective) < 1e-4:
            stall_count += 1
        else:
            stall_count = 0

        previous_objective = rmp_objective

        # --- 3b. Solve pricing subproblems ---
        new_patterns = []

        for bin_type in data["bin_types"]:
            patterns_for_type = solve_pricing_for_bin_type(
                bin_type=bin_type,
                dual_values=dual_values,
                data=data,
                config=config,
                next_pattern_id=next_pattern_id,
                pool_size=config.pricing_pool_size,
            )
            new_patterns.extend(patterns_for_type)
            next_pattern_id += len(patterns_for_type)

        # --- 3c. Sort by reduced cost ---
        for pattern in new_patterns:
            pattern.reduced_cost = (
                pattern.cost
                - sum(dual_values.get(block, 0.0) for block in pattern.blocks)
            )

        new_patterns.sort(key=lambda pattern: pattern.reduced_cost)

        # Limit number of patterns added in this iteration
        new_patterns = new_patterns[:config.max_patterns_per_iteration]

        # --- 3d. Add patterns to RMP ---
        n_added = 0
        for pattern in new_patterns:
            if rmp.add_pattern(pattern):
                n_added += 1

        total_patterns_added += n_added

        best_reduced_cost = new_patterns[0].reduced_cost if new_patterns else 0.0

        history["n_new_patterns"].append(n_added)
        history["best_reduced_cost"].append(best_reduced_cost)

        iteration_time = time.time() - iteration_start
        history["iteration_time"].append(iteration_time)

        # --- 3e. Logging ---
        elapsed = time.time() - start_time

        if (
            config.verbose >= 2
            or (config.verbose >= 1 and iteration % config.log_interval == 0)
        ):
            print(
                f"  Iter {iteration:4d} | "
                f"z_RMP = {rmp_objective:16,.1f} | "
                f"+{n_added:3d} cols | "
                f"RC* = {best_reduced_cost:12,.2f} | "
                f"t_iter = {iteration_time:.2f}s | "
                f"T = {elapsed:.0f}s"
            )

        # --- 3f. Convergence ---
        if n_added == 0 or best_reduced_cost >= -config.epsilon:
            print(f"\n  ✓ Convergence reached at iteration {iteration}")
            print(f"    z*_LP = {rmp_objective:,.2f}")
            print(f"    Best reduced cost = {best_reduced_cost:.6f}")
            print(f"    Total CG loop time: {elapsed:.1f}s")
            converged = True
            break

        if stall_count >= 20 and config.verbose >= 1:
            print(f"    ⚠ Stalling detected: {stall_count} iterations without improvement")

    else:
        print(f"\n  ⚠ Maximum iterations reached: {config.max_iterations}")

    # ======================================================================
    # STEP 4: Extract final results
    # ======================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Extracting results")
    print("-" * 70)

    lower_bound, final_dual_values = rmp.solve()
    lp_solution = rmp.get_lp_solution()

    gap_pct = (
        100.0 * (upper_bound - lower_bound) / lower_bound
        if lower_bound > 0
        else float("inf")
    )

    runtime = time.time() - start_time

    print(f"  Lower bound (LP):       {lower_bound:>20,.2f} cm³")
    print(f"  Upper bound (initial):  {upper_bound:>20,.2f} cm³")
    print(f"  Gap:                    {gap_pct:>20.2f}%")
    print(f"  Total patterns:         {len(rmp.patterns):>20,}")
    print(f"  Patterns in solution:   {len(lp_solution):>20,}")
    print(f"  CG columns added:       {total_patterns_added:>20,}")
    print(f"  Iterations:             {iteration:>20,}")
    print(f"  Total runtime:          {runtime:>20.2f}s")

    n_fractional = sum(
        1
        for value in lp_solution.values()
        if 0.001 < value < 0.999
    )
    n_integer_like = sum(
        1
        for value in lp_solution.values()
        if value > 0.999
    )

    print(f"  Integer-like variables: {n_integer_like:>20,}")
    print(f"  Fractional variables:   {n_fractional:>20,}")

    if history["iteration_time"]:
        avg_iteration_time = np.mean(history["iteration_time"])
        max_iteration_time = np.max(history["iteration_time"])

        print(f"  Avg iteration time:     {avg_iteration_time:>20.2f}s")
        print(f"  Max iteration time:     {max_iteration_time:>20.2f}s")

    result = ColumnGenerationResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gap_pct=gap_pct,
        n_iterations=iteration,
        n_patterns=len(rmp.patterns),
        runtime=runtime,
        patterns=rmp.patterns,
        dual_values=final_dual_values,
        lp_solution=lp_solution,
        converged=converged,
    )

    return result


# ============================================================================
# RESULT ANALYSIS AND EXPORT
# ============================================================================

def analyze_column_generation_result(
    result: ColumnGenerationResult,
    data: Dict,
):
    """Print detailed statistics for the column generation result."""
    print("\n" + "=" * 70)
    print("RESULT ANALYSIS")
    print("=" * 70)

    total_item_volume = data["total_item_volume"]

    print(f"\n  Total item volume:       {total_item_volume:>20,.2f} cm³")
    print(f"  Lower bound (LP):        {result.lower_bound:>20,.2f} cm³")

    theoretical_efficiency = (
        100.0 * total_item_volume / result.lower_bound
        if result.lower_bound > 0
        else 0.0
    )

    print(f"  Theoretical LP efficiency: {theoretical_efficiency:>17.2f}%")

    # Distribution of patterns in the LP solution
    type_counts = defaultdict(float)
    type_volume = defaultdict(float)

    pattern_by_id = {
        pattern.pattern_id: pattern
        for pattern in result.patterns
    }

    for pattern_id, y_value in result.lp_solution.items():
        pattern = pattern_by_id.get(pattern_id)

        if pattern is not None:
            type_counts[pattern.bin_type_id] += y_value
            type_volume[pattern.bin_type_id] += y_value * pattern.cost

    if type_counts:
        total_lp_bins = sum(type_counts.values())

        print(
            f"\n  Pattern distribution in LP solution "
            f"(total ≈ {total_lp_bins:,.1f} bins):"
        )

        sorted_types = sorted(
            type_counts.items(),
            key=lambda item: -item[1],
        )

        n_show = min(15, len(sorted_types))

        for bin_type, count in sorted_types[:n_show]:
            volume = data["bin_volume"][bin_type]
            print(f"    {bin_type}: {count:10.3f} bins (volume = {volume:,.0f} cm³)")

        if len(sorted_types) > n_show:
            remaining_count = sum(
                count
                for _, count in sorted_types[n_show:]
            )
            print(
                f"    ... + {len(sorted_types) - n_show} more types "
                f"({remaining_count:.1f} bins)"
            )

    # Dual-value statistics
    dual_values = list(result.dual_values.values())

    if dual_values:
        print("\n  Dual value statistics π*:")
        print(f"    min  = {min(dual_values):>12.2f}")
        print(f"    max  = {max(dual_values):>12.2f}")
        print(f"    mean = {np.mean(dual_values):>12.2f}")
        print(f"    std  = {np.std(dual_values):>12.2f}")
        print(f"    p10  = {np.percentile(dual_values, 10):>12.2f}")
        print(f"    p90  = {np.percentile(dual_values, 90):>12.2f}")


def save_column_generation_results(
    result: ColumnGenerationResult,
    data: Dict,
    config: ColumnGenerationConfig,
):
    """Save column generation results to disk."""
    if config.results_dir is not None:
        output_dir = config.results_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(repo_dir, "results", config.dataset_id)

    os.makedirs(output_dir, exist_ok=True)

    pattern_by_id = {
        pattern.pattern_id: pattern
        for pattern in result.patterns
    }

    # --- 1. LP solution patterns ---
    solution_rows = []

    for pattern_id, y_value in result.lp_solution.items():
        pattern = pattern_by_id.get(pattern_id)

        if pattern is None:
            continue

        solution_rows.append({
            "pattern_id": pattern_id,
            "bin_type_id": pattern.bin_type_id,
            "cost": pattern.cost,
            "n_blocks": len(pattern.blocks),
            "n_skus": len(pattern.skus),
            "total_x": pattern.total_x,
            "y_value": y_value,
            "blocks": ";".join(f"{sku}_{block_k}" for sku, block_k in pattern.blocks),
        })

    solution_df = pd.DataFrame(solution_rows)
    solution_path = os.path.join(
        output_dir,
        f"{config.dataset_id}_cg_solution.csv",
    )
    solution_df.to_csv(solution_path, index=False)
    print(f"  ✓ LP solution: {solution_path}")

    # --- 2. Summary ---
    summary = {
        "dataset_id": config.dataset_id,
        "n_skus": data["n_skus"],
        "n_blocks": data["n_blocks"],
        "n_bin_types": data["n_bin_types"],
        "max_skus_per_bin": config.max_skus_per_bin,
        "initialization_strategy": config.initialization_strategy,
        "pricing_strategy": config.pricing_strategy,
        "lower_bound_lp": result.lower_bound,
        "upper_bound_initial": result.upper_bound,
        "gap_pct": result.gap_pct,
        "n_iterations": result.n_iterations,
        "n_patterns_total": result.n_patterns,
        "n_patterns_in_solution": len(result.lp_solution),
        "runtime_s": result.runtime,
        "total_item_volume": data["total_item_volume"],
        "lp_efficiency": (
            100.0 * data["total_item_volume"] / result.lower_bound
            if result.lower_bound > 0
            else 0.0
        ),
        "converged": result.converged,
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(
        output_dir,
        f"{config.dataset_id}_cg_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Summary: {summary_path}")

    # --- 3. Dual values ---
    dual_rows = []

    for (sku, block_k), dual_value in result.dual_values.items():
        dual_rows.append({
            "sku_id": sku,
            "block_k": block_k,
            "pi": dual_value,
            "block_id": data["tuple_to_block_id"].get((sku, block_k), f"{sku}_{block_k}"),
        })

    duals_df = pd.DataFrame(dual_rows)
    duals_path = os.path.join(
        output_dir,
        f"{config.dataset_id}_cg_duals.csv",
    )
    duals_df.to_csv(duals_path, index=False)
    print(f"  ✓ Duals: {duals_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(
    dataset_id: str,
    max_skus_per_bin: int = 3,
    max_skus: Optional[int] = None,
    selected_skus: Optional[List[str]] = None,
    pricing_strategy: str = "per_type",
    initialization_strategy: str = "singleton",
    verbose: int = 1,
    pricing_time_limit: float = 60.0,
    max_iterations: int = 5000,
    pricing_max_blocks: int = 0,
    log_interval: int = 5,
    results_dir: Optional[str] = None,
):
    """Main entry point."""
    if not GUROBI_AVAILABLE:
        print("ERROR: Gurobi is not available")
        return None

    execution_start_time = datetime.now()

    print("=" * 70)
    print(f"COLUMN GENERATION — {dataset_id.upper()}")
    print("=" * 70)

    config = ColumnGenerationConfig(
        dataset_id=dataset_id,
        max_skus_per_bin=max_skus_per_bin,
        max_skus=max_skus,
        selected_skus=selected_skus,
        pricing_strategy=pricing_strategy,
        initialization_strategy=initialization_strategy,
        verbose=verbose,
        pricing_time_limit=pricing_time_limit,
        max_iterations=max_iterations,
        pricing_max_blocks=pricing_max_blocks,
        log_interval=log_interval,
        results_dir=results_dir,
    )

    # --- Load data ---
    print("\n" + "-" * 70)
    print("Loading data")
    print("-" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(repo_dir, "final_data", dataset_id)

    blocks_df, bin_types_df, x_optimal_df = load_model_data(dataset_id, data_dir)

    print(f"  Blocks: {len(blocks_df):,}")
    print(f"  SKUs: {blocks_df['sku_id'].nunique():,}")
    print(f"  Feasible pairs: {len(x_optimal_df):,}")

    # --- Filter SKUs if requested ---
    if selected_skus is not None:
        available_skus = set(blocks_df["sku_id"].unique())
        selected_skus = [
            sku
            for sku in selected_skus
            if sku in available_skus
        ]

        blocks_df = blocks_df[blocks_df["sku_id"].isin(selected_skus)]
        block_ids = set(blocks_df["block_id"].unique())
        x_optimal_df = x_optimal_df[x_optimal_df["block_id"].isin(block_ids)]

        print(
            f"  → Filtered to {len(selected_skus)} SKUs: "
            f"{len(blocks_df):,} blocks"
        )

    elif max_skus is not None:
        selected_sku_values = blocks_df["sku_id"].unique()[:max_skus]

        blocks_df = blocks_df[blocks_df["sku_id"].isin(selected_sku_values)]
        block_ids = set(blocks_df["block_id"].unique())
        x_optimal_df = x_optimal_df[x_optimal_df["block_id"].isin(block_ids)]

        print(
            f"  → Filtered to {max_skus} SKUs: "
            f"{len(blocks_df):,} blocks"
        )

    # --- Prepare data structures ---
    print("\n" + "-" * 70)
    print("Preparing data structures")
    print("-" * 70)

    data = prepare_column_generation_data(
        blocks_df,
        bin_types_df,
        x_optimal_df,
    )

    del blocks_df, x_optimal_df
    gc.collect()

    print(f"  SKUs: {data['n_skus']:,}")
    print(f"  Blocks: {data['n_blocks']:,}")
    print(f"  Bin types: {data['n_bin_types']}")
    print(f"  Feasible pairs: {len(data['feasible_pairs']):,}")
    print(f"  Total item volume: {data['total_item_volume']:,.0f} cm³")

    blocks_per_type_sizes = [
        len(blocks)
        for blocks in data["blocks_by_type"].values()
    ]

    if blocks_per_type_sizes:
        print(
            f"  Blocks/type: "
            f"min={min(blocks_per_type_sizes):,} "
            f"max={max(blocks_per_type_sizes):,} "
            f"mean={np.mean(blocks_per_type_sizes):,.0f}"
        )

    # --- Run column generation ---
    result = run_column_generation(data, config)

    # --- Analyze results ---
    analyze_column_generation_result(result, data)

    # --- Save results ---
    print("\n" + "-" * 70)
    print("Saving results")
    print("-" * 70)

    save_column_generation_results(result, data, config)

    # --- Final summary ---
    total_execution_time = datetime.now() - execution_start_time

    print("\n" + "=" * 70)
    print("EXECUTION COMPLETED")
    print("=" * 70)
    print(f"  Total time: {total_execution_time}")
    print(f"  LP lower bound: {result.lower_bound:,.2f} cm³")
    print(
        f"  Upper bound ({config.initialization_strategy}): "
        f"{result.upper_bound:,.2f} cm³"
    )
    print(f"  Gap: {result.gap_pct:.2f}%")

    lp_efficiency = (
        100.0 * data["total_item_volume"] / result.lower_bound
        if result.lower_bound > 0
        else 0.0
    )

    print(f"  LP efficiency: {lp_efficiency:.2f}%")
    print("=" * 70)

    return result


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # --- Default configuration ---
    DEFAULT_DATASET = "DS-1"
    DEFAULT_MAX_SKUS_PER_BIN = 4
    DEFAULT_MAX_SKUS = None
    DEFAULT_PRICING_STRATEGY = "per_type"
    DEFAULT_INITIALIZATION_STRATEGY = "bfd"
    DEFAULT_VERBOSE = 1

    # Specific SKUs for testing, same list used in build_model.py
    # SELECTED_SKUS = [
    #     "IN1122", "IN1149", "IN1111", "IN1026", "IN398", "IN395", "IN1309",
    #     "IN1150", "IN1171", "IN746", "IN130", "IN370", "IN384", "IN693",
    #     "IN617", "IN397", "IN114", "IN139", "IN133", "IN125", "IN140",
    #     "IN737", "IN597", "IN587", "IN392", "IN42", "IN425", "IN47",
    #     "IN430", "IN373", "IN431", "IN97", "IN394", "IN400", "IN109",
    #     "IN428", "IN1061", "IN1419", "IN590",
    # ]
    SELECTED_SKUS = [
        "1122", "1149", "1111"
    ]
    SELECTED_SKUS = list(set(SELECTED_SKUS))

    import argparse

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Column generation for bin sizing"
        )

        parser.add_argument("dataset_id", type=str)

        parser.add_argument(
            "--m",
            type=int,
            default=4,
            help="Maximum number of SKUs per bin (paper: m=4)",
        )

        parser.add_argument(
            "--max_skus",
            type=int,
            default=None,
            help="Maximum number of SKUs to include",
        )

        parser.add_argument(
            "--skus",
            type=str,
            default=None,
            help="Comma-separated list of SKUs",
        )

        parser.add_argument(
            "--pricing",
            type=str,
            default="per_type",
            choices=["per_type"],
            help="Pricing strategy",
        )

        parser.add_argument(
            "--init",
            type=str,
            default="bfd",
            choices=["singleton", "bfd"],
            help="Initialization strategy",
        )

        parser.add_argument(
            "--verbose",
            type=int,
            default=1,
            choices=[0, 1, 2],
            help="Verbosity level",
        )

        parser.add_argument(
            "--sp_time",
            type=float,
            default=60.0,
            help="Time limit per pricing subproblem, in seconds",
        )

        parser.add_argument(
            "--max_iter",
            type=int,
            default=5000,
            help="Maximum number of column generation iterations",
        )

        parser.add_argument(
            "--sp_max_blocks",
            type=int,
            default=0,
            help="Top-K blocks per pricing subproblem; 0 means no limit",
        )

        parser.add_argument(
            "--log_interval",
            type=int,
            default=5,
            help="Logging interval, in iterations",
        )

        args = parser.parse_args()

        selected_skus_from_args = (
            [sku.strip() for sku in args.skus.split(",")]
            if args.skus
            else None
        )

        result = main(
            dataset_id=args.dataset_id,
            max_skus_per_bin=args.m,
            max_skus=args.max_skus,
            selected_skus=selected_skus_from_args,
            pricing_strategy=args.pricing,
            initialization_strategy=args.init,
            verbose=args.verbose,
            pricing_time_limit=args.sp_time,
            max_iterations=args.max_iter,
            pricing_max_blocks=args.sp_max_blocks,
            log_interval=args.log_interval,
        )

    else:
        # Default execution: full DS-1 dataset with BFD warm start
        result = main(
            dataset_id=DEFAULT_DATASET,
            max_skus_per_bin=DEFAULT_MAX_SKUS_PER_BIN,
            max_skus=DEFAULT_MAX_SKUS,
            selected_skus=None,
            pricing_strategy=DEFAULT_PRICING_STRATEGY,
            initialization_strategy=DEFAULT_INITIALIZATION_STRATEGY,
            verbose=DEFAULT_VERBOSE,
            log_interval=5,
        )