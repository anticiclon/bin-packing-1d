# Bin Dimensioning in E-Commerce Fulfillment Centers

Code and datasets for the paper:

> **Bin Dimensioning in E-Commerce Fulfillment Centers**  
> Gabriel González, Mariá C. V. Nascimento  
> Instituto Tecnológico de Aeronáutica (ITA), Brazil

---

## Overview

This repository contains the full implementation of the BFD + Column Generation + Restricted IP pipeline for optimising bin dimensions in e-commerce warehouses, together with the four anonymised Mercado Libre datasets used in the paper.

The pipeline proceeds in four stages:

```
Data preparation  →  BFD heuristic  →  Column Generation  →  Restricted IP
```

Scalability experiments (Table 6–7 and Figures in the paper) are run through a dedicated script that automates the full pipeline across multiple instance sizes and seeds.

---

## Requirements

- Python 3.12
- [Gurobi](https://www.gurobi.com/) 12.0 (academic or commercial licence required)
- Python packages: `pandas`, `numpy`, `pyarrow`, `gurobipy`

Install Python dependencies:

```bash
pip install pandas numpy pyarrow gurobipy
```

---

## Repository structure

```
bin-packing-1d/
├── code/
│   ├── adding_blocks.py          # Stage 1a: aggregate SKU orders into blocks
│   ├── create_configurations.py  # Stage 1b: enumerate feasible bin configurations
│   ├── save_preprocessed_data.py # Stage 1c: save preprocessed data as Parquet
│   ├── bfd_heuristic.py          # Stage 2: Best-Fit-Decreasing heuristic
│   ├── column_generation.py      # Stage 3: Column Generation (LP relaxation)
│   ├── restricted_ip.py          # Stage 4: Restricted Integer Program
│   ├── run_scalability.py        # Scalability experiments (all stages)
│   ├── create_tables.py          # Generate tables and figures for the paper
│   └── build_model.py            # Shared MILP building utilities
├── data/
│   ├── DS-1_skus.csv             # Dataset 1 (1,556 SKUs)
│   ├── DS-2_skus.csv             # Dataset 2 (94,556 SKUs)
│   ├── DS-3_skus.csv             # Dataset 3 (503,931 SKUs)
│   └── DS-4_skus.csv             # Dataset 4 (1,520,442 SKUs)
├── final_data/                   # Preprocessed Parquet files (generated)
├── results/                      # Outputs (generated)
└── README.md
```

---

## Datasets

The four datasets are anonymised extracts from Mercado Libre's warehouse operations.
Each `DS-X_skus.csv` file contains one row per SKU with dimensions (length, width, height in cm) and demand.

| Dataset | SKUs      | Blocks    | Country |
|---------|-----------|-----------|---------|
| DS-1    | 1,556     | 11,830    | —       |
| DS-2    | 94,556    | 198,640   | —       |
| DS-3    | 503,931   | 553,818   | —       |
| DS-4    | 1,520,442 | 1,725,931 | —       |

---

## Reproducing the paper results

### Step 1 — Data preparation

Run once per dataset. Generates `data/DS-X_blocks.csv`, `data/DS-X_configurations.csv`, and `final_data/DS-X/*.parquet`.

```bash
cd code

for ds in DS-1 DS-2 DS-3 DS-4; do
    python adding_blocks.py ../data/${ds}_skus.csv
    python create_configurations.py ../data/${ds}_blocks.csv
    python save_preprocessed_data.py ${ds}
done
```

> **Note:** The `.db` files produced by `save_preprocessed_data.py` can be safely deleted after this step to save disk space. Only the `.parquet` files are needed downstream.

### Step 2 — Scalability experiments

Runs BFD + CG + RIP for every instance size and seed defined in the paper, writing results to `results/scalability_results_{DS-X}.csv`.

```bash
python run_scalability.py --datasets DS-1 DS-2 DS-3 DS-4
```

**CLI options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--datasets` | Datasets to run (`DS-1` … `DS-4`) | all |
| `--sizes` | Instance sizes (number of SKUs) | all |
| `--seeds` | Random seeds | all |

Example — run only DS-1, sizes 50 and 100, seeds 0 and 1:

```bash
python run_scalability.py --datasets DS-1 --sizes 50 100 --seeds 0 1
```

The script **resumes automatically** from the last completed row if interrupted: simply re-run the same command.

> **On HPC clusters (PBS/Torque):** make sure the Gurobi licence environment variable is exported in your job script:
> ```bash
> export GRB_LICENSE_FILE=/path/to/gurobi.lic
> ```

### Step 3 — Generate tables and figures

Once all scalability CSVs are available, run:

```bash
python create_tables.py
```

This produces the `.tex` tables and `.pdf` figures referenced in the paper under `results/`.

---

## Running individual pipeline stages

### BFD heuristic

```bash
python bfd_heuristic.py DS-1
python bfd_heuristic.py DS-1 --m 4 --theta 0.05 --fit residual
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--m` | Maximum SKUs per bin | 4 |
| `--theta` | Bin-closing threshold | 0.05 |
| `--fit` | Fit metric: `residual`, `volume`, `composite` | `residual` |

Outputs: `results/DS-X/DS-X_bfd_solution.csv`, `_bfd_bins.csv`, `_bfd_report.txt`

### Column Generation

```bash
python column_generation.py DS-1
python column_generation.py DS-1 --init bfd --max_iter 5000
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--init` | Initialisation: `bfd` or `singleton` | `bfd` |
| `--pricing` | Pricing strategy: `per_type` or `combined` | `per_type` |
| `--max_iter` | Maximum CG iterations | 5000 |
| `--sp_time` | Time limit per pricing subproblem (s) | 60 |

Outputs: `results/DS-X/DS-X_cg_solution.csv`, `_cg_summary.csv`, `_cg_duals.csv`

### Restricted Integer Program

Requires Column Generation to have been run first.

```bash
python restricted_ip.py DS-1
python restricted_ip.py DS-1 --time_limit 3600 --mip_gap 0.001
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--time_limit` | Gurobi time limit (s) | 3600 |
| `--mip_gap` | MIP optimality gap | 0.001 |

Outputs: `results/DS-X/DS-X_rip_solution.csv`, `_rip_summary.csv`

---

## Computational environment

All experiments in the paper were run on the [CeMEAI/ICMC-USP Euler cluster](http://euler.cemeai.icmc.usp.br) using:

- CPU: Intel Xeon E5-2650 v2 @ 2.60 GHz
- Python 3.12.8
- Gurobi 12.0.3

---

## Licence

MIT