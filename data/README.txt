======================================================================
BIN DIMENSIONING IN E-COMMERCE FULFILLMENT CENTERS
Datasets for reproducibility
======================================================================

CONTENTS
----------------------------------------------------------------------
For each dataset DS-X (X = 1, 2, 3, 4):

  DS-X_skus.csv
      SKU-level input data. Columns:
        sku_id       — anonymous integer identifier (1-based)
        length       — item dimension, cm (front-to-back / depth)
        width        — item dimension, cm (left-right / width)
        height       — item dimension, cm (vertical / height)
        total_items  — total units of this SKU in the inventory
        n_s          — maximum units of this SKU allowed per bin
        rotatable    — whether the item may be rotated (True/False)

  DS-X_bin_catalog.csv
      Available bin types. Columns:
        L_t          — bin size along axis X (largura / width),      cm
        W_t          — bin size along axis Y (profundidade / depth), cm
        H_t          — bin size along axis Z (altura / height),      cm
        volume_cm3   — bin volume, cm^3

  AXIS CONVENTION
        Distinct SKUs are placed side by side along axis X (the bin
        width / largura), so that every SKU remains directly accessible
        from the front opening. The capacity constraint bounds the sum
        of their X-extents by L_t:  sum_{(s,k)} X*_{sk,t} <= L_t.
        Units of the SAME SKU may additionally stack along axis Y
        (depth) and axis Z (height). For non-rotatable items, the
        'height' dimension is always kept on axis Z (vertical).

DATASETS
----------------------------------------------------------------------
  DS-1  (large products catalog, 30 bin types)
    SKUs:               1,556
    Items:            308,059
    n_s median:             9
    n_s range:   [1, 384]
    Rotatable:          97.0%

  DS-2  (large products catalog, 30 bin types)
    SKUs:              94,556
    Items:            905,637
    n_s median:             2
    n_s range:   [1, 200]
    Rotatable:          99.8%

  DS-3  (small products catalog, 24 bin types)
    SKUs:             503,931
    Items:          2,218,609
    n_s median:             2
    n_s range:   [1, 250]
    Rotatable:          96.0%

  DS-4  (small products catalog, 24 bin types)
    SKUs:           1,520,442
    Items:          4,034,196
    n_s median:             1
    n_s range:   [1, 250]
    Rotatable:         100.0%

REPRODUCIBILITY
----------------------------------------------------------------------
To reproduce the full pipeline from these files, run in order:

  1. adding_blocks.py        — constructs blocks from SKU data
  2. create_configurations.py — computes X*_{sk,t} for each
                               (block, bin type) pair
  3. bfd_heuristic.py        — Best-Fit-Decreasing heuristic
  4. column_generation.py    — column generation / LP lower bound

CONFIDENTIALITY
----------------------------------------------------------------------
Original SKU identifiers have been replaced by sequential integers.
No geographic, temporal, or product-category information is included.
Raw data cannot be provided due to confidentiality agreements with
Mercado Libre.

======================================================================