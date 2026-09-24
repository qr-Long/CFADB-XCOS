# CFADB-XCOS: Automated Cross-database Mapping of Atomic Energy-Level Data

CFADB-XCOS is the research software and standardized atomic energy-level archive accompanying the paper **“Automated Cross-database Mapping of Atomic Energy-Level Data for Astrophysical Plasma Models.”** It identifies corresponding atomic energy levels across AtomDB v3.1.3, CHIANTI v11.0.2, and SPEX v3.08.01 using a combined similarity score, referred to here as XCOS, based on electronic configuration, term quantum numbers and parity, and energy information, followed by threshold-constrained Hungarian matching. Standardized SPEX/SPEXACT v3.08.03 tables are also retained for supplementary comparison, but the principal analysis and default sequence-wide scan use SPEX v3.08.01.

The current review archive is available on [Zenodo Sandbox](https://sandbox.zenodo.org/records/574242). This is a temporary record; the production Zenodo record and DOI will be added when the formal release is published.

> **Start here.** Open the latest hosted Viewer through [Binder/Voilà](https://mybinder.org/v2/gh/qr-Long/CFADB-XCOS/main?urlpath=voila%2Frender%2Fviewer%2Fthr_viewer.ipynb), or run it locally from the repository root:
>
> ```bash
> voila ./viewer/thr_viewer.ipynb --show_tracebacks=True
> ```
>
> The browser interface lets users select the sequence, ion, database pair, and threshold grid without editing notebook code. The hosted link follows the current GitHub `main` branch; use the archived files and the local command above when reproducing a specific Zenodo release.

## Contents

- [Repository tree](#repository-tree)
- [1. Overview](#1-overview)
- [2. Quick Start](#2-quick-start)
  - [2.1 Install the dependencies and run from the repository root](#21-install-the-dependencies-and-run-from-the-repository-root)
  - [2.2 Select the Viewer parameters](#22-select-the-viewer-parameters)
  - [2.3 Main interface functions](#23-main-interface-functions)
  - [2.4 First example: load, validate, and export a map](#24-first-example-load-validate-and-export-a-map)
  - [2.5 Interpret the display and downloaded results](#25-interpret-the-display-and-downloaded-results)
- [3. Atomic Structure Data](#3-atomic-structure-data)
  - [3.1 Sequence coverage](#31-sequence-coverage)
  - [3.2 Ion-file naming convention](#32-ion-file-naming-convention)
  - [3.3 HDF5 datasets and level-table columns](#33-hdf5-datasets-and-level-table-columns)
- [4. Interactive Matching Viewer](#4-interactive-matching-viewer)
- [5. Configuration Model (`cfg`)](#5-configuration-model-cfg)
  - [`cfg_train_data.py`](#cfg_train_datapy)
  - [`cfg_ts.py`](#cfg_tspy)
  - [Other files in `cfg/`](#other-files-in-cfg)
- [6. Similarity Computation (`cos`)](#6-similarity-computation-cos)
- [7. Threshold-constrained Matching and Validation (`thr_hung`)](#7-threshold-constrained-matching-and-validation-thr_hung)
- [8. Supporting Packages](#8-supporting-packages)
- [9. Cleaning and Exact Level-Table Restoration](#9-cleaning-and-exact-level-table-restoration)
  - [9.1 Stored cleaning information](#91-stored-cleaning-information)
  - [9.2 Row positions and reconstruction](#92-row-positions-and-reconstruction)
  - [9.3 Field values and data types](#93-field-values-and-data-types)
  - [9.4 Empty tables and absent datasets](#94-empty-tables-and-absent-datasets)
  - [9.5 Restoration with the bundled function](#95-restoration-with-the-bundled-function)
- [10. Figure Reproduction and Main Outputs](#10-figure-reproduction-and-main-outputs)
- [11. Software Requirements and Reproducibility Notes](#11-software-requirements-and-reproducibility-notes)
  - [Main dependencies](#main-dependencies)
  - [Opening the other notebooks](#opening-the-other-notebooks)
  - [Common usage issues](#common-usage-issues)
  - [Reproducibility notes](#reproducibility-notes)
- [12. Dataset Identification, Rights, and Contact](#12-dataset-identification-rights-and-contact)
  - [Repository rights](#repository-rights)
  - [Packaged datasets and attribution](#packaged-datasets-and-attribution)
  - [Contact](#contact)

## Repository tree

The archive is organized by scientific function rather than by a single execution order. The principal Viewer uses the archived tables in `clean_data/`, the sequence-specific encoders in `map_data/cfg_ts/`, and supporting modules in `viewer/`, `common/`, `cos/`, `thr_hung/`, `tbl/`, `datacollect/`, `dataclean/`, and `util/`. The tree below lists the main files used or referenced by this guide; package initializer files are omitted for clarity.

```text
cfadb-xcos/
├── README.md                              # English project documentation
├── requirements.txt                      # Python dependency specification
├── runtime.txt                           # Binder Python runtime declaration
├── clean_data/                            # Retained and excluded energy-level records in HDF5
│   ├── 010101.h5                          # First H-like file: H-like hydrogen
│   ├── ...                                # Other H-like ions
│   ├── 013636.h5                          # Last H-like file: H-like krypton
│   ├── 020201.h5                          # First He-like file: He-like helium
│   ├── ...                                # Other He-like ions
│   ├── 023635.h5                          # Last He-like file: He-like krypton
│   ├── 030301.h5                          # First Li-like file: Li-like lithium
│   ├── ...                                # Other Li-like ions
│   ├── 033028.h5                          # Last Li-like file: Li-like zinc
│   ├── 040401.h5                          # First Be-like file: Be-like beryllium
│   ├── ...                                # Other Be-like ions
│   ├── 043027.h5                          # Last Be-like file: Be-like zinc
│   ├── 101001.h5                          # First Ne-like file: Ne-like neon
│   ├── ...                                # Other Ne-like ions
│   └── 103021.h5                          # Last Ne-like file: Ne-like zinc
├── viewer/                                # Interactive and static similarity viewers
│   ├── thr_viewer.ipynb                   # Main Voilà interface for threshold/Hungarian matching
│   ├── threshold_viewer.py                # Loading, similarity, matching, diagnostics, plotting, and downloads
│   └── sim_only_viewer.ipynb              # Total-similarity matrix viewer without assignment
├── cfg/                                   # Configuration-pair generation and encoder training
│   ├── cfg_train_data.py                  # Builds sequence-specific configuration training pairs
│   ├── cfg_ts.py                          # Trains the BiLSTM configuration encoder and draws t-SNE plots
│   ├── cfg_model_plt.ipynb                # Draws the configuration-similarity boxplot
│   └── ion_config.py                      # Shared sequence definitions, filenames, and shortening rules
├── cos/                                   # Direct XCOS similarity computation and sequence-wide scans
│   ├── xcos.py                            # Combines configuration, term/parity, and energy similarities
│   ├── term_score.py                      # Computes the discrete 2S+1/L/J/parity score
│   ├── energy_features.py                 # Builds absolute, band, and within-band energy features
│   └── cos_scan_match.py                  # Scans the publication database pairs and thresholds
├── thr_hung/                              # Threshold-constrained assignment and physical validation
│   ├── hungarian.py                       # Full and threshold-aware Hungarian assignment
│   ├── rules.py                           # Rule matrices, N_theo, and selected-match validation
│   └── threshold_scan.py                  # Threshold-cube construction and assignment-change analysis
├── common/                                # Shared configuration-model and pair-loading utilities
│   ├── cfg_embed.py                       # Loads the encoder and appends configuration embeddings
│   └── data_loader.py                     # Loads database pairs and builds a preliminary identifier-based map
├── datacollect/                           # Data-preparation and HDF5 utilities
│   ├── adb_to_h5.py                       # AtomDB table-preparation code
│   ├── cdb_to_h5.py                       # CHIANTI table-preparation code
│   ├── spex_to_h5.py                      # SPEX table-preparation code
│   └── h5_tools.py                        # HDF5 conversion, reading, inspection, and merge utilities
├── dataclean/                             # Standardizes collected database tables
│   ├── adb_clean.py                       # Cleans AtomDB tables
│   ├── cdb_clean.py                       # Cleans CHIANTI tables
│   ├── spex_clean.py                      # Cleans SPEX tables
│   ├── base_tools.py                      # Shared cleaning and configuration-parity utilities
│   └── seq_list.py                        # Selects files by sequence and ion range
├── tbl/                                   # Standardized table readers and preliminary level alignment
│   ├── show_tbl_lev.py                    # Reads and exports an energy-level table
│   └── show_tbl_cflev.py                  # Aligns identical spectroscopic states for reference diagnostics
├── plot/                                  # Energy-level comparison plots
│   ├── three_energy_views_plot.ipynb      # Draws the three energy-level distribution views
│   └── plot_tbl_cflev.py                  # Configuration grouping and energy plotting functions
├── diagnostics/                           # Detailed matching diagnostics
│   └── dmap_check_try.py                  # Compares predicted and preliminary reference assignments
├── list/                                  # Database-name metadata utility
│   └── list_database.py                   # Normalizes database names and version identifiers
├── util/                                  # General atomic-data utilities
│   ├── ptbl.py                            # Periodic-table and ion-name utilities
│   ├── quantbl.py                         # Angular-momentum and term-symbol utilities
│   └── unit_cnvt.py                       # Energy-unit utilities
└── map_data/                              # Archived models, vocabularies, pairs, and model figures
    └── cfg_ts/
        ├── train_data/                    # H-/He-/Li-/Be-/Ne-like configuration-pair TSV files
        ├── h-like/                        # H-like model and vocabulary
        ├── he-like/                       # He-like model and vocabulary
        ├── li-like/                       # Li-like model and vocabulary
        ├── be-like/                       # Be-like model and vocabulary
        ├── ne-like/                       # Ne-like model and vocabulary
        └── figs/                          # Configuration-model figure outputs
```

Matching-output directories are created as needed; the bundled configuration-model figure is stored separately under `map_data/cfg_ts/figs/`. `map_data/thr_view/` is created when pair data are loaded, `map_data/photos/` is created by `sim_only_viewer.ipynb`, `data/` is created by the standalone table and energy-plot workflows, and `cos_data/` is created by `cos_scan_match.py`.

## 1. Overview

Atomic databases used by plasma codes can differ in energy-level coverage, ordering, energies, configuration notation, and quantum-number assignments. CFADB-XCOS constructs reproducible partial one-to-one correspondences between these level tables while allowing levels without a reliable counterpart to remain unmatched.

For a selected ion and database pair, the principal workflow:

1. loads the standardized energy-level tables and the sequence-specific configuration encoder;
2. calculates configuration, term-quantum-number/parity, and energy similarities;
3. combines them into a total XCOS similarity matrix;
4. applies a similarity threshold and Hungarian assignment with dummy unmatched states;
5. evaluates the selected pairs using configuration, term, parity, and energy rules; and
6. presents the matrices, assignments, diagnostics, and downloadable tables in an interactive interface.

The paper applies this workflow to H-like, He-like, Li-like, Be-like, and Ne-like ions. Its purpose is to construct transparent cross-database energy-level correspondences rather than to determine which atomic database is more accurate.

## 2. Quick Start

### 2.1 Install the dependencies and run from the repository root

For ordinary matching, start with the bundled `clean_data/` files and trained models; no model training or data regeneration is needed. The repository root is the directory containing `README.md`, `requirements.txt`, `clean_data/`, and `viewer/`. Keep the packaged directory structure intact.

Enter the extracted repository directory (its name may differ for a GitHub download), create a Python 3.11 virtual environment, and install the dependencies:

```bash
cd cfadb-xcos
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
voila ./viewer/thr_viewer.ipynb --show_tracebacks=True
```

The commands above use the standard activation syntax for macOS and Linux and require Python 3.11 to be installed. In Windows Command Prompt, create the environment with `py -3.11 -m venv .venv`, activate it with `.venv\Scripts\activate`, and then run the same `python -m pip` and `voila` commands.

Open the local URL printed by Voilà if the browser does not open automatically. Keep that terminal running while using the interface; press `Ctrl+C` there to stop the server. For later sessions, activate the existing environment and run the Voilà command again.

### 2.2 Select the Viewer parameters

The Viewer opens with browser controls rather than an editable parameter cell:

| Control | Meaning |
|---|---|
| `Sequence` | Isoelectronic sequence and corresponding configuration model: `h-like`, `he-like`, `li-like`, `be-like`, or `ne-like`. |
| `Ion` | Six-digit ion identifier. The list is automatically restricted to the selected sequence. |
| `DB1`, `DB2` | Available database datasets discovered from `lev_<database>` entries in the selected HDF5 file. The two databases must be different. |
| `Threshold min`, `Threshold max`, `Threshold step` | Threshold grid evaluated lazily by the Viewer. The public interface accepts at most 101 values per load. |
| `Load Viewer` | Loads the selected data, computes the XCOS similarity matrix, and opens the matching interface. |

The principal publication settings remain fixed internally as `EDGE_ABLATION="xcos"`, `XCOS_KEEP="cfg,phys,energy"`, and `MODEL_PATH=None`. The default XCOS weights are 0.47 for configuration, 0.25 for term quantum numbers and parity, and 0.28 for energy.

The Viewer grid requires `0 <= min <= max <= 1` and a positive step. It includes `max` only if the step reaches it; for example, `min=0`, `max=0.99`, `step=0.01` gives 100 values. This differs from the batch scanner's exclusive `--threshold-stop`. After changing the sequence, ion, databases, or grid, click **Load Viewer** again to apply those selections to the matching interface.

Only retained `lev_<database>` datasets are listed; `_delete` archives are excluded. Sequence selection chooses the configuration encoder and ion list. It does **not** automatically apply the sequence-dependent validation parameters from Table 6; set the rule controls to the values in Section 6 when reproducing the paper.

### 2.3 Main interface functions

The interface displays the total similarity matrix, threshold-constrained Hungarian assignments, rule-consistency and mismatch information, selected-pair summaries, threshold-dependent statistics, level lookup, and tracing tools. Threshold assignments are calculated only when needed and then cached for the current Viewer session.

The current map DAT file is available immediately. Full similarity DAT files, changed-region DAT files, and PDF figures are prepared only after the corresponding **Prepare** button is clicked, which avoids repeatedly serializing large files during ordinary threshold or display changes.

A Binder session may require an initial environment build. Large ions, dense threshold grids, complete change spans, and complete traces require more computation. The formal release will be archived on production Zenodo; the hosted Viewer is a convenient execution entry point.

### 2.4 First example: load, validate, and export a map

1. Select `Sequence = he-like`, `Ion = 020605` (C V), `DB1 = atomdb313`, and `DB2 = spex30801`. Set the threshold minimum, maximum, and step to `0.00`, `0.99`, and `0.01`, then click **Load Viewer**.
2. In **Rule / Diff**, keep all six checks enabled: `cfg`, `2S+1`, `L`, `J`, `parity`, and `energy`. Set `cfg cos = 0.99`, `E rel = 0.008`, and `E mode = rel only`. `E abs/eV` is unused in this mode. The relative tolerance `0.008` means 0.8%, not 0.008%. Use the sequence-specific values in Section 6 for other sequences; the initial rule-control values are generic defaults.
3. In **Overview / Threshold**, use the `thr` slider to select a matching threshold, for example `0.90`. Check **Summary**: an acceptable result requires `theoretical > 0`, `matched = theoretical`, and `mismatch = 0` together.
4. Click **Download map** to save the selected correspondences. To save the current plot, choose the display settings, click **Prepare PDF**, and then use the generated download link.

With the bundled data, models, and default XCOS weights, this AtomDB–SPEX example at `thr = 0.90` gives `matched = theoretical = 204` and `mismatch = 0`.

The matching threshold `thr` changes the Hungarian assignment. The controls in **Rule / Diff** instead evaluate that assignment and determine the rule-based `theoretical` count; changing them does not recalculate the similarity matrix or force the assignment to obey those rules. Therefore, zero mismatches alone does not demonstrate complete recovery.

### 2.5 Interpret the display and downloaded results

| Summary label | Meaning |
|---|---|
| `matched` | Number of selected pairs, `N_match`. |
| `mismatch` | Number of selected pairs that fail at least one enabled rule, `N_mismatch`. |
| `theoretical` | Maximum number of one-to-one pairs satisfying all enabled rules, `N_theo`. |
| `acc` | Fraction of selected pairs satisfying all enabled rules when `matched > 0`; undefined for an empty selection. This is internal rule consistency. `acc = 1` alone does not guarantee `matched = theoretical`. |

Matrix rows correspond to DB1 and columns to DB2. The **Display** index sliders select the plotted region using database `idx` values; they do not restrict the full-table assignment, Summary counts, or **Download map**. The initial plot shows a limited index window, so the number of visible selected cells may be smaller than `matched`.

**Download map** is a tab-separated file with `idx1`, `idx2`, and `sim`: the original level identifiers in DB1 and DB2 and their total similarity. It includes all selected pairs at the current threshold, including any rule violations; unmatched levels have no row. Choose an acceptable threshold before treating the export as an accepted map. Browser downloads are saved to the browser's configured download location. **Prepare similarity download** creates the full similarity table; **Prepare PDF** exports the currently displayed figure. Prepare downloads again after changing the relevant settings.

For individual inspection, enter database `idx` values in **Pair details (by database idx)** and click **Update details**. In **Trace**, select a database and level index, then click **Update trace** to follow its counterpart across the loaded thresholds. In **Change**, choose a `span` and click **Update changes** before displaying or downloading changes for that interval. A `not computed` entry indicates an on-demand calculation that has not yet been requested.

## 3. Atomic Structure Data

### 3.1 Sequence coverage

`clean_data/` contains 147 HDF5 ion files. The table gives the number of files and the total number of retained energy-level records (excluding `_delete` archives). Database coverage differs among ions. SPEX v3.08.01 is used in the principal analysis; SPEX v3.08.03 is retained as a supplementary standardized dataset.

| Sequence | Prefix | Atomic-number range | Ion files | AtomDB v3.1.3 levels | CHIANTI v11.0.2 levels | SPEX v3.08.01 levels | SPEX v3.08.03 levels |
|---|---:|---:|---:|---:|---:|---:|---:|
| H-like | `01` | Z = 1–36 | 36 | 3,500 | 653 | 8,044 | 8,236 |
| He-like | `02` | Z = 2–36 | 35 | 6,109 | 8,896 | 19,135 | 19,567 |
| Li-like | `03` | Z = 3–30 | 28 | 25,696 | 13,486 | 19,074 | 19,258 |
| Be-like | `04` | Z = 4–30 | 27 | 39,507 | 3,341 | 17,379 | 17,037 |
| Ne-like | `10` | Z = 10–30 | 21 | 2,192 | 3,433 | 14,606 | 15,175 |
| **Total** | — | — | **147** | **77,004** | **29,809** | **78,238** | **79,273** |

### 3.2 Ion-file naming convention

Each file is named `SSZZII.h5`:

- `SS` is the electron-count/isoelectronic-sequence code (`01`, `02`, `03`, `04`, or `10`);
- `ZZ` is the atomic number; and
- `II` is the spectroscopic ionization stage, with the neutral atom represented by `01`.

For example, `010202.h5` contains H-like helium, `020807.h5` contains He-like oxygen, `030604.h5` contains Li-like carbon, and `042623.h5` contains Be-like iron.

### 3.3 HDF5 datasets and level-table columns

A file stores the available tables for one ion. The principal dataset names are:

```text
lev_atomdb313       lev_atomdb313_delete
lev_chianti1102     lev_chianti1102_delete
lev_spex30801       lev_spex30801_delete
lev_spex30803       lev_spex30803_delete
```

Only nonempty datasets are saved; each `_delete` table is optional. The retained
energy-level table has the following columns:

| Column | Description |
|---|---|
| `idx` | Database level identifier stored in the archived record. |
| `cfg` | Electronic-configuration string. |
| `2S+1` | Spin multiplicity. |
| `L` | Total orbital angular momentum represented numerically (`S`, `P`, `D`, `F`, ... correspond to 0, 1, 2, 3, ...). |
| `J` | Total angular momentum. |
| `pi` | Existing source value, if present; otherwise derived from `cfg`: 0 for even parity, 1 for odd parity, 2 for a `limit` entry (a reserved label, not physical parity), and -1 when parity cannot be determined. |
| `pcm` | Energy in cm⁻¹. |
| `ryd` | Energy in Rydberg. |
| `eV` | Energy in electronvolts. |
| `original_cfg` | Original configuration before cleaning; always the last field. |

The matching inputs are the retained `lev_<database>` tables. The `original_cfg` field and `_delete` tables are archival information and are excluded from the similarity features. Only retained levels contribute to the level-count statistics in the paper.

Deleted level records retain
the original fields, with `cfg` renamed to the final field `original_cfg`, and
`original_row` recording the zero-based source row position. No derived `pi` is
added to deleted records. Together with retained records, they reconstruct the
pre-clean source level table, including its row order. See [Section 9](#9-cleaning-and-exact-level-table-restoration) for the reconstruction equations, restoration function, and empty-table handling.

## 4. Interactive Matching Viewer

`viewer/thr_viewer.ipynb` is the Voilà-compatible launcher, while `viewer/threshold_viewer.py` implements the main workflow. Together they:

- load a selected database pair from one HDF5 ion file;
- add configuration embeddings from the corresponding stored encoder;
- calculate the requested XCOS similarity matrix;
- perform threshold-constrained Hungarian matching;
- calculate the rule matrices and theoretical maximum one-to-one overlap `N_theo`;
- report rule-consistent and mismatched selected pairs;
- provide level-index tracing and local submatrix inspection; and
- export figures and data tables through the interface.

Use `viewer/sim_only_viewer.ipynb` when only the total similarity matrix is needed, without threshold filtering or Hungarian assignment. It supports selection of sequence, ion, database pair, level-index ranges, and figure dimensions. It creates `map_data/photos/` when needed and saves the generated PDF files there.

## 5. Configuration Model (`cfg`)

The `cfg/` package constructs and trains the sequence-specific model used to compare electronic-configuration strings. This is an optional workflow. Pair generation and training write to the paths below and can replace the bundled TSV files, models, and vocabularies; use a separate working copy if you want to preserve the packaged reference files.

### `cfg_train_data.py`

This script reads configuration strings from retained `lev_*` datasets in `clean_data/`, excluding `_delete` archives, and generates training pairs for each sequence. In the present archive, this includes the supplementary SPEX v3.08.03 tables as well as the three publication databases. The pair categories include formatting-equivalent (`hard`), parent-term-reduced (`medium`), closed-core-shortened (`shortening`), and structurally related (`soft`) examples. The generated files are stored in:

```text
map_data/cfg_ts/train_data/h_like_pairs.tsv
map_data/cfg_ts/train_data/he_like_pairs.tsv
map_data/cfg_ts/train_data/li_like_pairs.tsv
map_data/cfg_ts/train_data/be_like_pairs.tsv
map_data/cfg_ts/train_data/ne_like_pairs.tsv
```

The archived TSV files are the reference training inputs for the bundled models and the reported training-pair statistics. Regenerating them can change the pair counts if the set of `lev_*` datasets or the pair-generation rules changes. Select the target sequence in the script and run:

```bash
python ./cfg/cfg_train_data.py
```

### `cfg_ts.py`

This script trains a character-level Bidirectional Long Short-Term Memory (BiLSTM) encoder using the generated positive pairs and dynamically sampled negative examples. The training objective combines a weighted InfoNCE-style contrastive loss and a cosine-margin triplet loss. Select the sequence, number of epochs, and visualization interval in the final `train_from_pairs(...)` call, then run:

```bash
python ./cfg/cfg_ts.py
```

For each sequence, the trained model and vocabulary are stored in:

```text
map_data/cfg_ts/<sequence>/cfg_ts_model.pth
map_data/cfg_ts/<sequence>/cfg_vocab.pkl
```

When `visualize_every` is enabled, the script also saves t-SNE plots of the learned configuration embeddings and a training-loss curve in the selected sequence directory. Setting the Li-like sequence and the selected epoch runs the procedure used for the paper's Li-like configuration-embedding t-SNE plot. Retraining is not guaranteed to reproduce identical embeddings or point coordinates; use the bundled encoder for the reported matching workflow.

### Other files in `cfg/`

- `ion_config.py` defines sequence aliases, electron counts, filename prefixes, training-pair names, model directories, and shortening rules.
- `cfg_model_plt.ipynb` uses the Li-like training pairs and trained encoder to draw the paper's boxplot of cosine similarity for hard, shortening, medium, soft, and negative configuration pairs.

## 6. Similarity Computation (`cos`)

The direct similarity used by the viewer is implemented in `cos/`.

- `xcos.py` calculates the three component matrices and their weighted total.
- `term_score.py` assigns a discrete pairwise score from differences in `2S+1` and `L`, while requiring compatible `J`, parity, and physically valid term combinations.
- `energy_features.py` represents each energy using transformed absolute energy, a shared energy-band indicator, and relative position within that band.
- `cos_scan_match.py` scans sequences, ions, thresholds, and available pairs among AtomDB v3.1.3, CHIANTI v11.0.2, and SPEX v3.08.01. A threshold succeeds only when `N_theo > 0`, the selected match count equals this rule-based theoretical overlap, and all selected pairs satisfy the validation rules. Cases with `N_theo = 0` have no successful thresholds and are reported as `fail`, with the zero theoretical overlap retained in the report. SPEX v3.08.03 is not included in its default target-database list.

For level `i` in one database and level `j` in another, the total similarity is

```text
S(i,j) = 0.47 S_cfg(i,j) + 0.25 S_SLJP(i,j) + 0.28 S_E(i,j),
```

where the three terms represent configuration similarity, term-quantum-number/parity similarity, and energy similarity.

By default, `cos_scan_match.py` writes its text and CSV reports to:

```text
cos_data/real_data_scan/<sequence>/
```

The script defaults to the **Ne-like** validation settings. Changing only
`--ion-type` does not change these rule parameters. To reproduce the five
sequences, use the following Table 6 settings; energy tolerances are fractions
for relative differences and eV for absolute differences.

| Sequence | `--cfg-cos-thr` | `--energy-rel-thr` | `--energy-abs-thr` | `--energy-rule-mode` |
|---|---:|---:|---:|---|
| H-like | 0.99 | 0.007 | Not used | `rel_only` |
| He-like | 0.99 | 0.008 | Not used | `rel_only` |
| Li-like | 0.85 | 0.018 | Not used | `rel_only` |
| Be-like | 0.80 | 0.064 | 1.0 | `rel_or_abs` |
| Ne-like | 0.76 | 0.023 | 4.4 | `rel_or_abs` |

For example, run the Be-like scan from the repository root:

```bash
python cos/cos_scan_match.py --ion-type be-like \
  --cfg-cos-thr 0.80 --energy-rel-thr 0.064 --energy-abs-thr 1.0 \
  --energy-rule-mode rel_or_abs \
  --threshold-start 0.0 --threshold-stop 0.991 --threshold-step 0.01
```

The stop value is exclusive; this command evaluates 0.00 through 0.99.
Thresholds are configurable. For the SPEX v3.08.03 comparison, replace
`spex30801` with `spex30803` in `TARGET_DBS` and use a different
`--output-prefix` to preserve the principal scan reports.

With the bundled data and the three principal databases, there are 314
available ion–database-pair cases: 71 H-like, 65 He-like, 64 Li-like, 64 Be-like,
and 50 Ne-like. A case requires retained level tables in both databases.

To test the Be-like command on one ion first, append `--ion-start 040603 --ion-end 040603`; these bounds are inclusive. The scanner evaluates every available pair among `TARGET_DBS` for the selected ions. Run the command separately for each sequence, using its row in the parameter table.

The default report names are `threshold_theoretical_scan_report.txt` and `threshold_theoretical_scan_report.csv` inside the sequence output directory. Reusing the same output directory and prefix overwrites those reports; use `--output-prefix` to name a separate run. In the CSV, `status=success` means at least one sampled threshold is acceptable, `fail` means none is acceptable on that grid, and `error` indicates a processing exception. Check `theoretical` to distinguish an empty reference from an unsuccessful nonempty case. `success_threshold_ranges` groups adjacent successful grid points; it does not establish acceptance at every unsampled intermediate value. The console reports the output paths when the scan finishes.

## 7. Threshold-constrained Matching and Validation (`thr_hung`)

The `thr_hung/` package converts a similarity matrix into a partial one-to-one map.

- `hungarian.py` supports ordinary one-to-one assignment and threshold-aware assignment with unmatched levels.
- `threshold_scan.py` repeats the assignment over a threshold grid and identifies rows or columns whose selected counterpart changes.
- `rules.py` constructs pairwise consistency matrices for configuration, `2S+1`, `L`, `J`, parity, and energy; calculates the maximum possible one-to-one rule-consistent overlap `N_theo`; and evaluates the pairs selected by the Hungarian solution.

The threshold prevents low-similarity forced matches when two databases differ in level number or physical coverage. Under the paper's acceptance condition, a threshold is acceptable only if `N_theo > 0`, `N_match = N_theo`, and `N_mismatch = 0`. An empty result with `N_theo = 0` does not establish any actual correspondence and must not be counted as a successful recovery. These conditions assess internal rule consistency, not accuracy against an independent external ground truth.

## 8. Supporting Packages

- `common/` provides shared loading of configuration encoders and cross-database level pairs.
- `tbl/` provides energy-level table readers and exporters. `show_tbl_cflev.py` also constructs a preliminary reference alignment for states with identical configuration and term identifiers, resolving duplicate states by minimizing energy differences; this alignment is used for diagnostics rather than as the final XCOS assignment.
- `plot/` groups levels by configuration and generates cross-database energy-level plots.
- `diagnostics/` contains detailed checks for incorrect or interchanged assignments.
- `list/` normalizes database names and identifiers.
- `util/` provides periodic-table information, term-symbol processing, and atomic energy-unit conversion.

For an available level table, read the packaged file directly:

```python
from datacollect.h5_tools import ds_to_df

levels = ds_to_df("clean_data/020605.h5", "lev_atomdb313")
```

`common.data_loader.load_pair_from_h5` is the pair-loading entry point used by the Viewer. `datacollect/h5_tools.py` provides the HDF5 reading and conversion helpers used by these workflows.

## 9. Cleaning and Exact Level-Table Restoration

The packaged `clean_data/` files contain the information needed to reconstruct their pre-clean level tables. In this section, **original table** means the ordered, structured level table immediately before cleaning. Restoration uses only the packaged HDF5 file and `dataclean/base_tools.py`.

### 9.1 Stored cleaning information

For a database label `<database>`, the archive stores:

| Dataset | Contents |
|---|---|
| `lev_<database>` | Retained records in their original relative order, with cleaned `cfg` and **`original_cfg` as the last field** |
| `lev_<database>_delete` | Excluded records, with zero-based `original_row` positions and **`original_cfg` as the last field and only configuration field** |

`base_tools.clean_configuration_strings` removes trailing dot padding and any whitespace following it: `1s........................` becomes `1s`, and `1s.2s...` becomes `1s.2s`. Internal separators and parent-term labels are preserved. The pre-clean configuration bytes are saved in `original_cfg`. Other pre-existing fields, including `idx`, energies, their data types, and any existing `pi`, are preserved. If `pi` was absent, it is derived for retained records only. The HDF5 cleaning path performs no energy conversion.

The AtomDB cleaner retains records satisfying `2S+1 >= 0` and `L >= 0`; the CHIANTI and SPEX cleaners retain all records under their current rules. Exclusion by this rule does not itself establish that an energy value is physically wrong.

### 9.2 Row positions and reconstruction

Let the original table be $R=(r_0,\ldots,r_{N-1})$, the retained table be $K=(k_0,\ldots,k_{n-1})$, and the deleted table be $D=(d_0,\ldots,d_{m-1})$. All positions are **zero-based**. For each deleted record, define

$$
p_j=D[j,\texttt{original\_row}],\qquad 0\leq j<m.
$$

A valid partition satisfies

$$
n+m=N,\qquad 0\leq p_j<N,\qquad p_j\neq p_\ell\ \text{for }j\neq\ell.
$$

Let $U=\{0,\ldots,N-1\}$ and $P=\{p_j:0\leq j<m\}$. List the complementary retained positions in increasing order:

$$
C=U\setminus P=\{c_0<\cdots<c_{n-1}\}.
$$

An index set is empty when its corresponding count is zero. Because cleaning preserves the relative order of retained records, the row at position $i$ is reconstructed as

$$
\widehat R[i]=
\begin{cases}
\Phi(d_j), & i=p_j,\quad 0\leq j<m,\\
\Phi(k_q), & i=c_q,\quad 0\leq q<n,
\end{cases}
$$

where $\Phi$ restores the original fields as defined in Section 9.3. The sets $P$ and $C$ are disjoint and cover $U$, so every original position receives exactly one record. Distinct deleted positions and the preserved retained-row order make this placement unique.

For example, if the original table has five rows and its second and third rows are excluded, then $N=5$, $P=\{1,2\}$, and $C=\{0,3,4\}$:

| Archived record | Restored zero-based row | Original row, counted from one |
|---|---:|---:|
| `K[0]` | 0 | 1 |
| `D[0]`, `original_row = 1` | 1 | 2 |
| `D[1]`, `original_row = 2` | 2 | 3 |
| `K[1]` | 3 | 4 |
| `K[2]` | 4 | 5 |

Thus, the three retained rows occupy positions 0, 1, and 2 in `K`, but return to positions 0, 3, and 4 in the reconstructed table. `original_row` denotes a row position, **not** the value of `idx`; `idx` itself is preserved. No sorting by `idx`, energy, or configuration is involved. Preserve the archived order of the retained table when restoring it.

### 9.3 Field values and data types

Let $F$ be the ordered list of pre-clean field names stored in `source_field_names`. For a retained or deleted record $x$ and each $f\in F$, define

$$
\Phi(x)[f]=
\begin{cases}
x[\texttt{original\_cfg}], & f=\texttt{cfg},\\
x[f], & f\neq\texttt{cfg}.
\end{cases}
$$

Only the fields in $F$ are included in the restored record. Consequently, `original_cfg` supplies the restored `cfg` value, while `original_row` and `original_cfg` are omitted from the restored field list. A `pi` field added during cleaning is also omitted; a `pi` field already present before cleaning is retained.

The original structured data type is rebuilt from the archived field types, `source_field_names`, `source_field_offsets`, and `source_itemsize`. The original row count is stored in `source_row_count`. The `columns` attribute describes the archived table's current fields, rather than the restored field list.

Combining the row placement in Section 9.2 with $\Phi$ gives $\widehat R=R$: the row order, original field order, field types, and each original field value are recovered. This equality refers to the structured level records. The cleaning implementation verifies it by checking the array shape and dtype and comparing each field's bytes, both before writing and after rereading the archive. This also preserves string bytes and floating-point bit patterns. The guarantee assumes that the archived records, retained-row order, and restoration metadata have not been modified.

The row count, bounds, and uniqueness conditions alone establish a valid placement; they cannot detect arbitrary later changes to a field value. They should not be interpreted as a general integrity check for an edited archive.

### 9.4 Empty tables and absent datasets

Zero-row datasets are not written. The restoration convention is:

| Archived state | Restoration |
|---|---|
| Retained table exists; `_delete` table is absent | Set $m=0$ and restore all rows from the retained table. |
| `_delete` table exists; retained table is absent | Set $n=0$ and restore all rows using `original_row`. |
| Both tables exist | Apply the partition and field mappings above. |
| Both tables are absent, but `empty_level:<table>` exists as a file attribute | Restore the typed empty table stored in that attribute; $N=0$. |
| Both tables and the empty-table attribute are absent | The requested table is unavailable; the restoration function raises `KeyError`. |

For a nonempty original table, restoration metadata is stored on each nonempty archived table, so either one supplies it. Pre-clean dataset attributes are retained under the `source_attr:` prefix; for an originally empty table, they are stored as file attributes under `empty_source_attr:<table>:`.

### 9.5 Restoration with the bundled function

Run the following from the repository root:

```python
from dataclean.base_tools import restore_level_table

original = restore_level_table("clean_data/020605.h5", "lev_atomdb313")
print(original.dtype.names)
print(len(original))
```

`restore_level_table` returns a NumPy structured array containing the reconstructed records. It reads the necessary row-position and schema information from the HDF5 archive without modifying that file. It does not create a new HDF5 dataset or attach the archived dataset attributes to the returned array.

For matching and training, use the retained `lev_<database>` tables. The database-discovery routines exclude `_delete` tables, and `original_cfg` is not selected as a `cfg_` embedding feature.

## 10. Figure Reproduction and Main Outputs

| Program | Paper figure or main output | Output location or form |
|---|---|---|
| `viewer/thr_viewer.ipynb` | Similarity matrices, threshold-constrained Hungarian maps, rule diagnostics, and matching figures | Interactive Voilà interface with downloadable PDF and DAT files; `map_data/thr_view/` is created at runtime for pair-specific working directories |
| `viewer/sim_only_viewer.ipynb` | Total similarity matrix of Fe XXIII between AtomDB v3.1.3 and SPEX v3.08.01 | PDF files under the automatically created `map_data/photos/` directory |
| `plot/three_energy_views_plot.ipynb` | Energy-level distributions of O VII in AtomDB v3.1.3, CHIANTI v11.0.2, and SPEX v3.08.01 | Creates `data/seq<SS>/<ION>/<database_combination>/three_energy_views/` and writes three PDF views there |
| `cfg/cfg_model_plt.ipynb` | Cosine-similarity boxplot for hard, shortening, medium, soft, and negative Li-like configuration pairs | `map_data/cfg_ts/figs/cosine_boxplot.pdf` |
| `cfg/cfg_ts.py` | Trained configuration encoder, vocabulary, loss curve, and t-SNE plot | `map_data/cfg_ts/<sequence>/` |
| `cos/cos_scan_match.py` | Sequence-wide threshold and rule-consistency scan | Text and CSV reports under the configured output directory |

Other threshold-overlay and similarity-submatrix figures in the paper can be generated from `thr_viewer.ipynb`. Select the ion, database pair, level-index range, threshold, and figure dimensions in the interface, then click **Prepare PDF** before downloading the current figure.

## 11. Software Requirements and Reproducibility Notes

### Main dependencies

Python 3.11 is the Binder target recorded in `runtime.txt`. The core runtime and notebook dependencies are specified in `requirements.txt` using compatible version ranges. Install them from the repository root with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency ranges are installation constraints, not an exact environment lock.

The dependency file includes NumPy, pandas, h5py, SciPy, Matplotlib, seaborn, PyTorch, scikit-learn, Astropy, IPython, Jupyter Notebook, ipywidgets, and Voilà.

PyTorch installations with a specific CUDA build may be installed separately before running `pip install -r requirements.txt`; the archived models and viewer can also be used on a CPU-only installation.

### Opening the other notebooks

With the environment active, run `jupyter notebook` from the repository root and open the desired `.ipynb` file. Execute its code cells in order. The Viewer notebooks provide browser controls; the plotting notebooks have parameters in their initial code cells.

For `cfg/cfg_model_plt.ipynb` and `plot/three_energy_views_plot.ipynb`, if the kernel starts inside their subdirectory, run the following in a new first code cell before the existing imports:

```python
from pathlib import Path
import os
import sys

root = Path.cwd()
if not (root / "clean_data").is_dir():
    root = root.parent
assert (root / "clean_data").is_dir(), "Select the CFADB-XCOS repository root."
os.chdir(root)
sys.path.insert(0, str(root))
```

### Common usage issues

| Symptom | What to check |
|---|---|
| `voila` or a Python module cannot be found | Activate the environment used for installation and install `requirements.txt` in that environment. |
| An ion, database, model, or file is unavailable | Keep the directory structure intact, choose an available ion and two distinct database entries, and confirm the working directory. Availability differs by ion. |
| The grid exceeds 101 thresholds | Increase the step or narrow the interval, then click **Load Viewer** again. |
| A new ion or grid selection leaves the old plot visible | Click **Load Viewer** to load the new selection. |
| No map satisfies the acceptance condition | Check the sequence-specific rule values and `E mode`, then inspect the chosen grid. A `fail` report is not itself a software error. |

### Reproducibility notes

- Run scripts from the repository root. The two Viewer notebooks locate the root automatically. Execute the other plotting notebooks with the repository root as their working directory; if Jupyter starts the kernel inside `cfg/` or `plot/`, change its working directory to the repository root before running imports.
- Use the archived `clean_data/` files and sequence-specific models under `map_data/cfg_ts/` to reproduce the matching workflow.
- Database coverage varies by ion; select a database pair available for the chosen ion.
- Retraining can produce small differences across hardware and library builds. Use the archived model files when reproducing the reported similarity results.

## 12. Dataset Identification, Rights, and Contact

### Repository rights

Licensing terms for the original CFADB-XCOS source code and documentation are defined by the Zenodo record. They do not apply to third-party atomic data under `clean_data/`. This README does not independently grant additional rights.

### Packaged datasets and attribution

The `clean_data/` archive uses `lev_atomdb313`, `lev_chianti1102`, `lev_spex30801`, and `lev_spex30803` for retained energy levels, with optional `lev_<database>_delete` tables for excluded records. Availability varies by ion. The database labels identify the versions listed in Section 3.

When reporting results obtained with this package, identify the database versions used and cite the accompanying article and the relevant archive record. Attribution to the AtomDB, CHIANTI, and SPEX teams remains applicable to their data in these tables; standardization does not alter the associated rights or attribution requirements. The article provides the corresponding scientific references and acknowledgements.

### Contact

For scientific questions, reuse requests, or questions concerning the archived data, contact **Qirui Long** at `longqr24@mails.tsinghua.edu.cn` or consult the corresponding-author information in the paper.
