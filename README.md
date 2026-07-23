# CFADB: Automated Cross-database Mapping of Atomic Energy-Level Data

CFADB is the research software and standardized atomic-data archive accompanying the paper **“Automated Cross-database Mapping of Atomic Energy-Level Data for Astrophysical Plasma Models.”** It identifies corresponding atomic energy levels across AtomDB v3.1.3, CHIANTI v11.0.2, and SPEX v3.08.01 by combining electronic-configuration similarity, term quantum numbers and parity, and energy similarity, followed by threshold-constrained Hungarian matching.

> **Start here.** Run the interactive Viewer from the repository root. The browser interface lets users select the sequence, ion, database pair, and threshold grid without editing notebook code:
>
> ```bash
> voila ./viewer/thr_viewer.ipynb --show_tracebacks=True
> ```
>
> The same notebook is ready to serve as the target of a Binder/Voilà launch URL. Add a real release-specific link only after the public GitHub repository and tag have been configured and the hosted workflow has been tested; no placeholder URL is included in this archive.

## Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Atomic Structure Data](#3-atomic-structure-data)
4. [Interactive Matching Viewer](#4-interactive-matching-viewer)
5. [Configuration Model (`cfg`)](#5-configuration-model-cfg)
6. [Similarity Computation (`cos`)](#6-similarity-computation-cos)
7. [Threshold-constrained Matching and Validation (`thr_hung`)](#7-threshold-constrained-matching-and-validation-thr_hung)
8. [Supporting Packages](#8-supporting-packages)
9. [Data Collection and Cleaning](#9-data-collection-and-cleaning)
10. [Figure Reproduction and Main Outputs](#10-figure-reproduction-and-main-outputs)
11. [Software Requirements and Reproducibility Notes](#11-software-requirements-and-reproducibility-notes)
12. [Data Sources and Contact](#12-data-sources-and-contact)

## Repository tree

The repository is organized by scientific function rather than by a single execution order. Users who only wish to inspect the matching results generally need `clean_data/`, `map_data/cfg_ts/`, `viewer/`, `cos/`, and `thr_hung/`. Generated results are stored separately from the source code, archived atomic data, and trained configuration models. Package initializer files and generated cache files are omitted below for clarity.

```text
cfadb/
├── README.md                              # Project documentation
├── requirements.txt                      # Python dependencies
├── LICENSE                               # License for original CFADB code and documentation
├── THIRD_PARTY_DATA.md                    # Sources and rights notice for external atomic data
├── clean_data/                            # Standardized HDF5 atomic-data files
│   ├── 010101.h5                          # First H-like file: H-like hydrogen ion
│   ├── ...                                # Other H-like ions
│   ├── 013636.h5                          # Last H-like file: H-like krypton ion
│   ├── 020201.h5                          # First He-like file: He-like helium ion
│   ├── ...                                # Other He-like ions
│   ├── 023635.h5                          # Last He-like file: He-like krypton ion
│   ├── 030301.h5                          # First Li-like file: Li-like lithium ion
│   ├── ...                                # Other Li-like ions
│   ├── 033028.h5                          # Last Li-like file: Li-like zinc ion
│   ├── 040401.h5                          # First Be-like file: Be-like beryllium ion
│   ├── ...                                # Other Be-like ions
│   ├── 043027.h5                          # Last Be-like file: Be-like zinc ion
│   ├── 101001.h5                          # First Ne-like file: Ne-like neon ion
│   ├── ...                                # Other Ne-like ions
│   └── 103021.h5                          # Last Ne-like file: Ne-like zinc ion
├── viewer/                                # Interactive and static similarity viewers
│   ├── thr_viewer.ipynb                   # Main Voila interface for threshold/Hungarian matching
│   ├── threshold_viewer.py                # Loading, similarity, matching, diagnostics, plotting, and downloads
│   └── sim_only_viewer.ipynb              # Total-similarity matrix viewer without assignment
├── cfg/                                   # Configuration-pair generation and encoder training
│   ├── cfg_train_data.py                  # Builds sequence-specific configuration training pairs
│   ├── cfg_ts.py                          # Trains the BiLSTM configuration encoder and draws t-SNE plots
│   ├── cfg_model_plt.ipynb                # Draws the configuration-similarity boxplot
│   └── ion_config.py                      # Shared sequence definitions and paths
├── cos/                                   # Direct XCOS similarity computation
│   ├── xcos.py                            # Combines configuration, term/parity, and energy similarities
│   ├── term_score.py                      # Computes 2S+1, L, J, and parity similarity
│   ├── energy_features.py                 # Builds absolute, band, and within-band energy features
│   └── cos_scan_match.py                  # Scans ions, database pairs, and thresholds
├── thr_hung/                              # Threshold-constrained assignment and physical validation
│   ├── hungarian.py                       # Full and threshold-aware Hungarian assignment
│   ├── rules.py                           # Rule matrices, N_theo, and selected-match validation
│   └── threshold_scan.py                  # Threshold-grid and assignment-change analysis
├── common/                                # Shared configuration-model and data-loading utilities
│   ├── cfg_embed.py                       # Loads the encoder and appends configuration embeddings
│   └── data_loader.py                     # Prepares cross-database atomic-level pairs
├── datacollect/                           # Converts source-database files to HDF5
│   ├── adb_to_h5.py                       # AtomDB FITS-to-HDF5 conversion
│   ├── cdb_to_h5.py                       # CHIANTI text-to-HDF5 conversion
│   ├── spex_to_h5.py                      # SPEX FITS-to-HDF5 conversion
│   └── h5_tools.py                        # HDF5 conversion, reading, inspection, and merge utilities
├── dataclean/                             # Standardizes collected database tables
│   ├── adb_clean.py                       # Cleans AtomDB tables
│   ├── cdb_clean.py                       # Cleans CHIANTI tables
│   ├── spex_clean.py                      # Cleans SPEX tables
│   ├── base_tools.py                      # Shared cleaning and configuration-parity utilities
│   └── seq_list.py                        # Selects files by sequence and ion range
├── tbl/                                   # Standardized and aligned table readers
│   ├── show_tbl_lev.py                    # Reads and exports an energy-level table
│   └── show_tbl_tran.py                   # Reads and exports a transition table
├── plot/                                  # Energy-level comparison plots
│   ├── three_energy_views_plot.ipynb      # Draws the three energy-level distribution views
│   └── plot_tbl_cflev.py                  # Configuration grouping and energy plotting functions
├── diagnostics/                           # Detailed matching diagnostics
│   └── dmap_check_try.py                  # Examines incorrect or interchanged assignments
├── list/                                  # Database-name metadata
│   └── list_database.py                   # Normalizes database names and identifiers
├── util/                                  # General atomic-data utilities
│   ├── ptbl.py                            # Periodic-table and ion-name utilities
│   ├── quantbl.py                         # Angular-momentum and term-symbol utilities
│   └── unit_cnvt.py                       # Energy and wavelength unit conversion
└── map_data/                              # Archived models, vocabularies, training pairs, and viewer data
    ├── cfg_ts/
    │   ├── train_data/                    # H-/He-/Li-/Be-/Ne-like configuration-pair TSV files
    │   ├── h-like/                        # H-like model and vocabulary
    │   ├── he-like/                       # He-like model, vocabulary, and training figures
    │   ├── li-like/                       # Li-like model and vocabulary
    │   ├── be-like/                       # Be-like model and vocabulary
    │   ├── ne-like/                       # Ne-like model and vocabulary
    │   └── figs/                          # Configuration-model figure outputs
    └── thr_view/                          # Data used or created by threshold-viewer workflows
```

## 1. Overview

Atomic databases used by plasma codes can differ in energy-level coverage, ordering, energies, configuration notation, and quantum-number assignments. CFADB constructs reproducible partial one-to-one correspondences between these level tables while allowing levels without a reliable counterpart to remain unmatched.

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

Enter the extracted repository directory, create a virtual environment, and install the archived dependency specification:

```bash
cd cfadb
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
voila ./viewer/thr_viewer.ipynb --show_tracebacks=True
```

The commands above use the standard activation syntax for macOS and Linux. On Windows, activate the environment with `.venv\Scripts\activate`.

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

Only databases with a `lev_<database>` dataset are listed. A missing `tran_<database>` dataset does not prevent level matching.

### 2.3 Main interface functions

The interface displays the total similarity matrix, threshold-constrained Hungarian assignments, rule-consistency and mismatch information, selected-pair summaries, threshold-dependent statistics, level lookup, and tracing tools. Threshold assignments are calculated only when needed and then cached for the current Viewer session.

The current map DAT file is available immediately. Full similarity DAT files, changed-region DAT files, and PDF figures are prepared only after the corresponding **Prepare** button is clicked, which avoids repeatedly serializing large files during ordinary threshold or display changes.

A Binder session may require an initial environment build. Large ions, dense threshold grids, complete change spans, and complete traces require more computation. Zenodo remains the permanent code and data archive; the hosted Viewer is a convenient execution entry point.

## 3. Atomic Structure Data

### 3.1 Sequence coverage

`clean_data/` contains 147 HDF5 ion files. The table gives the number of files and the total number of archived energy-level records for the three databases used in the paper. Database coverage differs among ions.

| Sequence | Prefix | Atomic-number range | Ion files | AtomDB v3.1.3 levels | CHIANTI v11.0.2 levels | SPEX v3.08.01 levels |
|---|---:|---:|---:|---:|---:|---:|
| H-like | `01` | Z = 1–36 | 36 | 3,500 | 653 | 8,044 |
| He-like | `02` | Z = 2–36 | 35 | 6,109 | 8,896 | 19,135 |
| Li-like | `03` | Z = 3–30 | 28 | 25,696 | 13,486 | 19,074 |
| Be-like | `04` | Z = 4–30 | 27 | 39,507 | 3,341 | 17,379 |
| Ne-like | `10` | Z = 10–30 | 21 | 2,192 | 3,433 | 14,606 |
| **Total** | — | — | **147** | **77,004** | **29,809** | **78,238** |

### 3.2 Ion-file naming convention

Each file is named `SSZZII.h5`:

- `SS` is the electron-count/isoelectronic-sequence code (`01`, `02`, `03`, `04`, or `10`);
- `ZZ` is the atomic number; and
- `II` is the spectroscopic ionization stage, with the neutral atom represented by `01`.

For example, `010202.h5` contains H-like helium, `020807.h5` contains He-like oxygen, `030604.h5` contains Li-like carbon, and `042623.h5` contains Be-like iron.

### 3.3 HDF5 datasets and level-table columns

A file stores the available tables for one ion. The principal dataset names are:

```text
lev_atomdb313       tran_atomdb313
lev_chianti1102     tran_chianti1102
lev_spex30801       tran_spex30801
```

The standardized energy-level table has the following columns:

| Column | Description |
|---|---|
| `idx` | Level index in the source database. |
| `cfg` | Electronic-configuration string. |
| `2S+1` | Spin multiplicity. |
| `L` | Total orbital angular momentum represented numerically (`S`, `P`, `D`, `F`, ... correspond to 0, 1, 2, 3, ...). |
| `J` | Total angular momentum. |
| `pi` | Parity derived from the configuration: 0 for even parity and 1 for odd parity. |
| `pcm` | Energy in cm⁻¹. |
| `ryd` | Energy in Rydberg. |
| `eV` | Energy in electronvolts. |

The `tran_*` datasets contain transition indices, wavelength, Einstein A coefficient, and `loggf`; they are archived for future work and are not used in the present paper.

## 4. Interactive Matching Viewer

`viewer/thr_viewer.ipynb` is the Voila-compatible launcher, while `viewer/threshold_viewer.py` implements the main workflow. Together they:

- load a selected database pair from one HDF5 ion file;
- add configuration embeddings from the corresponding stored encoder;
- calculate the requested XCOS similarity matrix;
- perform threshold-constrained Hungarian matching;
- calculate the rule matrices and theoretical maximum one-to-one overlap `N_theo`;
- report rule-consistent and mismatched selected pairs;
- provide level-index tracing and local submatrix inspection; and
- export figures and data tables through the interface.

Use `viewer/sim_only_viewer.ipynb` when only the total similarity matrix is needed, without threshold filtering or Hungarian assignment. It supports selection of sequence, ion, database pair, level-index ranges, and figure dimensions, and saves PDF files to `map_data/photos/`.

## 5. Configuration Model (`cfg`)

The `cfg/` package constructs and trains the sequence-specific model used to compare electronic-configuration strings.

### `cfg_train_data.py`

This script reads configuration strings from the available `lev_*` datasets in `clean_data/` and generates training pairs for each sequence. The pair categories include formatting-equivalent (`hard`), parent-term-reduced (`medium`), closed-core-shortened (`shortening`), and structurally related (`soft`) examples. The generated files are stored in:

```text
map_data/cfg_ts/train_data/h_like_pairs.tsv
map_data/cfg_ts/train_data/he_like_pairs.tsv
map_data/cfg_ts/train_data/li_like_pairs.tsv
map_data/cfg_ts/train_data/be_like_pairs.tsv
map_data/cfg_ts/train_data/ne_like_pairs.tsv
```

Select the target sequence in the script and run:

```bash
python ./cfg/cfg_train_data.py
```

### `cfg_ts.py`

This script trains a character-level Bidirectional Long Short-Term Memory (BiLSTM) encoder using the generated positive pairs and sampled negative examples. The training objective combines weighted contrastive and triplet-style losses. Select the sequence, number of epochs, and visualization interval in the final `train_from_pairs(...)` call, then run:

```bash
python ./cfg/cfg_ts.py
```

For each sequence, the trained model and vocabulary are stored in:

```text
map_data/cfg_ts/<sequence>/cfg_ts_model.pth
map_data/cfg_ts/<sequence>/cfg_vocab.pkl
```

When `visualize_every` is enabled, the script also saves t-SNE plots of the learned configuration embeddings and a training-loss curve in the selected sequence directory. Setting the Li-like sequence and the required epoch reproduces the paper's Li-like configuration-embedding t-SNE plot.

### Other files in `cfg/`

- `ion_config.py` defines sequence aliases, electron counts, filename prefixes, training-pair names, model directories, and shortening rules.
- `cfg_model_plt.ipynb` uses the Li-like training pairs and trained encoder to draw the paper's boxplot of cosine similarity for hard, shortening, medium, soft, and negative configuration pairs.

## 6. Similarity Computation (`cos`)

The direct similarity used by the viewer is implemented in `cos/`.

- `xcos.py` calculates the three component matrices and their weighted total.
- `term_score.py` compares `2S+1`, `L`, `J`, and the stored `pi` parity column.
- `energy_features.py` represents each energy using transformed absolute energy, a shared energy-band indicator, and relative position within that band.
- `cos_scan_match.py` scans sequences, ions, available database pairs, and thresholds, and records cases in which the selected match count equals the rule-based theoretical overlap and all selected pairs satisfy the validation rules.

For level `i` in one database and level `j` in another, the total similarity is

```text
S(i,j) = 0.47 S_cfg(i,j) + 0.25 S_SLJP(i,j) + 0.28 S_E(i,j),
```

where the three terms represent configuration similarity, term-quantum-number/parity similarity, and energy similarity.

By default, `cos_scan_match.py` writes its text and CSV reports to:

```text
cos_data/real_data_scan/<sequence>/
```

## 7. Threshold-constrained Matching and Validation (`thr_hung`)

The `thr_hung/` package converts a similarity matrix into a partial one-to-one map.

- `hungarian.py` supports ordinary one-to-one assignment and threshold-aware assignment with unmatched levels.
- `threshold_scan.py` repeats the assignment over a threshold grid and identifies rows or columns whose selected counterpart changes.
- `rules.py` constructs pairwise consistency matrices for configuration, `2S+1`, `L`, `J`, parity, and energy; calculates the maximum possible one-to-one rule-consistent overlap `N_theo`; and evaluates the pairs selected by the Hungarian solution.

The threshold prevents low-similarity forced matches when two databases differ in level number or physical coverage. The acceptable threshold range is therefore evaluated together with the rule-consistency conditions described in the paper.

## 8. Supporting Packages

- `common/` provides shared loading of configuration encoders and cross-database level pairs.
- `tbl/` reads and exports standardized energy-level and archived transition tables.
- `plot/` groups levels by configuration and generates cross-database energy-level plots.
- `diagnostics/` contains detailed checks for incorrect or interchanged assignments.
- `list/` normalizes database names and identifiers.
- `util/` provides periodic-table information, term-symbol processing, and atomic energy/wavelength unit conversion.

## 9. Data Collection and Cleaning

The HDF5 files in `clean_data/` are ready for use by the viewer and matching programs. The collection and cleaning packages document how these standardized files were generated.

### Data collection

- `datacollect/adb_to_h5.py` reads AtomDB/APED FITS data.
- `datacollect/cdb_to_h5.py` reads CHIANTI text files.
- `datacollect/spex_to_h5.py` reads SPEX/SPEXACT FITS data.
- `datacollect/h5_tools.py` provides common HDF5 conversion, inspection, reading, and merge functions.

The archive does not duplicate complete external source-database installations. Rerunning data collection requires locally configured source paths.

### Data cleaning

`dataclean/adb_clean.py`, `cdb_clean.py`, and `spex_clean.py` convert database-specific tables into the common archived schema. `base_tools.py` provides shared cleaning functions and derives the `pi` parity column from electronic configurations. `seq_list.py` selects ion files by isoelectronic sequence and optional ion range.

## 10. Figure Reproduction and Main Outputs

| Program | Paper figure or main output | Output location or form |
|---|---|---|
| `viewer/thr_viewer.ipynb` | Similarity matrices, threshold-constrained Hungarian maps, rule diagnostics, and matching figures | Interactive Voila interface with downloadable PDF and DAT files; workflow data under `map_data/thr_view/` |
| `viewer/sim_only_viewer.ipynb` | Total similarity matrix of Fe XXIII between AtomDB v3.1.3 and SPEX v3.08.01 | PDF files under `map_data/photos/` |
| `plot/three_energy_views_plot.ipynb` | Energy-level distributions of O VII in AtomDB v3.1.3, CHIANTI v11.0.2, and SPEX v3.08.01 | `data/seq<SS>/<ION>/<database_pair>/three_energy_views/` |
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

The dependency file includes NumPy, pandas, h5py, SciPy, Matplotlib, seaborn, PyTorch, scikit-learn, Astropy, IPython, Jupyter Notebook, ipywidgets, and Voila.

`pyatomdb` is not included in `requirements.txt` because it is needed only when rerunning AtomDB source-data collection. Install it separately for that workflow:

```bash
python -m pip install pyatomdb
```

PyTorch installations with a specific CUDA build may be installed separately before running `pip install -r requirements.txt`; the archived models and viewer can also be used on a CPU-only installation.

### Reproducibility notes

- Run scripts and notebooks from the repository root unless a notebook states otherwise.
- Use the archived `clean_data/` files and sequence-specific models under `map_data/cfg_ts/` to reproduce the matching workflow.
- Database coverage varies by ion; select a database pair available for the chosen ion.
- Retraining can produce small differences across hardware and library builds. Use the archived model files when reproducing the reported similarity results.
- Complete external source-database installations are required only when rerunning the data-collection stage.

## 12. Data Sources and Contact

### Primary atomic-data sources

- **AtomDB v3.1.3 / APED:** <https://www.atomdb.org/>
- **CHIANTI v11.0.2:** <https://www.chiantidatabase.org/chianti_download.html>
- **SPEX/SPEXACT v3.08.01:** <https://zenodo.org/records/12771915>
- **SPEX/SPEXACT v3.08.03:** supplementary standardized data retained in the archive

Relevant database papers are listed in the accompanying article.

### Contact

For scientific questions about the archive, contact **Qirui Long** at `longqr24@mails.tsinghua.edu.cn` or consult the corresponding-author information in the paper.
