# CFADB-XCOS: data and figure reproduction

Version 1.0.0.

CFADB-XCOS is the research software and standardized atomic energy-level archive accompanying the paper **“Automated Cross-database Mapping of Atomic Energy-Level Data for Astrophysical Plasma Models.”** It identifies corresponding atomic energy levels across AtomDB v3.1.3, CHIANTI v11.0.2, and SPEX v3.08.01 using a combined similarity score, referred to here as XCOS, based on electronic configuration, term quantum numbers and parity, and energy information, followed by threshold-constrained Hungarian matching. Standardized SPEX/SPEXACT v3.08.03 tables are also retained for supplementary comparison, but the principal analysis and default sequence-wide scan use SPEX v3.08.01.

We developed a web interface for generating and inspecting cross-database energy-level matching maps. [Launch the viewer online](https://mybinder.org/v2/gh/qr-Long/CFADB-XCOS/main?urlpath=voila%2Frender%2Fviewer%2Fthr_viewer.ipynb) (startup may take a few minutes), or follow the [local installation and startup instructions](#first-time-installation) below.

## Contents

| Directory or file | Contents |
|---|---|
| `clean_data/` | 147 HDF5 files containing standardized energy levels and cleaning records. |
| `models/` | Configuration models and vocabularies for five isoelectronic sequences; Li-like training pairs used by Figures 3–4 in `training_pairs/`. |
| `core/` | Shared data reading, model inference, similarity calculation and matching routines. |
| `viewer/` | Local matching webpage and its plotting functions. |
| `paper_plot/` | Programs for Figures 1–6 and 10; numerical plot data in `data/`; PDF output in `photos/`. |
| `weight_validation/` | Figure 11, archived scan results, and the resumable weight scan. |
| `paper_parameters.json` | Similarity weights, sequence validation rules, threshold grid and reference parameters for the paper figures. |
| `requirements.txt` | Tested Python dependencies. |
| `runtime.txt` | Python version for Binder. |
| `voila.json` | Local webpage settings; required widget extensions and compatible MathJax rendering. |

There are four shared modules: `core/data.py` (tables and model loading), `core/encoder.py` (configuration vectors), `core/similarity.py` (configuration, term/parity and energy similarities), and `core/matching.py` (Hungarian assignment and validation rules). Figure scripts and the viewer use these routines.

## Data

The main database labels are `atomdb313` (AtomDB 3.1.3), `chianti1102` (CHIANTI 11.0.2), and `spex30801` (SPEX 3.08.01). The archive also retains available `spex30803` tables for comparison. Availability varies by ion.

A filename has the form `EEZZII.h5`: electron count, atomic number, and spectroscopic ion number. For example, `030806.h5` is O VI. Each retained level table is named `lev_<database>`. The matching fields are `idx`, `cfg`, `2S+1`, `L`, `J`, `pi`, and `eV`. The `idx` field is the database level index. Deleted records, original configuration strings and dataset attributes are retained for tracing the cleaning process.

CHIANTI energies use valid observed values where available and theoretical values otherwise. Na IX level 81 uses the theoretical energy as the documented exception to this rule. The tables contain one adopted energy per level; they do not preserve both original CHIANTI energy columns. No transition tables are required by this package.

## First-time installation

Download and extract this package, and open a terminal in the extracted `cfadb-xcos` folder containing `requirements.txt`. Keep the directory structure intact. 

Run the commands for your operating system and terminal below **once** to create the `.venv` environment and download and install the dependencies listed in `requirements.txt`. 

**macOS / Linux:**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows — Command Prompt (CMD):**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

**Windows — PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

After installation, use the startup commands below; do not recreate the environment or reinstall the dependencies on each launch.


## Local matching webpage

### Each time you start the viewer

Open a terminal in the same `cfadb-xcos` folder and run the commands for your terminal:

**macOS / Linux:**

```sh
source .venv/bin/activate
python -m voila viewer/thr_viewer.ipynb
```

**Windows — Command Prompt (CMD):**

```bat
.venv\Scripts\activate.bat
python -m voila viewer/thr_viewer.ipynb
```

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m voila viewer/thr_viewer.ipynb
```

Keep the terminal running while using the webpage. Select a sequence, ion and database pair, then click **Load Viewer**. Adjust the matching threshold, validation rules and displayed index ranges to inspect a map. The rule switches include configuration, $S,~L,~J,~\pi$ and $E$. The page provides PDF and matching-table downloads. 

To reproduce Figures 7–9, select the ion and databases listed below, click **Load Viewer**, and enter the displayed ranges and parameters manually. DB1 is on the vertical axis and DB2 on the horizontal axis. Display ranges are inclusive database indices; changing the displayed range does not restrict the full matching calculation.

### Figures 7–9: displayed ranges and parameters

All panels use similarity weights `(configuration, SLJ and parity, energy) = (0.47, 0.25, 0.28)`. In the table, A = AtomDB 3.1.3, C = CHIANTI 11.0.2, and S = SPEX 3.08.01. The listed energy conditions are relative-only. All four term/parity quantities must agree exactly. The examples use their caption-specific rules, which can differ from the sequence rules used in Figures 10 and 11.

| Figure | Ion (file ID) | DB1 / DB2 | DB1 indices | DB2 indices | Matching threshold | Configuration threshold | Relative energy limit | labels | Diff |
|---|---|---|---|---|---|---|---|---|---|
| 7a | Mn XVI (`102516`) | A / S | 1–941 (113 retained levels) | 1–157 | 0.72 | 0.89 | 0.023 | idx | off |
| 7b | Mn XVI (`102516`) | A / S | 1–60 | 1–60 | 0.72 | 0.89 | 0.023 | idx | off |
| 8a | Fe XXIII (`042623`) | A / S | 309–322 | 1126–1139 | 0.89 | 0.80 | 0.11 | full | selected vs rules |
| 8b | O VI (`030806`) | C / S | 68–76 | 260–268 | 0.73 | 0.86 | 0.12 | full | selected vs rules |
| 9 | F VIII (`020908`) | A / S | 200–204 | 513–522 | 0.90 | 0.99 | 0.008 | full | selected vs rules |

For all rows above, set **E mode = rel only** and keep all six validation rules enabled. To retain the paper panel sizes, set **fig W = 6** and **fig H = 5.4, 5.4, 5.8, 5.5, 3.8** for Figures 7a, 7b, 8a, 8b and 9, respectively. Leave the remaining controls at their defaults. Download each PDF and use the filename listed below.

## Figure reproduction

To generate Figures 1–6 and 10:

```sh
python paper_plot/run_all.py
```

To generate a single figure, run its listed program, for example `python paper_plot/plot_fig6.py`. The PDFs are written to `paper_plot/photos/`, replacing previous files of the same name. Figure 11 has its own command and output directory.

| Figure | Input | Program or webpage parameters | PDF filename |
|---|---|---|---|
| 1a | O VII levels, `clean_data/020807.h5` | `paper_plot/plot_fig1.py` | `fig01a_o_vii_energy_full.pdf` |
| 1b | Same, lower energy region | Same | `fig01b_o_vii_energy_low.pdf` |
| 1c | Same, higher energy region | Same | `fig01c_o_vii_energy_high.pdf` |
| 2 | Retained tables for five sequences, atomic numbers 1–30 | `paper_plot/plot_fig2.py` | `fig02_level_counts.pdf` |
| 3 | Li-like training pairs, model and vocabulary in `models/` | `paper_plot/plot_fig3.py` | `fig03_configuration_similarity.pdf` |
| 4 | Same model and configuration pool; `paper_plot/data/fig04_coordinates.csv` | `paper_plot/plot_fig4.py` | `fig04_configuration_tsne.pdf` |
| 5a | C IV retained levels, `clean_data/030604.h5` | `paper_plot/plot_fig5.py` | `fig05a_c_iv_energy.pdf` |
| 5b | Analytic energy transformation, no atomic table | Same | `fig05b_energy_transform.pdf` |
| 6a | Fe XXIII levels and Be-like model | `paper_plot/plot_fig6.py` | `fig06a_fe_xxiii_similarity_full.pdf` |
| 6b | Same, indices 1–100 in each database | Same | `fig06b_fe_xxiii_similarity_zoom.pdf` |
| 7a | Mn XVI levels and Ne-like model | Webpage: enter the Figure 7a table row above | `fig07a_mn_xvi_matching_full.pdf` |
| 7b | Same | Webpage: enter the Figure 7b table row above | `fig07b_mn_xvi_matching_zoom.pdf` |
| 8a | Fe XXIII levels and Be-like model | Webpage: enter the Figure 8a table row above | `fig08a_fe_xxiii_matching.pdf` |
| 8b | O VI levels and Li-like model | Webpage: enter the Figure 8b table row above | `fig08b_o_vi_matching.pdf` |
| 9 | F VIII levels and He-like model | Webpage: enter the Figure 9 table row above | `fig09_f_viii_mismatches.pdf` |
| 10 | `paper_plot/data/threshold_ranges.csv` | `paper_plot/plot_fig10.py` | `fig10_threshold_ranges.pdf` |
| 11 | `weight_validation/data/weight_grid_results.csv` | `weight_validation/plot_fig11.py` | `fig11_weight_sensitivity.pdf` |

Figure 3 recalculates the similarities each time it runs. Figure 4 plots the saved t-SNE coordinates; to recalculate the point positions, run:

```sh
python paper_plot/plot_fig4.py --recompute
```

Figure 10 also uses saved threshold ranges. If you want to recalculate all 314 cases and update these ranges, run:

```sh
python paper_plot/plot_fig10.py --rescan
```

### Sequence rules for Figures 10 and 11

| Sequence | Configuration threshold | Relative energy limit | Absolute energy limit (eV) | Energy mode |
|---|---|---|---|---|
| H-like | 0.99 | 0.005 | — | relative only |
| He-like | 0.99 | 0.008 | — | relative only |
| Li-like | 0.85 | 0.013 | — | relative only |
| Be-like | 0.80 | 0.064 | — | relative only |
| Ne-like | 0.76 | 0.026 | 4.3 | relative OR absolute |

All four term/parity quantities must agree exactly. These settings are stored in `paper_parameters.json` and also serve as the defaults for custom viewer cases.

## Figure 11 and weight validation

To redraw Figure 11 from the completed scan distributed with the package:

```sh
python weight_validation/plot_fig11.py
```

The output is `weight_validation/output/figures/fig11_weight_sensitivity.pdf`. The archived scan covers 314 ion/database-pair cases and 67 weight settings: 66 points on a 0.1-step grid and the adopted weights `(0.47, 0.25, 0.28)`.

To repeat the calculation:

```sh
python weight_validation/run_scan.py
```

This is the long-running calculation. Completed case checkpoints are stored in `weight_validation/output/checkpoints/`. Repeating the same command resumes completed cases when their inputs and settings agree. An interrupted case restarts from its beginning. On successful completion, the program writes the results to `output/data/`, updates the PDF in `output/figures/`, and removes its completed checkpoints. Add `--keep-checkpoints` to retain them, or `--workers 2` to process two cases at once if memory permits.

The result files are:

- `case_list.csv`: the evaluated ion/database pairs.
- `weight_grid_case_details.csv`: the result for each case and weight setting.
- `weight_grid_results.csv`: the number of successful cases at each weight setting.
- `run_config.json`: completion status, settings and input/result fingerprints; retained to identify the calculation and validate the results.

New calculations replace files under `output/`; the distributed results under `data/` are preserved. To redraw a newly completed scan, use `python weight_validation/plot_fig11.py --input-dir weight_validation/output/data`.

## Citation and contact

Please cite the accompanying article, this archive record, and the relevant AtomDB, CHIANTI and SPEX references when using the package. Consult the archive's license metadata for reuse terms; the original atomic data retain their respective attribution and reuse conditions. No standalone license file is included.

Contact: Qirui Long, `longqr24@mails.tsinghua.edu.cn`.
