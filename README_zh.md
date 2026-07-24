# CFADB-XCOS：原子能级数据的跨数据库自动映射

CFADB-XCOS 是论文 **“Automated Cross-database Mapping of Atomic Energy-Level Data for Astrophysical Plasma Models”** 的配套研究软件和标准化原子数据归档。它使用 XCOS——一个综合电子组态、光谱项量子数与宇称以及能量信息的相似度框架——识别 AtomDB v3.1.3、CHIANTI v11.0.2 和 SPEX v3.08.01 之间相互对应的原子能级，随后执行带阈值约束的匈牙利匹配。

> **从这里开始。** 可以通过 [Binder/Voilà](https://mybinder.org/v2/gh/qr-Long/CFADB-XCOS/main?urlpath=voila%2Frender%2Fviewer%2Fthr_viewer.ipynb) 打开最新的在线 Viewer，也可以在仓库根目录本地运行：
>
> ```bash
> voila ./viewer/thr_viewer.ipynb --show_tracebacks=True
> ```
>
> 浏览器界面允许用户直接选择等电子序列、离子、数据库组合和阈值网格，无需编辑 notebook 代码。在线链接指向当前 GitHub `main` 分支；复现某个特定 Zenodo 版本时，请使用该版本归档中的文件并通过上述命令本地运行。

## 目录

1. [概述](#1-概述)
2. [快速开始](#2-快速开始)
3. [原子结构数据](#3-原子结构数据)
4. [交互式匹配 Viewer](#4-交互式匹配-viewer)
5. [电子组态模型（`cfg`）](#5-电子组态模型cfg)
6. [相似度计算（`cos`）](#6-相似度计算cos)
7. [带阈值约束的匹配与验证（`thr_hung`）](#7-带阈值约束的匹配与验证thr_hung)
8. [辅助程序包](#8-辅助程序包)
9. [数据收集与清洗](#9-数据收集与清洗)
10. [论文图片复现与主要输出](#10-论文图片复现与主要输出)
11. [软件要求与可复现性说明](#11-软件要求与可复现性说明)
12. [数据来源、权利说明与联系方式](#12-数据来源权利说明与联系方式)

## 仓库结构

本仓库按照科学功能组织，而不是按照单一执行顺序组织。只希望查看匹配结果的用户通常只需要 `clean_data/`、`map_data/cfg_ts/`、`viewer/`、`cos/` 和 `thr_hung/`。生成结果与源代码、归档原子数据和训练好的电子组态模型分开保存。为简化展示，以下目录树省略了包初始化文件和自动生成的缓存文件。

```text
CFADB-XCOS/
├── README.md                              # 英文项目说明
├── README_zh.md                           # 中文项目说明
├── requirements.txt                      # Python 依赖
├── clean_data/                            # 标准化 HDF5 原子数据文件
│   ├── 010101.h5                          # 第一个 H-like 文件：类氢氢离子
│   ├── ...                                # 其他 H-like 离子
│   ├── 013636.h5                          # 最后一个 H-like 文件：类氢氪离子
│   ├── 020201.h5                          # 第一个 He-like 文件：类氦氦离子
│   ├── ...                                # 其他 He-like 离子
│   ├── 023635.h5                          # 最后一个 He-like 文件：类氦氪离子
│   ├── 030301.h5                          # 第一个 Li-like 文件：类锂锂离子
│   ├── ...                                # 其他 Li-like 离子
│   ├── 033028.h5                          # 最后一个 Li-like 文件：类锂锌离子
│   ├── 040401.h5                          # 第一个 Be-like 文件：类铍铍离子
│   ├── ...                                # 其他 Be-like 离子
│   ├── 043027.h5                          # 最后一个 Be-like 文件：类铍锌离子
│   ├── 101001.h5                          # 第一个 Ne-like 文件：类氖氖离子
│   ├── ...                                # 其他 Ne-like 离子
│   └── 103021.h5                          # 最后一个 Ne-like 文件：类氖锌离子
├── viewer/                                # 交互式与静态相似度 Viewer
│   ├── thr_viewer.ipynb                   # 阈值/匈牙利匹配主 Voilà 界面
│   ├── threshold_viewer.py                # 加载、相似度、匹配、诊断、绘图和下载
│   └── sim_only_viewer.ipynb              # 不进行分配的总相似度矩阵 Viewer
├── cfg/                                   # 组态样本生成与编码器训练
│   ├── cfg_train_data.py                  # 构建按等电子序列划分的组态训练对
│   ├── cfg_ts.py                          # 训练 BiLSTM 组态编码器并绘制 t-SNE 图
│   ├── cfg_model_plt.ipynb                # 绘制组态相似度箱线图
│   └── ion_config.py                      # 共用的序列定义和路径
├── cos/                                   # XCOS 相似度直接计算
│   ├── xcos.py                            # 综合组态、光谱项/宇称和能量相似度
│   ├── term_score.py                      # 计算 2S+1、L、J 和宇称相似度
│   ├── energy_features.py                 # 构建绝对能量、能段和能段内相对位置特征
│   └── cos_scan_match.py                  # 扫描离子、数据库组合和阈值
├── thr_hung/                              # 带阈值约束的分配与物理验证
│   ├── hungarian.py                       # 完整及带阈值约束的匈牙利分配
│   ├── rules.py                           # 规则矩阵、N_theo 和已选匹配验证
│   └── threshold_scan.py                  # 阈值网格和分配变化分析
├── common/                                # 共用组态模型与数据加载工具
│   ├── cfg_embed.py                       # 加载编码器并添加组态嵌入
│   └── data_loader.py                     # 准备跨数据库原子能级对
├── datacollect/                           # 将源数据库文件转换为 HDF5
│   ├── adb_to_h5.py                       # AtomDB FITS 到 HDF5 的转换
│   ├── cdb_to_h5.py                       # CHIANTI 文本到 HDF5 的转换
│   ├── spex_to_h5.py                      # SPEX FITS 到 HDF5 的转换
│   └── h5_tools.py                        # HDF5 转换、读取、检查和合并工具
├── dataclean/                             # 标准化数据库表
│   ├── adb_clean.py                       # 清洗 AtomDB 表
│   ├── cdb_clean.py                       # 清洗 CHIANTI 表
│   ├── spex_clean.py                      # 清洗 SPEX 表
│   ├── base_tools.py                      # 共用清洗和组态宇称工具
│   └── seq_list.py                        # 按序列和离子范围选择文件
├── tbl/                                   # 标准化和对齐后的表读取工具
│   ├── show_tbl_lev.py                    # 读取并导出能级表
│   └── show_tbl_tran.py                   # 读取并导出跃迁表
├── plot/                                  # 能级比较图
│   ├── three_energy_views_plot.ipynb      # 绘制三种能级分布视图
│   └── plot_tbl_cflev.py                  # 组态分组和能量绘图函数
├── diagnostics/                           # 详细匹配诊断
│   └── dmap_check_try.py                  # 检查错误或交换的匹配
├── list/                                  # 数据库名称元数据
│   └── list_database.py                   # 统一数据库名称和标识符
├── util/                                  # 通用原子数据工具
│   ├── ptbl.py                            # 元素周期表和离子名称工具
│   ├── quantbl.py                         # 角动量和光谱项符号工具
│   └── unit_cnvt.py                       # 能量和波长单位换算
└── map_data/                              # 归档模型、词表、训练样本和 Viewer 数据
    ├── cfg_ts/
    │   ├── train_data/                    # H-/He-/Li-/Be-/Ne-like 组态训练对 TSV
    │   ├── h-like/                        # H-like 模型与词表
    │   ├── he-like/                       # He-like 模型、词表和训练图
    │   ├── li-like/                       # Li-like 模型与词表
    │   ├── be-like/                       # Be-like 模型与词表
    │   ├── ne-like/                       # Ne-like 模型与词表
    │   └── figs/                          # 组态模型图片输出
    └── thr_view/                          # 阈值 Viewer 工作流使用或生成的数据
```

## 1. 概述

等离子体模型所使用的原子数据库在能级覆盖范围、排列顺序、能量、电子组态记法和量子数赋值方面可能存在差异。CFADB-XCOS 在这些能级表之间构建可复现的部分一对一对应关系，同时允许没有可靠对应项的能级保持未匹配状态。

对于选定的离子和数据库组合，主要工作流程如下：

1. 加载标准化能级表和相应等电子序列的组态编码器；
2. 计算组态、光谱项量子数/宇称和能量相似度；
3. 将各分量组合为总 XCOS 相似度矩阵；
4. 应用相似度阈值，并使用包含虚拟未匹配状态的匈牙利分配；
5. 使用组态、光谱项、宇称和能量规则评估选中的配对；
6. 在交互式界面中展示矩阵、分配结果、诊断信息和可下载表格。

论文将该工作流程应用于 H-like、He-like、Li-like、Be-like 和 Ne-like 离子。其目的是构建透明的跨数据库能级对应关系，而不是判断哪个原子数据库更准确。

## 2. 快速开始

### 2.1 安装依赖并从仓库根目录运行

进入解压后的仓库目录，创建虚拟环境并安装归档的依赖配置：

```bash
cd CFADB-XCOS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
voila ./viewer/thr_viewer.ipynb --show_tracebacks=True
```

上述命令使用 macOS 和 Linux 的标准虚拟环境激活方式。Windows 用户请使用 `.venv\Scripts\activate` 激活环境。

### 2.2 选择 Viewer 参数

Viewer 使用浏览器控件，而不是需要编辑的参数代码单元：

| 控件 | 含义 |
|---|---|
| `Sequence` | 等电子序列及对应组态模型：`h-like`、`he-like`、`li-like`、`be-like` 或 `ne-like`。 |
| `Ion` | 六位离子标识符。列表会根据选定序列自动筛选。 |
| `DB1`、`DB2` | 从选定 HDF5 文件的 `lev_<database>` 数据集中发现的可用数据库。两个数据库必须不同。 |
| `Threshold min`、`Threshold max`、`Threshold step` | Viewer 延迟计算的阈值网格。公开界面每次加载最多接受 101 个阈值。 |
| `Load Viewer` | 加载选定数据，计算 XCOS 相似度矩阵并打开匹配界面。 |

论文采用的主要设置在内部固定为 `EDGE_ABLATION="xcos"`、`XCOS_KEEP="cfg,phys,energy"` 和 `MODEL_PATH=None`。XCOS 默认权重分别为：组态 0.47、光谱项量子数与宇称 0.25、能量 0.28。

界面只列出包含 `lev_<database>` 数据集的数据库。

### 2.3 主界面功能

界面展示总相似度矩阵、带阈值约束的匈牙利分配、规则一致性与不匹配信息、已选配对统计、阈值相关统计、能级查询和追踪工具。阈值分配仅在需要时计算，并在当前 Viewer 会话中缓存。

当前映射 DAT 文件可立即下载。完整相似度 DAT、变化区域 DAT 和 PDF 图片只有在点击相应的 **Prepare** 按钮后才会生成，从而避免在普通阈值或显示设置变化时反复序列化大型文件。

Binder 会话可能需要先构建运行环境。较大的离子、密集阈值网格、完整变化区间和完整追踪需要更多计算。Zenodo 是永久代码与数据归档，在线 Viewer 则提供便捷的运行入口。

## 3. 原子结构数据

### 3.1 序列覆盖范围

`clean_data/` 包含 147 个 HDF5 离子文件。下表列出论文所使用的三个数据库在各等电子序列中的文件数量和归档能级记录总数。不同离子的数据库覆盖范围可能不同。

| 等电子序列 | 前缀 | 原子序数范围 | 离子文件数 | AtomDB v3.1.3 能级数 | CHIANTI v11.0.2 能级数 | SPEX v3.08.01 能级数 |
|---|---:|---:|---:|---:|---:|---:|
| H-like | `01` | Z = 1–36 | 36 | 3,500 | 653 | 8,044 |
| He-like | `02` | Z = 2–36 | 35 | 6,109 | 8,896 | 19,135 |
| Li-like | `03` | Z = 3–30 | 28 | 25,696 | 13,486 | 19,074 |
| Be-like | `04` | Z = 4–30 | 27 | 39,507 | 3,341 | 17,379 |
| Ne-like | `10` | Z = 10–30 | 21 | 2,192 | 3,433 | 14,606 |
| **总计** | — | — | **147** | **77,004** | **29,809** | **78,238** |

### 3.2 离子文件命名规则

每个文件命名为 `SSZZII.h5`：

- `SS` 是电子数/等电子序列代码（`01`、`02`、`03`、`04` 或 `10`）；
- `ZZ` 是原子序数；
- `II` 是光谱学电离级数，其中中性原子记为 `01`。

例如，`010202.h5` 表示类氢氦，`020807.h5` 表示类氦氧，`030604.h5` 表示类锂碳，`042623.h5` 表示类铍铁。

### 3.3 HDF5 数据集和能级表字段

每个文件保存一个离子的可用表。主要数据集名称如下：

```text
lev_atomdb313       tran_atomdb313
lev_chianti1102     tran_chianti1102
lev_spex30801       tran_spex30801
```

标准化能级表包含以下字段：

| 字段 | 说明 |
|---|---|
| `idx` | 源数据库中的能级编号。 |
| `cfg` | 电子组态字符串。 |
| `2S+1` | 自旋多重度。 |
| `L` | 以数值表示的总轨道角动量（`S`、`P`、`D`、`F`……分别对应 0、1、2、3……）。 |
| `J` | 总角动量。 |
| `pi` | 根据电子组态计算的宇称：偶宇称为 0，奇宇称为 1。 |
| `pcm` | 以 cm⁻¹ 表示的能量。 |
| `ryd` | 以 Rydberg 表示的能量。 |
| `eV` | 以电子伏特表示的能量。 |

`tran_*` 数据集包含跃迁上下能级编号、波长、Einstein A 系数和 `loggf`；这些数据为未来研究归档，本文不使用。

## 4. 交互式匹配 Viewer

`viewer/thr_viewer.ipynb` 是兼容 Voilà 的启动程序，`viewer/threshold_viewer.py` 实现主要工作流程。二者共同完成：

- 从一个 HDF5 离子文件中加载选定的数据库组合；
- 使用相应的已保存编码器添加电子组态嵌入；
- 计算所需的 XCOS 相似度矩阵；
- 执行带阈值约束的匈牙利匹配；
- 计算规则矩阵和理论最大一对一重叠数 `N_theo`；
- 报告满足规则和不满足规则的已选配对；
- 提供能级编号追踪和局部子矩阵检查；
- 通过界面导出图片和数据表。

当只需要总相似度矩阵而不需要阈值筛选或匈牙利分配时，可使用 `viewer/sim_only_viewer.ipynb`。它支持选择等电子序列、离子、数据库组合、能级编号范围和图片尺寸，并将 PDF 保存到 `map_data/photos/`。

## 5. 电子组态模型（`cfg`）

`cfg/` 程序包构建并训练用于比较电子组态字符串的按序列划分模型。

### `cfg_train_data.py`

该脚本读取 `clean_data/` 中可用 `lev_*` 数据集里的电子组态字符串，并为每个等电子序列生成训练对。配对类别包括格式等价（`hard`）、母项约简（`medium`）、闭壳层缩写（`shortening`）以及结构相关（`soft`）样本。生成的文件保存在：

```text
map_data/cfg_ts/train_data/h_like_pairs.tsv
map_data/cfg_ts/train_data/he_like_pairs.tsv
map_data/cfg_ts/train_data/li_like_pairs.tsv
map_data/cfg_ts/train_data/be_like_pairs.tsv
map_data/cfg_ts/train_data/ne_like_pairs.tsv
```

在脚本中选择目标序列后运行：

```bash
python ./cfg/cfg_train_data.py
```

### `cfg_ts.py`

该脚本使用生成的正样本对和采样负样本训练字符级双向长短期记忆网络（Bidirectional Long Short-Term Memory，BiLSTM）编码器。训练目标结合加权对比损失和三元组形式损失。在最后的 `train_from_pairs(...)` 调用中选择序列、训练轮数和可视化间隔，然后运行：

```bash
python ./cfg/cfg_ts.py
```

每个序列的训练模型和词表保存在：

```text
map_data/cfg_ts/<sequence>/cfg_ts_model.pth
map_data/cfg_ts/<sequence>/cfg_vocab.pkl
```

启用 `visualize_every` 时，脚本还会在所选序列目录中保存所学习组态嵌入的 t-SNE 图和训练损失曲线。选择 Li-like 序列和论文使用的训练轮数，可复现论文中的 Li-like 组态嵌入 t-SNE 图。

### `cfg/` 中的其他文件

- `ion_config.py` 定义序列别名、电子数、文件名前缀、训练对名称、模型目录和缩写规则。
- `cfg_model_plt.ipynb` 使用 Li-like 训练对和训练好的编码器，绘制 hard、shortening、medium、soft 和 negative 组态对的余弦相似度箱线图。

## 6. 相似度计算（`cos`）

Viewer 使用的直接相似度在 `cos/` 中实现。

- `xcos.py` 计算三个分量矩阵及其加权总和。
- `term_score.py` 比较 `2S+1`、`L`、`J` 和保存的 `pi` 宇称字段。
- `energy_features.py` 使用变换后的绝对能量、共用能段标记和能段内相对位置表示能量。
- `cos_scan_match.py` 扫描等电子序列、离子、可用数据库组合和阈值，并记录匹配数量等于基于规则得到的理论重叠数且所有已选配对均满足验证规则的情况。

对于一个数据库中的能级 `i` 和另一个数据库中的能级 `j`，总相似度为：

```text
S(i,j) = 0.47 S_cfg(i,j) + 0.25 S_SLJP(i,j) + 0.28 S_E(i,j),
```

其中三个分量分别表示电子组态相似度、光谱项量子数/宇称相似度和能量相似度。

默认情况下，`cos_scan_match.py` 将文本和 CSV 报告写入：

```text
cos_data/real_data_scan/<sequence>/
```

## 7. 带阈值约束的匹配与验证（`thr_hung`）

`thr_hung/` 程序包将相似度矩阵转换为部分一对一映射。

- `hungarian.py` 支持普通一对一分配，以及包含未匹配状态的阈值感知分配。
- `threshold_scan.py` 在阈值网格上重复分配，并识别已选对应项发生变化的行或列。
- `rules.py` 构建组态、`2S+1`、`L`、`J`、宇称和能量的配对一致性矩阵；计算可能达到的最大一对一规则一致重叠数 `N_theo`；并评估匈牙利算法选中的配对。

当两个数据库的能级数量或物理覆盖范围不同时，阈值可避免强制选择低相似度匹配。因此，可接受阈值范围需要与论文所述规则一致性条件共同评估。

## 8. 辅助程序包

- `common/` 提供组态编码器和跨数据库能级对的共用加载功能。
- `tbl/` 读取并导出标准化能级表和归档跃迁表。
- `plot/` 按电子组态对能级分组，并生成跨数据库能级图。
- `diagnostics/` 包含对错误或交换匹配的详细检查。
- `list/` 统一数据库名称和标识符。
- `util/` 提供元素周期表信息、光谱项符号处理，以及原子能量/波长单位换算。

## 9. 数据收集与清洗

`clean_data/` 中的 HDF5 文件已可直接用于 Viewer 和匹配程序。数据收集与清洗程序包记录了这些标准化文件的生成过程。

### 数据收集

- `datacollect/adb_to_h5.py` 读取 AtomDB/APED FITS 数据。
- `datacollect/cdb_to_h5.py` 读取 CHIANTI 文本文件。
- `datacollect/spex_to_h5.py` 读取 SPEX/SPEXACT FITS 数据。
- `datacollect/h5_tools.py` 提供通用 HDF5 转换、检查、读取和合并功能。

本归档不会复制完整的外部源数据库安装。重新运行数据收集流程时，需要在本地配置源文件路径。

### 数据清洗

`dataclean/adb_clean.py`、`cdb_clean.py` 和 `spex_clean.py` 将各数据库特有的表转换为统一归档格式。`base_tools.py` 提供共用清洗函数，并根据电子组态生成 `pi` 宇称字段。`seq_list.py` 按等电子序列和可选离子范围选择离子文件。

## 10. 论文图片复现与主要输出

| 程序 | 论文图片或主要输出 | 输出位置或形式 |
|---|---|---|
| `viewer/thr_viewer.ipynb` | 相似度矩阵、带阈值约束的匈牙利映射、规则诊断和匹配图 | 交互式 Voilà 界面，可下载 PDF 和 DAT；工作流数据位于 `map_data/thr_view/` |
| `viewer/sim_only_viewer.ipynb` | AtomDB v3.1.3 与 SPEX v3.08.01 之间 Fe XXIII 的总相似度矩阵 | PDF 位于 `map_data/photos/` |
| `plot/three_energy_views_plot.ipynb` | AtomDB v3.1.3、CHIANTI v11.0.2 和 SPEX v3.08.01 中 O VII 的能级分布 | `data/seq<SS>/<ION>/<database_pair>/three_energy_views/` |
| `cfg/cfg_model_plt.ipynb` | hard、shortening、medium、soft 和 negative Li-like 组态对的余弦相似度箱线图 | `map_data/cfg_ts/figs/cosine_boxplot.pdf` |
| `cfg/cfg_ts.py` | 训练好的组态编码器、词表、损失曲线和 t-SNE 图 | `map_data/cfg_ts/<sequence>/` |
| `cos/cos_scan_match.py` | 整个序列范围内的阈值和规则一致性扫描 | 配置输出目录中的文本和 CSV 报告 |

论文中的其他阈值叠加图和相似度子矩阵图也可通过 `thr_viewer.ipynb` 生成。在界面中选择离子、数据库组合、能级编号范围、阈值和图片尺寸，然后点击 **Prepare PDF** 下载当前图片。

## 11. 软件要求与可复现性说明

### 主要依赖

`runtime.txt` 将 Python 3.11 记录为 Binder 目标版本。核心运行和 notebook 依赖在 `requirements.txt` 中通过兼容版本范围指定。在仓库根目录使用以下命令安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖文件包含 NumPy、pandas、h5py、SciPy、Matplotlib、seaborn、PyTorch、scikit-learn、Astropy、IPython、Jupyter Notebook、ipywidgets 和 Voilà。

`requirements.txt` 不包含 `pyatomdb`，因为只有重新运行 AtomDB 源数据收集流程时才需要它。该工作流需要单独安装：

```bash
python -m pip install pyatomdb
```

需要特定 CUDA 构建的 PyTorch 可以在执行 `pip install -r requirements.txt` 前单独安装；归档模型和 Viewer 也可在仅使用 CPU 的环境中运行。

### 可复现性说明

- 除非 notebook 另有说明，否则请从仓库根目录运行脚本和 notebook。
- 使用归档的 `clean_data/` 文件以及 `map_data/cfg_ts/` 中按序列保存的模型复现匹配工作流。
- 不同离子的数据库覆盖范围不同；请选择该离子实际包含的数据库组合。
- 在不同硬件和库版本上重新训练可能产生细微差异。复现论文报告的相似度结果时，请使用归档模型文件。
- 只有重新执行数据收集阶段时才需要安装完整的外部源数据库。

## 12. 数据来源、权利说明与联系方式

### 仓库权利说明

CFADB-XCOS 公开发布的目的是支持科学审阅、可复现性研究以及配套交互式 Viewer 的使用。本仓库未为原创 CFADB-XCOS 源代码和文档提供单独的开源许可证。除非另有说明，这些材料的著作权由作者保留。

代码和文档在 GitHub 或 Zenodo 上公开，并不自动授予他人重新发布、修改或将其中大部分内容整合进其他项目的权利。此类复用应事先联系作者。仓库的访问和使用同时受相应托管平台条款的约束。

### 第三方原子数据

`clean_data/` 下的标准化 HDF5 文件包含源自以下资源的原子能级和跃迁数据：

- **AtomDB/APED v3.1.3：** <https://www.atomdb.org/>
- **CHIANTI v11.0.2：** <https://www.chiantidatabase.org/chianti_download.html>
- **SPEX/SPEXACT v3.08.01：** <https://zenodo.org/records/12771915>
- **SPEX/SPEXACT v3.08.03：** 本归档中保留的补充标准化数据

源数据被转换为统一的 CFADB-XCOS HDF5 格式，以进行跨数据库科学比较。该标准化过程不会转移或改变底层原子数据的所有权、著作权、数据库权利、许可条件或署名要求。

底层数据的相关权利仍属于各数据库开发者、机构和原始数据提供者。本仓库不会额外授予第三方原子数据的使用权。用户仍需自行遵守原始提供方适用的条款和署名要求。

CHIANTI 用户应按照 CHIANTI 团队的要求进行致谢和引用。SPEX 团队已确认允许在本研究归档中收录由 SPEX 3.08 数据衍生的内容；用户应引用相应的 SPEX Zenodo 仓库和相关 SPEX 论文。使用 AtomDB 衍生数据时，应引用相关 AtomDB/APED 论文并注明所使用的数据库版本。

配套论文包含 AtomDB、CHIANTI 和 SPEX 的主要科学参考文献、数据库说明和致谢。

### 衍生数据集名称

标准化数据集使用以下名称：

- `lev_atomdb313` 和 `tran_atomdb313`
- `lev_chianti1102` 和 `tran_chianti1102`
- `lev_spex30801` 和 `tran_spex30801`
- `lev_spex30803` 和 `tran_spex30803`

由于不同数据库对各离子的覆盖范围不同，部分离子文件不包含所有数据集。

### 联系方式

如有科学问题、复用请求或关于归档数据的问题，请联系 **Qirui Long**：`longqr24@mails.tsinghua.edu.cn`，或查阅论文中的通讯作者信息。
