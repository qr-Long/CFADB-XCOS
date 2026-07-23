# -*- coding: utf-8 -*-
"""
cfg_ts.py

Keep the legacy API and output paths intact so existing callers continue to work.
The training flow now uses pairs.tsv (Hard/Medium/Shortening-A/Soft) and explicitly uses negative examples.

Input:
  ./map_data/cfg_ts/train_data/{h/he/li/be/ne}_like_pairs.tsv

Output (kept compatible with the legacy layout):
  ./map_data/cfg_ts/{ion-like}/cfg_vocab.pkl
  ./map_data/cfg_ts/{ion-like}/cfg_ts_model.pth
"""
import os, re, sys, pickle, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
try:
    from .ion_config import (
        ION_CONFIG,
        get_electron_count,
        get_ion_info,
        get_shorten_prefixes,
        normalize_ion_type,
    )
except ImportError:  # Allow running the script directly with python cfg_ts.py
    from ion_config import (  # type: ignore
        ION_CONFIG,
        get_electron_count,
        get_ion_info,
        get_shorten_prefixes,
        normalize_ion_type,
    )

# Preserve the legacy sys.path behavior
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directory)

SPECIALS = ["[PAD]", "[UNK]"]
ALLOWED  = list("abcdefghijklmnopqrstuvwxyz0123456789.() ")

def get_last_orbital(cfg: str) -> str:
    parts = str(cfg).split(".")
    return parts[-1] if parts else ""

def last_n(cfg: str):
    m = re.match(r"(\d+)", get_last_orbital(cfg))
    return int(m.group(1)) if m else None

def build_vocab_from_pairs(df_pairs: pd.DataFrame):
    corpus = (df_pairs["anchor"].astype(str).str.lower() + df_pairs["positive"].astype(str).str.lower()).sum()
    chars_from_data = sorted(set(corpus))
    vocab = SPECIALS + sorted(set(chars_from_data) | set(ALLOWED))
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    assert char2idx["[PAD]"] == 0 and char2idx["[UNK]"] == 1
    return char2idx, idx2char, vocab

def tokenize_cfg(cfg, char2idx, max_len=30):
    cfg = str(cfg).strip().lower()
    tokens = [char2idx.get(ch, char2idx["[UNK]"]) for ch in cfg]
    if len(tokens) < max_len:
        tokens += [char2idx["[PAD]"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens)

# ---- Preserve the legacy interface: CfgDataset ----
class CfgDataset(Dataset):
    def __init__(self, cfg_list, char2idx, max_len=30):
        self.cfg_list = cfg_list
        self.char2idx = char2idx
        self.max_len = max_len
    def __len__(self): return len(self.cfg_list)
    def __getitem__(self, idx):
        cfg = str(self.cfg_list[idx]).strip().lower()
        return tokenize_cfg(cfg, self.char2idx, self.max_len), cfg

# ---- Preserve the legacy structure: BiLSTM -> mean -> fc -> 128 ----
class CfgEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = out.mean(dim=1)
        return self.fc(out)

# ---- Visualization helpers are retained (not used by default training) ----
_ORBITAL_PATTERN_FOR_PLOT = re.compile(
    r"(\d+)\s*([spdfghijklmno])\s*(\d*)",
    re.IGNORECASE
)


def _infer_seq_from_begin(begin):
    """Infer a common iso-electronic sequence from the electron count; used only for automatic plot completion."""
    try:
        b = int(begin)
    except Exception:
        return None
    return {
        1: "h-like",
        2: "he-like",
        3: "li-like",
        4: "be-like",
        10: "ne-like",
    }.get(b)


def _resolve_plot_seq_and_len(seq=None, begin=None, default_len=2):
    """
    Resolve the plotting sequence and target electron count consistently.
    Priority: explicit seq > inferred from begin > begin > default_len.
    """
    plot_seq = None
    if seq is not None:
        try:
            plot_seq = normalize_ion_type(seq)
        except Exception:
            plot_seq = None

    if plot_seq is None:
        plot_seq = _infer_seq_from_begin(begin)

    if plot_seq is not None:
        try:
            target_len = get_electron_count(plot_seq)
            return plot_seq, int(target_len)
        except Exception:
            pass

    try:
        target_len = int(begin) if begin is not None else int(default_len)
    except Exception:
        target_len = int(default_len)
    return plot_seq, target_len


def _clean_cfg_for_plot_parse(cfg: str) -> str:
    """
    Clean cfg strings for plot-label parsing:
    - Prefer the existing _drop_terms_for_key helper to remove terms like (3P) and (2D);
    - Keep and expand real configuration groups such as 1s.(2s.2p2.(2D)).
    """
    text = str(cfg).replace(" ", "").lower()
    drop_terms = globals().get("_drop_terms_for_key")
    if callable(drop_terms):
        try:
            cleaned = drop_terms(text)
            if cleaned:
                text = cleaned
        except Exception:
            pass
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\.+", ".", text)
    return text.strip(".")


def _is_limit_marker_for_plot(token: str) -> bool:
    """Treat limit as a special placeholder in plot labels rather than as a normal orbital."""
    return str(token).strip().lower() == "limit"


def _cfg_items_for_plot(cfg: str):
    """
    Parse a cfg string into sequence elements for plot labels.

    Normal orbitals are expanded to their principal quantum number n; limit is preserved as the string "limit".
    Examples:
      1s.2p.limit -> [1, 2, "limit"]
      2p.limit    -> [2, "limit"]
      1s2.limit   -> [1, 1, "limit"]
    """
    text = _clean_cfg_for_plot_parse(cfg)
    items = []
    for token in [x for x in text.split(".") if x]:
        if _is_limit_marker_for_plot(token):
            items.append("limit")
            continue
        for n_str, _, occ_str in _ORBITAL_PATTERN_FOR_PLOT.findall(token):
            n = int(n_str)
            occ = int(occ_str) if occ_str not in ("", None) else 1
            # This is only to expand the n-sequence and apply a permissive upper bound to avoid an excessively long list if the string is malformed.
            occ = max(1, min(occ, 30))
            items.extend([n] * occ)
    return items


def _electron_n_list_for_plot(cfg: str):
    """Compatibility helper: return only the principal quantum numbers of ordinary electrons, excluding limit."""
    return [x for x in _cfg_items_for_plot(cfg) if x != "limit"]


def _split_cfg_tokens_for_plot(cfg: str):
    text = _clean_cfg_for_plot_parse(cfg)
    return [x for x in text.split(".") if x]


def _prefix_overlaps_current_cfg(prefix: str, cfg: str) -> bool:
    """Avoid prepending 2s2 to a cfg that already starts with 2s2."""
    p_tokens = _split_cfg_tokens_for_plot(prefix)
    c_tokens = [x for x in _split_cfg_tokens_for_plot(cfg) if not _is_limit_marker_for_plot(x)]
    if not p_tokens or not c_tokens:
        return False
    if c_tokens[:len(p_tokens)] == p_tokens:
        return True
    return c_tokens[0] == p_tokens[-1]


def _format_plot_sequence(items):
    """Format [1, 2, 'limit'] as (1, 2, limit) to avoid Python string quoting issues."""
    return "(" + ", ".join("limit" if x == "limit" else str(x) for x in items) + ")"


def _complete_n_sequence_with_closed_core(cfg: str, seq=None, begin=None, default_len=2, strict_seq=True):
    """
    Parse the n-sequence / limit-sequence used for plotting.

    Main behavior:
    1. When a cfg is abbreviated, prefer filling in the closed-core prefix according to the current seq.
       For example Li-like: 2p -> 1s2.2p -> (1, 1, 2).
    2. .limit is treated as the final special placeholder.
       For example Li-like: 1s.2p.limit -> (1, 2, limit);
       Be-like: 2p.limit -> 1s2.2p.limit -> (1, 1, 2, limit).
    """
    plot_seq, target_len = _resolve_plot_seq_and_len(seq=seq, begin=begin, default_len=default_len)
    items = _cfg_items_for_plot(cfg)

    if not items:
        return None, "UNK"

    if len(items) == target_len:
        return items, None

    if plot_seq is not None and len(items) < target_len:
        has_limit = any(x == "limit" for x in items)
        missing = target_len - len(items)
        for prefix in get_shorten_prefixes(plot_seq):
            prefix_items = _cfg_items_for_plot(prefix)
            if not prefix_items:
                continue

            # Ordinary cfg: accept only a full closed-core prefix that fills the gap exactly.
            # Limit cfg: limit occupies the last sequence position, so sometimes only part of the closed-core prefix is needed.
            # For example Li-like: 2p.limit -> 1s.2p.limit -> (1, 2, limit);
            #      Be-like: 2p.limit -> 1s2.2p.limit -> (1, 1, 2, limit).
            if len(prefix_items) == missing:
                prefix_fill = prefix_items
            elif has_limit and len(prefix_items) > missing:
                prefix_fill = prefix_items[:missing]
            else:
                continue

            if _prefix_overlaps_current_cfg(prefix, cfg):
                continue
            return prefix_fill + items, None

    if strict_seq:
        return None, f"INVALID({len(items)}e/{target_len}e)"

    # Compatibility behavior: if seq is not checked strictly, fill the remaining positions with the last ordinary n; if the sequence ends in limit, keep limit at the end.
    if len(items) < target_len:
        limit_tail = items[-1] == "limit"
        if limit_tail:
            body = items[:-1]
            if body:
                body = body + [body[-1]] * (target_len - len(items))
                items = body + ["limit"]
        else:
            items = items + [items[-1]] * (target_len - len(items))
    return items[:target_len], None


def _parse_cfg_n_sequence(cfg, begin=None, default_len=2, seq=None, strict_seq=True):
    items, special_label = _complete_n_sequence_with_closed_core(
        cfg,
        seq=seq,
        begin=begin,
        default_len=default_len,
        strict_seq=strict_seq,
    )
    if special_label is not None:
        return special_label
    return _format_plot_sequence(items)

def plot_cfg_features_tsne_past(features, cfg_list, sources=None, epoch=0, perplexity=30,
                          random_state=42, max_points=None, begin=None, save_dir="./map_data/cfg_ts",
                          seq=None, strict_seq=True):
    if hasattr(features, "detach"): features = features.detach().cpu().numpy()
    else: features = np.asarray(features)
    if sources is None: sources = ["default"]*len(cfg_list)
    if max_points is not None and max_points < len(features):
        features = features[:max_points]; cfg_list = cfg_list[:max_points]; sources = sources[:max_points]
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(features)
    labels = [_parse_cfg_n_sequence(c, begin=begin, seq=seq, strict_seq=strict_seq) for c in cfg_list]
    unique_labels_sorted = sorted(set(labels))
    tab20_colors = plt.cm.get_cmap("tab20").colors[:20]
    custom_colors = ["#ff1414", "#7601b1", "#ff4d00", "#800000"]
    colors_24 = list(tab20_colors) + custom_colors
    cmap = ListedColormap(colors_24, name="my24")
    label_to_color = {lb: cmap(i % cmap.N) for i, lb in enumerate(unique_labels_sorted)}
    marker_list = ["o","^","s","D","*","P","X","v","<",">"]
    unique_sources = sorted(set(sources))
    source_to_marker = {src: marker_list[i % len(marker_list)] for i, src in enumerate(unique_sources)}
    plt.figure(figsize=(16,9))
    for point,label,src in zip(reduced,labels,sources):
        plt.scatter(point[0],point[1],color=label_to_color.get(label,"gray"),
                    marker=source_to_marker[src],s=40,alpha=0.75,edgecolor="k",linewidth=0.2)
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tsne_epoch{epoch}.pdf"))
    return reduced

def plot_cfg_features_tsne(
    features,
    cfg_list,
    sources=None,
    ions=None,
    databases=None,
    epoch=0,
    perplexity=30,
    random_state=42,
    max_points=None,
    begin=None,
    save_dir="./map_data/cfg_ts",
    show_source_legend=False,
    max_label_legend=None,   # None = show all labels
    seq=None,
    strict_seq=True,
    figsize=None,
    fig_width=16,
    fig_height=9,
    label_legend_ncol=2,
    legend_right=0.68,
    print_invalid_cfg=True,
    invalid_log_path=None,
):
    """
    t-SNE visualization:
    - Colors: correspond to the n-sequence label; abbreviated cfg values are auto-completed with the closed-core prefix based on seq.
    - Point shapes: correspond to the source (if multiple sources are provided)
    - Right-side legend: clearly shows which n-sequence each color corresponds to; defaults to two columns
    - figsize / fig_width / fig_height control the figure size; default is (16, 9)
    - ions / databases can be passed optionally; if INVALID/UNK labels remain, they are printed and saved to invalid_cfg_epoch*.tsv
    """

    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    else:
        features = np.asarray(features)

    if sources is None:
        sources = ["default"] * len(cfg_list)

    if ions is None:
        ions = [str(seq) if seq is not None else "unknown"] * len(cfg_list)

    if databases is None:
        databases = list(sources)

    if max_points is not None and max_points < len(features):
        features = features[:max_points]
        cfg_list = cfg_list[:max_points]
        sources = sources[:max_points]
        ions = ions[:max_points]
        databases = databases[:max_points]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto"
    )
    reduced = tsne.fit_transform(features)

    # Color labels are based on the n-sequence; limit is displayed as a special label such as (1, 2, limit)
    labels = [_parse_cfg_n_sequence(c, begin=begin, seq=seq, strict_seq=strict_seq) for c in cfg_list]

    invalid_rows = []
    for i, (cfg, label, ion, db, src) in enumerate(zip(cfg_list, labels, ions, databases, sources)):
        if str(label).startswith("INVALID") or str(label) == "UNK":
            invalid_rows.append({
                "index": i,
                "cfg": cfg,
                "label": label,
                "ion": ion,
                "database": db,
                "source": src,
                "seq": seq,
            })

    if invalid_rows and print_invalid_cfg:
        print("\n[plot_cfg_features_tsne] Found invalid cfg labels:")
        for row in invalid_rows:
            print(
                f"  index={row['index']}, "
                f"cfg={row['cfg']}, "
                f"label={row['label']}, "
                f"ion={row['ion']}, "
                f"database={row['database']}, "
                f"source={row['source']}, "
                f"seq={row['seq']}"
            )

    if invalid_rows:
        if invalid_log_path is None:
            invalid_log_path = os.path.join(save_dir, f"invalid_cfg_epoch{epoch}.tsv")
        invalid_dir = os.path.dirname(invalid_log_path)
        if invalid_dir:
            os.makedirs(invalid_dir, exist_ok=True)
        pd.DataFrame(invalid_rows).to_csv(invalid_log_path, sep="\t", index=False)
        if print_invalid_cfg:
            print(f"[plot_cfg_features_tsne] Invalid cfg list saved to: {invalid_log_path}")

    unique_labels_sorted = sorted(set(labels))

    # Use more colors so categories do not repeat too quickly when there are many of them
    colors = (
        list(plt.cm.tab20.colors)
        + list(plt.cm.tab20b.colors)
        + list(plt.cm.tab20c.colors)
    )
    cmap = ListedColormap(colors, name="my_many_colors")
    label_to_color = {lb: cmap(i % cmap.N) for i, lb in enumerate(unique_labels_sorted)}

    # source -> marker
    marker_list = ["o", "^", "s", "D", "*", "P", "X", "v", "<", ">"]
    unique_sources = sorted(set(sources))
    source_to_marker = {src: marker_list[i % len(marker_list)] for i, src in enumerate(unique_sources)}

    if figsize is None:
        figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each point individually
    for point, label, src in zip(reduced, labels, sources):
        ax.scatter(
            point[0],
            point[1],
            color=label_to_color.get(label, "gray"),
            marker=source_to_marker[src],
            s=42,
            alpha=0.78,
            edgecolor="k",
            linewidth=0.25
        )

    ax.set_title(f"t-SNE Visualization (epoch {epoch})", fontsize=16)
    ax.set_xlabel("t-SNE Dim 1", fontsize=12)
    ax.set_ylabel("t-SNE Dim 2", fontsize=12)
    ax.grid(True, alpha=0.25)

    # -------------------------
    # Color legend: n-sequence labels
    # -------------------------
    legend_labels = unique_labels_sorted
    if max_label_legend is not None:
        legend_labels = legend_labels[:max_label_legend]

    label_handles = []
    for lb in legend_labels:
        handle = Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=label_to_color[lb],
            markeredgecolor="k",
            markeredgewidth=0.3,
            markersize=8,
            label=lb
        )
        label_handles.append(handle)

    if max_label_legend is not None and len(unique_labels_sorted) > max_label_legend:
        label_handles.append(
            Line2D([0], [0], linestyle="", label="...")
        )

    legend1 = ax.legend(
        handles=label_handles,
        title="n-sequence labels",
        bbox_to_anchor=(1.02, 0.95),
        loc="upper left",
        frameon=True,
        borderaxespad=0.0,
        fontsize=10,
        title_fontsize=11,
        ncol=label_legend_ncol
    )
    ax.add_artist(legend1)

    # -------------------------
    # Optional: source legend
    # -------------------------
    legend2 = None
    if show_source_legend and len(unique_sources) > 1:
        source_handles = []
        for src in unique_sources:
            handle = Line2D(
                [0], [0],
                marker=source_to_marker[src],
                linestyle="",
                color="gray",
                markeredgecolor="k",
                markeredgewidth=0.3,
                markersize=8,
                label=str(src)
            )
            source_handles.append(handle)

        legend2 = ax.legend(
            handles=source_handles,
            title="Data sources",
            bbox_to_anchor=(1.02, 0.45),
            loc="upper left",
            frameon=True,
            borderaxespad=0.0,
            fontsize=10,
            title_fontsize=11,
            ncol=1
        )

    os.makedirs(save_dir, exist_ok=True)

    # Reserve extra space on the right for the legend to make it more stable than tight_layout; legend_right can be tuned based on the figure width.
    fig.subplots_adjust(right=legend_right)

    # png_path = os.path.join(save_dir, f"tsne_epoch{epoch}.png")
    pdf_path = os.path.join(save_dir, f"tsne_epoch{epoch}.pdf")

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        bbox_extra_artists=tuple(x for x in (legend1, legend2) if x is not None)
    )
    plt.close(fig)

    return reduced

def plot_losses(loss_history, save_dir="./map_data/cfg_ts"):
    epochs = range(1, len(loss_history)+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, loss_history, label="Total Loss", color="black", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "cfg_ts_loss.png"), dpi=300)

# ---- When sampling negatives, avoid using members of the same group as negatives: ignore terms and group by the current seq's closed-core shortening ----
def _clean_dots_for_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.+", ".", s)
    return s.strip(".")


def _find_matching_paren_for_key(s: str, start: int) -> int:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _is_term_content_for_key(inner: str) -> bool:
    inner = str(inner).strip().lower()
    if not inner or "." in inner or "(" in inner or ")" in inner:
        return False
    return bool(re.fullmatch(r"\d*[a-z]+\d*", inner))


def _flatten_group_parens_for_key(cfg: str) -> str:
    s = _clean_dots_for_key(cfg)

    def rec(text: str) -> str:
        out = []
        i = 0
        while i < len(text):
            if text[i] != "(":
                out.append(text[i])
                i += 1
                continue
            j = _find_matching_paren_for_key(text, i)
            if j < 0:
                out.append(text[i])
                i += 1
                continue
            inner = rec(text[i + 1:j])
            if _is_term_content_for_key(inner):
                out.append(f"({inner})")
            else:
                out.append(inner)
            i = j + 1
        return "".join(out)

    return _clean_dots_for_key(rec(s))


def _drop_terms_for_key(cfg: str) -> str:
    s = _flatten_group_parens_for_key(cfg).replace(".(", "(")
    s = re.sub(r"\([^()]*\)", "", s)
    return _clean_dots_for_key(s)


def _contains_limit_for_key(cfg: str) -> bool:
    s = _drop_terms_for_key(cfg)
    return any(tok == "limit" or tok.endswith("limit") for tok in s.split("."))


def _strip_prefix_once_for_key(s: str, prefix: str) -> str:
    s = _clean_dots_for_key(s)
    prefix = _clean_dots_for_key(prefix)
    if not prefix or s == prefix:
        return ""
    p = prefix + "."
    if s.startswith(p):
        return _clean_dots_for_key(s[len(p):])
    return ""


def medium_key(cfg: str, seq: str = "li-like") -> str:
    """
    Used for negative-sampling grouping:
      - Ignore parent-term / term markers
      - Expand nested cfg groups
      - Group by the current seq's closed-core shortening family
      - .limit remains distinct and is not merged with ordinary cfg entries such as 2p
    """
    seq = normalize_ion_type(seq)
    s = _drop_terms_for_key(cfg)
    if not s:
        return ""
    if _contains_limit_for_key(s):
        return f"limit:{s}"

    prefixes = get_shorten_prefixes(seq)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            rest = _strip_prefix_once_for_key(s, prefix)
            if rest:
                s = rest
                changed = True
                break
    return _clean_dots_for_key(s)


class PairDataset(Dataset):
    def __init__(self, df_pairs: pd.DataFrame, char2idx, max_len=30, seed=0, seq="li-like"):
        self.df = df_pairs.reset_index(drop=True)
        self.char2idx = char2idx
        self.max_len = max_len
        self.rng = random.Random(seed)
        self.seq = normalize_ion_type(seq)
        pool = set(self.df["anchor"].astype(str).str.lower()) | set(self.df["positive"].astype(str).str.lower())
        self.pool = sorted(pool)
        self.by_n = {}
        for i, c in enumerate(self.pool):
            n = last_n(c)
            if n is not None:
                self.by_n.setdefault(n, []).append(i)

    def __len__(self): return len(self.df)

    def _sample_neg(self, anchor_cfg: str):
        n0 = last_n(anchor_cfg)
        k0 = medium_key(anchor_cfg, self.seq)
        cand_ns = [n for n in self.by_n.keys() if (n0 is not None and abs(n - n0) >= 2)] if n0 is not None else []
        if not cand_ns:
            cand_ns = list(self.by_n.keys())
        for _ in range(60):
            nt = self.rng.choice(cand_ns)
            idx = self.rng.choice(self.by_n[nt])
            c = self.pool[idx]
            if medium_key(c, self.seq) != k0:
                return c
        return self.rng.choice(self.pool)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a = str(row["anchor"]).lower()
        p = str(row["positive"]).lower()
        w = float(row["weight"])
        n = self._sample_neg(a)
        return (tokenize_cfg(a, self.char2idx, self.max_len),
                tokenize_cfg(p, self.char2idx, self.max_len),
                tokenize_cfg(n, self.char2idx, self.max_len),
                torch.tensor(w, dtype=torch.float32))

def weighted_info_nce(emb_a, emb_p, w, temperature=0.1):
    a = F.normalize(emb_a, dim=-1); p = F.normalize(emb_p, dim=-1)
    sim = (a @ p.T) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_vec = F.cross_entropy(sim, labels, reduction="none")
    w = w.to(loss_vec.device)
    return (loss_vec * w).sum() / (w.sum() + 1e-8)

def triplet_cos_margin(emb_a, emb_p, emb_n, w, margin=0.2):
    a = F.normalize(emb_a, dim=-1); p = F.normalize(emb_p, dim=-1); n = F.normalize(emb_n, dim=-1)
    sim_pos = (a*p).sum(dim=1); sim_neg = (a*n).sum(dim=1)
    loss_vec = torch.clamp(margin + sim_neg - sim_pos, min=0.0)
    w = w.to(loss_vec.device)
    return (loss_vec * w).sum() / (w.sum() + 1e-8)

def train_from_pairs(seq="li-like",
                     pairs_path=None,
                     begin=None,
                     epochs=100, lr=1e-3, batch_size=256, max_len=30,
                     temperature=0.1, triplet_alpha=0.5,
                     visualize_every=0, device=None,
                     cfg_ts_root="./map_data/cfg_ts"):

    # Normalize seq notation: h / 01 / h_like / h-like all become h-like
    seq = normalize_ion_type(seq)

    # Reuse the single ion configuration defined in ion_config.py
    ion_info = get_ion_info(seq)

    # begin is inferred from seq by default
    if begin is None:
        begin = ion_info["electron_count"]

    # pairs_path is inferred from seq by default
    if pairs_path is None:
        pairs_path = os.path.join(
            cfg_ts_root,
            "train_data",
            ion_info["pair_file"]
        )

    # The model output directory is inferred from seq by default
    out_dir = os.path.join(
        cfg_ts_root,
        ion_info["model_dir_name"]
    )
    os.makedirs(out_dir, exist_ok=True)

    vocab_path = os.path.join(out_dir, "cfg_vocab.pkl")
    model_path = os.path.join(out_dir, "cfg_ts_model.pth")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"seq        = {seq}")
    print(f"begin      = {begin}")
    print(f"pairs_path = {pairs_path}")
    print(f"out_dir    = {out_dir}")
    print(f"device     = {device}")

    df = pd.read_csv(pairs_path, sep="\t")

    for c in ["anchor", "positive", "weight"]:
        if c not in df.columns:
            raise ValueError(f"pairs file missing column: {c}")

    char2idx, idx2char, vocab = build_vocab_from_pairs(df)

    with open(vocab_path, "wb") as f:
        pickle.dump((char2idx, idx2char), f)

    model = CfgEncoder(
        vocab_size=len(vocab),
        emb_dim=64,
        hidden_dim=128,
        padding_idx=char2idx["[PAD]"]
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)

    ds = PairDataset(df, char2idx, max_len=max_len, seed=0, seq=seq)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    loss_hist = []

    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0.0
        nb = 0

        for a, p, n, w in dl:
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)
            w = w.to(device)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = weighted_info_nce(
                emb_a,
                emb_p,
                w,
                temperature=temperature
            ) + triplet_alpha * triplet_cos_margin(
                emb_a,
                emb_p,
                emb_n,
                w,
                margin=0.2
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += float(loss.item())
            nb += 1

        avg = tot / max(1, nb)
        loss_hist.append(avg)

        print(f"Epoch {epoch}/{epochs}, loss={avg:.4f}")

        if (visualize_every and epoch % visualize_every == 0) or (epoch == epochs):
            with torch.no_grad():
                rng = random.Random(0)
                vis_n = min(1200, len(ds.pool))
                cfg_list = rng.sample(ds.pool, vis_n) if len(ds.pool) > vis_n else ds.pool

                toks = torch.stack([
                    tokenize_cfg(c, char2idx, max_len)
                    for c in cfg_list
                ]).to(device)

                feats = model(toks)

                plot_cfg_features_tsne(
                    feats,
                    cfg_list,
                    sources=["train"] * len(cfg_list),
                    epoch=epoch,
                    begin=begin,
                    seq=seq,
                    save_dir=cfg_ts_root,
                    show_source_legend=False,
                    max_label_legend=None
                )

    plot_losses(loss_hist, save_dir=cfg_ts_root)

    torch.save(model, model_path)

    print(f"✅ saved vocab to {vocab_path}")
    print(f"✅ saved model to {model_path}")

    return model

if __name__ == "__main__":
    train_from_pairs(
        seq="ne-like",
        epochs=100,
        visualize_every=50
    )