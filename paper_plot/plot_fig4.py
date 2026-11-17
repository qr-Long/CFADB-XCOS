"""Figure 4: Li-like configuration embeddings in a fixed t-SNE projection."""
from pathlib import Path
import sys
import argparse
import hashlib,json
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import os,re,random,pickle
RECOMPUTE = False
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from core.encoder import load_cfg_ts_model, tokenize_cfg

_ORBITAL_PATTERN_FOR_PLOT = re.compile(r'(\d+)\s*([spdfghijklmno])\s*(\d*)', re.I)
def normalize_ion_type(seq):
    return str(seq).lower().replace('_','-')
def get_electron_count(seq):
    return {'h-like':1,'he-like':2,'li-like':3,'be-like':4,'ne-like':10}[seq]
def get_shorten_prefixes(seq):
    return ('1s2',) if seq=='li-like' else ()
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
    save_dir=str(HERE/"photos"),
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
    coordinate_path = HERE/'data/fig04_coordinates.csv'
    metadata_path = coordinate_path.with_suffix('.json')
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT/'models/li-like/cfg_ts_model.pth', ROOT/'models/li-like/cfg_vocab.pkl', ROOT/'models/training_pairs/li_like_pairs.tsv']}
    if coordinate_path.exists() and not RECOMPUTE:
        metadata = json.loads(metadata_path.read_text())
        if metadata['inputs'] != inputs or metadata['csv_sha256'] != hashlib.sha256(coordinate_path.read_bytes()).hexdigest():
            raise ValueError('Stored t-SNE coordinates or inputs changed; use --recompute.')
        cached = pd.read_csv(coordinate_path)
        if cached['cfg'].tolist() != list(cfg_list):
            raise ValueError('Stored t-SNE samples differ; use --recompute.')
        reduced = cached[['x','y']].to_numpy()
    else:
        reduced = tsne.fit_transform(features)
        coordinate_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'cfg':cfg_list,'x':reduced[:,0],'y':reduced[:,1]}).to_csv(coordinate_path,index=False)
        metadata_path.write_text(json.dumps(dict(inputs=inputs,
            csv_sha256=hashlib.sha256(coordinate_path.read_bytes()).hexdigest(),
            sample_seed=0, tsne_seed=random_state, perplexity=perplexity,
            sklearn_version=sklearn.__version__, torch_version=torch.__version__),indent=2)+'\n')

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
    pdf_path = os.path.join(save_dir, "fig04_configuration_tsne.pdf")

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        bbox_extra_artists=tuple(x for x in (legend1, legend2) if x is not None)
    )
    plt.close(fig)

    return reduced


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


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute',action='store_true',help='Recalculate the t-SNE coordinates from the archived model.')
    RECOMPUTE=parser.parse_args().recompute
    pairs=pd.read_csv(ROOT/'models/training_pairs/li_like_pairs.tsv',sep='\t')
    pool=sorted(set(pairs.anchor.astype(str).str.lower())|set(pairs.positive.astype(str).str.lower()))
    configs=random.Random(0).sample(pool,min(1200,len(pool))) if len(pool)>1200 else pool
    with open(ROOT/'models/li-like/cfg_vocab.pkl','rb') as stream: vocabulary,_=pickle.load(stream)
    model=load_cfg_ts_model(ROOT/'models/li-like/cfg_ts_model.pth')
    with torch.no_grad(): features=model(torch.stack([tokenize_cfg(c,vocabulary) for c in configs]))
    plot_cfg_features_tsne(features,configs,sources=['train']*len(configs),epoch=100,begin=3,seq='li-like',save_dir=str(HERE/'photos'),print_invalid_cfg=False)
    print(HERE/'photos/fig04_configuration_tsne.pdf')
