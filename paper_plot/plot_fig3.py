"""Figure 3: Li-like configuration-pair cosine similarities."""
from pathlib import Path
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import os
import re
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import __main__

from core.encoder import CfgEncoder

# -------------------------
# paths
# -------------------------
VOCAB = str(ROOT/"models/li-like/cfg_vocab.pkl")
MODEL = str(ROOT/"models/li-like/cfg_ts_model.pth")
PAIRS = str(ROOT/"models/training_pairs/li_like_pairs.tsv")
FIG_DIR = str(HERE/"photos")
os.makedirs(FIG_DIR, exist_ok=True)

device = "cpu"
MAX_LEN = 30
SEED = 0
PER_REL = 3000
NEG_N = 5000

# -------------------------
# helpers
# -------------------------
def tokenize_cfg(cfg, char2idx, max_len=30):
    cfg = str(cfg).strip().lower()
    pad = char2idx["[PAD]"]
    unk = char2idx["[UNK]"]
    ids = [char2idx.get(ch, unk) for ch in cfg]
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

def encode_cfg_list(model, cfg_list, char2idx, max_len, device, batch_size=512):
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(cfg_list), batch_size):
            batch = cfg_list[i:i+batch_size]
            toks = torch.stack([tokenize_cfg(c, char2idx, max_len) for c in batch]).to(device)
            emb = model(toks)
            emb = F.normalize(emb, dim=1)
            outs.append(emb.cpu())
    return torch.cat(outs, dim=0)

def last_n(cfg: str):
    s = str(cfg).strip().lower()
    last = s.split(".")[-1] if "." in s else s
    m = re.match(r"(\d+)", last)
    return int(m.group(1)) if m else None
# -------------------------
# load model
# -------------------------
with open(VOCAB, "rb") as f:
    char2idx, idx2char = pickle.load(f)

__main__.CfgEncoder = CfgEncoder
model = torch.load(MODEL, map_location=device, weights_only=False)
model.eval()

df = pd.read_csv(PAIRS, sep="	")
df["anchor"] = df["anchor"].astype(str).str.lower()
df["positive"] = df["positive"].astype(str).str.lower()
df["relation"] = df["relation"].astype(str)

# -------------------------
# positive relations
# -------------------------
scores = {}
rels = ["hard", "shortening", "medium", "soft"]

for rel in rels:
    d = df[df["relation"] == rel]
    if len(d) == 0:
        scores[rel] = np.array([])
        continue

    d = d.sample(n=min(PER_REL, len(d)), random_state=SEED)
    A = d["anchor"].tolist()
    P = d["positive"].tolist()

    embA = encode_cfg_list(model, A, char2idx, MAX_LEN, device)
    embP = encode_cfg_list(model, P, char2idx, MAX_LEN, device)

    cos = (embA * embP).sum(dim=1).numpy()
    scores[rel] = cos

# -------------------------
# negative pairs: |Δn_last| >= 2
# -------------------------
rng = random.Random(SEED)
pool = sorted(set(df["anchor"].tolist()) | set(df["positive"].tolist()))
pool_n = [last_n(c) for c in pool]

by_n = {}
for i, n in enumerate(pool_n):
    if n is not None:
        by_n.setdefault(n, []).append(i)

neg_a = []
neg_b = []

for _ in range(NEG_N):
    a = rng.choice(pool)
    n0 = last_n(a)

    if n0 is None or not by_n:
        b = rng.choice(pool)
    else:
        cand_ns = [n for n in by_n.keys() if abs(n - n0) >= 2]
        if not cand_ns:
            b = rng.choice(pool)
        else:
            nt = rng.choice(cand_ns)
            b = pool[rng.choice(by_n[nt])]

    neg_a.append(a)
    neg_b.append(b)

embA = encode_cfg_list(model, neg_a, char2idx, MAX_LEN, device)
embB = encode_cfg_list(model, neg_b, char2idx, MAX_LEN, device)
scores["negative"] = (embA * embB).sum(dim=1).numpy()
# Keep the plotted numerical values alongside the reproduction script.
pd.DataFrame([(rel,float(value)) for rel,values in scores.items() for value in values],
    columns=['relation','cosine_similarity']).to_csv(HERE/'data/fig03_scores.csv', index=False)
# -------------------------
# plot: boxplot
# -------------------------
order = ["hard", "shortening", "medium", "soft", "negative"]
display_names = {
    "hard": "hard",
    "shortening": "shortening",
    "medium": "medium",
    "soft": "soft",
    "negative": "negative",
}
plot_order = [k for k in order if len(scores[k]) > 0]
data = [scores[k] for k in plot_order]
labels = [display_names[k] for k in plot_order]
medians = [np.median(scores[k]) for k in plot_order]
mins = [np.min(scores[k]) for k in plot_order]
maxs = [np.max(scores[k]) for k in plot_order]

fig, ax = plt.subplots(figsize=(6, 4))
box = ax.boxplot(data, labels=labels, showfliers=True)
ax.set_ylabel("Cosine similarity")
ax.grid(axis="y", alpha=0.3)
ax.set_xlim(0.5, len(plot_order) + 0.9)
all_vals = np.concatenate(data)
ymin, ymax = np.min(all_vals), np.max(all_vals)
ypad = (ymax - ymin) * 0.05 if ymax > ymin else 0.05
ax.set_ylim(ymin - ypad, ymax + ypad)

# Annotate each box with the median, maximum, and minimum values to two decimal places
for i, (med, vmin, vmax) in enumerate(zip(medians, mins, maxs), start=1):
    ax.text(i + 0.28, med, f"{med:.2f}", va="center", ha="left", fontsize=9)
    ax.text(i-0.12, vmax+0.015, f"{vmax:.2f}", va="bottom", ha="left", fontsize=9)
    ax.text(i-0.12, vmin-0.015, f"{vmin:.2f}", va="top", ha="left", fontsize=9)

fig.tight_layout()

pdf_path = os.path.join(FIG_DIR, "fig03_configuration_similarity.pdf")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

for k in plot_order:
    arr = scores[k]
    print(f"{display_names[k]:12s} mean={arr.mean():.4f}  median={np.median(arr):.4f}  n={len(arr)}")

print("saved:", pdf_path)
