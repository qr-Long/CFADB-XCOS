"""
cfg_embed.py

Load the trained electronic-configuration encoder and append learned
configuration embeddings to atomic energy-level tables used by the CFADB
cross-database mapping workflow.
"""

from __future__ import annotations

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Support imports from either the project root or a direct script directory.
try:
    from cfg.cfg_ts import CfgEncoder, CfgDataset
except Exception:  # pragma: no cover
    from cfg_ts import CfgEncoder, CfgDataset  # type: ignore


directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_VOCAB_PATH = os.path.join(directory, 'map_data', 'cfg_ts', 'cfg_vocab.pkl')
_cfg_vocab_cache = None


def load_cfg_vocab(vocab_path=CFG_VOCAB_PATH):
    """Load the character-to-index and index-to-character mappings."""
    global _cfg_vocab_cache
    if _cfg_vocab_cache is not None and vocab_path == CFG_VOCAB_PATH:
        return _cfg_vocab_cache
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Cannot find configuration vocabulary: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    if not isinstance(vocab, tuple) or len(vocab) != 2:
        raise ValueError('cfg_vocab.pkl does not contain the expected two-item tuple')
    if vocab_path == CFG_VOCAB_PATH:
        _cfg_vocab_cache = vocab
    return vocab


def load_cfg_ts_model(model_path):
    """
    Load a trained configuration encoder and set it to evaluation mode.

    Parameters
    ----------
    model_path : str or path-like
        Path to a model saved with ``torch.save``.

    Returns
    -------
    torch.nn.Module
        The loaded configuration encoder.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find configuration model: {model_path}")

    # Expose the class name required by models serialized from script execution.
    try:
        sys.modules["__main__"].CfgEncoder = CfgEncoder
    except Exception:
        pass
    # The archived model files are trusted CFADB assets saved as complete
    # CfgEncoder objects. PyTorch 2.6 and later default to weights_only=True,
    # so explicitly request full-object loading for these bundled files.
    try:
        model = torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        # Older supported PyTorch releases do not accept weights_only.
        model = torch.load(model_path, map_location=torch.device("cpu"))

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Configuration model is not a torch.nn.Module: {type(model).__name__}"
        )
    for attr in ("embedding", "lstm", "fc"):
        if not hasattr(model, attr):
            raise AttributeError(
                f"Configuration model is missing the expected attribute: {attr}"
            )

    model.eval()

    print(f"Loaded configuration model: {model_path}")
    return model


def process_cfg(lev_df, model, char2idx=None, max_len=30, vocab_path=CFG_VOCAB_PATH):
    """
    Encode the ``cfg`` column and append configuration feature columns.

    Parameters
    ----------
    lev_df : pandas.DataFrame
        Atomic level table containing a ``cfg`` column.
    model : torch.nn.Module
        Trained configuration encoder.
    char2idx : dict, optional
        Character-to-index mapping. It is loaded from ``vocab_path`` when not
        provided.
    max_len : int, optional
        Maximum token-sequence length.
    vocab_path : str or path-like, optional
        Path to the serialized vocabulary.

    Returns
    -------
    pandas.DataFrame
        A copy of the level table with ``cfg_0``, ``cfg_1``, ... feature
        columns appended.
    """
    if char2idx is None:
        char2idx, _ = load_cfg_vocab(vocab_path)
    if "[UNK]" not in char2idx:
        raise KeyError('char2idx does not contain the [UNK] token')

    current_dataset = CfgDataset(
        cfg_list=lev_df['cfg'].tolist(),
        char2idx=char2idx,
        max_len=max_len
    )
    dataloader = DataLoader(current_dataset, batch_size=32, shuffle=False)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    def find_oov_chars(cfg_list, char2idx, vsize, limit=20):
        """Find configuration strings that map to indices outside the model vocabulary."""
        bad = []
        unk = char2idx.get('[UNK]', None)
        for s in cfg_list:
            for ch in str(s):
                idx = char2idx.get(ch, unk if unk is not None else vsize)
                if idx is None or idx >= vsize:
                    bad.append((s, ch, idx))
                    break
            if len(bad) >= limit:
                break
        return bad

    vsize = model.embedding.num_embeddings
    bad = find_oov_chars(lev_df['cfg'].tolist(), char2idx, vsize)
    if len(bad) > 0:
        print(f"Found {len(bad)} configurations with out-of-range tokens (showing up to 20):")
        for s, ch, idx in bad[:20]:
            print(f"  cfg='{s}' has '{ch}' -> idx {idx} (>= {vsize})")

    all_features = []
    with torch.no_grad():
        for tokens, _ in dataloader:
            tokens = tokens.to(device)
            features = model(tokens)
            all_features.append(features.detach().cpu().numpy())

    cfg_features = np.vstack(all_features)

    lev_df_cfg = lev_df.copy()
    for i in range(cfg_features.shape[1]):
        lev_df_cfg[f'cfg_{i}'] = cfg_features[:, i]

    return lev_df_cfg
