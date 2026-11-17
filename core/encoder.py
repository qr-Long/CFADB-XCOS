"""Load the archived configuration encoders and calculate configuration vectors."""
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def tokenize_cfg(cfg, char2idx, max_len=30):
    cfg = str(cfg).strip().lower()
    tokens = [char2idx.get(ch, char2idx["[UNK]"]) for ch in cfg]
    if len(tokens) < max_len:
        tokens += [char2idx["[PAD]"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens)


class CfgDataset(Dataset):
    def __init__(self, cfg_list, char2idx, max_len=30):
        self.cfg_list = cfg_list
        self.char2idx = char2idx
        self.max_len = max_len
    def __len__(self): return len(self.cfg_list)
    def __getitem__(self, idx):
        cfg = str(self.cfg_list[idx]).strip().lower()
        return tokenize_cfg(cfg, self.char2idx, self.max_len), cfg


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


def load_cfg_vocab(vocab_path):
    with open(vocab_path, 'rb') as stream:
        return pickle.load(stream)

def load_cfg_ts_model(model_path):
    # The distributed 0.3.0 models store this class under __main__.
    sys.modules['__main__'].CfgEncoder = CfgEncoder
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    return model

def process_cfg(lev_df, model, char2idx=None, max_len=30, vocab_path=None):
    if char2idx is None:
        char2idx, _ = load_cfg_vocab(vocab_path)
    configs = lev_df['cfg'].astype(str).tolist()
    device = next(model.parameters()).device
    features = []
    with torch.no_grad():
        for start in range(0, len(configs), 32):
            batch = torch.stack([tokenize_cfg(c, char2idx, max_len) for c in configs[start:start+32]])
            features.append(model(batch.to(device)).cpu().numpy())
    values = np.vstack(features) if features else np.empty((0, model.fc.out_features))
    columns = pd.DataFrame(values, columns=[f'cfg_{i}' for i in range(values.shape[1])], index=lev_df.index)
    return pd.concat([lev_df.copy(), columns], axis=1)
