"""Shared data loading: builds padded symbol-ID inputs and spectrogram targets
from the cached features, using the fixed splits. All models call these so they
train and evaluate on identical data."""
import os
import json
import numpy as np

from src import config as C
from src import text as T
from src.preprocess import load_split, load_metadata, feat_path


def get_stats():
    with open(os.path.join(C.PROCESSED, "stats.json")) as f:
        return json.load(f)


def _load_split_arrays(name, meta, kind="lin"):
    ids = load_split(name)
    seqs = [T.text_to_sequence(meta[cid], seed=C.SEED + i) for i, cid in enumerate(ids)]
    specs = [np.load(feat_path(cid, kind)) for cid in ids]
    return ids, seqs, specs


def load_all(kind="lin"):
    """Returns dict with padded arrays for train/dev/test.
    X: (N, L_max) int symbol ids ; Y: (N, T_max, F) float32 ; plus lengths."""
    C.set_seed()
    meta = load_metadata()
    data = {}
    all_seqs = []
    T_max = 0
    F = None
    raw = {}
    for name in ("train", "dev", "test"):
        ids, seqs, specs = _load_split_arrays(name, meta, kind)
        raw[name] = (ids, seqs, specs)
        all_seqs += seqs
        T_max = max(T_max, max(s.shape[0] for s in specs))
        F = specs[0].shape[1]
    L_max = max(len(s) for s in all_seqs)
    for name in ("train", "dev", "test"):
        ids, seqs, specs = raw[name]
        X = T.pad_sequences(seqs, max_len=L_max)
        Y = np.zeros((len(specs), T_max, F), dtype=np.float32)
        Tlen = np.zeros(len(specs), dtype=np.int64)
        for i, s in enumerate(specs):
            Y[i, :s.shape[0]] = s
            Tlen[i] = s.shape[0]
        Xlen = np.array([min(len(s), L_max) for s in seqs], dtype=np.int64)
        data[name] = dict(ids=ids, X=X, Y=Y, Xlen=Xlen, Tlen=Tlen,
                          texts=[meta[c] for c in ids])
    data["L_max"] = L_max
    data["T_max"] = T_max
    data["F"] = F
    return data
