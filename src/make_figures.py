"""Generate all result figures into results/figures/ at dpi 200."""
import os
import sys
import csv
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.preprocess import load_split, load_metadata, feat_path
from src import text as TXT

DPI = 200
MODELS = [("svr", "SVR"), ("nn", "NN"), ("seq2seq", "Seq2Seq"), ("dctts", "DCTTS")]


def _metrics():
    rows = {}
    with open(os.path.join(C.RESULTS, "metrics.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


def fig_bars():
    rows = _metrics()
    tags = [t for t, _ in MODELS if t in rows]
    names = [n for t, n in MODELS if t in rows]
    utmos = [float(rows[t]["utmos_mean"]) for t in tags]
    cer = [float(rows[t]["cer_mean"]) for t in tags]
    x = np.arange(len(tags))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - 0.2, utmos, 0.4, label="UTMOS (↑)", color="#4C72B0")
    ax1.set_ylabel("UTMOS"); ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, cer, 0.4, label="CER (↓)", color="#C44E52")
    ax2.set_ylabel("CER")
    ax1.set_title("Model comparison: UTMOS vs CER")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "compare_bars.png"), dpi=DPI)
    plt.close(fig)


def fig_spectrograms():
    cid = load_split("test")[0]
    panels = [("ground truth", np.load(feat_path(cid, "lin")))]
    for tag, name in MODELS:
        p = os.path.join(C.AUDIO_OUT, tag, f"{cid}.spec.npy")
        if os.path.exists(p):
            panels.append((name, np.load(p)))
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n))
    if n == 1: axes = [axes]
    for ax, (name, S) in zip(axes, panels):
        ax.imshow(S.T, origin="lower", aspect="auto", cmap="magma")
        ax.set_ylabel(name, fontsize=9); ax.set_yticks([])
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"Spectrograms — {cid}")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "spectrograms.png"), dpi=DPI)
    plt.close(fig)


def fig_attention():
    s = sorted(glob.glob(os.path.join(C.FIGURES, "attention", "seq2seq_*.attn.npy")))
    d = sorted(glob.glob(os.path.join(C.FIGURES, "attention", "dctts_*.attn.npy")))
    if not s and not d:
        print("[figs] no attention maps"); return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if s:
        A = np.load(s[0])                       # (T,L)
        axes[0].imshow(A.T, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("Seq2Seq attention"); axes[0].set_xlabel("decoder frame"); axes[0].set_ylabel("encoder symbol")
    if d:
        A = np.load(d[0])                       # (L,T)
        axes[1].imshow(A, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("DCTTS attention (guided)"); axes[1].set_xlabel("decoder frame"); axes[1].set_ylabel("encoder symbol")
    fig.suptitle("Attention alignment: Seq2Seq vs DCTTS")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "attention_compare.png"), dpi=DPI)
    plt.close(fig)


def fig_loss_curves():
    fig, ax = plt.subplots(figsize=(7, 4))
    nnh = os.path.join(C.CHECKPOINTS, "nn_history.json")
    if os.path.exists(nnh):
        h = json.load(open(nnh))["history"]
        ax.plot([e["train_mse"] for e in h], label="NN train MSE")
    sh = os.path.join(C.CHECKPOINTS, "seq2seq_history.json")
    if os.path.exists(sh):
        h = json.load(open(sh))["history"]
        ax.plot([e["train_l1"] for e in h], label="Seq2Seq train L1")
    dh = os.path.join(C.CHECKPOINTS, "dctts_history.json")
    if os.path.exists(dh):
        h = json.load(open(dh))
        ax.plot(h["t2m_loss"], label="DCTTS Text2Mel loss")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title("Training loss curves"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "loss_curves.png"), dpi=DPI)
    plt.close(fig)


def fig_eda():
    meta = load_metadata()
    ids = load_split("train") + load_split("dev") + load_split("test")
    durs, slens = [], []
    for cid in ids:
        S = np.load(feat_path(cid, "lin"))
        durs.append(S.shape[0] * C.HOP / C.SR)
        slens.append(len(TXT.text_to_sequence(meta[cid], seed=C.SEED)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(durs, bins=20, color="#4C72B0"); axes[0].set_title("Clip duration (s)"); axes[0].set_xlabel("seconds")
    axes[1].hist(slens, bins=20, color="#55A868"); axes[1].set_title("Symbol-sequence length"); axes[1].set_xlabel("symbols")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "eda.png"), dpi=DPI)
    plt.close(fig)


def build_all():
    for fn in (fig_eda, fig_bars, fig_spectrograms, fig_attention, fig_loss_curves):
        try:
            fn(); print("[figs]", fn.__name__, "ok")
        except Exception as e:
            print("[figs]", fn.__name__, "FAILED:", e)


if __name__ == "__main__":
    build_all()
