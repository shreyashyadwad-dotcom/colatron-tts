"""Build results/comparison_table.md (Reported vs Ours) from results/metrics.csv,
plus a Wilcoxon paired test between DCTTS and Seq2Seq."""
import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

REPORTED_MOS = {"svr": "1.0", "nn": "1.7", "seq2seq": "2.5"}
DISPLAY = [
    ("ground_truth", "Ground truth (ref)"),
    ("gt_griffinlim", "GT + Griffin-Lim (vocoder ceiling, ref)"),
    ("svr", "SVR (poly)"),
    ("svr_linear", "SVR (linear)"),
    ("nn", "Simple NN"),
    ("seq2seq", "Seq2Seq + attention"),
    ("dctts", "DCTTS (extension)"),
]


def _load_metrics():
    path = os.path.join(C.RESULTS, "metrics.csv")
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


def _f(r, k, nd=3):
    try:
        v = float(r[k])
        if np.isnan(v):
            return "—"
        return f"{v:.{nd}f}"
    except Exception:
        return "—"


def _human_mos():
    agg = os.path.join(C.RESULTS, "human_mos.json")
    if os.path.exists(agg):
        with open(agg) as f:
            return json.load(f)
    return None


def _wilcoxon():
    try:
        from scipy.stats import wilcoxon
        out = {}
        for metric in ("cer", "utmos"):
            pa = os.path.join(C.RESULTS, "per_utt_dctts.json")
            pb = os.path.join(C.RESULTS, "per_utt_seq2seq.json")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                return None
            a = json.load(open(pa))[metric]; b = json.load(open(pb))[metric]
            n = min(len(a), len(b))
            if n < 2:
                continue
            stat, p = wilcoxon(a[:n], b[:n])
            out[metric] = {"stat": float(stat), "p": float(p),
                           "dctts_mean": float(np.mean(a[:n])),
                           "seq2seq_mean": float(np.mean(b[:n]))}
        return out
    except Exception as e:
        return {"error": str(e)}


def build():
    rows = _load_metrics()
    hmos = _human_mos()
    lines = []
    lines.append("# Reported vs Ours — Colatron reproduction + DCTTS extension\n")
    lines.append("Human MOS (paper) is from 10 Stanford raters (Table 2). Our primary "
                 "automated stand-in is **UTMOS** (predicted MOS); a small human MOS test "
                 "is optional and reported only if collected. All objective metrics are on "
                 "the 25 held-out test clips. Lower MCD/CER/WER/specL1 = better; higher "
                 "UTMOS = better.\n")
    hdr = ("| Model | Reported MOS (paper) | Our UTMOS | Our human MOS | MCD | CER | WER | spec L1 |")
    sep = "|" + "---|" * 8
    lines.append(hdr); lines.append(sep)
    for tag, name in DISPLAY:
        if tag not in rows:
            continue
        r = rows[tag]
        rep = REPORTED_MOS.get(tag, "—")
        hm = "—"
        if hmos and tag in hmos:
            hm = f"{hmos[tag]['mean']:.2f}±{hmos[tag]['ci95']:.2f} (n={hmos[tag]['n']})"
        elif tag in REPORTED_MOS or tag == "dctts":
            hm = "pending"
        lines.append(f"| {name} | {rep} | {_f(r,'utmos_mean',2)} | {hm} | "
                     f"{_f(r,'mcd_mean',2)} | {_f(r,'cer_mean')} | {_f(r,'wer_mean')} | "
                     f"{_f(r,'specL1_mean',4)} |")
    lines.append("\n*Reference: the paper also cites Tacotron's MOS of 3.82 for context only.*\n")

    # ranking check
    def um(tag):
        try: return float(rows[tag]["utmos_mean"])
        except Exception: return float("nan")
    lines.append("## Ranking check\n")
    lines.append(f"- Paper ranking: SVR (1.0) < NN (1.7) < Seq2Seq (2.5).")
    order = sorted([t for t in ("svr", "nn", "seq2seq") if t in rows], key=um)
    lines.append(f"- Our UTMOS ranking (worst→best): {' < '.join(order)}.")
    lines.append("")

    w = _wilcoxon()
    if w and "error" not in w:
        lines.append("## DCTTS vs Seq2Seq — Wilcoxon signed-rank (paired, 25 utts)\n")
        for metric, d in w.items():
            lines.append(f"- **{metric.upper()}**: DCTTS={d['dctts_mean']:.3f} vs "
                         f"Seq2Seq={d['seq2seq_mean']:.3f}; W={d['stat']:.1f}, p={d['p']:.4f}")
        lines.append("")
        with open(os.path.join(C.RESULTS, "wilcoxon.json"), "w") as f:
            json.dump(w, f, indent=2)

    out = os.path.join(C.RESULTS, "comparison_table.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[tables] wrote", out)
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n".join(lines)).encode("utf-8", "replace"))


if __name__ == "__main__":
    build()
