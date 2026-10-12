# Reported vs Ours — Colatron reproduction + DCTTS extension

Human MOS (paper) is from 10 Stanford raters (Table 2). Our primary automated stand-in is **UTMOS** (predicted MOS); a small human MOS test is optional and reported only if collected. All objective metrics are on the 25 held-out test clips. Lower MCD/CER/WER/specL1 = better; higher UTMOS = better.

| Model | Reported MOS (paper) | Our UTMOS | Our human MOS | MCD | CER | WER | spec L1 |
|---|---|---|---|---|---|---|---|
| Ground truth (ref) | — | 4.34 | — | 0.05 | 0.040 | 0.052 | — |
| GT + Griffin-Lim (vocoder ceiling, ref) | — | 3.01 | — | 23.16 | 0.041 | 0.055 | — |
| SVR (poly) | 1.0 | 1.31 | pending | 42.41 | 0.834 | 0.985 | 0.1612 |
| SVR (linear) | — | 1.30 | — | 53.03 | 0.891 | 0.993 | 0.2130 |
| Simple NN | 1.7 | 1.39 | pending | 40.99 | 0.934 | 0.996 | 0.1570 |
| Seq2Seq + attention | 2.5 | 1.63 | pending | 51.34 | 0.929 | 0.996 | 0.1689 |
| DCTTS (extension) | — | 1.44 | pending | 51.44 | 0.915 | 1.000 | 0.1886 |

*Reference: the paper also cites Tacotron's MOS of 3.82 for context only.*

## Ranking check

- Paper ranking: SVR (1.0) < NN (1.7) < Seq2Seq (2.5).
- Our UTMOS ranking (worst→best): svr < nn < seq2seq.

## DCTTS vs Seq2Seq — Wilcoxon signed-rank (paired, 25 utts)

- **CER**: DCTTS=0.915 vs Seq2Seq=0.929; W=91.0, p=0.3945
- **UTMOS**: DCTTS=1.441 vs Seq2Seq=1.633; W=71.0, p=0.0125
