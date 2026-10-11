# Colatron — Reproducing & Extending End-to-End TTS

**UE24CS352A Machine Learning — Mini-Project #20 · Team 23 · PES University, Section H**
Members: **PES2UG24CS495**, **PES2UG24CS493**

Reproduction + extension of **Wang, Yang & Li, "End-to-End Text to Speech Synthesis"**
(CS229, Autumn 2018) — [report](https://cs229.stanford.edu/proj2018/report/15.pdf) ·
[poster](https://cs229.stanford.edu/proj2018/poster/15.pdf). Both PDFs are in `reference/`,
and the authors' released source is reconciled in `reference/original_code/`.

## Overview
The paper maps text directly to a waveform and compares three models — a per-timestep **SVR**,
a flattened-output **MLP**, and an **LSTM seq2seq with attention** — trained on only ~200 LJSpeech
sentences, scored by human **MOS** (SVR 1.0 < NN 1.7 < Seq2Seq 2.5; Table 2). Its main finding is
that the seq2seq *repeats words* because attention alignment drifts.

- **Part 1 (reproduce):** all three models, the same data / preprocessing / Griffin-Lim, scored on a
  Reported-vs-Ours table with a specific explanation for every gap.
- **Part 2 (extend):** **DCTTS** (Tachibana 2018) — fully-convolutional Text2Mel + SSRN with a
  **guided-attention loss** that forces a near-diagonal alignment, directly attacking the paper's
  failure mode. Not used by the paper; same data, split and metrics.

MOS needs humans, so we approximate it **two ways and say so everywhere**: **UTMOS** (a neural
predicted-MOS) as the primary automated column, plus an optional small human listening test. We add
objective metrics: **MCD** (DTW), **CER/WER** (Whisper ASR — a direct proxy for "repeated/incomplete
words"), and spectrogram L1.

## Dataset & split
LJ Speech 1.1 (single speaker). Streamed from the HuggingFace parquet mirror (no 2.6 GB tar needed).
Fixed **200 train / 25 dev / 25 test** (`data/splits/*.txt`), **seed 42**, restricted to **1.5–5.0 s**
clips so zero-padding to the longest clip stays tractable on CPU — the same memory wall the authors
report. The audio front-end (sr 20 kHz, n_fft 2048/1025 bins, hop 250, win 1000, pre-emphasis 0.97,
Griffin-Lim 60 iters) is taken **directly from the authors' code**. All models use exactly these splits.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install --no-build-isolation openai-whisper==20231117
```
`ffmpeg` is **not** required (Whisper reads our generated WAVs via soundfile). CPU-only is fine; a GPU
is used automatically if present. First evaluation downloads the UTMOS and Whisper-base weights.

## Run
```bash
python main.py --all          # download → preprocess → train (4 models) → synth → eval → tables → figures
python main.py --demo "Hello from team twenty three"   # all 4 models, <60s on CPU, wavs in results/demo/
python main.py --step seq2seq # run a single stage: data|prep|svr|nn|seq2seq|dctts|eval|tables|figures
```
Every stage skips itself if its outputs already exist (use `--force` to recompute).

## Results
Full, auto-generated table: **`results/comparison_table.md`** (regenerated from `results/metrics.csv`).

| Model | Reported MOS (paper) | Our UTMOS | MCD | CER | WER | spec L1 |
|---|---|---|---|---|---|---|
| Ground truth (ref) | — | 4.34 | 0.05 | 0.040 | 0.052 | — |
| GT + Griffin-Lim (vocoder ceiling, ref) | — | 3.01 | 23.16 | 0.041 | 0.055 | — |
| SVR (poly) | 1.0 | 1.31 | 42.41 | 0.834 | 0.985 | 0.161 |
| SVR (linear) | — | 1.30 | 53.03 | 0.891 | 0.993 | 0.213 |
| Simple NN | 1.7 | 1.39 | 40.99 | 0.934 | 0.996 | 0.157 |
| Seq2Seq + attention | 2.5 | 1.63 | 51.34 | 0.929 | 0.996 | 0.169 |
| DCTTS (extension) | — | 1.44 | 51.44 | 0.915 | 1.000 | 0.189 |

*UTMOS is a predicted-MOS stand-in (higher = better); lower MCD/CER/WER/spec-L1 = better. Numbers
come from `results/metrics.csv`; the "human MOS" columns stay `pending` (paper used n=10 raters).*

**Headline findings (honest):**
- Our UTMOS **recovers the paper's SVR < NN < Seq2Seq ranking** (1.31 < 1.39 < 1.63).
- **Extension verdict is mixed:** DCTTS's guided attention produces a *far more diagonal/monotonic
  alignment* than seq2seq (`attention_compare.png`) — it fixes the mechanism the paper blames for
  repeated words — but on this 200-clip, CPU-shrunk budget it did **not** improve perceptual quality:
  UTMOS is significantly *lower* than seq2seq (paired Wilcoxon p=0.012) and CER is statistically
  indistinguishable (p=0.39). This matches the paper's own conclusion: 200 sentences is the binding
  constraint, not the model.

Key figures (`results/figures/`, dpi 200): `attention_compare.png` (Seq2Seq vs DCTTS — the key
figure), `compare_bars.png`, `spectrograms.png`, `loss_curves.png`, `eda.png`.

## Deviations from the paper
All flagged `ASSUMPTION` in code and tabulated in `docs/paper_spec.md`. The main ones:
- **Clip band 1.5–5.0 s, fixed 200/25/25 (seed 42)** — the paper never says which 200 sentences.
- **MOS ≈ UTMOS + a small human test** — human MOS can't be fully automated.
- **SVR regresses top-16 PCA components** of the spectrum (then inverts) for CPU tractability; the
  linear kernel is iteration-capped. SVR params (poly, C=1, ε=0.2) match the authors' code.
- **Seq2Seq latent 512→256, Bahdanau attention** (scoring fn unspecified) for the CPU budget.
- **DCTTS** uses reduced channels, and SSRN is channel-expanding (mel & linear share the hop).
- Every training run is **time-boxed with checkpoints**; numbers come from code that actually ran.

## Repository layout
```
src/            config, audio, text, dataset, download, preprocess, evaluate,
                make_figures, make_tables ; models/
main.py         orchestrator (--all / --demo / --step)
data/splits/    the fixed 200/25/25 lists (regenerated deterministically)
results/        metrics.csv, comparison_table.md, audio/<model>/, figures/
docs/           paper_spec.md, writeup.pdf, slides.pptx, qa_prep.md
reference/      the two paper PDFs + the authors' original_code
checkpoints/    trained model weights (small ones kept for --demo)
```
Large/regenerable artifacts are git-ignored; regenerate with `python main.py --all`.

## Team
PES2UG24CS495 · PES2UG24CS493 — see `docs/qa_prep.md` for the per-member work split.
