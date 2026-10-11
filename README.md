# Colatron — Reproducing & Extending End-to-End TTS

> Reproduction and extension of Wang, Yang & Li, *"End-to-End Text to Speech Synthesis"* (CS229, 2018).

| | |
|---|---|
| **Course** | UE24CS352A — Machine Learning |
| **Project** | Mini-Project #20 · Team 23 |
| **Institution** | PES University, Section H |
| **Authors** | Shriram Chandrasekar (PES2UG24CS495) · Shreyash Yadwad (PES2UG24CS493) |
| **Paper** | [report](https://cs229.stanford.edu/proj2018/report/15.pdf) · [poster](https://cs229.stanford.edu/proj2018/poster/15.pdf) (both PDFs in `reference/`) |

---

## Overview

The paper maps text directly to a waveform and compares three models trained on only ~200 LJSpeech
sentences, scored by human **MOS**. Its main finding: the seq2seq model *repeats words* because
attention alignment drifts.

- **Part 1 — Reproduce:** all three models on identical data, preprocessing and Griffin-Lim, reported
  on a *Reported-vs-Ours* table with an explanation for every gap.
- **Part 2 — Extend:** **DCTTS** (Tachibana 2018) — a fully-convolutional Text2Mel + SSRN with a
  **guided-attention loss** that forces a near-diagonal alignment, directly attacking the paper's
  failure mode. Not used by the paper; same data, split and metrics.

MOS needs humans, so we approximate it **two ways and say so**: **UTMOS** (neural predicted-MOS) as the
primary automated column, plus an optional small human listening test. Objective metrics add **MCD**
(DTW), **CER/WER** (Whisper ASR — a direct proxy for repeated/incomplete words) and spectrogram **L1**.

---

## Models

| Model | Architecture | Role |
|---|---|---|
| **SVR** (poly / linear) | Per-timestep support-vector regression on PCA-compressed spectra | Paper baseline |
| **Simple NN** | Flattened-output MLP (symbol IDs → full spectrogram) | Paper baseline |
| **Seq2Seq + Attention** | BiLSTM encoder → LSTM decoder with Bahdanau attention | Paper main model |
| **DCTTS** | Convolutional Text2Mel + SSRN with guided attention | **Our extension** |

---

## Dataset & Split

| Property | Value |
|---|---|
| Source | LJ Speech 1.1 (single speaker), streamed from the HuggingFace parquet mirror |
| Split | **200 train / 25 dev / 25 test** — `data/splits/*.txt`, **seed 42** |
| Clip band | **1.5–5.0 s** (keeps CPU zero-padding tractable) |
| Front-end | sr 20 kHz · n_fft 2048 (1025 bins) · hop 250 · win 1000 · pre-emphasis 0.97 · Griffin-Lim 60 |

The audio front-end values are taken **directly from the authors' code**, and every model uses exactly
these splits.

---

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install --no-build-isolation openai-whisper==20231117
```

`ffmpeg` is **not** required (Whisper reads generated WAVs via soundfile). CPU-only is fine; a GPU is
used automatically if present. First evaluation downloads the UTMOS and Whisper-base weights.

---

## Usage

| Command | Action |
|---|---|
| `python main.py --all` | Full pipeline: download → preprocess → train 4 models → synth → eval → tables → figures |
| `python main.py --demo "your sentence"` | Synthesize one sentence with all 4 models (<60 s, CPU) |
| `python main.py --step <name>` | Run one stage: `data`,`prep`,`svr`,`nn`,`seq2seq`,`dctts`,`eval`,`tables`,`figures` |
| `python main.py ... --force` | Recompute even if outputs already exist |

Every stage skips itself if its outputs already exist.

---

## Results

Auto-generated from `results/metrics.csv` (full table: `results/comparison_table.md`).

| Model | Reported MOS | UTMOS ↑ | MCD ↓ | CER ↓ | WER ↓ | spec L1 ↓ |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Ground truth *(ref)* | — | 4.34 | 0.05 | 0.040 | 0.052 | — |
| GT + Griffin-Lim *(ceiling, ref)* | — | 3.01 | 23.16 | 0.041 | 0.055 | — |
| SVR (poly) | 1.0 | 1.31 | 42.41 | 0.834 | 0.985 | 0.161 |
| SVR (linear) | — | 1.30 | 53.03 | 0.891 | 0.993 | 0.213 |
| Simple NN | 1.7 | 1.39 | 40.99 | 0.934 | 0.996 | 0.157 |
| Seq2Seq + attention | 2.5 | 1.63 | 51.34 | 0.929 | 0.996 | 0.169 |
| DCTTS *(extension)* | — | 1.44 | 51.44 | 0.915 | 1.000 | 0.189 |

<sub>UTMOS is a predicted-MOS stand-in (higher = better); lower MCD/CER/WER/spec-L1 = better. "Human MOS" columns stay `pending` (paper used n=10 raters).</sub>

**Findings**

- UTMOS **recovers the paper's ranking**: SVR < NN < Seq2Seq (1.31 < 1.39 < 1.63).
- **DCTTS verdict is mixed (and honest):** its guided attention gives a far more diagonal, monotonic
  alignment than seq2seq (`attention_compare.png`) — fixing the *mechanism* the paper blames for
  repeated words — but on this 200-clip, CPU-shrunk budget it did **not** improve perceptual quality
  (UTMOS significantly lower, paired Wilcoxon p = 0.012; CER indistinguishable, p = 0.39).
- Matches the paper's conclusion: **200 sentences is the binding constraint, not the model.**

Key figures in `results/figures/` (dpi 200): `attention_compare.png`, `compare_bars.png`,
`spectrograms.png`, `loss_curves.png`, `eda.png`.

---

## Deviations from the Paper

All flagged `ASSUMPTION` in code and tabulated in `docs/paper_spec.md`.

| Deviation | Reason |
|---|---|
| Clip band 1.5–5.0 s, fixed 200/25/25 (seed 42) | Paper never states *which* 200 sentences |
| MOS ≈ UTMOS + small human test | Human MOS can't be fully automated |
| SVR regresses top-16 PCA components; linear kernel iteration-capped | CPU tractability (params poly, C=1, ε=0.2 match authors) |
| Seq2Seq latent 512 → 256 | CPU budget (Bahdanau scoring fn unspecified in paper) |
| DCTTS reduced channels; SSRN channel-expanding | CPU budget (mel & linear share the hop) |
| Training time-boxed with checkpoints | Reproducible on CPU; numbers come from code that actually ran |

---

## Repository Layout

| Path | Contents |
|---|---|
| `src/` | `config`, `audio`, `text`, `dataset`, `download`, `preprocess`, `evaluate`, `make_figures`, `make_tables`, `models/` |
| `main.py` | Orchestrator (`--all` / `--demo` / `--step`) |
| `data/splits/` | Fixed 200/25/25 lists (deterministic, seed 42) |
| `results/` | `metrics.csv`, `comparison_table.md`, `audio/<model>/`, `figures/` |
| `docs/` | `paper_spec.md`, `writeup.pdf`, `slides.pptx` |
| `reference/` | Paper PDFs + authors' `original_code` |
| `checkpoints/` | Trained weights (small ones kept for `--demo`) |

Large / regenerable artifacts are git-ignored — rebuild with `python main.py --all`.

---

## Contributions

| Component | Shriram Chandrasekar<br>(PES2UG24CS495) | Shreyash Yadwad<br>(PES2UG24CS493) |
|---|:--:|:--:|
| Audio front-end & config (`audio.py`, `config.py`) | ● | |
| Text / CMU-phoneme front-end (`text.py`) | ● | |
| Data download, split & preprocessing | ● | |
| Dataset loader (`dataset.py`) | ● | |
| SVR baseline | ● | |
| Simple NN baseline | ● | |
| CLI orchestrator (`main.py`) | ● | |
| Seq2Seq + Bahdanau attention | | ● |
| DCTTS extension (guided attention) | | ● |
| Evaluation harness (UTMOS / MCD / CER-WER / spec-L1) | | ● |
| Results — tables & figures | | ● |
| Documentation (write-up, slides, paper spec) | | ● |

*Both members jointly reviewed the reference paper, the final comparison, and the reproduced results.*
