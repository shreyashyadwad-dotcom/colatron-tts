"""Global config. Seeds fixed to 42; see docs/paper_spec.md for the assumptions."""
import os
import random
import numpy as np

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
PROCESSED = os.path.join(DATA, "processed")
SPLITS = os.path.join(DATA, "splits")
RESULTS = os.path.join(ROOT, "results")
AUDIO_OUT = os.path.join(RESULTS, "audio")
FIGURES = os.path.join(RESULTS, "figures")
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
for _d in (DATA, RAW, PROCESSED, SPLITS, RESULTS, AUDIO_OUT, FIGURES, CHECKPOINTS):
    os.makedirs(_d, exist_ok=True)

# Audio front-end values are taken from the authors' params_dict.py / audio.py.
SR = 20000
N_FREQ = 1025
N_FFT = (N_FREQ - 1) * 2
HOP = int(12.5 / 1000 * SR)   # 250 samples (12.5 ms shift)
WIN = int(50.0 / 1000 * SR)   # 1000 samples (50 ms window)
PREEMPH = 0.97
REF_DB = 20.0
MIN_DB = -100.0
GL_ITERS = 60
N_MELS = 80
POWER = 1.5

N_TRAIN = 200
N_DEV = 25
N_TEST = 25
# Keep clips in a 1.5-5 s band so padding to the longest stays tractable on CPU.
MIN_DUR = 1.5
MAX_DUR = 5.0

PHONEME_PROB = 0.5   # map each word to phonemes w.p. 0.5, else letters
