"""Compute and cache audio features (linear spectrogram + mel) for every split
clip, and run the Griffin-Lim 'vocoder ceiling' sanity check on 3 clips."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import audio as A


def load_split(name):
    with open(os.path.join(C.SPLITS, f"{name}.txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_metadata():
    meta = {}
    with open(os.path.join(C.RAW, "metadata_selected.csv"), encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                cid, txt = line.rstrip("\n").split("|", 1)
                meta[cid] = txt
    return meta


def wav_path(cid):
    return os.path.join(C.RAW, "wavs", f"{cid}.wav")


def feat_path(cid, kind):
    return os.path.join(C.PROCESSED, f"{cid}.{kind}.npy")


def preprocess_all():
    C.set_seed()
    ids = load_split("train") + load_split("dev") + load_split("test")
    Tmax = 0
    for cid in ids:
        y = A.load_wav(wav_path(cid))
        spec = A.wav_to_spectrogram(y)      # (T,1025)
        mel = A.wav_to_mel(y)               # (T,80)
        np.save(feat_path(cid, "lin"), spec)
        np.save(feat_path(cid, "mel"), mel)
        Tmax = max(Tmax, spec.shape[0])
    # Save dataset-level stats used by all models (T_max over TRAIN only)
    train_ids = load_split("train")
    Tmax_train = max(np.load(feat_path(c, "lin")).shape[0] for c in train_ids)
    stats = {"T_max_all": int(Tmax), "T_max_train": int(Tmax_train),
             "n_freq": C.N_FREQ, "n_mels": C.N_MELS}
    with open(os.path.join(C.PROCESSED, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[preprocess] cached", len(ids), "clips;", stats)
    _sanity_check()
    return stats


def _sanity_check():
    """Vocoder ceiling: reconstruct 3 GT clips through STFT->Griffin-Lim."""
    out = os.path.join(C.AUDIO_OUT, "gt_recon")
    os.makedirs(out, exist_ok=True)
    for cid in load_split("test")[:3]:
        y = A.load_wav(wav_path(cid))
        A.save_wav(os.path.join(out, f"{cid}.gt.wav"), y)
        spec = A.wav_to_spectrogram(y)
        rec = A.spectrogram_to_wav(spec)
        A.save_wav(os.path.join(out, f"{cid}.gl.wav"), rec)
    print("[preprocess] wrote vocoder-ceiling reconstructions to", out)


if __name__ == "__main__":
    preprocess_all()
