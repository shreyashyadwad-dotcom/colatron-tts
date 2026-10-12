"""Evaluation harness. For a model's output folder computes, on the 25 test clips:
  - UTMOS  : predicted MOS (automated stand-in for the paper's human MOS)
  - MCD    : mel-cepstral distortion with DTW alignment vs ground truth
  - CER/WER: Whisper-base ASR of the synthesis vs the input text (intelligibility)
  - specL1 : L1 between predicted and GT normalised spectrogram
Appends a row to results/metrics.csv and saves per-utterance values for sig tests.
"""
import os
import sys
import csv
import json
import re
import warnings
import numpy as np
import librosa

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import audio as A
from src.preprocess import load_split, load_metadata, feat_path, wav_path

_utmos = None
_whisper = None


def _get_utmos():
    global _utmos
    if _utmos is None:
        import torch
        _utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                trust_repo=True)
        _utmos.eval()
    return _utmos


def _get_whisper():
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper.load_model("base")
    return _whisper


def utmos_score(y, sr=C.SR):
    import torch
    model = _get_utmos()
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        wav = torch.from_numpy(y).float().unsqueeze(0)
        return float(model(wav, 16000).item())


def _norm_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[m]


def cer_wer(hyp, ref):
    ref_n, hyp_n = _norm_text(ref), _norm_text(hyp)
    rc, hc = list(ref_n), list(hyp_n)
    cer = _edit_distance(hc, rc) / max(1, len(rc))
    rw, hw = ref_n.split(), hyp_n.split()
    wer = _edit_distance(hw, rw) / max(1, len(rw))
    return min(cer, 1.0), min(wer, 1.0)


def _mcc(y, sr, n_mcc=13):
    """Mel-cepstral coefficients: DCT of natural-log mel magnitude (standard MCD basis)."""
    import scipy.fftpack
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP,
                                         n_mels=40, power=1.0)
    logmel = np.log(np.maximum(mel, 1e-8))                 # natural log of magnitude
    mcc = scipy.fftpack.dct(logmel, axis=0, type=2, norm="ortho")[:n_mcc]
    return mcc                                             # (n_mcc, T)


def mcd_dtw(y_ref, y_syn, sr=C.SR, n_mcc=13):
    """Mel-cepstral distortion (dB) with DTW alignment, 0th coeff excluded."""
    m_ref = _mcc(y_ref, sr, n_mcc)[1:]
    m_syn = _mcc(y_syn, sr, n_mcc)[1:]
    _, wp = librosa.sequence.dtw(X=m_ref, Y=m_syn, metric="euclidean")
    k = 10.0 / np.log(10) * np.sqrt(2)
    dists = [k * np.sqrt(np.sum((m_ref[:, i] - m_syn[:, j]) ** 2)) for i, j in wp]
    return float(np.mean(dists))


def evaluate_model(tag, wavdir, specdir=None, do_asr=True, do_utmos=True):
    C.set_seed()
    meta = load_metadata()
    test_ids = load_split("test")
    per = {"utmos": [], "mcd": [], "cer": [], "wer": [], "specL1": []}
    whisper_model = _get_whisper() if do_asr else None
    for cid in test_ids:
        wavf = os.path.join(wavdir, f"{cid}.wav")
        if not os.path.exists(wavf):
            continue
        y = A.load_wav(wavf)
        y_ref = A.load_wav(wav_path(cid))
        if do_utmos:
            try: per["utmos"].append(utmos_score(y))
            except Exception as e: print("[eval] utmos fail", e)
        try: per["mcd"].append(mcd_dtw(y_ref, y))
        except Exception as e: print("[eval] mcd fail", e)
        if do_asr:
            try:
                # Decode ourselves (16k mono float32) and pass the array to Whisper so it
                # does NOT shell out to ffmpeg (which isn't installed on this box).
                y16 = librosa.resample(y, orig_sr=C.SR, target_sr=16000).astype(np.float32)
                txt = whisper_model.transcribe(y16, fp16=False, language="en")["text"]
                c, w = cer_wer(txt, meta[cid])
                per["cer"].append(c); per["wer"].append(w)
            except Exception as e:
                print("[eval] asr fail", e)
        if specdir:
            sf = os.path.join(specdir, f"{cid}.spec.npy")
            if os.path.exists(sf):
                pred = np.load(sf); gt = np.load(feat_path(cid, "lin"))
                n = min(pred.shape[0], gt.shape[0])
                per["specL1"].append(float(np.mean(np.abs(pred[:n] - gt[:n]))))
    row = {"model": tag, "n": len(test_ids)}
    for k, v in per.items():
        row[k + "_mean"] = float(np.mean(v)) if v else float("nan")
        row[k + "_std"] = float(np.std(v)) if v else float("nan")
    _append_csv(row)
    with open(os.path.join(C.RESULTS, f"per_utt_{tag}.json"), "w") as f:
        json.dump(per, f, indent=2)
    print(f"[eval] {tag}: UTMOS={row['utmos_mean']:.3f} MCD={row['mcd_mean']:.2f} "
          f"CER={row['cer_mean']:.3f} WER={row['wer_mean']:.3f} specL1={row['specL1_mean']}")
    return row


FIELDS = ["model", "n", "utmos_mean", "utmos_std", "mcd_mean", "mcd_std",
          "cer_mean", "cer_std", "wer_mean", "wer_std", "specL1_mean", "specL1_std"]


def _append_csv(row):
    path = os.path.join(C.RESULTS, "metrics.csv")
    exists = os.path.exists(path)
    # de-dup: drop any prior row for this model
    rows = []
    if exists:
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["model"] != row["model"]]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        w.writerow({k: row.get(k, "") for k in FIELDS})


def evaluate_references():
    """Ground-truth and GT->Griffin-Lim ceiling rows."""
    gt_dir = os.path.join(C.AUDIO_OUT, "ground_truth"); os.makedirs(gt_dir, exist_ok=True)
    gl_dir = os.path.join(C.AUDIO_OUT, "gt_griffinlim"); os.makedirs(gl_dir, exist_ok=True)
    for cid in load_split("test"):
        y = A.load_wav(wav_path(cid))
        A.save_wav(os.path.join(gt_dir, f"{cid}.wav"), y)
        spec = A.wav_to_spectrogram(y)
        A.save_wav(os.path.join(gl_dir, f"{cid}.wav"), A.spectrogram_to_wav(spec))
    evaluate_model("ground_truth", gt_dir)
    evaluate_model("gt_griffinlim", gl_dir)


if __name__ == "__main__":
    tag = sys.argv[1]
    evaluate_model(tag, os.path.join(C.AUDIO_OUT, tag), os.path.join(C.AUDIO_OUT, tag))
