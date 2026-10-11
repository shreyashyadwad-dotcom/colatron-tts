"""Colatron reproduction - top-level orchestrator.

  python main.py --all                 full pipeline (skips steps whose outputs exist)
  python main.py --demo "a sentence"   synth one sentence with all 4 models (<60s CPU)
  python main.py --step <name>         run one stage: data|prep|svr|nn|seq2seq|dctts|eval|tables|figures
"""
import os
import sys
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C
from src import audio as A


def _exists_audio(tag, n_expected=25):
    d = os.path.join(C.AUDIO_OUT, tag)
    return os.path.isdir(d) and len([f for f in os.listdir(d) if f.endswith(".wav")]) >= n_expected


def step_data(force=False):
    if not force and os.path.exists(os.path.join(C.SPLITS, "train.txt")):
        print("[main] splits exist, skip download"); return
    from src.download_data import main as dl; dl()


def step_prep(force=False):
    if not force and os.path.exists(os.path.join(C.PROCESSED, "stats.json")):
        print("[main] features exist, skip preprocess"); return
    from src.preprocess import preprocess_all; preprocess_all()


def step_svr(force=False):
    if not force and _exists_audio("svr"):
        print("[main] svr outputs exist, skip"); return
    from src.models.svr import train_and_synth; train_and_synth()


def step_nn(force=False):
    if not force and _exists_audio("nn"):
        print("[main] nn outputs exist, skip"); return
    from src.models.nn import train_and_synth; train_and_synth()


def step_seq2seq(force=False):
    if not force and _exists_audio("seq2seq"):
        print("[main] seq2seq outputs exist, skip"); return
    from src.models.seq2seq import train_and_synth; train_and_synth()


def step_dctts(force=False):
    if not force and _exists_audio("dctts"):
        print("[main] dctts outputs exist, skip"); return
    from src.models.dctts import train_and_synth; train_and_synth()


def step_eval(force=False):
    from src import evaluate as E
    E.evaluate_references()
    for tag in ("svr", "svr_linear", "nn", "seq2seq", "dctts"):
        if _exists_audio(tag, 1):
            E.evaluate_model(tag, os.path.join(C.AUDIO_OUT, tag),
                             os.path.join(C.AUDIO_OUT, tag))


def step_tables(force=False):
    from src.make_tables import build; build()


def step_figures(force=False):
    from src.make_figures import build_all; build_all()


def run_all():
    step_data(); step_prep()
    step_svr(); step_nn(); step_seq2seq(); step_dctts()
    step_eval(); step_tables(); step_figures()
    print("\n[main] pipeline complete. See results/comparison_table.md")


def demo(sentence):
    """Load each model's checkpoint and synth one sentence with all four."""
    import time, json, joblib, torch
    from src import text as TXT
    from src import dataset as D
    out = os.path.join(C.RESULTS, "demo"); os.makedirs(out, exist_ok=True)
    C.set_seed()
    seq = TXT.text_to_sequence(sentence, seed=C.SEED)
    print(f"[demo] '{sentence}'  ({len(seq)} symbols)")
    t_all = time.time()

    # SVR (poly)
    try:
        ck = joblib.load(os.path.join(C.CHECKPOINTS, "svr_poly.joblib"))
        X = TXT.pad_sequences([seq], max_len=ck["L_max"]).astype(np.float64)
        Tm, Fd = ck["T_max"], ck["F"]
        spec = np.zeros((Tm, Fd), np.float32)
        for t in range(Tm):
            spec[t] = ck["pca"].inverse_transform(ck["regs"][t].predict(X))[0]
        _write_demo(out, "svr", np.clip(spec, 0, 1))
    except Exception as e:
        print("[demo] svr skipped:", e)

    # NN
    try:
        from src.models.nn import SimpleNN
        ck = torch.load(os.path.join(C.CHECKPOINTS, "nn.pt"), map_location="cpu")
        m = SimpleNN(ck["L_max"], ck["T_max"], ck["F"], ck["hidden"]); m.load_state_dict(ck["state"]); m.eval()
        X = torch.tensor(TXT.pad_sequences([seq], max_len=ck["L_max"]))
        with torch.no_grad():
            spec = m(X)[0].numpy()
        _write_demo(out, "nn", np.clip(spec, 0, 1))
    except Exception as e:
        print("[demo] nn skipped:", e)

    # Seq2Seq
    try:
        from src.models.seq2seq import Seq2Seq
        ck = torch.load(os.path.join(C.CHECKPOINTS, "seq2seq.pt"), map_location="cpu")
        m = Seq2Seq(ck["F"]); m.load_state_dict(ck["state"]); m.eval()
        X = torch.tensor(TXT.pad_sequences([seq]))
        L = torch.tensor([len(seq)])
        with torch.no_grad():
            pred, _ = m(X, L, max_T=ck["stop_T"], teacher=False)
        spec = _trim(pred[0].numpy())
        _write_demo(out, "seq2seq", spec)
    except Exception as e:
        print("[demo] seq2seq skipped:", e)

    # DCTTS
    try:
        from src.models.dctts import Text2Mel, SSRN, MEL
        t2m = Text2Mel(); t2m.load_state_dict(torch.load(os.path.join(C.CHECKPOINTS, "dctts_t2m.pt"), map_location="cpu")["state"]); t2m.eval()
        ssrn = SSRN(); ssrn.load_state_dict(torch.load(os.path.join(C.CHECKPOINTS, "dctts_ssrn.pt"), map_location="cpu")["state"]); ssrn.eval()
        X = torch.tensor(TXT.pad_sequences([seq]))
        stop_T = 180
        mel = torch.zeros(1, MEL, stop_T)
        with torch.no_grad():
            for t in range(1, stop_T):
                pred, _ = t2m(X, mel[:, :, :t]); mel[:, :, t] = pred[:, :, t - 1]
            lin = ssrn(mel)[0].transpose(0, 1).numpy()
        _write_demo(out, "dctts", _trim(lin))
    except Exception as e:
        print("[demo] dctts skipped:", e)

    print(f"[demo] done in {time.time()-t_all:.1f}s. Wavs in {out}")


def _trim(spec):
    energy = spec.mean(1); keep = np.where(energy > 0.02)[0]
    end = (keep[-1] + 1) if len(keep) else spec.shape[0]
    return np.clip(spec[:max(end, 10)], 0, 1)


def _write_demo(out, tag, spec):
    np.save(os.path.join(out, f"{tag}.spec.npy"), spec)
    A.save_wav(os.path.join(out, f"{tag}.wav"), A.spectrogram_to_wav(spec))
    print(f"[demo]   {tag}: results/demo/{tag}.wav")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--demo", type=str, default=None)
    ap.add_argument("--step", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.demo is not None:
        demo(a.demo)
    elif a.step:
        globals()[f"step_{a.step}"](a.force) if f"step_{a.step}" in globals() else print("unknown step")
    elif a.all:
        run_all()
    else:
        ap.print_help()
