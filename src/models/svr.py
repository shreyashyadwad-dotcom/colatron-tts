"""Model 1: SVR baseline (paper Sec IV.A). One multi-output regressor per output
time step, mapping the padded symbol-ID sequence to that frame's spectrum. Poly is
the paper's final kernel; we also run the linear one as they did.
We regress the top-k PCA components of each spectrum rather than all 1025 bins, so
the per-frame SVR stays tractable on CPU - see docs/paper_spec.md."""
import os
import sys
import time
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C
from src import audio as A
from src import dataset as D
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

PCA_K = 16  # each spectrum compressed to 16 PCA comps to keep the SVR tractable


def _fit_kernel(data, kernel):
    X = data["train"]["X"].astype(np.float64)
    Y = data["train"]["Y"]          # (N,T,F)
    N, Tm, F = Y.shape
    # Global PCA over all train frames
    frames = Y.reshape(N * Tm, F)
    pca = PCA(n_components=PCA_K, random_state=C.SEED).fit(frames)
    Yc = pca.transform(frames).reshape(N, Tm, PCA_K)   # (N,T,K)

    regs = []
    t0 = time.time()
    for t in range(Tm):
        # cap linear-kernel iterations; unbounded SVR(linear) never converges here
        max_it = 3000 if kernel == "linear" else -1
        # hyperparams from the authors' code: poly, C=1.0, gamma=scale, epsilon=0.2
        base = SVR(kernel=kernel, degree=3, C=1.0, gamma="scale", epsilon=0.2,
                   max_iter=max_it, cache_size=500)
        reg = MultiOutputRegressor(base)
        reg.fit(X, Yc[:, t, :])
        regs.append(reg)
        if t == 4:  # ETA after 5 frames
            eta = (time.time() - t0) / 5 * Tm
            print(f"[svr-{kernel}] ETA ~{eta/60:.1f} min for {Tm} frames")
    return regs, pca


def _predict(regs, pca, X, Tm, F):
    X = X.astype(np.float64)
    out = np.zeros((X.shape[0], Tm, F), dtype=np.float32)
    for t in range(Tm):
        comps = regs[t].predict(X)                 # (N,K)
        out[:, t, :] = pca.inverse_transform(comps)
    return np.clip(out, 0.0, 1.0)


def train_and_synth(kernels=("poly", "linear")):
    C.set_seed()
    data = D.load_all(kind="lin")
    Tm, F = data["T_max"], data["F"]
    results = {}
    for kernel in kernels:
        print(f"[svr] fitting kernel={kernel} ...")
        regs, pca = _fit_kernel(data, kernel)
        joblib.dump({"regs": regs, "pca": pca, "L_max": data["L_max"],
                     "T_max": Tm, "F": F},
                    os.path.join(C.CHECKPOINTS, f"svr_{kernel}.joblib"))
        # Training loss (teacher-forced L1 on train)
        pred_tr = _predict(regs, pca, data["train"]["X"], Tm, F)
        tr_l1 = float(np.mean(np.abs(pred_tr - data["train"]["Y"])))
        # synth test set (poly is the paper's final; synth both, poly -> main folder)
        tag = "svr" if kernel == "poly" else "svr_linear"
        outdir = os.path.join(C.AUDIO_OUT, tag)
        os.makedirs(outdir, exist_ok=True)
        pred_te = _predict(regs, pca, data["test"]["X"], Tm, F)
        for i, cid in enumerate(data["test"]["ids"]):
            spec = pred_te[i, :data["test"]["Tlen"][i]]
            np.save(os.path.join(outdir, f"{cid}.spec.npy"), spec)
            A.save_wav(os.path.join(outdir, f"{cid}.wav"), A.spectrogram_to_wav(spec))
        results[tag] = {"train_l1": tr_l1}
        print(f"[svr-{kernel}] train L1={tr_l1:.4e}; test wavs -> {outdir}")
    jpath = os.path.join(C.CHECKPOINTS, "svr_train.json")
    merged = {}
    if os.path.exists(jpath):
        merged = json.load(open(jpath))
    merged.update(results)
    with open(jpath, "w") as f:
        json.dump(merged, f, indent=2)
    return results


if __name__ == "__main__":
    train_and_synth()
