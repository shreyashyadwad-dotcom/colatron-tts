"""Model 2: simple neural network (paper Sec IV.B), matching the authors' code
(NN/simple_neural_network_tts.py): padded symbol-ID vector (no embedding) ->
Dense(L_max) -> Dense(L_max) -> Dense(T_max*1025, sigmoid), Adam + MSE, hidden
width = input width. We add dev-based early stopping and a time budget for CPU
safety, seed 42."""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C
from src import audio as A
from src import dataset as D

BATCH = 10
MAX_EPOCHS = 100          # authors' code: epochs=100
TIME_BUDGET_S = 300
PATIENCE = 40


class SimpleNN(nn.Module):
    def __init__(self, L_max, T_max, F, hidden=None):
        super().__init__()
        hidden = hidden or L_max            # authors: hidden width = max_input_len
        self.T_max, self.F, self.hidden = T_max, F, hidden
        self.net = nn.Sequential(
            nn.Linear(L_max, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, T_max * F), nn.Sigmoid(),
        )

    def forward(self, x):                   # x: (B, L_max) int ids
        out = self.net(x.float())
        return out.view(-1, self.T_max, self.F)


def train_and_synth():
    C.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = D.load_all(kind="lin")
    L_max, T_max, F = data["L_max"], data["T_max"], data["F"]
    model = SimpleNN(L_max, T_max, F).to(device)
    print(f"[nn] device={device} L_max/hidden={L_max} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    Xtr = torch.tensor(data["train"]["X"], device=device)
    Ytr = torch.tensor(data["train"]["Y"], device=device)
    Xdv = torch.tensor(data["dev"]["X"], device=device)
    Ydv = torch.tensor(data["dev"]["Y"], device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    N = Xtr.shape[0]
    best_dev = float("inf"); best_state = None; bad = 0; history = []
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(N, device=device); ep_loss = 0.0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= N
        model.eval()
        with torch.no_grad():
            dev_loss = loss_fn(model(Xdv), Ydv).item()
        history.append({"epoch": ep, "train_mse": ep_loss, "dev_mse": dev_loss})
        if dev_loss < best_dev - 1e-8:
            best_dev = dev_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if ep % 10 == 0:
            print(f"[nn] ep{ep} train_mse={ep_loss:.4e} dev_mse={dev_loss:.4e} t={time.time()-t0:.0f}s")
        if bad >= PATIENCE or (time.time() - t0) > TIME_BUDGET_S:
            print(f"[nn] stop at ep{ep}"); break
    if best_state:
        model.load_state_dict(best_state)
    torch.save({"state": model.state_dict(), "L_max": L_max, "T_max": T_max,
                "F": F, "hidden": model.hidden}, os.path.join(C.CHECKPOINTS, "nn.pt"))
    with open(os.path.join(C.CHECKPOINTS, "nn_history.json"), "w") as f:
        json.dump({"history": history, "best_dev_mse": best_dev}, f, indent=2)
    _synth(model, data, device)
    print(f"[nn] best dev MSE={best_dev:.4e}; final train MSE={history[-1]['train_mse']:.4e}")
    return {"best_dev_mse": best_dev, "final_train_mse": history[-1]["train_mse"]}


def _synth(model, data, device):
    outdir = os.path.join(C.AUDIO_OUT, "nn"); os.makedirs(outdir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(data["test"]["X"], device=device)).cpu().numpy()
    for i, cid in enumerate(data["test"]["ids"]):
        spec = np.clip(pred[i, :data["test"]["Tlen"][i]], 0, 1)
        np.save(os.path.join(outdir, f"{cid}.spec.npy"), spec)
        A.save_wav(os.path.join(outdir, f"{cid}.wav"), A.spectrogram_to_wav(spec))


if __name__ == "__main__":
    train_and_synth()
