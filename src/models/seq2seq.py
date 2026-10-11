"""Model 3: seq2seq with attention (paper Sec IV.C, the main model). Embedding ->
BiLSTM encoder -> LSTM decoder with Bahdanau attention, decoder initialised from
the encoder states. Teacher-forced on the frame-shifted spectrogram; autoregressive
at inference with a stop rule. Latent reduced from the paper's 512 to fit the CPU
budget - see docs/paper_spec.md."""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C
from src import audio as A
from src import dataset as D
from src import text as TXT

EMBED_DIM = 128
HIDDEN = 256
BATCH = 16
MAX_EPOCHS = 300
TIME_BUDGET_S = 600
PATIENCE = 20


class Seq2Seq(nn.Module):
    def __init__(self, Fdim, hidden=HIDDEN):
        super().__init__()
        self.F = Fdim; self.H = hidden
        self.embed = nn.Embedding(TXT.N_SYMBOLS, EMBED_DIM, padding_idx=TXT.PAD_ID)
        self.encoder = nn.LSTM(EMBED_DIM, hidden, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(hidden * 2, hidden)
        self.h_bridge = nn.Linear(hidden * 2, hidden)
        self.c_bridge = nn.Linear(hidden * 2, hidden)
        self.frame_in = nn.Linear(Fdim, hidden)
        self.decoder = nn.LSTMCell(hidden + hidden, hidden)
        # Bahdanau attention
        self.W1 = nn.Linear(hidden, hidden, bias=False)
        self.W2 = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.out = nn.Linear(hidden + hidden, Fdim)

    def encode(self, x, xlen):
        e = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(e, xlen.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        outs, (h, c) = self.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(outs, batch_first=True,
                                                  total_length=x.shape[1])
        enc = self.enc_proj(enc)                               # (B,L,H)
        h0 = torch.tanh(self.h_bridge(torch.cat([h[0], h[1]], -1)))
        c0 = torch.tanh(self.c_bridge(torch.cat([c[0], c[1]], -1)))
        return enc, h0, c0

    def attend(self, enc, dec_h, mask):
        score = self.v(torch.tanh(self.W1(enc) + self.W2(dec_h).unsqueeze(1))).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        w = F.softmax(score, dim=1)                            # (B,L)
        ctx = torch.bmm(w.unsqueeze(1), enc).squeeze(1)        # (B,H)
        return ctx, w

    def forward(self, x, xlen, y=None, max_T=None, teacher=True):
        B = x.shape[0]
        enc, h, c = self.encode(x, xlen)
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < xlen[:, None]
        T = y.shape[1] if teacher else max_T
        prev = torch.zeros(B, self.F, device=x.device)         # GO frame
        outs, attns = [], []
        for t in range(T):
            ctx, w = self.attend(enc, h, mask)
            dec_in = torch.cat([self.frame_in(prev), ctx], -1)
            h, c = self.decoder(dec_in, (h, c))
            frame = torch.sigmoid(self.out(torch.cat([h, ctx], -1)))
            outs.append(frame); attns.append(w)
            prev = y[:, t, :] if (teacher and y is not None) else frame
        return torch.stack(outs, 1), torch.stack(attns, 1)


def train_and_synth():
    C.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = D.load_all(kind="lin")
    Fdim = data["F"]
    model = Seq2Seq(Fdim).to(device)
    print(f"[seq2seq] device={device} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    max_T_train = int(data["train"]["Tlen"].max())
    stop_T = int(1.2 * max_T_train)

    Xtr = torch.tensor(data["train"]["X"], device=device)
    Ytr = torch.tensor(data["train"]["Y"], device=device)
    Ltr = torch.tensor(data["train"]["Xlen"], device=device)
    Xdv = torch.tensor(data["dev"]["X"], device=device)
    Ydv = torch.tensor(data["dev"]["Y"], device=device)
    Ldv = torch.tensor(data["dev"]["Xlen"], device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = Xtr.shape[0]
    best_dev = float("inf"); best_state = None; bad = 0; history = []
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            pred, _ = model(Xtr[idx], Ltr[idx], y=Ytr[idx], teacher=True)
            loss = F.l1_loss(pred, Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= N
        model.eval()
        with torch.no_grad():
            dv, _ = model(Xdv, Ldv, y=Ydv, teacher=True)
            dev_loss = F.l1_loss(dv, Ydv).item()
        history.append({"epoch": ep, "train_l1": ep_loss, "dev_l1": dev_loss})
        if dev_loss < best_dev - 1e-7:
            best_dev = dev_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if ep % 5 == 0:
            print(f"[seq2seq] ep{ep} train_l1={ep_loss:.4e} dev_l1={dev_loss:.4e} t={time.time()-t0:.0f}s")
        if bad >= PATIENCE or (time.time() - t0) > TIME_BUDGET_S:
            print(f"[seq2seq] stop at ep{ep}")
            break
    if best_state:
        model.load_state_dict(best_state)
    torch.save({"state": model.state_dict(), "F": Fdim, "stop_T": stop_T},
               os.path.join(C.CHECKPOINTS, "seq2seq.pt"))
    with open(os.path.join(C.CHECKPOINTS, "seq2seq_history.json"), "w") as f:
        json.dump({"history": history, "best_dev_l1": best_dev, "stop_T": stop_T}, f, indent=2)

    _synth(model, data, device, stop_T)
    print(f"[seq2seq] best dev L1={best_dev:.4e}; final train L1={history[-1]['train_l1']:.4e}")
    return {"best_dev_l1": best_dev, "final_train_l1": history[-1]["train_l1"]}


def _synth(model, data, device, stop_T):
    outdir = os.path.join(C.AUDIO_OUT, "seq2seq"); os.makedirs(outdir, exist_ok=True)
    attndir = os.path.join(C.FIGURES, "attention"); os.makedirs(attndir, exist_ok=True)
    model.eval()
    X = torch.tensor(data["test"]["X"], device=device)
    L = torch.tensor(data["test"]["Xlen"], device=device)
    with torch.no_grad():
        for i, cid in enumerate(data["test"]["ids"]):
            pred, attn = model(X[i:i+1], L[i:i+1], max_T=stop_T, teacher=False)
            spec = pred[0].cpu().numpy()
            # trim trailing near-silence frames
            energy = spec.mean(1)
            keep = np.where(energy > 0.02)[0]
            end = (keep[-1] + 1) if len(keep) else spec.shape[0]
            spec = np.clip(spec[:max(end, 10)], 0, 1)
            np.save(os.path.join(outdir, f"{cid}.spec.npy"), spec)
            A.save_wav(os.path.join(outdir, f"{cid}.wav"), A.spectrogram_to_wav(spec))
            if i < 3:
                np.save(os.path.join(attndir, f"seq2seq_{cid}.attn.npy"),
                        attn[0, :spec.shape[0]].cpu().numpy())


if __name__ == "__main__":
    train_and_synth()
