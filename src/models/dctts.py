"""Extension: DCTTS (Tachibana et al., 2018), a fully-convolutional TTS with a
guided-attention loss. The paper's seq2seq fails on attention alignment (repeated
words); guided attention forces a near-diagonal alignment and trains fast on CPU.
Text2Mel predicts an 80-mel, SSRN expands it to a 1025-bin linear spectrogram, so
the output shares the same Griffin-Lim and eval path as the other models.
Channel widths are reduced from the paper to fit the CPU budget."""
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

D_DIM = 128          # reduced from 256 for the CPU budget
C_DIM = 256          # SSRN hidden
MEL = C.N_MELS
FREQ = C.N_FREQ
GUIDE_G = 0.2
TIME_BUDGET_T2M = 330
TIME_BUDGET_SSRN = 180
BATCH = 16
SYNTH_MAX_T = 340          # cap autoregressive length (synth is O(T^2) on CPU)


def causal_pad(x, k, d):
    return F.pad(x, ((k - 1) * d, 0))


class CConv(nn.Module):
    """1D conv; causal (left-pad) or same-pad."""
    def __init__(self, ci, co, k=1, d=1, causal=False, act=None):
        super().__init__()
        self.k, self.d, self.causal = k, d, causal
        self.conv = nn.Conv1d(ci, co, k, dilation=d,
                              padding=0 if causal else ((k - 1) * d) // 2)
        self.act = act

    def forward(self, x):
        if self.causal and self.k > 1:
            x = causal_pad(x, self.k, self.d)
        x = self.conv(x)
        return self.act(x) if self.act else x


class HighwayConv(nn.Module):
    def __init__(self, ch, k, d=1, causal=False):
        super().__init__()
        self.conv = CConv(ch, 2 * ch, k, d, causal)
        self.ch = ch

    def forward(self, x):
        h = self.conv(x)
        h1, h2 = h[:, :self.ch], h[:, self.ch:]
        g = torch.sigmoid(h1)
        return g * h2 + (1 - g) * x


class TextEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TXT.N_SYMBOLS, D_DIM, padding_idx=TXT.PAD_ID)
        layers = [CConv(D_DIM, 2 * D_DIM, 1, act=nn.ReLU()), CConv(2 * D_DIM, 2 * D_DIM, 1)]
        for _ in range(2):
            for dd in (1, 3, 9, 27):
                layers.append(HighwayConv(2 * D_DIM, 3, dd))
        for _ in range(2):
            layers.append(HighwayConv(2 * D_DIM, 3, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.embed(x).transpose(1, 2)
        h = self.net(h)
        K, V = h[:, :D_DIM], h[:, D_DIM:]
        return K, V


class AudioEnc(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [CConv(MEL, D_DIM, 1, causal=True, act=nn.ReLU()),
                  CConv(D_DIM, D_DIM, 1, causal=True, act=nn.ReLU()),
                  CConv(D_DIM, D_DIM, 1, causal=True)]
        for _ in range(2):
            for dd in (1, 3, 9, 27):
                layers.append(HighwayConv(D_DIM, 3, dd, causal=True))
        for _ in range(2):
            layers.append(HighwayConv(D_DIM, 3, 3, causal=True))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s)


class AudioDec(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [CConv(2 * D_DIM, D_DIM, 1, causal=True)]
        for dd in (1, 3, 9, 27):
            layers.append(HighwayConv(D_DIM, 3, dd, causal=True))
        for _ in range(2):
            layers.append(HighwayConv(D_DIM, 3, 1, causal=True))
        for _ in range(3):
            layers.append(CConv(D_DIM, D_DIM, 1, causal=True, act=nn.ReLU()))
        layers.append(CConv(D_DIM, MEL, 1, causal=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class Text2Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_enc = TextEnc()
        self.audio_enc = AudioEnc()
        self.audio_dec = AudioDec()

    def forward(self, text, mel_prev):
        K, V = self.text_enc(text)            # (B,d,L)
        Q = self.audio_enc(mel_prev)          # (B,d,T)
        A_ = torch.bmm(K.transpose(1, 2), Q) / np.sqrt(D_DIM)   # (B,L,T)
        A_ = F.softmax(A_, dim=1)
        R = torch.bmm(V, A_)                  # (B,d,T)
        Rp = torch.cat([R, Q], dim=1)         # (B,2d,T)
        mel = self.audio_dec(Rp)              # (B,MEL,T)
        return mel, A_


class SSRN(nn.Module):
    """mel (80) -> linear (1025), same time length (same hop)."""
    def __init__(self):
        super().__init__()
        layers = [CConv(MEL, C_DIM, 1)]
        for dd in (1, 3):
            layers.append(HighwayConv(C_DIM, 3, dd))
        for _ in range(2):
            layers.append(HighwayConv(C_DIM, 3, 1))
        layers += [CConv(C_DIM, 2 * C_DIM, 1)]
        for dd in (1, 3):
            layers.append(HighwayConv(2 * C_DIM, 3, dd))
        layers += [CConv(2 * C_DIM, FREQ, 1, act=nn.ReLU()),
                   CConv(FREQ, FREQ, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, mel):
        return torch.sigmoid(self.net(mel))


def guided_attention_loss(A_, Llen, Tlen):
    B, L, T = A_.shape
    device = A_.device
    n = torch.arange(L, device=device).view(1, L, 1).float()
    t = torch.arange(T, device=device).view(1, 1, T).float()
    Ld = Llen.view(B, 1, 1).float().clamp(min=1)
    Td = Tlen.view(B, 1, 1).float().clamp(min=1)
    W = 1.0 - torch.exp(-((n / Ld - t / Td) ** 2) / (2 * GUIDE_G ** 2))
    return (A_ * W).mean()


def _shift(mel):
    prev = torch.zeros_like(mel)
    prev[:, :, 1:] = mel[:, :, :-1]
    return prev


def train_text2mel(device, data):
    model = Text2Mel().to(device)
    print(f"[dctts/t2m] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    Xtr = torch.tensor(data["train"]["X"], device=device)
    mel_np = np.stack([_pad_np(np.load(os.path.join(C.PROCESSED, f"{c}.mel.npy")), data["T_max"])
                       for c in data["train"]["ids"]])          # (N,T,MEL)
    Mtr = torch.tensor(mel_np, device=device).transpose(1, 2)   # (N,MEL,T)
    Ltr = torch.tensor(data["train"]["Xlen"], device=device)
    Ttr = torch.tensor(data["train"]["Tlen"], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    N = Xtr.shape[0]; t0 = time.time(); hist = []
    ep = 0
    while time.time() - t0 < TIME_BUDGET_T2M:
        model.train(); perm = torch.randperm(N, device=device); el = 0.0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            mel = Mtr[idx]; prev = _shift(mel)
            opt.zero_grad()
            pred, A_ = model(Xtr[idx], prev)
            # L1 + guided attention; BCE dropped (it diverges when the sigmoid saturates)
            l1 = F.l1_loss(pred, mel)
            gl = guided_attention_loss(A_, Ltr[idx], Ttr[idx])
            loss = l1 + gl
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            el += loss.item() * len(idx)
        el /= N; hist.append(el)
        if ep % 10 == 0:
            print(f"[dctts/t2m] ep{ep} loss={el:.4e} t={time.time()-t0:.0f}s")
        ep += 1
    torch.save({"state": model.state_dict()}, os.path.join(C.CHECKPOINTS, "dctts_t2m.pt"))
    print(f"[dctts/t2m] done ep{ep} final loss={hist[-1]:.4e}")
    return model, hist


def train_ssrn(device, data):
    model = SSRN().to(device)
    print(f"[dctts/ssrn] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    mel_np = np.stack([_pad_np(np.load(os.path.join(C.PROCESSED, f"{c}.mel.npy")), data["T_max"])
                       for c in data["train"]["ids"]])          # (N,T,MEL)
    mel = torch.tensor(mel_np, device=device).transpose(1, 2)   # (N,MEL,T)
    lin = torch.tensor(data["train"]["Y"], device=device).transpose(1, 2)  # (N,FREQ,T)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    N = mel.shape[0]; t0 = time.time(); hist = []; ep = 0
    while time.time() - t0 < TIME_BUDGET_SSRN:
        model.train(); perm = torch.randperm(N, device=device); el = 0.0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            pred = model(mel[idx])
            loss = F.l1_loss(pred, lin[idx])     # L1 only (BCE drops for stability)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            el += loss.item() * len(idx)
        el /= N; hist.append(el)
        if ep % 10 == 0:
            print(f"[dctts/ssrn] ep{ep} loss={el:.4e} t={time.time()-t0:.0f}s")
        ep += 1
    torch.save({"state": model.state_dict()}, os.path.join(C.CHECKPOINTS, "dctts_ssrn.pt"))
    print(f"[dctts/ssrn] done ep{ep} final loss={hist[-1]:.4e}")
    return model, hist


def _pad_T(x, T):
    if x.shape[1] >= T:
        return x[:, :T]
    pad = torch.zeros(x.shape[0], T - x.shape[1], x.shape[2], device=x.device)
    return torch.cat([x, pad], 1)


def _pad_np(x, T):
    """Pad/trim (T0,F) numpy spectrogram to (T,F)."""
    if x.shape[0] >= T:
        return x[:T]
    return np.concatenate([x, np.zeros((T - x.shape[0], x.shape[1]), dtype=x.dtype)], 0)


def synth(t2m, ssrn, device, data, tag="dctts", save_attn=True):
    outdir = os.path.join(C.AUDIO_OUT, tag); os.makedirs(outdir, exist_ok=True)
    attndir = os.path.join(C.FIGURES, "attention"); os.makedirs(attndir, exist_ok=True)
    t2m.eval(); ssrn.eval()
    X = torch.tensor(data["test"]["X"], device=device)
    stop_T = min(int(1.2 * int(data["train"]["Tlen"].max())), SYNTH_MAX_T)
    with torch.no_grad():
        for i, cid in enumerate(data["test"]["ids"]):
            x = X[i:i + 1]
            mel = torch.zeros(1, MEL, stop_T, device=device)
            A_last = None
            for t in range(1, stop_T):
                pred, A_ = t2m(x, mel[:, :, :t])
                mel[:, :, t] = pred[:, :, t - 1]
                A_last = A_
            lin = ssrn(mel)[0].transpose(0, 1).cpu().numpy()   # (T,FREQ)
            energy = lin.mean(1); keep = np.where(energy > 0.02)[0]
            end = (keep[-1] + 1) if len(keep) else lin.shape[0]
            spec = np.clip(lin[:max(end, 10)], 0, 1)
            np.save(os.path.join(outdir, f"{cid}.spec.npy"), spec)
            A.save_wav(os.path.join(outdir, f"{cid}.wav"), A.spectrogram_to_wav(spec))
            if save_attn and i < 3 and A_last is not None:
                np.save(os.path.join(attndir, f"dctts_{cid}.attn.npy"),
                        A_last[0].cpu().numpy())
    print(f"[dctts] synthesised {len(data['test']['ids'])} test clips -> {outdir}")


def train_and_synth():
    C.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = D.load_all(kind="lin")
    t2m, h1 = train_text2mel(device, data)
    ssrn, h2 = train_ssrn(device, data)
    synth(t2m, ssrn, device, data)
    with open(os.path.join(C.CHECKPOINTS, "dctts_history.json"), "w") as f:
        json.dump({"t2m_loss": h1, "ssrn_loss": h2}, f, indent=2)
    return {"t2m_final": h1[-1], "ssrn_final": h2[-1]}


if __name__ == "__main__":
    train_and_synth()
