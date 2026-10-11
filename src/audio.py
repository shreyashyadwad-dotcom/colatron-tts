"""Shared audio front-end: pre-emphasis -> STFT -> log-mag dB -> [0,1], and back
via Griffin-Lim. Every model uses this same pipeline so the comparison is fair."""
import numpy as np
import librosa
import scipy.signal
import soundfile as sf
from . import config as C


def load_wav(path, sr=C.SR):
    y, _ = librosa.load(path, sr=sr)
    return y.astype(np.float32)


def save_wav(path, y, sr=C.SR):
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y.astype(np.float32), sr)


def preemphasis(y, coef=C.PREEMPH):
    return scipy.signal.lfilter([1, -coef], [1], y).astype(np.float32)


def inv_preemphasis(y, coef=C.PREEMPH):
    return scipy.signal.lfilter([1], [1, -coef], y).astype(np.float32)


def _stft(y):
    return librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP, win_length=C.WIN,
                        window="hann")


def _istft(mag_complex):
    return librosa.istft(mag_complex, hop_length=C.HOP, win_length=C.WIN,
                         window="hann")


def _amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S_db):
    """Map dB spectrogram to [0,1] for the sigmoid-output networks."""
    return np.clip((S_db - C.REF_DB - C.MIN_DB) / (-C.MIN_DB), 0.0, 1.0)


def _denormalize(S_norm):
    return S_norm * (-C.MIN_DB) + C.MIN_DB + C.REF_DB


def wav_to_spectrogram(y):
    """Returns linear log-magnitude spectrogram normalised to [0,1], shape (T, 1025)."""
    y = preemphasis(y)
    mag = np.abs(_stft(y))                 # (1025, T)
    S_db = _amp_to_db(mag)
    S = _normalize(S_db)                   # (1025, T)
    return S.T.astype(np.float32)          # (T, 1025)


def spectrogram_to_wav(S, n_iter=C.GL_ITERS):
    """Invert a normalised [0,1] log-magnitude spectrogram (T,1025) back to a waveform
    with Griffin-Lim phase recovery, then inverse pre-emphasis."""
    S = np.asarray(S, dtype=np.float32).T              # (1025, T)
    S_db = _denormalize(S)
    mag = _db_to_amp(S_db)                             # linear magnitude
    mag = np.power(np.maximum(mag, 1e-8), C.POWER)     # GL emphasis power
    y = _griffin_lim(mag, n_iter=n_iter)
    y = inv_preemphasis(y)
    m = np.max(np.abs(y)) + 1e-8
    return (y / m * 0.97).astype(np.float32)


def _griffin_lim(mag, n_iter=C.GL_ITERS):
    angles = np.exp(2j * np.pi * np.random.rand(*mag.shape)).astype(np.complex64)
    S = mag.astype(np.complex64) * angles
    y = _istft(S)
    for _ in range(n_iter):
        angles = np.exp(1j * np.angle(_stft(y)))
        S = mag.astype(np.complex64) * angles.astype(np.complex64)
        y = _istft(S)
    return y.astype(np.float32)


# Mel helpers (DCTTS only), derived from the same STFT.
_mel_basis = None

def _get_mel_basis():
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(sr=C.SR, n_fft=C.N_FFT, n_mels=C.N_MELS)
    return _mel_basis


def wav_to_mel(y):
    """Normalised [0,1] log-mel spectrogram, shape (T, 80)."""
    y = preemphasis(y)
    mag = np.abs(_stft(y))
    mel = np.dot(_get_mel_basis(), mag)
    mel_db = _amp_to_db(mel)
    mel = _normalize(mel_db)
    return mel.T.astype(np.float32)
