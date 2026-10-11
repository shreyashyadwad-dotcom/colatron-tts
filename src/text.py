"""Text front-end: CMU dict + symbol set, keithito-style random phoneme/letter
mapping (p=0.5 per word). Produces integer symbol-ID sequences."""
import re
import random
import numpy as np
import cmudict
from . import config as C

# Symbol set
_pad = "_"
_eos = "~"
_chars = "abcdefghijklmnopqrstuvwxyz!'(),-.:;? "
# ARPAbet phoneme inventory (CMU), prefixed with '@' to keep them distinct from letters
_arpabet = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
_arpabet = ["@" + p for p in _arpabet]

symbols = [_pad, _eos] + list(_chars) + _arpabet
_sym_to_id = {s: i for i, s in enumerate(symbols)}
PAD_ID = _sym_to_id[_pad]
EOS_ID = _sym_to_id[_eos]
N_SYMBOLS = len(symbols)

_cmu = None

def _dict():
    global _cmu
    if _cmu is None:
        _cmu = cmudict.dict()
    return _cmu


def _word_to_phonemes(word):
    entry = _dict().get(word.lower())
    if not entry:
        return None
    phones = entry[0]  # first pronunciation
    return ["@" + re.sub(r"\d", "", p) for p in phones]  # strip stress digits


def text_to_sequence(text, seed=None):
    """Map text -> list of symbol IDs. Each word is randomly rendered as its
    CMU phonemes (p=0.5) or its letters (keithito style)."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    text = text.strip()
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']", text)  # words and single punctuation/spaces
    out = []
    for tok in tokens:
        if re.match(r"[A-Za-z']+", tok):
            use_phonemes = rng.random() < C.PHONEME_PROB
            phones = _word_to_phonemes(tok) if use_phonemes else None
            if phones:
                out.extend(_sym_to_id[p] for p in phones if p in _sym_to_id)
            else:
                out.extend(_sym_to_id[c] for c in tok.lower() if c in _sym_to_id)
        else:
            if tok in _sym_to_id:
                out.append(_sym_to_id[tok])
    out.append(EOS_ID)
    return out


def pad_sequences(seqs, max_len=None):
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    arr = np.full((len(seqs), max_len), PAD_ID, dtype=np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), max_len)
        arr[i, :n] = s[:n]
    return arr
