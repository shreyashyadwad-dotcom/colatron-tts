"""Download the LJSpeech-1.1 clips we need and build the fixed 200/25/25 split
(seed 42). Streams only the required clips from the Hugging Face parquet mirror
rather than the full 2.6 GB tar, trying a few mirror repos in order."""
import io
import os
import sys
import random

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

CANDIDATE_REPOS = [
    ("keithito/lj_speech", "refs/convert/parquet"),
    ("keithito/lj_speech", "main"),
    ("lj_speech", "refs/convert/parquet"),
    ("MikhailT/lj-speech", "main"),
]

WAV_DIR = os.path.join(C.RAW, "wavs")
os.makedirs(WAV_DIR, exist_ok=True)


def _find_parquet_files():
    from huggingface_hub import HfApi
    api = HfApi()
    for repo, rev in CANDIDATE_REPOS:
        try:
            files = api.list_repo_files(repo_id=repo, repo_type="dataset", revision=rev)
            pq = sorted(f for f in files if f.endswith(".parquet"))
            if pq:
                print(f"[data] using repo={repo} rev={rev}: {len(pq)} parquet shard(s)")
                return repo, rev, pq
        except Exception as e:
            print(f"[data] repo {repo}@{rev} unavailable: {e}")
    raise RuntimeError("No LJSpeech parquet mirror reachable.")


def _iter_clips(limit_shards=3):
    """Yield (clip_id, text, waveform, sr) from parquet shards."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, rev, files = _find_parquet_files()
    for fname in files[:limit_shards]:
        local = hf_hub_download(repo_id=repo, repo_type="dataset", revision=rev,
                                filename=fname)
        table = pq.read_table(local)
        cols = table.column_names
        audio_col = "audio" if "audio" in cols else None
        text_col = next((c for c in ("normalized_text", "text", "transcription")
                         if c in cols), None)
        id_col = next((c for c in ("id", "file", "name") if c in cols), None)
        data = table.to_pylist()
        for i, row in enumerate(data):
            try:
                a = row[audio_col]
                if isinstance(a, dict) and a.get("bytes"):
                    y, sr = sf.read(io.BytesIO(a["bytes"]))
                elif isinstance(a, dict) and a.get("array") is not None:
                    y, sr = np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
                else:
                    continue
                if y.ndim > 1:
                    y = y.mean(axis=1)
                cid = str(row.get(id_col) or f"{fname}_{i}").replace("/", "_").replace(".wav", "")
                txt = (row.get(text_col) or "").strip()
                if txt:
                    yield cid, txt, np.asarray(y, dtype=np.float32), sr
            except Exception:
                continue


def main():
    C.set_seed()
    need = C.N_TRAIN + C.N_DEV + C.N_TEST
    print(f"[data] collecting clips in [{C.MIN_DUR},{C.MAX_DUR}]s, need {need}...")
    pool = []  # (cid, text, path, dur)
    import librosa
    for cid, txt, y, sr in _iter_clips():
        if sr != C.SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=C.SR)
        dur = len(y) / C.SR
        if C.MIN_DUR <= dur <= C.MAX_DUR:
            path = os.path.join(WAV_DIR, f"{cid}.wav")
            sf.write(path, y.astype(np.float32), C.SR)
            pool.append((cid, txt, path, dur))
        if len(pool) >= need + 50:   # small cushion for a clean seeded sample
            break
    print(f"[data] collected {len(pool)} candidate clips")
    if len(pool) < need:
        raise RuntimeError(f"Only {len(pool)} clips in band; need {need}.")

    rng = random.Random(C.SEED)
    rng.shuffle(pool)
    sel = pool[:need]
    train = sel[:C.N_TRAIN]
    dev = sel[C.N_TRAIN:C.N_TRAIN + C.N_DEV]
    test = sel[C.N_TRAIN + C.N_DEV:need]

    meta = {cid: txt for cid, txt, _, _ in sel}
    with open(os.path.join(C.RAW, "metadata_selected.csv"), "w", encoding="utf-8") as f:
        for cid, txt in meta.items():
            f.write(f"{cid}|{txt}\n")

    for name, part in (("train", train), ("dev", dev), ("test", test)):
        with open(os.path.join(C.SPLITS, f"{name}.txt"), "w", encoding="utf-8") as f:
            for cid, _, _, _ in part:
                f.write(cid + "\n")
        durs = [d for _, _, _, d in part]
        print(f"[data] {name}: {len(part)} clips, dur mean={np.mean(durs):.2f}s "
              f"min={np.min(durs):.2f} max={np.max(durs):.2f}")
    print("[data] splits written to", C.SPLITS)


if __name__ == "__main__":
    main()
