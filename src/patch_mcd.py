"""Recompute ONLY the MCD metric (fast, no ASR) for every model folder and patch
results/metrics.csv + per_utt_*.json in place. Used after an eval run so we don't
redo the slow Whisper pass just to fix MCD."""
import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import audio as A
from src.evaluate import mcd_dtw, FIELDS
from src.preprocess import load_split, wav_path

FOLDERS = ["ground_truth", "gt_griffinlim", "svr", "svr_linear", "nn", "seq2seq", "dctts"]


def main():
    test_ids = load_split("test")
    csvp = os.path.join(C.RESULTS, "metrics.csv")
    rows = {}
    with open(csvp, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    for tag in FOLDERS:
        wavdir = os.path.join(C.AUDIO_OUT, tag)
        if not os.path.isdir(wavdir) or tag not in rows:
            continue
        vals = []
        for cid in test_ids:
            wf = os.path.join(wavdir, f"{cid}.wav")
            if not os.path.exists(wf):
                continue
            try:
                vals.append(mcd_dtw(A.load_wav(wf), A.load_wav(wav_path(cid))))
            except Exception as e:
                print("mcd fail", tag, cid, e)
        if vals:
            rows[tag]["mcd_mean"] = f"{np.mean(vals):.6f}"
            rows[tag]["mcd_std"] = f"{np.std(vals):.6f}"
            # patch per-utt json too
            pj = os.path.join(C.RESULTS, f"per_utt_{tag}.json")
            if os.path.exists(pj):
                d = json.load(open(pj)); d["mcd"] = vals
                json.dump(d, open(pj, "w"), indent=2)
            print(f"[patch_mcd] {tag}: MCD={np.mean(vals):.2f}")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for tag in FOLDERS:
            if tag in rows:
                w.writerow({k: rows[tag].get(k, "") for k in FIELDS})
    print("[patch_mcd] metrics.csv updated")


if __name__ == "__main__":
    main()
