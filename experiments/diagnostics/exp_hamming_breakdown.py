"""Why does the emotion route have the worst Hamming loss? Errors per genre and route.

Hamming loss is the share of individual clip-genre decisions that are wrong. On the
five-genre subset the emotion route scores 0.406, worse than direct VGGish (0.358), AST
(0.311), the base-rate random guess (0.323) and the most-frequent baseline that predicts
only Drama (0.218), although it has the best macro-F1. This script splits the Hamming
loss of each route into its genres and into false positives and false negatives, and
counts how many genres each route predicts per clip against the 1.54 true ones.

Nothing is re-fitted. The counts are derived from what `film_level.json` already stores
for every route (clip-level precision and recall per genre, averaged over the ten
repeats) and the number of clips per genre:

    TP = recall * positives,   FN = positives - TP,   FP = TP / precision - TP

Because precision and recall are averaged over repeats before this is applied, the
derived Hamming loss is an approximation; the script checks it against the value stored
in `metrics_stability.json` and fails if they differ by more than 0.005.

Run:  python experiments/diagnostics/exp_hamming_breakdown.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features import load_set1  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

ARMS = {
    "ground-truth emotion(11) [ceiling]": "ratings (ceiling)",
    "VGGish -> predicted emotion(11), OOF-trained": "emotion route",
    "VGGish-128 (direct)": "direct VGGish",
    "AST-768": "direct AST",
    "PCA-8(VGGish) [control]": "PCA-8 control",
    "dummy": "most frequent",
}


def main() -> None:
    fl = json.load(open(RES / "film_level.json", encoding="utf-8"))["subset5"]
    stored = json.load(open(RES / "metrics_stability.json", encoding="utf-8"))
    stored = stored["metrics"]["hamming_loss"]
    genres = fl["genres"]

    df = load_set1()
    Y = df[genres].astype(int).values
    Y = Y[Y.sum(1) > 0]
    n, pos = len(Y), Y.sum(0)
    base = pos / n

    out = {
        "what": "Hamming loss of each route split by genre and into false positives and "
                "false negatives (five-genre subset, clip level, seed 42, mean of ten "
                "repeats); derived from the per-genre clip precision and recall stored in "
                "film_level.json",
        "n_clips": int(n),
        "genres": genres,
        "positives_per_genre": dict(zip(genres, pos.tolist())),
        "base_rate": dict(zip(genres, np.round(base, 3).tolist())),
        "true_genres_per_clip": round(float(Y.sum(1).mean()), 3),
        "random_guess_hamming": round(float(np.mean(2 * base * (1 - base))), 3),
        "arms": {},
    }
    for key, label in ARMS.items():
        a = fl["arms"][key]
        prec = np.array(a["clip_per_genre_precision"])
        rec = np.array(a["clip_per_genre_recall"])
        tp = rec * pos
        fn = pos - tp
        fp = np.where(prec > 0, tp / np.maximum(prec, 1e-12) - tp, 0.0)
        derived = float((fp.sum() + fn.sum()) / (len(genres) * n))
        ref = stored["arms"][key]["mean"]
        assert abs(derived - ref) < 0.005, (label, derived, ref)
        out["arms"][label] = {
            "hamming_loss_stored": round(ref, 3),
            "hamming_loss_derived": round(derived, 3),
            "predicted_genres_per_clip": round(float((tp + fp).sum() / n), 2),
            "false_positives_per_clip": round(float(fp.sum() / n), 2),
            "false_negatives_per_clip": round(float(fn.sum() / n), 2),
            "error_rate_per_genre": dict(zip(genres, np.round((fp + fn) / n, 3).tolist())),
            "predicted_clips_per_genre": dict(zip(genres, np.round(tp + fp).astype(int).tolist())),
            "false_positives_per_genre": dict(zip(genres, np.round(fp).astype(int).tolist())),
            "false_negatives_per_genre": dict(zip(genres, np.round(fn).astype(int).tolist())),
        }

    text = json.dumps(out, indent=2) + "\n"
    with io.open(RES / "hamming_breakdown.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"true genres per clip {out['true_genres_per_clip']}, "
          f"random guess Hamming {out['random_guess_hamming']}")
    for label, r in out["arms"].items():
        print(f"{label:18} Hamming {r['hamming_loss_stored']:.3f} "
              f"(derived {r['hamming_loss_derived']:.3f})  "
              f"predicted/clip {r['predicted_genres_per_clip']:.2f}  "
              f"FP/clip {r['false_positives_per_clip']:.2f}  "
              f"FN/clip {r['false_negatives_per_clip']:.2f}")
        print(f"{'':18} error per genre {r['error_rate_per_genre']}")


if __name__ == "__main__":
    main()
