"""The all-positive classifier under every metric of the metric table (Appendix A.2).

The classifier that assigns every genre to every clip has learned nothing, yet it reaches a
macro-F1 of 0.434 over films on the five-genre subset (``per_genre_analysis.json``). The
author asked on 9 Nov for it to stand beside the most-frequent baseline and the random guess
in the metric table, so that each metric's weakness is visible: macro-F1 is fooled by saying
yes to everything, Hamming loss and exact match by saying only "Drama", and ROC-AUC by
neither. Its predictions do not depend on any training, so every value follows from the
labels alone; it is computed with the metric functions of ``exp_metrics_stability.py``.
The constant score gives every genre the same rank for every clip; any per-genre threshold
tuned on it keeps all clips positive, since that beats predicting none on F1.

Run:  python experiments/evaluation/exp_all_positive_metrics.py
Writes results/all_positive_metrics.json.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.features import genre_matrix, load_set1  # noqa: E402
from exp_metrics_stability import METRICS  # noqa: E402

OUT = config.RESULTS_DIR / "all_positive_metrics.json"
PER_GENRE = config.RESULTS_DIR / "per_genre_analysis.json"


def main() -> None:
    df = load_set1()
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    Y = Y8[keep][:, idx]
    films = df[keep]["soundtrack"].to_numpy().astype(str)
    P = np.ones(Y.shape)                  # the same score for every clip and genre
    H = np.ones(Y.shape, dtype=int)       # every genre for every clip
    scores = {m: float(fn(Y, P, H, H, films)) for m, fn in METRICS.items()}

    ref = json.load(open(PER_GENRE, encoding="utf-8"))["all_positive_floor"]["subset5"]
    assert abs(scores["film_level_macro_f1"] - ref["film_macro"]) < 1e-9
    assert abs(scores["macro_f1"] - ref["clip_macro"]) < 1e-9

    out = {"n_clips": int(len(Y)), "n_films": int(len(np.unique(films))),
           "genres": list(config.GENRE_SUBSET), "metrics": scores}
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    for m, v in scores.items():
        print(f"  {m:28}{v:.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
