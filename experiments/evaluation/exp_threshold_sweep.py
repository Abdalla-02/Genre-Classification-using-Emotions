"""What happens to precision, recall and F1 when the global decision threshold moves?

Every genre result of the thesis cuts the film-averaged probability at 0.5. The author asked
on 9 Nov whether another cut gives better precision while F1 stays high. This script runs
the pipelines of ``exp_film_level.py`` on the same 10x5 folds (seed 42), keeps their
out-of-fold probabilities, and scores them at film level for one global threshold between
0.20 and 0.80. The same threshold is used for every genre and chosen by nobody: this is a
description of the trade-off, not a tuned result (the threshold tuned per genre on the
training folds is in ``metrics_stability.json``). At 0.5 every arm must reproduce
``film_level.json``.

Run:  python experiments/evaluation/exp_threshold_sweep.py [--jobs -1]
Writes results/threshold_sweep.json.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.utils import set_seed  # noqa: E402
from exp_cv_corrected import CEIL, MUSICNN, PCA8, PRED_OOF, VGG  # noqa: E402
from exp_film_level import Films, run  # noqa: E402

OUT = config.RESULTS_DIR / "threshold_sweep.json"
FILM_LEVEL = config.RESULTS_DIR / "film_level.json"
THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.801, 0.05)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    set_seed()

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    films = dfk["soundtrack"].to_numpy().astype(str)
    E = dfk[config.EMOTIONS].to_numpy(float)
    Xvgg = assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    arms = {
        CEIL: ("plain", dfk[config.FEATURE_COLS].to_numpy(float), None),
        PRED_OOF: ("pred_oof", Xvgg, E),
        VGG: ("plain", Xvgg, None),
        MUSICNN: ("plain", assemble_from_cache(dfk, "set1", cache_dir=config.MUSICNN_EMBEDDINGS_DIR), None),
        PCA8: ("pca8", Xvgg, None),
    }
    runs, _ = run(arms, Y, films, config.SEED, args.jobs)
    fl = Films(Y, films)
    ref = json.load(open(FILM_LEVEL, encoding="utf-8"))["subset5"]["arms"]

    out = {"design": {"unit": "film (clip probabilities averaged per film)",
                      "protocol": "exp_film_level.py arms and folds, seed 42, mean of ten repeats",
                      "threshold": "one global cut for every genre, not tuned"},
           "thresholds": THRESHOLDS, "true_genres_per_film": float(fl.Y.sum(1).mean()), "arms": {}}
    for a, r in runs.items():
        rows = []
        for t in THRESHOLDS:
            D = [(fl.M @ p >= t).astype(int) for p in r["P"]]
            rows.append({
                "threshold": t,
                "precision": float(np.mean([precision_score(fl.Y, d, average="macro", zero_division=0) for d in D])),
                "recall": float(np.mean([recall_score(fl.Y, d, average="macro", zero_division=0) for d in D])),
                "f1": float(np.mean([f1_score(fl.Y, d, average="macro", zero_division=0) for d in D])),
                "genres_per_film": float(np.mean([d.sum(1).mean() for d in D])),
            })
        at_half = next(x for x in rows if x["threshold"] == 0.5)["f1"]
        assert abs(at_half - ref[a]["mean"]) < 1e-9, (a, at_half, ref[a]["mean"])
        out["arms"][a] = rows
        print(f"\n{a}")
        for x in rows:
            print(f"  t={x['threshold']:.2f}  P={x['precision']:.3f}  R={x['recall']:.3f}  "
                  f"F1={x['f1']:.3f}  genres/film={x['genres_per_film']:.2f}")
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
