"""Learning curve: would a larger dataset improve genre classification?

Trains on increasing fractions of the FILMS (subsampled by soundtrack so grouping is
preserved), evaluates Macro-F1 on the held-out GroupKFold folds, for the emotion (11)
and AST (768) feature sets. A still-rising curve at 100% => the dataset size limits
performance (more data would help); a flat curve => signal-limited.

Run:  python notebooks/exp_learning_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.models import build_classifier  # noqa: E402
from src.utils import set_seed  # noqa: E402

FRACS = [0.4, 0.55, 0.7, 0.85, 1.0]
N_SUBSAMPLES = 4  # random film subsets per fraction (except 1.0), for a stable estimate


def learning_curve(X, Y, groups):
    out = []
    for frac in FRACS:
        f1s = []
        for tr, te in GroupKFold(5).split(X, Y, groups):
            films = np.unique(groups[tr])
            seeds = [0] if frac >= 0.999 else range(N_SUBSAMPLES)
            for s in seeds:
                rng = np.random.RandomState(s)
                k = max(2, int(round(frac * len(films))))
                sel = set(rng.choice(films, k, replace=False))
                idx = tr[np.array([g in sel for g in groups[tr]])]
                clf = build_classifier("logreg")
                clf.fit(X[idx], Y[idx])
                pred = np.asarray(clf.predict(X[te]))
                f1s.append(f1_score(Y[te], pred, average="macro", zero_division=0))
        out.append((frac, float(np.mean(f1s)), float(np.std(f1s))))
    return out


def main():
    set_seed()
    df = add_derived_features(load_set1())
    Y = genre_matrix(df)
    groups = df["soundtrack"].to_numpy()
    E = df[config.FEATURE_COLS].to_numpy(float)
    A = assemble_from_cache(df, "set1")
    n_train_films = df["soundtrack"].nunique() * 0.8  # approx films per train fold

    le = learning_curve(E, Y, groups)
    la = learning_curve(A, Y, groups)
    print(f"{'frac':>6}{'~films':>8}{'~clips':>8} | {'emotion F1':>13} | {'AST F1':>11}")
    print("-" * 52)
    for (f, me, se), (_, ma, sa) in zip(le, la):
        print(f"{f:>6.2f}{f * n_train_films:>8.0f}{f * len(df) * 0.8:>8.0f} | "
              f"{me:>8.3f}±{se:.2f} | {ma:>6.3f}±{sa:.2f}")
    print("\nRising at frac=1.0 => dataset-size-limited (more data would help); "
          "flat => signal-limited.")


if __name__ == "__main__":
    main()
