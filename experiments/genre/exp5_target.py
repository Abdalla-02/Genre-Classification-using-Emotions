"""Experiment 5 (sanity check): AST embedding -> 12 discrete-emotion TARGET classes.

Single-label, class-balanced (30 clips x 12 categories = 360). This is the cheapest
end-to-end validation of the audio -> embedding -> classifier pipeline before the
harder multi-label genre work. Reports Accuracy and Macro-F1 under both
StratifiedKFold and GroupKFold(by soundtrack); the gap between them quantifies
film-identity leakage (the same GroupKFold-vs-Stratified logic used for genre).

Run (after experiments/features/extract_features.py):  python experiments/genre/exp5_target.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import assemble_from_cache, load_set1  # noqa: E402
from src.utils import set_seed  # noqa: E402

N_SPLITS = 5


def _clf(name: str):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=config.SEED),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, random_state=config.SEED, n_jobs=-1
        )
    raise ValueError(name)


def _cv_scores(clf_name, X, y, groups, cv, use_groups):
    accs, f1s = [], []
    split_iter = cv.split(X, y, groups) if use_groups else cv.split(X, y)
    for train_idx, test_idx in split_iter:
        clf = _clf(clf_name)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))
    return np.array(accs), np.array(f1s)


def main() -> None:
    set_seed()
    df = load_set1(clean=False)  # all 360 clips (TARGET is genre-independent)
    assert df["TARGET"].nunique() == 12, df["TARGET"].value_counts()

    X = assemble_from_cache(df, "set1")
    y = df["TARGET"].to_numpy()
    groups = df["soundtrack"].to_numpy()
    print(f"X={X.shape}  classes={df['TARGET'].nunique()}  "
          f"chance={1/df['TARGET'].nunique():.3f}  soundtracks={df['soundtrack'].nunique()}")

    strat = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.SEED)
    group = GroupKFold(n_splits=N_SPLITS)

    for clf_name in ("logreg", "rf"):
        s_acc, s_f1 = _cv_scores(clf_name, X, y, groups, strat, use_groups=False)
        g_acc, g_f1 = _cv_scores(clf_name, X, y, groups, group, use_groups=True)
        print(f"\n=== {clf_name} ===")
        print(f"  StratifiedKFold : acc={s_acc.mean():.3f}+/-{s_acc.std():.3f}  "
              f"macroF1={s_f1.mean():.3f}")
        print(f"  GroupKFold(film): acc={g_acc.mean():.3f}+/-{g_acc.std():.3f}  "
              f"macroF1={g_f1.mean():.3f}")
        print(f"  GroupKFold-Gap  : acc {s_acc.mean() - g_acc.mean():+.3f}  "
              f"(leakage from same-film clips)")


if __name__ == "__main__":
    main()
