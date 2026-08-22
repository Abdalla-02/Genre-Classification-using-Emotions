"""Per-genre error analysis for the multi-label genre classification (GroupKFold OOF).

For each genre reports, from out-of-fold predictions:
  support, base rate, recall, FN% (=missed positives), precision, FP% (of predicted
  positives that are wrong), FPR (=negatives wrongly flagged), F1.
Run for WITH-emotion (11 ground-truth features) and the WITHOUT/AST baseline (768-d).

Run:  python experiments/exp_error_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.models import build_classifier  # noqa: E402
from src.utils import set_seed  # noqa: E402


def per_genre_report(name, X, Y, groups):
    pred = cross_val_predict(build_classifier("logreg"), X, Y,
                             cv=GroupKFold(5).split(X, Y, groups))
    G = config.PRIMARY_GENRES
    print(f"\n{'=' * 86}\n{name}  (X={X.shape}) | avg true labels/clip="
          f"{Y.sum(1).mean():.2f}  avg predicted/clip={pred.sum(1).mean():.2f}\n{'=' * 86}")
    hdr = (f"{'genre':11}{'supp':>5}{'base':>6}{'recall':>7}{'FN%':>6}"
           f"{'prec':>6}{'FP%':>6}{'FPR':>6}{'F1':>6}")
    print(hdr + "\n" + "-" * len(hdr))
    for j, g in enumerate(G):
        yt, yp = Y[:, j], pred[:, j]
        TP = int(((yt == 1) & (yp == 1)).sum()); FP = int(((yt == 0) & (yp == 1)).sum())
        FN = int(((yt == 1) & (yp == 0)).sum()); TN = int(((yt == 0) & (yp == 0)).sum())
        recall = TP / (TP + FN) if TP + FN else 0
        fn_pct = FN / (TP + FN) if TP + FN else 0
        prec = TP / (TP + FP) if TP + FP else 0
        fp_pct = FP / (TP + FP) if TP + FP else 0        # of predicted-positive, share wrong
        fpr = FP / (FP + TN) if FP + TN else 0           # of true-negatives, share flagged
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0
        print(f"{g:11}{TP + FN:>5}{yt.mean():>6.2f}{recall:>7.2f}{fn_pct:>6.0%}"
              f"{prec:>6.2f}{fp_pct:>6.0%}{fpr:>6.0%}{f1:>6.2f}")


def main():
    set_seed()
    df = add_derived_features(load_set1())
    Y = genre_matrix(df)
    groups = df["soundtrack"].to_numpy()
    per_genre_report("WITH emotion (11 ground-truth features)",
                     df[config.FEATURE_COLS].to_numpy(float), Y, groups)
    per_genre_report("WITHOUT emotion (768-d AST embedding)",
                     assemble_from_cache(df, "set1"), Y, groups)


if __name__ == "__main__":
    main()
