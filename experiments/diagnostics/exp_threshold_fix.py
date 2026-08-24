"""Fix test: does curbing the balanced model's over-prediction recover Macro-F1?

Diagnosis (see exp_error_analysis.py): balanced LogReg predicts 3.49 labels/clip vs 1.87
true -> flooded with false positives -> low precision -> low Macro-F1. Here we compute
out-of-fold (GroupKFold) probabilities for balanced vs unbalanced LogReg on the
WITH-emotion features, then sweep the decision threshold and report Macro-F1, macro
precision/recall, and predicted labels/clip.

Run:  python experiments/diagnostics/exp_threshold_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import add_derived_features, genre_matrix, load_set1  # noqa: E402
from src.utils import set_seed  # noqa: E402


def oof_proba(X, Y, groups, balanced):
    """Out-of-fold P(genre) via GroupKFold; single-class folds -> train prevalence."""
    P = np.zeros(Y.shape, dtype=float)
    cw = "balanced" if balanced else None
    for tr, te in GroupKFold(5).split(X, Y, groups):
        for j in range(Y.shape[1]):
            y = Y[tr, j]
            if len(np.unique(y)) < 2:
                P[te, j] = y.mean()
            else:
                m = make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=2000, class_weight=cw, random_state=config.SEED))
                m.fit(X[tr], y)
                P[te, j] = m.predict_proba(X[te])[:, 1]
    return P


def scores_at(P, Y, thr):
    pred = (P >= thr).astype(int)
    return (f1_score(Y, pred, average="macro", zero_division=0),
            precision_score(Y, pred, average="macro", zero_division=0),
            recall_score(Y, pred, average="macro", zero_division=0),
            pred.sum(1).mean())


def main():
    set_seed()
    df = add_derived_features(load_set1())
    X = df[config.FEATURE_COLS].to_numpy(float)
    Y = genre_matrix(df)
    groups = df["soundtrack"].to_numpy()
    G = config.PRIMARY_GENRES
    print(f"true labels/clip = {Y.sum(1).mean():.2f}\n")

    hdr = f"{'config':28}{'MacroF1':>8}{'mP':>6}{'mR':>6}{'pred/clip':>10}"
    print(hdr + "\n" + "-" * len(hdr))
    configs = [("balanced (current) thr=0.5", True, 0.5),
               ("balanced thr=0.6", True, 0.6),
               ("balanced thr=0.7", True, 0.7),
               ("unbalanced thr=0.5", False, 0.5)]
    P_bal = oof_proba(X, Y, groups, True)
    P_unb = oof_proba(X, Y, groups, False)
    for name, bal, thr in configs:
        f1, mp, mr, card = scores_at(P_bal if bal else P_unb, Y, thr)
        print(f"{name:28}{f1:>8.3f}{mp:>6.2f}{mr:>6.2f}{card:>10.2f}")

    # per-genre F1-optimal threshold on the balanced OOF probs (OPTIMISTIC upper bound:
    # threshold chosen on the eval set -> a ceiling, not an honest estimate)
    best = np.zeros(len(G)); thr_star = np.zeros(len(G))
    grid = np.linspace(0.1, 0.9, 33)
    for j in range(len(G)):
        f1s = [f1_score(Y[:, j], (P_bal[:, j] >= t).astype(int), zero_division=0) for t in grid]
        k = int(np.argmax(f1s)); best[j] = f1s[k]; thr_star[j] = grid[k]
    print(f"\n{'per-genre F1-tuned (OPTIMISTIC upper bound)':43}MacroF1={best.mean():.3f}")

    # report per-genre F1 for the best realistic config vs baseline
    def pergenre(P, thr):
        pred = (P >= thr).astype(int)
        return {G[j]: f1_score(Y[:, j], pred[:, j], zero_division=0) for j in range(len(G))}
    base = pergenre(P_bal, 0.5)
    print("\nper-genre F1:  baseline(bal,0.5) -> best fixed config")
    best_name = "unbalanced thr=0.5"
    bestP, bestT = P_unb, 0.5
    fixcfg = pergenre(bestP, bestT)
    for g in G:
        print(f"  {g:12} {base[g]:.2f} -> {fixcfg[g]:.2f}")
    print(f"\nbest realistic: '{best_name}'  (see table)")


if __name__ == "__main__":
    main()
