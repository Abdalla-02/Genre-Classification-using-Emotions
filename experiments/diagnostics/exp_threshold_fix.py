"""Fix test: does curbing the balanced model's over-prediction recover Macro-F1?

Diagnosis (see exp_error_analysis.py): balanced LogReg predicts 3.49 labels/clip vs 1.87
true -> flooded with false positives -> low precision -> low Macro-F1. Here we compute
out-of-fold (GroupKFold) probabilities for balanced vs unbalanced LogReg on the
WITH-emotion features, then sweep the decision threshold and report Macro-F1, macro
precision/recall, and predicted labels/clip.

Reported at two operating points: sklearn's default C=1.0 (the originally reported
Section 7c numbers) and the nested-CV-selected C (Section 11b).

Run:  python experiments/diagnostics/exp_threshold_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import add_derived_features, genre_matrix, load_set1  # noqa: E402
from src.models import build_binary_logreg  # noqa: E402
from src.utils import set_seed  # noqa: E402

TUNED_C = 0.003  # C selected by the nested CV in Section 11b


def oof_proba(X, Y, groups, balanced, C=1.0):
    """Out-of-fold P(genre) via GroupKFold; single-class folds -> train prevalence.

    Uses the shared ``build_binary_logreg`` definition so this analysis tracks any change
    to the thesis's logistic regression. ``C`` defaults to sklearn's 1.0 for continuity
    with the originally reported numbers; the tuned value is passed in separately (see
    Section 11b -- the default is measurably under-regularised).
    """
    P = np.zeros(Y.shape, dtype=float)
    for tr, te in GroupKFold(5).split(X, Y, groups):
        for j in range(Y.shape[1]):
            y = Y[tr, j]
            if len(np.unique(y)) < 2:
                P[te, j] = y.mean()
            else:
                m = build_binary_logreg(C=C, balanced=balanced).fit(X[tr], y)
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
    configs = [("balanced (current) thr=0.5", True, 0.5),
               ("balanced thr=0.6", True, 0.6),
               ("balanced thr=0.7", True, 0.7),
               ("unbalanced thr=0.5", False, 0.5)]

    # Run the comparison at BOTH operating points. The originally reported numbers
    # (Section 7c) used sklearn's default C=1.0, which Section 11b showed to be badly
    # under-regularised for this corpus; TUNED_C is the value the nested CV selects.
    # "Does curbing over-prediction help?" has to be answered at the operating point the
    # thesis actually reports, not at the default.
    probs = {}
    for label, C in [("A. default C=1.0", 1.0), (f"B. tuned C={TUNED_C:g}", TUNED_C)]:
        print(f"\n### {label} ###")
        print(hdr + "\n" + "-" * len(hdr))
        pb = oof_proba(X, Y, groups, True, C=C)
        pu = oof_proba(X, Y, groups, False, C=C)
        for name, bal, thr in configs:
            f1, mp, mr, card = scores_at(pb if bal else pu, Y, thr)
            print(f"{name:28}{f1:>8.3f}{mp:>6.2f}{mr:>6.2f}{card:>10.2f}")
        probs[label] = (pb, pu)

    # everything below uses the TUNED operating point
    P_bal, P_unb = probs[f"B. tuned C={TUNED_C:g}"]

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
