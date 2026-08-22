"""Genre classification on the curated 5-genre subset (Action, Crime, Drama, Comedy,
Horror) -- the genres with an emotional signature and precedent in the literature
(Austin 2010: 4; Ma 2021: 6 at comparable scale). Clips are kept if they have at least
one of the five genres (mirrors Ma et al.). Compares WITH-emotion (11) vs AST (768) on
Macro-F1, Exact Match (subset accuracy) and Hamming loss under GroupKFold, against the
full 8-genre result.

Run:  python experiments/exp_genre_subset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.utils import set_seed  # noqa: E402

SUBSET = ["Action", "Crime", "Drama", "Comedy", "Horror"]
CLFS = ["logreg", "rf", "dummy"]


def _block(title, X, Y, groups, strat, genres):
    print(f"\n{'=' * 70}\n{title}   (X={X.shape}, {len(genres)} genres)\n{'=' * 70}")
    print(f"{'clf':7} | {'MacroF1':>8} | {'ExactMatch':>11} | {'Hamming':>8}")
    print("-" * 44)
    for clf in CLFS:
        r = evaluate(X, Y, groups, strat, clf, "group")
        f1, em, ha = r["macro_f1"], r["exact_match"], r["hamming_loss"]
        print(f"{clf:7} | {f1[0]:.3f}    | {em[0]:.3f}       | {ha[0]:.3f}")


def main():
    set_seed()
    df = add_derived_features(load_set1())          # 346 clips, 8 genres
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(g) for g in SUBSET]
    Y5_all = Y8[:, idx]

    keep = Y5_all.sum(axis=1) >= 1                    # keep clips with >=1 of the 5
    dfk = df[keep].reset_index(drop=True)
    Y = Y5_all[keep]
    groups = dfk["soundtrack"].to_numpy()
    strat = np.array([SUBSET[i] for i in Y.argmax(axis=1)])  # first present of the 5

    print(f"5-genre subset: {SUBSET}")
    print(f"clips kept: {len(dfk)} of {len(df)}  (dropped {len(df) - len(dfk)} with none "
          f"of the 5)  |  soundtracks={dfk['soundtrack'].nunique()}")
    print("positives/genre: " + "  ".join(f"{g}={int(Y[:, j].sum())}"
                                           for j, g in enumerate(SUBSET)))
    print(f"avg labels/clip: {Y.sum(1).mean():.2f}")

    Xemo = dfk[config.FEATURE_COLS].to_numpy(float)
    Xast = assemble_from_cache(dfk, "set1")
    _block("WITH emotion (11 features)", Xemo, Y, groups, strat, SUBSET)
    _block("WITHOUT emotion (768-d AST)", Xast, Y, groups, strat, SUBSET)

    # paired significance test: emotion vs AST Macro-F1 across the same GroupKFold folds
    from scipy.stats import ttest_rel  # noqa: E402
    from src.evaluation import evaluate_folds  # noqa: E402
    fe = evaluate_folds(Xemo, Y, groups, strat, "logreg", "group")["macro_f1"]
    fa = evaluate_folds(Xast, Y, groups, strat, "logreg", "group")["macro_f1"]
    diff = fe - fa
    p = ttest_rel(fe, fa).pvalue if not np.allclose(diff, 0) else float("nan")
    print(f"\n{'=' * 70}\nPaired test (LogReg, Macro-F1 over GroupKFold folds)\n{'=' * 70}")
    print(f"  emotion {fe.mean():.3f} vs AST {fa.mean():.3f}  "
          f"diff={diff.mean():+.3f}+/-{diff.std():.2f}  paired-t p={p:.3f}  (5 folds)")

    # context: same models on the full 8 genres (all 346 clips)
    print(f"\n{'=' * 70}\nCONTEXT: full 8 genres (all 346 clips), LogReg MacroF1\n{'=' * 70}")
    g8 = df["soundtrack"].to_numpy()
    s8 = df["first_genre"].to_numpy()
    for name, X in [("emotion", df[config.FEATURE_COLS].to_numpy(float)),
                    ("AST", assemble_from_cache(df, "set1"))]:
        f1 = evaluate(X, Y8, g8, s8, "logreg", "group")["macro_f1"][0]
        print(f"  {name:8} 8-genre MacroF1 = {f1:.3f}")


if __name__ == "__main__":
    main()
