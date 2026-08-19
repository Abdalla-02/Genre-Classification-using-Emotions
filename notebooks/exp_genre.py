"""Genre classification: WITH emotion ratings vs WITHOUT (direct AST baseline).

  * WITH   : 11 features = 8 CSV emotion ratings + 3 engineered (ground-truth emotions
             -> ceiling of the emotion->genre relationship, RQ1).
  * WITHOUT: 768-dim AST embedding (direct audio->genre baseline, RQ2b).

For each feature set: 4 classifiers x 3 CV schemes reporting Exact Match, Hamming Loss
and Macro-F1 (mean +/- std over folds):
  * group  -- GroupKFold by film (leakage-safe, REPORTED)
  * kfold  -- plain shuffled KFold (leaky, unstratified)
  * strat  -- StratifiedKFold on first_genre (leaky, stratified; thesis comparator)
Two gaps isolate the cause of inflation: leakage = (kfold - group); the strat gap
adds any stratification effect. A paired t-test across folds checks whether WITH beats
WITHOUT (underpowered at 5 folds -- reported with that caveat).

Run:  python notebooks/exp_genre.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluation import evaluate_folds  # noqa: E402
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.models import CLASSIFIERS  # noqa: E402
from src.utils import set_seed  # noqa: E402

SCHEMES = ["group", "kfold", "strat"]


def _run_feature_set(X, Y, groups, strat):
    """clf -> scheme -> {metric: per-fold array}."""
    return {clf: {s: evaluate_folds(X, Y, groups, strat, clf, s) for s in SCHEMES}
            for clf in CLASSIFIERS}


def _print_block(title, X, res):
    print(f"\n{'=' * 78}\n{title}   (X={X.shape})\n{'=' * 78}")
    hdr = f"{'clf':7} {'scheme':6} {'ExactMatch':>13} {'Hamming':>13} {'MacroF1':>13}"
    print(hdr + "\n" + "-" * len(hdr))
    for clf in CLASSIFIERS:
        for s in SCHEMES:
            f = res[clf][s]
            def ms(m):
                return f"{f[m].mean():.3f}±{f[m].std():.2f}"
            print(f"{clf:7} {s:6} {ms('exact_match'):>13} {ms('hamming_loss'):>13} "
                  f"{ms('macro_f1'):>13}")
        g = res[clf]["group"]["macro_f1"].mean()
        leak = res[clf]["kfold"]["macro_f1"].mean() - g
        strat = res[clf]["strat"]["macro_f1"].mean() - g
        print(f"{'':7} {'gaps':6}  MacroF1 leakage(kfold-group)={leak:+.3f}  "
              f"strat-group={strat:+.3f}")
        print("-" * len(hdr))


def main():
    set_seed()
    df = add_derived_features(load_set1())
    Y = genre_matrix(df)
    groups = df["soundtrack"].to_numpy()
    strat = df["first_genre"].to_numpy()

    print(f"clips={len(df)}  genres={Y.shape[1]}  soundtracks={df['soundtrack'].nunique()}")
    print("genre base rates: " + "  ".join(
        f"{g}={r:.2f}" for g, r in zip(config.PRIMARY_GENRES, Y.mean(axis=0))))

    X_emo = df[config.FEATURE_COLS].to_numpy(dtype=float)
    X_ast = assemble_from_cache(df, "set1")

    res_emo = _run_feature_set(X_emo, Y, groups, strat)
    res_ast = _run_feature_set(X_ast, Y, groups, strat)
    _print_block("WITH emotion ratings (11 features, ground truth)", X_emo, res_emo)
    _print_block("WITHOUT emotion (768-dim AST embedding, direct baseline)", X_ast, res_ast)

    print(f"\n{'=' * 78}\nWITH vs WITHOUT -- paired across GroupKFold folds (Macro-F1)\n"
          f"{'=' * 78}")
    print("  (5 folds -> underpowered; read as effect size, not proof)")
    for clf in CLASSIFIERS:
        a = res_emo[clf]["group"]["macro_f1"]
        b = res_ast[clf]["group"]["macro_f1"]
        diff = a - b
        if np.allclose(diff, 0):
            p = float("nan")
        else:
            p = ttest_rel(a, b).pvalue
        print(f"  {clf:7}: emotion {a.mean():.3f} vs AST {b.mean():.3f}  "
              f"diff={diff.mean():+.3f}+/-{diff.std():.2f}  paired-t p={p:.3f}")


if __name__ == "__main__":
    main()
