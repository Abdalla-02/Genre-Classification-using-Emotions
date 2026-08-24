"""Statistical-power pass: repeated GroupKFold + confidence intervals for every
headline comparison.

Why this exists
---------------
All previous comparisons (emotion vs AST, predicted-emotion vs PCA-8 control, ...)
rested on ONE 5-fold GroupKFold run. Five folds is far too few to separate a +0.05
Macro-F1 margin from fold noise, so several real-looking effects were reported as
"within noise" -- an artefact of the experiment's resolution, not evidence of no
effect. Here every arm is re-run under 10 x 5 repeated GroupKFold (50 leakage-safe
folds, film -> fold assignment re-randomised each repeat) and each comparison gets:

  * mean Macro-F1 with a 95% t-interval over the 10 per-repeat means (conservative),
  * the Nadeau & Bengio (2003) corrected resampled paired t-test (the naive paired
    t-test over 50 correlated folds would be anti-conservative),
  * a nonparametric win rate (fraction of the 50 folds the first arm wins).

All arms are scored on the SAME folds, so every comparison is properly paired.

Run:  python experiments/exp_statistical_power.py [--repeats 10] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluation.repeated import (  # noqa: E402
    N_REPEATS,
    N_SPLITS,
    RepeatedGroupKFold,
    corrected_paired_t,
    diff_ci,
    nb_p_limit,
    repeat_ci,
    win_rate,
)
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.models import build_classifier  # noqa: E402
from src.utils import set_seed  # noqa: E402

SUBSET = ["Action", "Crime", "Drama", "Comedy", "Horror"]

CEIL = "ground-truth emotion(11) [ceiling]"
PRED = "VGGish -> PREDICTED emotion(11)"
PCA8 = "PCA-8(VGGish) [control]"
VGG = "VGGish-128 (direct)"
AST = "AST-768"
EMO8 = "ground-truth emotion(11)"


# --------------------------------------------------------------------------- #
# Arms: each returns the predicted label matrix for one fold. Everything an arm
# learns (regressor, PCA, classifier) is fit on the training fold only.
# --------------------------------------------------------------------------- #
def build_11(E: np.ndarray) -> np.ndarray:
    """8 raw emotions -> the 11-feature genre-stage input (same recipe as elsewhere)."""
    d = {e: E[:, i] for i, e in enumerate(config.EMOTIONS)}
    return np.column_stack([E, d["valence"] * d["energy"],
                            np.mean([d["anger"], d["fear"], d["tension"], d["sad"]], 0),
                            np.mean([d["happy"], d["tender"], d["valence"]], 0)])


def arm_plain(X, clf_name="logreg"):
    """Fixed feature matrix -> classifier."""
    def run(Y, tr, te):
        clf = build_classifier(clf_name).fit(X[tr], Y[tr])
        return np.asarray(clf.predict(X[te]))
    return run


def arm_pca(X, n_components=8):
    """PCA compression of X, FIT ON THE TRAINING FOLD ONLY (the bottleneck control)."""
    def run(Y, tr, te):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=n_components,
                  random_state=config.SEED).fit(sc.transform(X[tr]))
        clf = build_classifier("logreg").fit(pca.transform(sc.transform(X[tr])), Y[tr])
        return np.asarray(clf.predict(pca.transform(sc.transform(X[te]))))
    return run


def arm_predicted_emotion(X, Emo, n_estimators=300):
    """X -> RandomForest emotion regressor (fit on train) -> 11 features -> genre."""
    def run(Y, tr, te):
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=config.SEED,
                                    n_jobs=-1).fit(X[tr], Emo[tr])
        clf = build_classifier("logreg").fit(build_11(reg.predict(X[tr])), Y[tr])
        return np.asarray(clf.predict(build_11(reg.predict(X[te]))))
    return run


def score_arms(arms: dict, Y, folds) -> dict[str, np.ndarray]:
    """Score every arm on the SAME folds -> {arm: per-fold Macro-F1 array}."""
    out = {name: np.zeros(len(folds)) for name in arms}
    for i, (tr, te) in enumerate(folds):
        for name, run in arms.items():
            pred = run(Y, tr, te)
            out[name][i] = f1_score(Y[te], pred, average="macro", zero_division=0)
        print(f"  fold {i + 1}/{len(folds)} done", end="\r", flush=True)
    print(" " * 40, end="\r")
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report_arms(title, scores, n_splits, order=None):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'arm':40}{'MacroF1':>9}  {'95% CI (per-repeat)':>22}")
    print("-" * 78)
    rows = {}
    for name in (order or scores):
        ci = repeat_ci(scores[name], n_splits)
        rows[name] = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi,
                      "fold_std": float(np.std(scores[name]))}
        print(f"{name:40}{ci.mean:>9.3f}  [{ci.lo:>8.3f}, {ci.hi:>8.3f}]")
    return rows


def report_comparisons(title, scores, pairs, n_splits, n_train, n_test):
    labels = [f"{a} vs {b}" for a, b in pairs]
    wide = max([len("comparison")] + [len(x) for x in labels]) + 2
    line = "-" * (wide + 52)
    print(f"\n{'=' * len(line)}\n{title}\n{'=' * len(line)}")
    print(f"{'comparison':{wide}}{'diff':>7}  {'95% CI of diff':>18} "
          f"{'p':>8} {'p_lim':>8} {'win':>6}")
    print(line)
    rows = []
    for (a, b), label in zip(pairs, labels):
        sa, sb = scores[a], scores[b]
        d = diff_ci(sa, sb, n_splits)
        _, p = corrected_paired_t(sa, sb, n_train, n_test)
        plim = nb_p_limit(sa, sb, n_train, n_test)
        w = win_rate(sa, sb)
        sig = "*" if (p == p and p < 0.05) else " "
        print(f"{label:{wide}}{d.mean:>+7.3f}  [{d.lo:>+6.3f}, {d.hi:>+6.3f}] "
              f"{p:>8.3f}{sig}{plim:>8.3f} {w:>5.0%}")
        rows.append({"a": a, "b": b, "diff": d.mean, "ci_lo": d.lo, "ci_hi": d.hi,
                     "p_corrected": p, "p_limit": plim, "win_rate": w})
    print("\n  p     = Nadeau-Bengio corrected resampled paired t-test; * = p<0.05.")
    print("  p_lim = the p this test converges to with INFINITE repeats. p_lim close to")
    print("          p means extra repeats cannot help; p_lim>0.05 means only more")
    print("          films can make this comparison significant.")
    print("  CI    = 95% t-interval over the per-repeat mean differences.")
    print("  win   = fraction of the folds the first arm wins.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--quick", action="store_true",
                    help="skip the expensive predicted-emotion arm")
    args = ap.parse_args()
    set_seed()

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    Xemo8 = df[config.FEATURE_COLS].to_numpy(float)
    Xast8 = assemble_from_cache(df, "set1")
    g8 = df["soundtrack"].to_numpy()

    idx = [config.PRIMARY_GENRES.index(x) for x in SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y5 = Y8[keep][:, idx]
    g5 = dfk["soundtrack"].to_numpy()
    Xemo5 = dfk[config.FEATURE_COLS].to_numpy(float)
    Xast5 = assemble_from_cache(dfk, "set1")
    Xvgg5 = assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Emo5 = dfk[config.EMOTIONS].to_numpy(float)

    cv = RepeatedGroupKFold(args.splits, args.repeats)
    print(f"Repeated GroupKFold: {args.repeats} repeats x {args.splits} folds = "
          f"{args.repeats * args.splits} leakage-safe folds "
          f"(film -> fold re-randomised per repeat)")

    # ---------------- 5-genre subset (the reported set) ---------------- #
    folds5 = list(cv.split(np.zeros(len(dfk)), None, g5))
    n_test5 = float(np.mean([len(te) for _, te in folds5]))
    n_train5 = len(dfk) - n_test5
    print(f"\n5-genre subset {SUBSET}: n={len(dfk)} clips, "
          f"{dfk['soundtrack'].nunique()} films, mean test fold {n_test5:.0f}")

    arms5 = {
        CEIL: arm_plain(Xemo5),
        AST: arm_plain(Xast5),
        VGG: arm_plain(Xvgg5),
        PCA8: arm_pca(Xvgg5),
        "dummy": arm_plain(Xvgg5, "dummy"),
    }
    if not args.quick:
        arms5[PRED] = arm_predicted_emotion(Xvgg5, Emo5)
    print("scoring 5-genre arms ...")
    s5 = score_arms(arms5, Y5, folds5)

    order5 = [k for k in [PRED, CEIL, PCA8, AST, VGG, "dummy"] if k in s5]
    rows5 = report_arms(f"5-GENRE SUBSET (n={len(dfk)}), repeated GroupKFold Macro-F1",
                        s5, args.splits, order5)

    pairs5 = [(CEIL, AST), (CEIL, VGG), (AST, VGG)]
    if not args.quick:
        pairs5 = [(PRED, PCA8), (PRED, VGG), (PRED, AST), (PRED, CEIL)] + pairs5
    cmp5 = report_comparisons("5-GENRE SUBSET: paired comparisons", s5, pairs5,
                              args.splits, n_train5, n_test5)

    # ---------------- full 8 genres ---------------- #
    folds8 = list(cv.split(np.zeros(len(df)), None, g8))
    n_test8 = float(np.mean([len(te) for _, te in folds8]))
    n_train8 = len(df) - n_test8
    arms8 = {
        EMO8: arm_plain(Xemo8),
        AST: arm_plain(Xast8),
        "dummy": arm_plain(Xast8, "dummy"),
    }
    print(f"\nscoring 8-genre arms (n={len(df)}) ...")
    s8 = score_arms(arms8, Y8, folds8)
    rows8 = report_arms(f"FULL 8 GENRES (n={len(df)}), repeated GroupKFold Macro-F1",
                        s8, args.splits)
    cmp8 = report_comparisons("FULL 8 GENRES: paired comparisons", s8,
                              [(EMO8, AST)], args.splits, n_train8, n_test8)

    out = {
        "design": {"n_repeats": args.repeats, "n_splits": args.splits,
                   "scheme": "RepeatedGroupKFold by soundtrack", "metric": "macro_f1",
                   "seed": config.SEED},
        "subset5": {"n_clips": int(len(dfk)), "n_films": int(dfk["soundtrack"].nunique()),
                    "genres": SUBSET, "arms": rows5, "comparisons": cmp5,
                    "per_fold": {k: v.tolist() for k, v in s5.items()}},
        "full8": {"n_clips": int(len(df)), "n_films": int(df["soundtrack"].nunique()),
                  "genres": config.PRIMARY_GENRES, "arms": rows8, "comparisons": cmp8,
                  "per_fold": {k: v.tolist() for k, v in s8.items()}},
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "statistical_power.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
