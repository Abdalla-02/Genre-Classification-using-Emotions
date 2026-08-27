"""Statistical-power pass: repeated GroupKFold, confidence intervals, corrected
significance tests, and a nested-CV-tuned protocol for every headline comparison.

Why this exists
---------------
All earlier comparisons (emotion vs AST, predicted-emotion vs PCA-8 control, ...) rested
on ONE 5-fold GroupKFold run with sklearn's default hyper-parameters. Five folds cannot
separate a +0.05 Macro-F1 margin from partition noise, so real and consistent effects
were reported as "within noise" -- an artefact of the experiment's resolution, not
evidence of no effect. This script fixes both problems:

* **Repetition.** The film -> fold assignment is re-randomised 10 times, giving
  10 x 5 = 50 leakage-safe folds. Every arm is scored on the SAME folds, so all
  comparisons are paired.
* **Tuning.** Section B re-runs every arm with the logistic regression's ``C`` chosen by
  an INNER GroupKFold inside each training fold (nested CV). sklearn's default C=1.0 is
  measurably too weak a penalty here, and tuning it on the test folds would be cheating;
  nested CV is the honest version. Section A keeps C=1.0 so the effect of tuning is
  visible and the older reported numbers remain comparable.

Reported per arm: mean Macro-F1 with a 95% t-interval over the 10 per-repeat means
(folds within a repeat share training data and are not independent). Reported per
comparison: the Nadeau & Bengio (2003) corrected resampled paired t-test, its infinite-
repeat limit ``p_lim``, and a nonparametric win rate.

Run:  python experiments/evaluation/exp_statistical_power.py [--repeats 10] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from src.features import (  # noqa: E402
    add_derived_features,
    assemble_from_cache,
    build_emotion_features,
    genre_matrix,
    load_set1,
)
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

CEIL = "ground-truth emotion(11) [ceiling]"
PRED = "VGGish -> PREDICTED emotion(11)"
PCA8 = "PCA-8(VGGish) [control]"
VGG = "VGGish-128 (direct)"
AST = "AST-768"
CLAP = "CLAP-512"
DUMMY = "dummy"


# --------------------------------------------------------------------------- #
# Arms. Each returns predictions for one fold; everything an arm learns (emotion
# regressor, PCA, chosen C, classifier) is fit on the TRAINING fold only.
# --------------------------------------------------------------------------- #
def _fit_logreg(Xtr, Ytr, gtr, tune: bool, record: list):
    """Fit the BR logistic regression, optionally choosing C by inner GroupKFold."""
    C = select_logreg_C(Xtr, Ytr, gtr) if tune else 1.0
    record.append(C)
    return build_classifier("logreg", C=C).fit(Xtr, Ytr)


def arm_plain(X, clf_name="logreg", tune=False):
    """Fixed feature matrix -> classifier."""
    record: list = []

    def run(Y, tr, te, groups):
        if clf_name != "logreg":
            clf = build_classifier(clf_name).fit(X[tr], Y[tr])
        else:
            clf = _fit_logreg(X[tr], Y[tr], groups[tr], tune, record)
        return np.asarray(clf.predict(X[te]))
    run.record = record
    return run


def arm_pca(X, n_components=8, tune=False):
    """PCA compression of X, FIT ON THE TRAINING FOLD ONLY (the bottleneck control)."""
    record: list = []

    def run(Y, tr, te, groups):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=n_components,
                  random_state=config.SEED).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        clf = _fit_logreg(Ztr, Y[tr], groups[tr], tune, record)
        return np.asarray(clf.predict(Zte))
    run.record = record
    return run


def arm_predicted_emotion(X, Emo, n_estimators=300, tune=False):
    """X -> RandomForest emotion regressor (fit on train) -> 11 features -> genre."""
    record: list = []

    def run(Y, tr, te, groups):
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=config.SEED,
                                    n_jobs=-1).fit(X[tr], Emo[tr])
        Ztr, Zte = build_emotion_features(reg.predict(X[tr])), build_emotion_features(reg.predict(X[te]))
        clf = _fit_logreg(Ztr, Y[tr], groups[tr], tune, record)
        return np.asarray(clf.predict(Zte))
    run.record = record
    return run


def score_arms(arms: dict, Y, folds, groups) -> dict[str, np.ndarray]:
    """Score every arm on the SAME folds -> {arm: per-fold Macro-F1 array}."""
    out = {name: np.zeros(len(folds)) for name in arms}
    for i, (tr, te) in enumerate(folds):
        for name, run in arms.items():
            pred = run(Y, tr, te, groups)
            out[name][i] = f1_score(Y[te], pred, average="macro", zero_division=0)
        print(f"  fold {i + 1}/{len(folds)} done", end="\r", flush=True)
    print(" " * 40, end="\r")
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report_arms(title, scores, n_splits, order=None, arms=None):
    names = list(order or scores)
    wide = max([len("arm")] + [len(n) for n in names]) + 2
    line = "-" * (wide + 34)
    print(f"\n{'=' * len(line)}\n{title}\n{'=' * len(line)}")
    print(f"{'arm':{wide}}{'MacroF1':>9}  {'95% CI':>20}  {'C':>6}")
    print(line)
    rows = {}
    for name in names:
        ci = repeat_ci(scores[name], n_splits)
        rec = getattr(arms.get(name), "record", []) if arms else []
        cstr = ""
        if rec:
            mode, cnt = Counter(rec).most_common(1)[0]
            cstr = f"{mode:g}" + ("" if cnt == len(rec) else "*")
        rows[name] = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi,
                      "fold_std": float(np.std(scores[name])),
                      "C_selected": sorted(Counter(rec).items()) if rec else None}
        print(f"{name:{wide}}{ci.mean:>9.3f}  [{ci.lo:>7.3f}, {ci.hi:>7.3f}]  {cstr:>6}")
    if any(getattr(arms.get(n), "record", []) for n in names) if arms else False:
        print("\n  C = most frequently selected inner-CV value "
              "(* = not selected in every fold).")
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


def make_arms(Xemo, Xast, Xclap, Xvgg, Emo, tune, quick):
    arms = {
        CEIL: arm_plain(Xemo, tune=tune),
        AST: arm_plain(Xast, tune=tune),
        CLAP: arm_plain(Xclap, tune=tune),
        VGG: arm_plain(Xvgg, tune=tune),
        PCA8: arm_pca(Xvgg, tune=tune),
        DUMMY: arm_plain(Xvgg, DUMMY),
    }
    if not quick:
        arms[PRED] = arm_predicted_emotion(Xvgg, Emo, tune=tune)
    return arms


def run_protocol(label, arms, Y, folds, groups, n_splits, n_train, n_test, n_clips, quick):
    print(f"\nscoring: {label} ...")
    s = score_arms(arms, Y, folds, groups)
    order = [k for k in [PRED, CEIL, PCA8, VGG, AST, CLAP, DUMMY] if k in s]
    rows = report_arms(f"{label} -- 5-genre subset (n={n_clips}), Macro-F1",
                       s, n_splits, order, arms)
    pairs = [(CEIL, AST), (CEIL, CLAP), (CEIL, VGG), (AST, CLAP)]
    if not quick:
        pairs = [(PRED, PCA8), (PRED, VGG), (PRED, AST), (PRED, CEIL)] + pairs
    cmp = report_comparisons(f"{label} -- paired comparisons", s, pairs,
                             n_splits, n_train, n_test)
    return s, rows, cmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--quick", action="store_true",
                    help="skip the expensive predicted-emotion arm")
    ap.add_argument("--no-tuned", action="store_true",
                    help="skip section B (nested-CV tuned protocol)")
    args = ap.parse_args()
    set_seed()

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    g8 = df["soundtrack"].to_numpy()
    Xemo8 = df[config.FEATURE_COLS].to_numpy(float)
    Xast8 = assemble_from_cache(df, "set1")

    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y5 = Y8[keep][:, idx]
    g5 = dfk["soundtrack"].to_numpy()
    Xemo5 = dfk[config.FEATURE_COLS].to_numpy(float)
    Xast5 = assemble_from_cache(dfk, "set1")
    Xclap5 = assemble_from_cache(dfk, "set1", cache_dir=config.CLAP_EMBEDDINGS_DIR)
    Xvgg5 = assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Emo5 = dfk[config.EMOTIONS].to_numpy(float)

    cv = RepeatedGroupKFold(args.splits, args.repeats)
    print(f"Repeated GroupKFold: {args.repeats} repeats x {args.splits} folds = "
          f"{args.repeats * args.splits} leakage-safe folds "
          f"(film -> fold re-randomised per repeat)")

    folds5 = list(cv.split(np.zeros(len(dfk)), None, g5))
    n_test5 = float(np.mean([len(te) for _, te in folds5]))
    n_train5 = len(dfk) - n_test5
    print(f"\n5-genre subset {config.GENRE_SUBSET}: n={len(dfk)} clips, "
          f"{dfk['soundtrack'].nunique()} films, mean test fold {n_test5:.0f}")

    out = {"design": {"n_repeats": args.repeats, "n_splits": args.splits,
                      "scheme": "RepeatedGroupKFold by soundtrack", "metric": "macro_f1",
                      "seed": config.SEED},
           "subset5": {"n_clips": int(len(dfk)),
                       "n_films": int(dfk["soundtrack"].nunique()),
                       "genres": config.GENRE_SUBSET}}

    # ---- Section A: sklearn defaults (C=1.0), comparable to earlier reports ---- #
    armsA = make_arms(Xemo5, Xast5, Xclap5, Xvgg5, Emo5, tune=False, quick=args.quick)
    sA, rowsA, cmpA = run_protocol("A. DEFAULT C=1.0", armsA, Y5, folds5, g5,
                                   args.splits, n_train5, n_test5, len(dfk), args.quick)
    out["subset5"]["default"] = {"arms": rowsA, "comparisons": cmpA,
                                 "per_fold": {k: v.tolist() for k, v in sA.items()}}

    # ---- Section B: nested CV, C chosen inside each training fold ---- #
    if not args.no_tuned:
        armsB = make_arms(Xemo5, Xast5, Xclap5, Xvgg5, Emo5, tune=True, quick=args.quick)
        sB, rowsB, cmpB = run_protocol("B. NESTED-CV TUNED C", armsB, Y5, folds5, g5,
                                       args.splits, n_train5, n_test5, len(dfk),
                                       args.quick)
        out["subset5"]["tuned"] = {"arms": rowsB, "comparisons": cmpB,
                                   "per_fold": {k: v.tolist() for k, v in sB.items()}}
        print(f"\n{'=' * 60}\nEFFECT OF TUNING (B - A), Macro-F1\n{'=' * 60}")
        for k in rowsB:
            print(f"  {k:44}{rowsA[k]['mean']:.3f} -> {rowsB[k]['mean']:.3f}  "
                  f"({rowsB[k]['mean'] - rowsA[k]['mean']:+.3f})")

    # ---- full 8 genres (default + tuned emotion vs AST) ---- #
    folds8 = list(cv.split(np.zeros(len(df)), None, g8))
    n_test8 = float(np.mean([len(te) for _, te in folds8]))
    n_train8 = len(df) - n_test8
    EMO8 = "ground-truth emotion(11)"
    arms8 = {EMO8: arm_plain(Xemo8, tune=not args.no_tuned),
             AST: arm_plain(Xast8, tune=not args.no_tuned),
             DUMMY: arm_plain(Xast8, DUMMY)}
    print(f"\nscoring 8-genre arms (n={len(df)}) ...")
    s8 = score_arms(arms8, Y8, folds8, g8)
    proto = "nested-CV tuned" if not args.no_tuned else "default C=1.0"
    rows8 = report_arms(f"FULL 8 GENRES (n={len(df)}), {proto}, Macro-F1",
                        s8, args.splits, None, arms8)
    cmp8 = report_comparisons("FULL 8 GENRES: paired comparisons", s8, [(EMO8, AST)],
                              args.splits, n_train8, n_test8)
    out["full8"] = {"n_clips": int(len(df)), "n_films": int(df["soundtrack"].nunique()),
                    "genres": config.PRIMARY_GENRES, "protocol": proto, "arms": rows8,
                    "comparisons": cmp8,
                    "per_fold": {k: v.tolist() for k, v in s8.items()}}

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # A reduced run (fewer repeats than the default, or --quick) is a debug run:
    # write it to a scratch name so it cannot silently replace the canonical
    # results file that the progress log and the LaTeX snippets quote.
    full = args.repeats >= N_REPEATS and not args.quick
    path = config.RESULTS_DIR / ("statistical_power.json" if full else "statistical_power.partial.json")
    if not full:
        print("  (reduced run -> written to a .partial.json scratch file)")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
