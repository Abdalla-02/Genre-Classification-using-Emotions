"""Which emotions carry the genre signal? Ablation over the 8 emotion features.

Supervisor request: "remove some of the emotions and test if it makes a big difference."
Two kinds of ablation, both on the ground-truth ratings (the clean test of whether the
INFORMATION matters, uncontaminated by how well each emotion can be predicted):

  1. Leave-one-out: drop each emotion in turn. A large drop marks an emotion the
     classifier depends on; no drop marks one that is redundant given the other seven.
  2. Theory-motivated subsets. The Fundamentals chapter argues that valence and arousal
     alone cannot separate Action from Horror and that the discrete categories carry
     the genre-discriminating information. That is a testable claim:
        - valence + energy            the two-dimensional circumplex model
        - valence + energy + tension  Eerola & Vuoskoski's three-dimensional model
        - the five discrete emotions  anger, fear, happy, sad, tender
        - fear alone                  section 7b's "master discriminator"
        - fear + valence              the two strongest axes from the Cohen's d analysis

Protocol as everywhere: 5-genre subset, 5x5 repeated GroupKFold by film, nested-CV-tuned
C, Macro-F1, Nadeau-Bengio corrected paired test against the full 8-emotion model.
Derived features (valence x energy, composites) are omitted throughout so that every arm
is a clean subset of the same 8 raw ratings; the 11-feature model is reported once as a
reference row.

Run:  python experiments/genre/exp_emotion_ablation.py [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.repeated import (  # noqa: E402
    RepeatedGroupKFold,
    corrected_paired_t,
    diff_ci,
    repeat_ci,
    win_rate,
)
from src.features import add_derived_features, genre_matrix, load_set1  # noqa: E402
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

RESULTS = config.RESULTS_DIR / "emotion_ablation.json"
E = config.EMOTIONS

SUBSETS = {
    "all 8 emotions [reference]": E,
    "valence + energy  (2-d circumplex)": ["valence", "energy"],
    "valence + energy + tension  (3-d dimensional)": ["valence", "energy", "tension"],
    "5 discrete only  (anger fear happy sad tender)": ["anger", "fear", "happy", "sad", "tender"],
    "fear only": ["fear"],
    "fear + valence": ["fear", "valence"],
}


def score_arms(arms: dict[str, np.ndarray], Y, folds, groups):
    out = {k: np.zeros(len(folds)) for k in arms}
    for i, (tr, te) in enumerate(folds):
        for name, X in arms.items():
            C = select_logreg_C(X[tr], Y[tr], groups[tr])
            pred = build_classifier("logreg", C=C).fit(X[tr], Y[tr]).predict(X[te])
            out[name][i] = f1_score(Y[te], pred, average="macro", zero_division=0)
        print(f"  fold {i + 1}/{len(folds)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return out


def report(title, scores, ref, splits, n_train, n_test):
    w = max(len(k) for k in scores) + 2
    print(f"\n{title}")
    print(f"{'features':{w}}{'MacroF1':>9}  {'95% CI':>18}{'vs ref':>9}{'p':>8}{'win':>6}")
    print("-" * (w + 52))
    rows = {}
    for name in scores:
        ci = repeat_ci(scores[name], splits)
        if name == ref:
            print(f"{name:{w}}{ci.mean:>9.3f}  [{ci.lo:.3f}, {ci.hi:.3f}]")
            rows[name] = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi}
            continue
        d = diff_ci(scores[name], scores[ref], splits)
        _, p = corrected_paired_t(scores[name], scores[ref], n_train, n_test)
        wr = win_rate(scores[ref], scores[name])
        print(f"{name:{w}}{ci.mean:>9.3f}  [{ci.lo:.3f}, {ci.hi:.3f}]{d.mean:>+9.3f}"
              f"{p:>8.3f}{'*' if p < 0.05 else ' '}{wr:>5.0%}")
        rows[name] = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi,
                      "diff_vs_ref": d.mean, "p_corrected": p, "ref_win_rate": wr}
    print("\n  vs ref = change relative to all 8 emotions; p = Nadeau-Bengio corrected;")
    print("  win = fraction of folds on which the full 8-emotion model beats this arm.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    set_seed()

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    g = dfk["soundtrack"].to_numpy()
    folds = list(RepeatedGroupKFold(args.splits, args.repeats).split(np.zeros(len(dfk)), None, g))
    n_test = float(np.mean([len(te) for _, te in folds]))
    n_train = len(dfk) - n_test

    def feats(cols): return dfk[list(cols)].to_numpy(float)

    print("=" * 78)
    print(f"EMOTION ABLATION -- 5-genre subset, n={len(dfk)}, {args.repeats}x{args.splits} "
          f"repeated GroupKFold, tuned C")
    print("=" * 78)

    # ---- leave-one-out ---------------------------------------------------------
    REF = "all 8 emotions [reference]"
    loo = {REF: feats(E)}
    for e in E:
        loo[f"without {e}"] = feats([x for x in E if x != e])
    loo["11 features (8 + derived) [headline model]"] = dfk[config.FEATURE_COLS].to_numpy(float)
    print("\nscoring leave-one-out ...")
    s1 = score_arms(loo, Y, folds, g)
    rows1 = report("1. LEAVE-ONE-EMOTION-OUT", s1, REF, args.splits, n_train, n_test)

    # ---- theory-motivated subsets ----------------------------------------------
    sub = {k: feats(v) for k, v in SUBSETS.items()}
    print("\nscoring theory-motivated subsets ...")
    s2 = score_arms(sub, Y, folds, g)
    rows2 = report("2. THEORY-MOTIVATED SUBSETS", s2, REF, args.splits, n_train, n_test)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "design": {"n_repeats": args.repeats, "n_splits": args.splits,
                   "genres": config.GENRE_SUBSET, "n_clips": int(len(dfk)),
                   "features": "ground-truth ratings", "metric": "macro_f1"},
        "leave_one_out": {"arms": rows1, "per_fold": {k: v.tolist() for k, v in s1.items()}},
        "subsets": {"arms": rows2, "per_fold": {k: v.tolist() for k, v in s2.items()}},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
