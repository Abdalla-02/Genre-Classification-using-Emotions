"""Model / feature search -- can anything beat plain LogReg on the 11 emotion features?

This reproduces the comparison recorded in Section 7d of the progress log, which until now
existed only as a table there: no script in the repository produced it, so the numbers
could not be re-derived or re-checked. It also repairs a real problem with that table.
The original figures were all measured with scikit-learn's default ``C=1.0``, which
Section 11b later showed to be the *worst* setting for every feature set here. The claim
that "0.281 is a robust ceiling" was therefore made at the wrong operating point -- and
proper regularisation alone exceeds it.

Every arm is scored twice:

  A  default C=1.0        -- comparable with the originally reported Section 7d numbers
  B  nested-CV tuned C    -- C chosen by an inner GroupKFold inside each training fold

Arms without a ``C`` (gradient boosting, classifier chains) are unchanged between the two
protocols and are reported once per protocol for completeness.

8 genres, repeated GroupKFold by film, Macro-F1.

Run:  python experiments/diagnostics/exp_model_search.py [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.repeated import (  # noqa: E402
    RepeatedGroupKFold,
    corrected_paired_t,
    diff_ci,
    repeat_ci,
    win_rate,
)
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.models import build_binary_logreg, build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

BASE = "LogReg + emotion(11)  [baseline]"
AST = "LogReg + AST(768)"
FUSE = "emotion + AST fusion (779)"
FUSEP = "emotion + PCA-8(AST) fusion (19)"
GB = "HistGradientBoosting + emotion"
CHAIN = "ClassifierChain(LogReg) + emotion"


def _logreg(Xtr, Ytr, gtr, tune, rec):
    C = select_logreg_C(Xtr, Ytr, gtr) if tune else 1.0
    rec.append(C)
    return build_classifier("logreg", C=C).fit(Xtr, Ytr)


def arm_logreg(X, tune):
    rec: list = []

    def run(Y, tr, te, g):
        return np.asarray(_logreg(X[tr], Y[tr], g[tr], tune, rec).predict(X[te]))
    run.record = rec
    return run


def arm_fusion_pca(Xe, Xa, n_components, tune):
    """Emotion features concatenated with a PCA compression of AST (PCA fit on train)."""
    rec: list = []

    def run(Y, tr, te, g):
        sc = StandardScaler().fit(Xa[tr])
        pca = PCA(n_components=n_components, random_state=config.SEED).fit(sc.transform(Xa[tr]))
        Ztr = np.hstack([Xe[tr], pca.transform(sc.transform(Xa[tr]))])
        Zte = np.hstack([Xe[te], pca.transform(sc.transform(Xa[te]))])
        return np.asarray(_logreg(Ztr, Y[tr], g[tr], tune, rec).predict(Zte))
    run.record = rec
    return run


def arm_gb(X):
    """Binary-relevance histogram gradient boosting (one model per genre)."""
    def run(Y, tr, te, g):
        out = np.zeros((len(te), Y.shape[1]), dtype=int)
        for j in range(Y.shape[1]):
            y = Y[tr, j]
            if len(np.unique(y)) < 2:
                out[:, j] = int(y[0])
                continue
            m = HistGradientBoostingClassifier(random_state=config.SEED).fit(X[tr], y)
            out[:, j] = m.predict(X[te])
        return out
    return run


def arm_chain(X, C=1.0):
    """Classifier chains: each genre also sees the predictions of earlier genres."""
    def run(Y, tr, te, g):
        # chains need every link to see both classes; drop all-constant genres
        keep = [j for j in range(Y.shape[1]) if len(np.unique(Y[tr, j])) > 1]
        out = np.zeros((len(te), Y.shape[1]), dtype=int)
        for j in range(Y.shape[1]):
            if j not in keep:
                out[:, j] = int(Y[tr, j][0])
        if keep:
            cc = ClassifierChain(build_binary_logreg(C=C, balanced=True),
                                 order="random", random_state=config.SEED)
            cc.fit(X[tr], Y[tr][:, keep])
            out[:, keep] = np.asarray(cc.predict(X[te])).astype(int)
        return out
    return run


def score(arms, Y, folds, groups):
    out = {k: np.zeros(len(folds)) for k in arms}
    for i, (tr, te) in enumerate(folds):
        for k, run in arms.items():
            out[k][i] = f1_score(Y[te], run(Y, tr, te, groups),
                                 average="macro", zero_division=0)
        print(f"  fold {i + 1}/{len(folds)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return out


def report(title, scores, n_splits, arms, order):
    wide = max(len(x) for x in order) + 2
    line = "-" * (wide + 34)
    print(f"\n{'=' * len(line)}\n{title}\n{'=' * len(line)}")
    print(f"{'arm':{wide}}{'MacroF1':>9}  {'95% CI':>20}  {'C':>6}")
    print(line)
    rows = {}
    for k in order:
        ci = repeat_ci(scores[k], n_splits)
        rec = getattr(arms[k], "record", [])
        cs = ""
        if rec:
            vals, counts = np.unique(rec, return_counts=True)
            cs = f"{vals[counts.argmax()]:g}"
        rows[k] = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi}
        print(f"{k:{wide}}{ci.mean:>9.3f}  [{ci.lo:>7.3f}, {ci.hi:>7.3f}]  {cs:>6}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    set_seed()

    df = add_derived_features(load_set1())
    Y = genre_matrix(df)
    g = df["soundtrack"].to_numpy()
    Xe = df[config.FEATURE_COLS].to_numpy(float)
    Xa = assemble_from_cache(df, "set1")
    Xf = np.hstack([Xe, Xa])

    folds = list(RepeatedGroupKFold(args.splits, args.repeats).split(np.zeros(len(df)), None, g))
    n_test = float(np.mean([len(te) for _, te in folds]))
    n_train = len(df) - n_test
    print(f"Model/feature search -- 8 genres, n={len(df)} clips, "
          f"{args.repeats}x{args.splits} repeated GroupKFold")

    order = [BASE, AST, FUSE, FUSEP, GB, CHAIN]
    out = {"design": {"n_repeats": args.repeats, "n_splits": args.splits,
                      "n_clips": int(len(df)), "metric": "macro_f1"}}
    results = {}
    for label, tune, key in [("A. DEFAULT C=1.0", False, "default"),
                             ("B. NESTED-CV TUNED C", True, "tuned")]:
        arms = {
            BASE: arm_logreg(Xe, tune),
            AST: arm_logreg(Xa, tune),
            FUSE: arm_logreg(Xf, tune),
            FUSEP: arm_fusion_pca(Xe, Xa, 8, tune),
            GB: arm_gb(Xe),
            CHAIN: arm_chain(Xe, C=0.003 if tune else 1.0),
        }
        print(f"\nscoring {label} ...")
        s = score(arms, Y, folds, g)
        rows = report(f"{label} -- 8 genres, Macro-F1", s, args.splits, arms, order)
        results[key] = s
        out[key] = {"arms": rows, "per_fold": {k: v.tolist() for k, v in s.items()}}

    # did tuning change the conclusion?
    print(f"\n{'=' * 66}\nEFFECT OF TUNING (B - A)\n{'=' * 66}")
    for k in order:
        a, b = results["default"][k].mean(), results["tuned"][k].mean()
        print(f"  {k:38}{a:.3f} -> {b:.3f}  ({b - a:+.3f})")

    print(f"\n{'=' * 66}\nDoes anything beat the tuned baseline?\n{'=' * 66}")
    base = results["tuned"][BASE]
    cmp_rows = []
    for k in order[1:]:
        d = diff_ci(results["tuned"][k], base, args.splits)
        _, p = corrected_paired_t(results["tuned"][k], base, n_train, n_test)
        w = win_rate(results["tuned"][k], base)
        print(f"  {k:38}{d.mean:>+7.3f}  p={p:>6.3f}  wins {w:>4.0%}")
        cmp_rows.append({"arm": k, "diff_vs_baseline": d.mean, "p_corrected": p,
                         "win_rate": w})
    out["comparisons_vs_tuned_baseline"] = cmp_rows

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # A reduced run (fewer repeats than the default, or --quick) is a debug run:
    # write it to a scratch name so it cannot silently replace the canonical
    # results file that the progress log and the LaTeX snippets quote.
    full = args.repeats >= 5
    path = config.RESULTS_DIR / ("model_search.json" if full else "model_search.partial.json")
    if not full:
        print("  (reduced run -> written to a .partial.json scratch file)")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
