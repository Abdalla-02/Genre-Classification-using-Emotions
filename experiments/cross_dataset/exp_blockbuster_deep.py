"""Blockbuster, properly: repeated CV, cue-level (MIL) representations, full MIR set.

Why this exists
---------------
Every Blockbuster number reported so far (VGGish 0.582 vs MFCC 0.455 in section 3;
predicted-emotion 0.565 vs PCA-8 0.559 in section 10) came from ONE 5-fold split at
sklearn's default C -- precisely the flaw that section 11 diagnosed and fixed for the
Eerola corpus, and then never applied to this one. Five folds cannot separate the margins
at stake, and the default C was shown to be the worst setting for every feature set. This
script re-runs the target corpus under the same protocol used for the primary results:
repeated CV, C chosen by an inner CV inside each training fold, per-repeat confidence
intervals and the Nadeau-Bengio corrected paired test.

It also uses two things the earlier scripts discarded:

* **The full 140-feature MIR set.** ``load_blockbuster`` keeps only the 78 MFCC-family
  columns, because the supervisor's question was specifically "VGGish vs MFCC". The
  remaining 62 columns (roughness, key strength, pulse clarity, spectral shape, ...) are
  exactly the descriptors most associated with musical affect, so excluding them
  understates the hand-crafted baseline.
* **The cue level.** Each film is a bag of individually scored music cues -- 4664 in
  total, median 39 per film, median duration 19 s. Collapsing a film to the mean of its
  cues throws that away, and it creates the domain shift that limits cross-corpus
  transfer: an Eerola clip is ~17 s of music, whereas a film-level mean over 39 cues is a
  far smoother object (per-dimension SD 50.4 vs 59.9). Predicting emotion PER CUE and
  aggregating afterwards keeps the model on the unit it was trained on.

Fold design: the evaluation unit is always the FILM. Part A splits films directly. In the
cue-level parts the split is still over films and cues follow their film, since cues of
one film share a label -- splitting cues at random would leak the bag label.

Run:  python experiments/cross_dataset/exp_blockbuster_deep.py
      python experiments/cross_dataset/exp_blockbuster_deep.py --repeats 3   # quick
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation import macro_prf  # noqa: E402
from src.evaluation.repeated import (  # noqa: E402
    corrected_paired_t,
    diff_ci,
    nb_p_limit,
    repeat_ci,
    win_rate,
)
from src.features import (  # noqa: E402
    assemble_from_cache,
    build_emotion_features,
    load_set1,
)
from src.features.blockbuster import (  # noqa: E402
    aggregate_cues,
    load_blockbuster,
    load_blockbuster_cues,
    mir_feature_names,
)
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

RESULTS = config.RESULTS_DIR / "blockbuster_deep.json"
_MFCC_PREFIXES = ("mfcc", "deltamfcc", "deltadeltamfcc")


# --------------------------------------------------------------------------- #
# Emotion regressor: trained ONCE on all of Eerola. Blockbuster never trains it,
# so it may be fitted outside the CV loop without leaking anything.
# --------------------------------------------------------------------------- #
def eerola_emotion_regressor() -> RandomForestRegressor:
    df = load_set1(clean=False)
    X = assemble_from_cache(df, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    E = df[config.EMOTIONS].to_numpy(float)
    return RandomForestRegressor(n_estimators=300, random_state=config.SEED,
                                 n_jobs=-1).fit(X, E)


# --------------------------------------------------------------------------- #
# Arms. Each maps (train films, test films) -> predicted label matrix for the test films.
# --------------------------------------------------------------------------- #
def arm_film_level(X):
    """Fixed film-level feature matrix -> tuned Binary-Relevance logistic regression."""
    rec: list = []

    def run(Y, tr, te):
        C = select_logreg_C(X[tr], Y[tr], np.arange(len(tr)))
        rec.append(C)
        return np.asarray(build_classifier("logreg", C=C).fit(X[tr], Y[tr]).predict(X[te]))
    run.record = rec
    return run


def arm_film_level_rf(X):
    def run(Y, tr, te):
        return np.asarray(build_classifier("rf").fit(X[tr], Y[tr]).predict(X[te]))
    run.record = []
    return run


def arm_pca(X, n=8, projection="pca"):
    """Generic low-dimensional bottleneck control, fitted on the training films only."""
    rec: list = []

    def run(Y, tr, te):
        sc = StandardScaler().fit(X[tr])
        if projection == "pca":
            red = PCA(n_components=n, random_state=config.SEED).fit(sc.transform(X[tr]))
        else:
            red = GaussianRandomProjection(
                n_components=n, random_state=config.SEED).fit(sc.transform(X[tr]))
        Ztr, Zte = red.transform(sc.transform(X[tr])), red.transform(sc.transform(X[te]))
        C = select_logreg_C(Ztr, Y[tr], np.arange(len(tr)))
        rec.append(C)
        return np.asarray(build_classifier("logreg", C=C).fit(Ztr, Y[tr]).predict(Zte))
    run.record = rec
    return run


def arm_imv(Xc, cue_film, n_films):
    """Instance Majority Voting (Ma et al.): train on cues, vote per film.

    Cues inherit their film's label. The fold split is over films, so a film's cues are
    entirely in train or entirely in test -- otherwise the bag label leaks.
    """
    rec: list = []

    def run(Y, tr, te):
        m_tr = np.isin(cue_film, tr)
        m_te = np.isin(cue_film, te)
        Yc = Y[cue_film[m_tr]]
        C = select_logreg_C(Xc[m_tr], Yc, cue_film[m_tr])
        rec.append(C)
        clf = build_classifier("logreg", C=C).fit(Xc[m_tr], Yc)
        pred_cue = np.asarray(clf.predict(Xc[m_te]))
        # majority vote within each test film
        votes = aggregate_cues(pred_cue.astype(float), cue_film[m_te], n_films)
        return (votes[te] > 0.5).astype(int)
    run.record = rec
    return run


def arm_fixed(F):
    """A feature matrix that is already film-level (e.g. cue-averaged emotions)."""
    return arm_film_level(F)


# --------------------------------------------------------------------------- #
def score(arms, Y, folds) -> dict[str, np.ndarray]:
    out = {k: np.zeros(len(folds)) for k in arms}
    for i, (tr, te) in enumerate(folds):
        for name, run in arms.items():
            out[name][i] = f1_score(Y[te], run(Y, tr, te), average="macro",
                                    zero_division=0)
        print(f"  fold {i + 1}/{len(folds)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return out


def report(title, scores, n_splits, arms=None, order=None):
    names = list(order or sorted(scores, key=lambda k: -scores[k].mean()))
    w = max(len(n) for n in names) + 2
    print(f"\n{'=' * (w + 34)}\n{title}\n{'=' * (w + 34)}")
    print(f"{'arm':{w}}{'MacroF1':>9}  {'95% CI':>20}{'C':>7}")
    print("-" * (w + 38))
    rows = {}
    for n in names:
        ci = repeat_ci(scores[n], n_splits)
        rec = getattr(arms.get(n), "record", []) if arms else []
        c = f"{Counter(rec).most_common(1)[0][0]:g}" if rec else ""
        rows[n] = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi}
        print(f"{n:{w}}{ci.mean:>9.3f}  [{ci.lo:>7.3f}, {ci.hi:>7.3f}]{c:>7}")
    return rows


def compare(title, scores, pairs, n_splits, n_train, n_test):
    labels = [f"{a}  vs  {b}" for a, b in pairs]
    w = max(len(x) for x in labels) + 2
    print(f"\n{title}")
    print(f"{'comparison':{w}}{'diff':>8}{'p':>9}{'p_lim':>8}{'win':>6}")
    print("-" * (w + 31))
    rows = []
    for (a, b), lab in zip(pairs, labels):
        d = diff_ci(scores[a], scores[b], n_splits)
        _, p = corrected_paired_t(scores[a], scores[b], n_train, n_test)
        pl = nb_p_limit(scores[a], scores[b], n_train, n_test)
        wr = win_rate(scores[a], scores[b])
        print(f"{lab:{w}}{d.mean:>+8.3f}{p:>9.3f}{'*' if p < 0.05 else ' '}{pl:>7.3f}{wr:>6.0%}")
        rows.append({"a": a, "b": b, "diff": d.mean, "ci_lo": d.lo, "ci_hi": d.hi,
                     "p_corrected": p, "p_limit": pl, "win_rate": wr})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--rf-repeats", type=int, default=3)
    args = ap.parse_args()
    set_seed()

    film = load_blockbuster()
    cue = load_blockbuster_cues()
    Y, films = film["Y"], film["films"]
    n_films = len(films)
    names = mir_feature_names()
    mfcc_idx = [i for i, n in enumerate(names) if n.startswith(_MFCC_PREFIXES)]

    print(f"Blockbuster: {n_films} films, {Y.shape[1]} genres, "
          f"{cue['X_vggish'].shape[0]} VGGish cues / {cue['X_mir'].shape[0]} MIR cues")
    print(f"MIR features: {len(names)} total, {len(mfcc_idx)} of them MFCC-family\n")

    # ---- film-level feature matrices -------------------------------------- #
    Xv = film["X_vggish"]
    Xmfcc = film["X_mfcc"]
    Xmir140 = aggregate_cues(cue["X_mir"], cue["cue_film_mir"], n_films)

    # ---- the emotion bridge, at two granularities ------------------------- #
    reg = eerola_emotion_regressor()
    F_film = build_emotion_features(reg.predict(Xv))                 # pool THEN predict
    Ecue = reg.predict(cue["X_vggish"])                              # predict THEN pool
    F_cue = build_emotion_features(
        aggregate_cues(Ecue, cue["cue_film_vggish"], n_films))

    VGG, MFCC, MIR140 = "VGGish-128", "MFCC-78", "MIR-140 (full)"
    EMO_C, EMO_F = "emotion(11), per-cue -> pooled", "emotion(11), pooled -> per-film"
    PCA8, RND8, IMV = "PCA-8(VGGish) [control]", "random-8(VGGish) [control]", \
        "VGGish, instance majority voting"
    DUM = "dummy"

    arms = {
        VGG: arm_film_level(Xv),
        MFCC: arm_film_level(Xmfcc),
        MIR140: arm_film_level(Xmir140),
        EMO_C: arm_fixed(F_cue),
        EMO_F: arm_fixed(F_film),
        PCA8: arm_pca(Xv),
        RND8: arm_pca(Xv, projection="rand"),
        IMV: arm_imv(cue["X_vggish"], cue["cue_film_vggish"], n_films),
        DUM: lambda Y_, tr, te: np.asarray(
            build_classifier("dummy").fit(Xv[tr], Y_[tr]).predict(Xv[te])),
    }

    rng = np.random.default_rng(config.SEED)
    folds = []
    for r in range(args.repeats):
        kf = KFold(args.splits, shuffle=True, random_state=config.SEED + r)
        folds += list(kf.split(np.arange(n_films)))
    n_test = float(np.mean([len(te) for _, te in folds]))
    n_train = n_films - n_test
    print(f"protocol: {args.repeats} x {args.splits} repeated KFold over films "
          f"= {len(folds)} folds, nested-CV-tuned C")
    print("(films are independent rows -- unlike Eerola clips there is nothing to group "
          "by;\n cue-level arms still split by FILM so a bag label cannot leak)\n")

    s = score(arms, Y, folds)
    rows = report(f"A. Blockbuster in-domain, {args.repeats}x{args.splits} repeated "
                  f"KFold, Macro-F1", s, args.splits, arms)
    cmp_rows = compare("paired comparisons (Nadeau-Bengio corrected)", s, [
        (VGG, MFCC), (MIR140, MFCC), (VGG, MIR140),
        (EMO_C, VGG), (EMO_C, PCA8), (EMO_C, EMO_F), (EMO_F, PCA8),
        (PCA8, RND8), (IMV, VGG),
    ], args.splits, n_train, n_test)

    # ---- random-guess floor, Ma et al.'s definition ----------------------- #
    p = Y.mean(0)
    rg = float(np.mean([macro_prf(Y, (rng.random(Y.shape) < p).astype(int))["macro_f1"]
                        for _ in range(300)]))
    print(f"\n  floors: dummy (most-frequent) {rows[DUM]['mean']:.3f} | "
          f"random-guess by base rate {rg:.3f} (Ma et al. report .32)")

    # ---- Part B: does the classifier choice change the ordering? ---------- #
    print(f"\nscoring RandomForest robustness check ({args.rf_repeats}x{args.splits}) ...")
    folds_rf = folds[: args.rf_repeats * args.splits]
    arms_rf = {VGG: arm_film_level_rf(Xv), MFCC: arm_film_level_rf(Xmfcc),
               MIR140: arm_film_level_rf(Xmir140), EMO_C: arm_film_level_rf(F_cue)}
    s_rf = score(arms_rf, Y, folds_rf)
    rows_rf = report(f"B. Same arms with RandomForest ({args.rf_repeats}x{args.splits})",
                     s_rf, args.splits, arms_rf)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "design": {"n_repeats": args.repeats, "n_splits": args.splits,
                   "scheme": "RepeatedKFold over films", "metric": "macro_f1",
                   "seed": config.SEED, "n_films": n_films,
                   "n_cues_vggish": int(cue["X_vggish"].shape[0]),
                   "n_cues_mir": int(cue["X_mir"].shape[0])},
        "logreg": {"arms": rows, "comparisons": cmp_rows,
                   "per_fold": {k: v.tolist() for k, v in s.items()}},
        "random_forest": {"arms": rows_rf,
                          "per_fold": {k: v.tolist() for k, v in s_rf.items()}},
        "random_guess_floor": rg,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
