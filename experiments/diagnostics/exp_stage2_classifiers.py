"""Stage 2 of the emotion pipeline: is the logistic regression the right genre classifier?

Methods 4.4 says that more flexible classifiers were tried on the emotion features and that
none outperformed the regularised logistic regression. Until this script, only two of the
four alternatives had a saved result (``model_search.json``: classifier chains and gradient
boosting), and on an older protocol (eight genres, clip level, human ratings as input). The
random forest and the multi-layer perceptron rested on an early experiment of log section 4
without a results file. This script backs the sentence on the current protocol.

Protocol (the one of ``exp_film_level.py``, unchanged)
-----------------------------------------------------
Five genres, 329 clips / 40 films, 10x5 RepeatedGroupKFold by film, seed 42. Input: the 11
emotion features predicted from VGGish by the random forest regressor, trained on
out-of-fold emotions of the training clips (the thesis pipeline). The emotion features of a
fold are computed once and shared by every classifier, so the arms differ only in Stage 2.
Headline: film-level macro-F1 (clip probabilities averaged per film, genre at >= 0.5), with
clip-level macro-F1 from the same predictions beside it. Significance: paired film
bootstrap (2000 draws) against the logistic regression; clip-level Nadeau-Bengio beside it.
A second block repeats the comparison on the human ratings (the ceiling input).

Every alternative gets an inner grid search, as the logistic regression gets its C: the
same inner GroupKFold (3 folds by film, training data only), macro-F1, ties to the first
(simplest) setting. Arms:

  logistic regression   binary relevance, class-balanced, C tuned (the thesis classifier)
  random forest         400 trees, class-balanced, min_samples_leaf tuned
  MLP                   one hidden layer of 32 units, L2 penalty alpha tuned, early stopping
  classifier chain      the same logistic regression, each genre also sees the earlier
                        genres' predictions (random order), C tuned
  gradient boosting     HistGradientBoosting per genre, class-balanced, depth/rate tuned

The logistic-regression arm must reproduce ``film_level.json`` (VGGish -> predicted
emotion, OOF-trained) exactly at film level; the script checks this.

Run:  python experiments/diagnostics/exp_stage2_classifiers.py [--jobs -1] [--boot 2000]
Writes results/stage2_classifiers.json (a reduced run writes *.partial.json instead).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "evaluation"))

from src import config  # noqa: E402
from src.evaluation.repeated import RepeatedGroupKFold  # noqa: E402
from src.features import (  # noqa: E402
    add_derived_features,
    assemble_from_cache,
    build_emotion_features,
    genre_matrix,
    load_set1,
)
from src.models import build_binary_logreg, build_classifier, select_logreg_C  # noqa: E402
from src.models.classifiers import _SingleClassSafe  # noqa: E402
from src.utils import set_seed  # noqa: E402
from exp_cv_corrected import PRED_OOF, FilmBootstrap  # noqa: E402
from exp_film_level import Films, compare, summarise  # noqa: E402
from exp_metrics_stability import oof_emotions  # noqa: E402

RESULTS = config.RESULTS_DIR / "stage2_classifiers.json"
FILM_LEVEL = config.RESULTS_DIR / "film_level.json"
N_REPEATS, N_SPLITS = 10, 5

LR, RF, MLP, CHAIN, GB = ("logistic regression", "random forest", "MLP",
                          "classifier chain", "gradient boosting")
ARMS = (LR, RF, MLP, CHAIN, GB)
GRIDS = {
    RF: [{"min_samples_leaf": v} for v in (10, 5, 1)],
    MLP: [{"alpha": v} for v in (10.0, 1.0, 0.1, 0.01)],
    GB: [{"max_depth": d, "learning_rate": r} for d, r in ((2, 0.05), (3, 0.1), (None, 0.1))],
}


# --------------------------------------------------------------------------- #
# The classifiers
# --------------------------------------------------------------------------- #
def make(arm, p, seed):
    if arm == RF:
        return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                      random_state=seed, n_jobs=1, **p)
    if arm == MLP:
        return Pipeline([("scale", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                                               early_stopping=True, validation_fraction=0.2,
                                               random_state=seed, **p))])
    if arm == GB:
        return MultiOutputClassifier(_SingleClassSafe(HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed, **p)))
    if arm == CHAIN:
        # guarded like the logistic regression: in a few outer folds every film of a
        # genre is in the test fold, so the training data hold no positive of it
        return ClassifierChain(_SingleClassSafe(build_binary_logreg(C=p["C"], balanced=True)),
                               order="random", random_state=seed)
    if arm == LR:
        return build_classifier("logreg", C=p["C"])
    raise ValueError(arm)


def proba(clf, Z):
    """(n, genres) P(genre) for every arm (lists of per-genre arrays or one matrix)."""
    p = clf.predict_proba(Z)
    if isinstance(p, list):
        return np.column_stack([q[:, 1] if q.shape[1] == 2 else np.zeros(len(Z)) for q in p])
    return np.asarray(p)


def select(arm, Z, Y, groups, seed):
    """Inner grid search on the training data only (as select_logreg_C)."""
    scores = []
    for p in GRIDS[arm]:
        f = []
        for tr, te in GroupKFold(n_splits=3).split(Z, Y, groups):
            clf = make(arm, p, seed).fit(Z[tr], Y[tr])
            f.append(f1_score(Y[te], proba(clf, Z[te]) >= 0.5, average="macro", zero_division=0))
        scores.append(float(np.mean(f)))
    return GRIDS[arm][int(np.argmax(scores))]


def fit_fold(X, E, Y, tr, te, groups, seed, predicted):
    """Stage 1 once (if predicted), then every Stage-2 classifier -> {arm: (P, params)}."""
    if predicted:
        reg = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1).fit(X[tr], E[tr])
        Zte = build_emotion_features(reg.predict(X[te]))
        Ztr = build_emotion_features(oof_emotions(X[tr], E[tr], groups[tr], seed))
    else:
        Ztr, Zte = X[tr], X[te]
    C = select_logreg_C(Ztr, Y[tr], groups[tr])
    out = {}
    for arm in ARMS:
        p = {"C": C} if arm in (LR, CHAIN) else select(arm, Ztr, Y[tr], groups[tr], seed)
        clf = make(arm, p, seed).fit(Ztr, Y[tr])
        out[arm] = (proba(clf, Zte), p)
    return out


def run(X, E, Y, groups, seed, jobs, predicted):
    folds = list(RepeatedGroupKFold(N_SPLITS, N_REPEATS, random_state=seed)
                 .split(np.zeros(len(Y)), None, groups))
    res = Parallel(n_jobs=jobs)(delayed(fit_fold)(X, E, Y, tr, te, groups, seed, predicted)
                                for tr, te in folds)
    runs = {a: {"P": np.zeros((N_REPEATS,) + Y.shape),
                "hard": np.zeros((N_REPEATS,) + Y.shape, dtype=int),
                "fold_f1": np.zeros(len(folds))} for a in ARMS}
    params = {a: [] for a in ARMS}
    for i, ((_, te), r) in enumerate(zip(folds, res)):
        for a in ARMS:
            P, p = r[a]
            runs[a]["P"][i // N_SPLITS][te] = P
            runs[a]["hard"][i // N_SPLITS][te] = (P >= 0.5).astype(int)
            runs[a]["fold_f1"][i] = f1_score(Y[te], P >= 0.5, average="macro", zero_division=0)
            params[a].append(p)
    n_test = float(np.mean([len(te) for _, te in folds]))
    return runs, params, n_test


def block(X, E, Y, groups, args, predicted):
    runs, params, n_test = run(X, E, Y, groups, config.SEED, args.jobs, predicted)
    fl = Films(Y, groups)
    bf = FilmBootstrap(fl.Y, np.arange(len(fl.ids)), args.boot)
    bc = FilmBootstrap(Y, groups, args.boot)
    arms, rf, rc, df_, dc, fold = summarise(Y, runs, fl, bf, bc, N_SPLITS)
    comps = compare([(a, LR) for a in ARMS if a != LR], rf, df_, dc, fold,
                    len(Y) - n_test, n_test)
    chosen = {}
    for a in ARMS:
        keys = [json.dumps(p, sort_keys=True) for p in params[a]]
        chosen[a] = {k: keys.count(k) for k in sorted(set(keys))}
    return {"n_clips": int(len(Y)), "n_films": int(len(fl.ids)), "arms": arms,
            "comparisons": comps, "chosen_params_per_fold": chosen,
            "per_repeat_film": {k: v.tolist() for k, v in rf.items()},
            "per_repeat_clip": {k: v.tolist() for k, v in rc.items()}}


def show(title, b):
    print(f"\n{title}")
    for a in ARMS:
        v = b["arms"][a]
        print(f"  {a:22}{v['mean']:7.3f}  [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
              f"   clip {v['clip_level_mean']:.3f}")
    for c in b["comparisons"]:
        print(f"  {c['a']:22} - LR {c['diff']:+.3f}  p={c['p_two_sided']:.3f}"
              f"  wins {c['repeat_win_rate']:.1f}  (clip {c['clip_level_diff']:+.3f},"
              f" NB p={c['clip_level_nb_p']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    set_seed()
    t0 = time.time()
    out_path = RESULTS if args.boot >= 2000 else RESULTS.with_suffix(".partial.json")

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y5 = Y8[keep][:, idx]
    g5 = dfk["soundtrack"].to_numpy().astype(str)
    E5 = dfk[config.EMOTIONS].to_numpy(float)
    Xvgg = assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)

    out = {"design": {
        "question": "does any more flexible Stage-2 classifier beat the logistic regression "
                    "on the emotion features?",
        "protocol": "5 genres, 10x5 RepeatedGroupKFold by film, seed 42 (the folds of "
                    "film_level.json); emotion features predicted from VGGish by the random "
                    "forest, OOF-trained; computed once per fold and shared by every arm",
        "metric": "film-level macro-F1 (clip probabilities averaged per film, >= 0.5), "
                  "clip-level macro-F1 beside it",
        "significance": f"paired film bootstrap ({args.boot} draws) against the logistic "
                        "regression; clip-level Nadeau-Bengio beside it",
        "tuning": "every arm tuned by an inner 3-fold GroupKFold on the training data: "
                  "logistic regression and chain C in (0.003 ... 1); " + "; ".join(
                      f"{a}: {GRIDS[a]}" for a in GRIDS),
        "genres": list(config.GENRE_SUBSET)}}

    print(f"predicted emotions: {len(dfk)} clips / {len(set(g5))} films ...", flush=True)
    out["predicted_emotions"] = block(Xvgg, E5, Y5, g5, args, predicted=True)
    show("PREDICTED EMOTIONS (film level, 95% CI; clip level)", out["predicted_emotions"])
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    out["human_ratings"] = block(dfk[config.FEATURE_COLS].to_numpy(float), None, Y5, g5, args,
                                 predicted=False)
    show("HUMAN RATINGS (film level, 95% CI; clip level)", out["human_ratings"])

    fl = json.loads(FILM_LEVEL.read_text(encoding="utf-8"))["subset5"]
    want = fl["per_repeat_film"][PRED_OOF]
    got = out["predicted_emotions"]["per_repeat_film"][LR]
    ok = bool(np.allclose(want, got, atol=1e-12, rtol=0))
    out["reproduction_check"] = {
        "what": "logistic-regression arm vs film_level.json subset5 " + PRED_OOF,
        "film_level_per_repeat_match": ok}
    print(f"\nREPRODUCTION CHECK (film_level.json): {'identical' if ok else 'MISMATCH'}")

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {out_path}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
