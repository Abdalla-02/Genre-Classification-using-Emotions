"""The emotion pipeline for every audio representation and every emotion regressor.

So far the emotion pipeline has a genre result for two inputs only (VGGish and wav2vec 2.0,
both with the random forest), and the regressor was chosen on emotion R^2 alone, on VGGish
only (ridge 0.39, SVR 0.54, random forest 0.55). This script fills the grid:

  six representations   VGGish, AST, CLAP, MusiCNN, hand-crafted MIR, wav2vec 2.0
  x three regressors    random forest (300 trees; the thesis pipeline),
                        ridge (standardised; alpha chosen by film-grouped 3-fold CV inside
                        the training data, from 0.1 to 10^5),
                        SVR with an RBF kernel (standardised; C = 10, gamma = "scale", one
                        model per emotion; the setting of the original comparison, untuned)

Every cell is the full emotion pipeline of ``exp_film_level.py``: the regressor is fitted on
the training films, the training clips get out-of-fold emotions from the same regressor
(inner film-grouped 5-fold CV), the eleven emotion features go to the logistic regression
with nested C, and genre is scored per film (clip probabilities averaged, >= 0.5). The six
direct pipelines and the ratings (ceiling) run on the same folds, so every comparison is
paired. The random-forest cells for VGGish and wav2vec 2.0 and the direct and ceiling arms
must reproduce ``film_level.json``; the script checks this.

Also per cell: film-level ROC AUC (threshold-free, mean over genres, mean over repeats) and
the emotion R^2 of the regressor (mean over the eight emotions, out of fold, on the first
repeat's five outer folds).

Paired film bootstrap: every emotion cell against the direct use of its representation;
ridge and SVR against the random forest on the same representation; every random-forest
cell against the VGGish one.

Run:  python experiments/emotion/exp_emotion_pipeline_grid.py [--jobs 12]
      (about an hour; the six random-forest cells dominate)
Writes results/emotion_pipeline_grid.json.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from src import config  # noqa: E402
from src.evaluation.repeated import RepeatedGroupKFold  # noqa: E402
from src.features import (  # noqa: E402
    add_derived_features, assemble_from_cache, build_emotion_features, genre_matrix, load_set1,
)
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402
from exp_cv_corrected import AST, CEIL, CLAP, MIR, MUSICNN, PRED_OOF, VGG, W2V, W2V_OOF  # noqa: E402
from exp_metrics_stability import proba  # noqa: E402
import exp_film_level  # noqa: E402
from exp_film_level import N_REPEATS, N_SPLITS, block  # noqa: E402

OUT = config.RESULTS_DIR / "emotion_pipeline_grid.json"
FILM_LEVEL = config.RESULTS_DIR / "film_level.json"
REPS = ("VGGish", "AST", "CLAP", "MusiCNN", "MIR", "wav2vec 2.0")
DIRECT = dict(zip(REPS, (VGG, AST, CLAP, MUSICNN, MIR, W2V)))
REGRESSORS = ("random forest", "ridge", "SVR")
_fit_fold = exp_film_level.fit_fold


def cell(rep: str, reg: str) -> str:
    # the two random-forest cells that film_level.json holds keep its arm names
    if reg == "random forest" and rep == "VGGish":
        return PRED_OOF
    if reg == "random forest" and rep == "wav2vec 2.0":
        return W2V_OOF
    return f"{rep} -> {reg} -> emotion(11)"


def fit_regressor(reg: str, X, E, groups, seed):
    if reg == "random forest":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1).fit(X, E)
    if reg == "ridge":
        cv = list(GroupKFold(n_splits=3).split(X, E, groups))
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-1, 5, 13), cv=cv)).fit(X, E)
    if reg == "SVR":
        return make_pipeline(StandardScaler(),
                             MultiOutputRegressor(SVR(C=10, gamma="scale"))).fit(X, E)
    raise ValueError(reg)


def fit_fold_grid(kind, X, Y, tr, te, groups, emo, seed):
    """'emo:<regressor>' = the emotion pipeline with that regressor; the random forest is
    delegated to exp_film_level so that its cells reproduce film_level.json exactly."""
    if not kind.startswith("emo:"):
        return _fit_fold(kind, X, Y, tr, te, groups, emo, seed)
    reg = kind[4:]
    if reg == "random forest":
        return _fit_fold("pred_oof", X, Y, tr, te, groups, emo, seed)
    Zte = build_emotion_features(fit_regressor(reg, X[tr], emo[tr], groups[tr], seed).predict(X[te]))
    Etr = np.zeros_like(emo[tr])
    for itr, ite in GroupKFold(n_splits=5).split(X[tr], emo[tr], groups[tr]):
        m = fit_regressor(reg, X[tr][itr], emo[tr][itr], groups[tr][itr], seed)
        Etr[ite] = m.predict(X[tr][ite])
    Ztr = build_emotion_features(Etr)
    C = select_logreg_C(Ztr, Y[tr], groups[tr])
    clf = build_classifier("logreg", C=C).fit(Ztr, Y[tr])
    return proba(clf, Zte), np.asarray(clf.predict(Zte))


def macro_auc(runs, fl) -> dict:
    out = {}
    for a, r in runs.items():
        per_rep = []
        for p in r["P"]:
            s = np.round(fl.M @ p, 10)
            per_rep.append(np.mean([roc_auc_score(fl.Y[:, g], s[:, g]) for g in range(fl.Y.shape[1])]))
        out[a] = float(np.mean(per_rep))
    return out


def stage1_r2(X, E, reg, folds, seed) -> float:
    P = np.zeros_like(E)
    for tr, te in folds:
        # the groups are not needed for prediction; ridge's inner CV uses the training films
        P[te] = fit_regressor(reg, X[tr], E[tr], FILMS[tr], seed).predict(X[te])
    return float(np.mean([r2_score(E[:, k], P[:, k]) for k in range(E.shape[1])]))


def main() -> None:
    global FILMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    set_seed()
    t0 = time.time()

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    FILMS = dfk["soundtrack"].to_numpy().astype(str)
    E = dfk[config.EMOTIONS].to_numpy(float)
    genres = list(config.GENRE_SUBSET)

    def cache(d):
        return assemble_from_cache(dfk, "set1", cache_dir=d)

    X = dict(zip(REPS, (cache(config.VGGISH_EMBEDDINGS_DIR), cache(config.EMBEDDINGS_DIR),
                        cache(config.CLAP_EMBEDDINGS_DIR), cache(config.MUSICNN_EMBEDDINGS_DIR),
                        cache(config.MIR_EMBEDDINGS_DIR), cache(config.W2V_EMBEDDINGS_DIR))))

    arms = {CEIL: ("plain", dfk[config.FEATURE_COLS].to_numpy(float), None)}
    arms |= {DIRECT[r]: ("plain", X[r], None) for r in REPS}
    arms |= {cell(r, g): (f"emo:{g}", X[r], E) for g in REGRESSORS for r in REPS}
    pairs = [(cell(r, g), DIRECT[r]) for g in REGRESSORS for r in REPS]
    pairs += [(cell(r, g), cell(r, "random forest")) for g in REGRESSORS[1:] for r in REPS]
    pairs += [(cell(r, "random forest"), PRED_OOF) for r in REPS[1:]]

    exp_film_level.fit_fold = fit_fold_grid
    print(f"{len(arms)} arms on the 10x5 folds ...", flush=True)
    b, runs, fl = block(Y, FILMS, genres, arms, pairs, args)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    ref = json.load(open(FILM_LEVEL, encoding="utf-8"))["subset5"]["arms"]
    repro = {a: bool(abs(b["arms"][a]["mean"] - ref[a]["mean"]) < 1e-9) for a in arms if a in ref}
    assert all(repro.values()), f"does not reproduce film_level.json: {repro}"

    auc = macro_auc(runs, fl)
    folds = list(RepeatedGroupKFold(N_SPLITS, N_REPEATS, random_state=config.SEED)
                 .split(np.zeros(len(Y)), None, FILMS))[:N_SPLITS]
    r2 = {r: {g: stage1_r2(X[r], E, g, folds, config.SEED) for g in REGRESSORS} for r in REPS}
    print(f"  stage 1 [{time.time() - t0:.0f}s]", flush=True)

    def row(a):
        v = b["arms"][a]
        return {"film_f1": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                "clip_f1": v["clip_level_mean"], "macro_auc": auc[a],
                "per_genre_f1": dict(zip(genres, v["per_genre_f1"]))}

    out = {
        "design": {"protocol": "the folds, seed and fitting code of exp_film_level.py "
                               "(10x5 RepeatedGroupKFold by film, OOF-trained emotions, nested C)",
                   "regressors": {"random forest": "300 trees",
                                  "ridge": "standardised, alpha in 10^-1..10^5 by film-grouped 3-fold CV",
                                  "SVR": "standardised, RBF, C=10, gamma=scale, untuned"},
                   "metric": "film-level macro-F1 (clip level beside it); film-level ROC AUC "
                             "mean over genres and repeats; stage-1 R^2 mean over eight emotions, "
                             "first repeat's outer folds",
                   "all_positive_floor_film": 0.434},
        "representations": list(REPS), "regressors": list(REGRESSORS),
        "grid": {r: {g: row(cell(r, g)) | {"stage1_r2": r2[r][g]} for g in REGRESSORS} for r in REPS},
        "direct": {r: row(DIRECT[r]) for r in REPS},
        "ceiling": row(CEIL),
        "comparisons": b["comparisons"],
        "reproduces_film_level_json": repro,
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"\nfilm F1 / AUC / R2\n{'representation':16}{'direct':>12}" + "".join(f"{g:>20}" for g in REGRESSORS))
    for r in REPS:
        d = out["direct"][r]
        cells = "".join("{film_f1:>10.3f}{macro_auc:>5.2f}{stage1_r2:>5.2f}".format(**out["grid"][r][g])
                        for g in REGRESSORS)
        print(f"{r:16}{d['film_f1']:7.3f}{d['macro_auc']:5.2f}{cells}")
    print(f"ceiling {out['ceiling']['film_f1']:.3f} (AUC {out['ceiling']['macro_auc']:.3f})")
    for c in b["comparisons"]:
        s = "*" if c["p_two_sided"] < 0.05 else " "
        print(f"  {c['a'][:40]:40} vs {c['b'][:40]:40} {c['diff']:+.3f} p={c['p_two_sided']:.3f}{s} "
              f"won {c['repeat_win_rate']:.1f}  clip {c['clip_level_diff']:+.3f} NB {c['clip_level_nb_p']:.3f}")
    print(f"\n[{time.time() - t0:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
