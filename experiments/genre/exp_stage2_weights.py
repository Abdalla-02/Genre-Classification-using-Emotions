"""The weights of the Stage-2 genre classifier: what the emotion pipeline actually uses.

The thesis argues that the emotion pipeline is interpretable because its logistic regression has
one weight per emotion and genre. The second examiner's report (20 Nov 2026) noted that these
weights were never shown. This script reports them.

Protocol: the pipeline of ``film_level.json`` exactly (five genres, 329 clips / 40 films, 10x5
RepeatedGroupKFold by film, seed 42; emotions predicted from VGGish by the random forest, trained on
out-of-fold emotions of the training clips; C tuned in an inner loop; balanced class weights). The
inputs are standardised inside the classifier, so each weight is the change in the log-odds of a
genre per standard deviation of the feature. For every genre and feature the script stores the mean
weight over the 50 training folds, its standard deviation, and the share of folds in which the
weight has the sign of the mean. A second block does the same with the human ratings as input.

Caveat: three of the eleven features are combinations of the eight emotions (valence x energy, a
negative and a positive composite), so the weights of correlated features share the effect between
them; a single weight should be read together with its neighbours.

Run:  python experiments/genre/exp_stage2_weights.py      (a few minutes)
Writes results/stage2_weights.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

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
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402
from exp_metrics_stability import oof_emotions  # noqa: E402

OUT = config.RESULTS_DIR / "stage2_weights.json"
N_REPEATS, N_SPLITS = 10, 5


def weights(clf, n_genres, n_feat):
    """(genres, features) standardised weights; NaN for a genre without positives in the fold."""
    W = np.full((n_genres, n_feat), np.nan)
    for j, est in enumerate(clf.named_steps["clf"].estimators_):
        if getattr(est, "_fitted", None) is not None:
            W[j] = est._fitted.coef_[0]
    return W


def fit_fold(X, E, Y, tr, groups, seed, predicted):
    if predicted:
        Ztr = build_emotion_features(oof_emotions(X[tr], E[tr], groups[tr], seed))
    else:
        Ztr = X[tr]
    C = select_logreg_C(Ztr, Y[tr], groups[tr])
    clf = build_classifier("logreg", C=C).fit(Ztr, Y[tr])
    return weights(clf, Y.shape[1], Ztr.shape[1]), C


def block(X, E, Y, groups, jobs, predicted):
    folds = list(RepeatedGroupKFold(N_SPLITS, N_REPEATS, random_state=config.SEED)
                 .split(np.zeros(len(Y)), None, groups))
    res = Parallel(n_jobs=jobs)(delayed(fit_fold)(X, E, Y, tr, groups, config.SEED, predicted)
                                for tr, _ in folds)
    W = np.stack([r[0] for r in res])                     # folds x genres x features
    mean = np.nanmean(W, 0)
    sd = np.nanstd(W, 0, ddof=1)
    same = np.nanmean(np.sign(W) == np.sign(mean)[None], 0)
    Cs = [r[1] for r in res]
    return mean, sd, same, {str(c): Cs.count(c) for c in sorted(set(Cs))}


def main() -> None:
    set_seed()
    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    g = dfk["soundtrack"].to_numpy().astype(str)
    E = dfk[config.EMOTIONS].to_numpy(float)
    X = assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    feats = list(config.FEATURE_COLS)
    genres = list(config.GENRE_SUBSET)

    out = {"what": "standardised logistic-regression weights of the Stage-2 genre classifier, mean over "
                   "the 50 training folds of film_level.json (5 genres, seed 42), their SD, and the share "
                   "of folds with the sign of the mean",
           "features": feats, "genres": genres}
    for name, Xin, pred in (("predicted_emotions", X, True),
                            ("human_ratings", dfk[config.FEATURE_COLS].to_numpy(float), False)):
        mean, sd, same, Cs = block(Xin, E, Y, g, -1, pred)
        out[name] = {"mean": {gn: dict(zip(feats, np.round(mean[j], 4).tolist())) for j, gn in enumerate(genres)},
                     "sd": {gn: dict(zip(feats, np.round(sd[j], 4).tolist())) for j, gn in enumerate(genres)},
                     "sign_share": {gn: dict(zip(feats, np.round(same[j], 3).tolist())) for j, gn in enumerate(genres)},
                     "chosen_C": Cs}
        print(f"\n{name}  (chosen C per fold: {Cs})")
        print(f"{'':14}" + "".join(f"{gn[:7]:>9}" for gn in genres))
        for i, fname in enumerate(feats):
            print(f"{fname[:14]:14}" + "".join(f"{mean[j, i]:+9.2f}" for j in range(len(genres))))
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
