"""Why is Horror the weakest genre of the emotion pipeline? Rated vs predicted emotions.

On the five-genre subset the emotion pipeline reaches a film-level F1 of 0.23 on Horror,
against 0.41 when the same genre classifier is given the listeners' ratings instead of
emotions predicted from audio (`film_level.json`). The genre stage is identical in both,
so the loss must come from Stage 1, the emotion regression. This script shows where.

For every clip of the five-genre subset it predicts the eight emotions out of fold, exactly
as the pipeline does at test time (random forest on VGGish, 300 trees, folds grouped by
film; the 10x5 folds of the main analysis, averaged over the ten repeats). It then
compares the rated and the predicted emotions:

* **Signatures:** Cohen's d of every emotion, genre against all other clips, once from
  the ratings and once from the predictions. If the regression flattens the emotional
  profile of a genre, d shrinks.
* **Fear by genre:** the mean rated and the mean predicted fear per genre. A regressor
  trained on few extreme clips pulls their predictions towards the mean (regression to the
  mean), so the genre with the highest fear loses most of its distance to the others.
* **Fear on Horror clips:** R^2 and the slope of predicted on rated fear within the
  Horror clips.

Run:  python experiments/diagnostics/exp_horror_diagnosis.py   (about a minute)
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.repeated import RepeatedGroupKFold  # noqa: E402
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.utils import set_seed  # noqa: E402

OUT = config.RESULTS_DIR / "horror_diagnosis.json"
N_REPEATS, N_SPLITS = 10, 5


def cohens_d(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a, b = values[mask], values[~mask]
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / sp


def main() -> None:
    set_seed()
    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(g) for g in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    groups = dfk["soundtrack"].to_numpy()
    E = dfk[config.EMOTIONS].to_numpy(float)
    X = assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)

    folds = list(RepeatedGroupKFold(N_SPLITS, N_REPEATS).split(np.zeros(len(dfk)), None, groups))
    preds = np.zeros((N_REPEATS,) + E.shape)
    for k, (tr, te) in enumerate(folds):
        reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
        preds[k // N_SPLITS, te] = reg.fit(X[tr], E[tr]).predict(X[te])

    genres, emos = config.GENRE_SUBSET, config.EMOTIONS
    fear = emos.index("fear")
    horror = Y[:, genres.index("Horror")].astype(bool)

    d_rated = {g: dict(zip(emos, np.round(cohens_d(E, Y[:, j].astype(bool)), 3).tolist()))
               for j, g in enumerate(genres)}
    d_pred_rep = np.array([[cohens_d(P, Y[:, j].astype(bool)) for j in range(len(genres))]
                           for P in preds])               # repeats x genres x emotions
    d_pred = {g: dict(zip(emos, np.round(d_pred_rep[:, j].mean(0), 3).tolist()))
              for j, g in enumerate(genres)}

    fear_by_genre = {}
    for j, g in enumerate(genres):
        m = Y[:, j].astype(bool)
        fear_by_genre[g] = {"clips": int(m.sum()),
                            "rated_mean": round(float(E[m, fear].mean()), 3),
                            "predicted_mean": round(float(preds[:, m, fear].mean()), 3)}
    rest = ~horror
    fear_by_genre["all non-Horror clips"] = {
        "clips": int(rest.sum()),
        "rated_mean": round(float(E[rest, fear].mean()), 3),
        "predicted_mean": round(float(preds[:, rest, fear].mean()), 3)}

    pf = preds[:, :, fear].mean(0)
    within = {}
    for name, m in (("Horror clips", horror), ("all clips", np.ones(len(E), bool))):
        r, p = E[m, fear], pf[m]
        ss_res, ss_tot = ((r - p) ** 2).sum(), ((r - r.mean()) ** 2).sum()
        within[name] = {"r2": round(float(1 - ss_res / ss_tot), 3),
                        "slope_pred_on_rated": round(float(np.polyfit(r, p, 1)[0]), 3)}

    # How many non-Horror clips does each source put above the Horror clips' median fear?
    thr_r = float(np.median(E[horror, fear]))
    thr_p = float(np.median(pf[horror]))
    overlap = {"rated: non-Horror clips above the Horror median": int((E[rest, fear] >= thr_r).sum()),
               "predicted: non-Horror clips above the Horror median": int((pf[rest] >= thr_p).sum())}

    out = {
        "what": "rated vs out-of-fold predicted emotions on the five-genre subset (VGGish, "
                "random forest, the 10x5 folds of the main analysis, mean over repeats)",
        "n_clips": int(len(E)), "n_horror_clips": int(horror.sum()),
        "cohens_d_rated": d_rated, "cohens_d_predicted": d_pred,
        "fear_by_genre": fear_by_genre, "fear_within": within, "fear_overlap": overlap,
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=2) + "\n")

    print(f"Horror clips: {out['n_horror_clips']} of {out['n_clips']}")
    print("Cohen's d, Horror vs rest:   emotion   rated  predicted")
    for e in emos:
        print(f"                              {e:8} {d_rated['Horror'][e]:+.2f}   {d_pred['Horror'][e]:+.2f}")
    print("fear per genre (rated / predicted):")
    for g, v in fear_by_genre.items():
        print(f"  {g:22} n={v['clips']:3}  {v['rated_mean']:.2f} / {v['predicted_mean']:.2f}")
    print("fear within:", within)
    print("overlap:", overlap)
    print("largest |d| per genre, rated vs predicted:")
    for g in genres:
        er = max(emos, key=lambda e: abs(d_rated[g][e]))
        print(f"  {g:7} rated {er} {d_rated[g][er]:+.2f} | predicted {d_pred[g][er]:+.2f}")


if __name__ == "__main__":
    main()
