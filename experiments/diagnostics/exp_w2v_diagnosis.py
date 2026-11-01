"""Why is wav2vec 2.0 the weakest representation used directly but a strong emotion input?

On the five-genre subset (`film_level.json`), wav2vec 2.0 given directly to the genre
classifier reaches a film-level macro-F1 of 0.261, the lowest of the six representations.
Given to the emotion pipeline instead (random forest -> eight emotions -> the same
classifier) it reaches 0.472, nominally above the VGGish emotion pipeline (0.460), although
its Stage-1 R^2 is by far the lowest (0.299 against 0.553). This script tests three
explanations.

1. **Is the wav2vec pipeline really better?** The paired film bootstrap between the two
   emotion pipelines, which `film_level.json` does not contain.
2. **Is it compression alone?** A PCA-8 control on wav2vec 2.0 (standardise, eight
   principal components, both fitted on the training fold), the counterpart of the VGGish
   PCA-8 control. If compression were the reason, it would close most of the gap.
3. **What does the bottleneck remove?** Out of fold, for both embeddings: how the
   predicted emotions relate to the ratings (R^2, Pearson r, the spread of the predictions
   relative to the ratings), how well they still separate the genres (Cohen's d per genre
   and emotion, compared with the d of the ratings), and how strongly each raw
   representation encodes *which film* a clip comes from (the share of a clip's nearest
   neighbours from its own film, and the share of variance between films). A representation
   dominated by film identity lets a high-dimensional classifier learn film-specific
   directions that do not carry over to unseen films; the emotion target keeps only what
   is shared across films. The Pearson r of the predictions is also computed on film
   means, the unit that is scored for genre.
4. **Is it only non-linearity?** The emotion pipeline puts a non-linear model (the random
   forest) in front of the linear classifier. A random-forest genre classifier on the
   embedding directly (one forest per genre, balanced class weights) tests whether any
   non-linear model would rescue wav2vec 2.0 without the emotion target.

Parts 1, 2 and 4 use the folds, seeds and fitting code of `exp_film_level.py`, so the five
arms shared with it must reproduce `film_level.json`; the script checks this.

Run:  python experiments/diagnostics/exp_w2v_diagnosis.py [--jobs 12] [--reuse-genre]
      (about 25 minutes; --reuse-genre takes parts 1 and 2 from an existing results file
      and needs about 5)
Writes results/w2v_diagnosis.json.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from src import config  # noqa: E402
from src.evaluation.repeated import RepeatedGroupKFold  # noqa: E402
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.utils import set_seed  # noqa: E402
from exp_cv_corrected import PCA8, PRED_OOF, VGG, W2V, W2V_OOF  # noqa: E402
import exp_film_level  # noqa: E402
from exp_film_level import N_REPEATS, N_SPLITS, block  # noqa: E402

OUT = config.RESULTS_DIR / "w2v_diagnosis.json"
FILM_LEVEL = config.RESULTS_DIR / "film_level.json"
PCA8_W2V = "PCA-8(wav2vec2) [control]"
RF_W2V = "random forest on wav2vec2-768 (direct)"
RF_VGG = "random forest on VGGish-128 (direct)"
_fit_fold = exp_film_level.fit_fold


def cohens_d(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a, b = values[mask], values[~mask]
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / sp


def oof_predictions(X, E, folds, jobs):
    """(repeats, clips, emotions) out-of-fold random-forest predictions."""
    def one(tr, te):
        reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=1)
        return reg.fit(X[tr], E[tr]).predict(X[te])
    res = Parallel(n_jobs=jobs)(delayed(one)(tr, te) for tr, te in folds)
    P = np.zeros((N_REPEATS,) + E.shape)
    for k, ((_, te), p) in enumerate(zip(folds, res)):
        P[k // N_SPLITS, te] = p
    return P


def stage1_profile(P, E, emos, films):
    """R^2, Pearson r and the spread ratio per emotion, averaged over the repeats; the
    Pearson r also on film means (the unit that is scored for genre)."""
    uniq, inv = np.unique(films, return_inverse=True)
    M = np.zeros((len(uniq), len(films)))
    M[inv, np.arange(len(films))] = 1
    M /= M.sum(1, keepdims=True)
    Ef = M @ E
    r2, r, sd, rf = [], [], [], []
    for p in P:
        ss_res = ((E - p) ** 2).sum(0)
        ss_tot = ((E - E.mean(0)) ** 2).sum(0)
        r2.append(1 - ss_res / ss_tot)
        r.append([np.corrcoef(E[:, j], p[:, j])[0, 1] for j in range(E.shape[1])])
        sd.append(p.std(0) / E.std(0))
        pf = M @ p
        rf.append([np.corrcoef(Ef[:, j], pf[:, j])[0, 1] for j in range(E.shape[1])])
    f = lambda a: dict(zip(emos, np.round(np.mean(a, 0), 3).tolist()))  # noqa: E731
    return {"r2": f(r2), "pearson_r": f(r), "sd_pred_over_sd_rated": f(sd),
            "pearson_r_film_means": f(rf),
            "mean_r2": round(float(np.mean(r2)), 3), "mean_r": round(float(np.mean(r)), 3),
            "mean_r_film_means": round(float(np.mean(rf)), 3),
            "mean_sd_ratio": round(float(np.mean(sd)), 3)}


def fit_fold_with_rf(kind, X, Y, tr, te, groups, emo, seed):
    """`exp_film_level.fit_fold` plus one arm kind: a random-forest genre classifier on the
    embedding directly (one forest per genre, balanced class weights, no tuning), the
    non-linear counterpart of the direct logistic regression."""
    if kind != "rf_direct":
        return _fit_fold(kind, X, Y, tr, te, groups, emo, seed)
    P = np.zeros((len(te), Y.shape[1]))
    for j in range(Y.shape[1]):
        if Y[tr, j].min() == Y[tr, j].max():          # no positive film in this fold
            P[:, j] = Y[tr, j][0]
            continue
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=1).fit(X[tr], Y[tr, j])
        P[:, j] = clf.predict_proba(X[te])[:, 1]
    return P, (P >= 0.5).astype(int)


def signature_profile(P, E, Y, genres, emos):
    """Cohen's d per genre and emotion from the predictions vs from the ratings."""
    d_rated = np.array([cohens_d(E, Y[:, j].astype(bool)) for j in range(len(genres))])
    d_pred = np.mean([[cohens_d(p, Y[:, j].astype(bool)) for j in range(len(genres))]
                      for p in P], 0)
    return d_rated, d_pred, {
        "corr_with_rated_d": round(float(np.corrcoef(d_rated.ravel(), d_pred.ravel())[0, 1]), 3),
        "mean_abs_d": round(float(np.abs(d_pred).mean()), 3),
        "mean_abs_d_rated": round(float(np.abs(d_rated).mean()), 3),
        "d_predicted": {g: dict(zip(emos, np.round(d_pred[j], 3).tolist()))
                        for j, g in enumerate(genres)},
    }


def film_identity(X, films, k=5):
    """How strongly a representation encodes the film: kNN same-film share, between-film variance."""
    Z = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    same = float((films[idx] == films[:, None]).mean())
    uniq, inv = np.unique(films, return_inverse=True)
    means = np.array([Z[inv == i].mean(0) for i in range(len(uniq))])
    between = float((means[inv] ** 2).sum() / (Z ** 2).sum())     # Z has zero mean
    chance = float(np.mean([(films == f).sum() - 1 for f in films]) / (len(films) - 1))
    return {"knn_same_film_share": round(same, 3), "between_film_variance_share": round(between, 3),
            "chance_same_film_share": round(chance, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--reuse-genre", action="store_true",
                    help="take parts 1 and 2 from an existing w2v_diagnosis.json (they take ~20 min)")
    args = ap.parse_args()
    set_seed()
    t0 = time.time()

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    films = dfk["soundtrack"].to_numpy().astype(str)
    E = dfk[config.EMOTIONS].to_numpy(float)
    genres, emos = config.GENRE_SUBSET, config.EMOTIONS

    def cache(d):
        return assemble_from_cache(dfk, "set1", cache_dir=d)

    reps = {"VGGish": cache(config.VGGISH_EMBEDDINGS_DIR), "wav2vec 2.0": cache(config.W2V_EMBEDDINGS_DIR),
            "AST": cache(config.EMBEDDINGS_DIR), "CLAP": cache(config.CLAP_EMBEDDINGS_DIR),
            "MusiCNN": cache(config.MUSICNN_EMBEDDINGS_DIR), "hand-crafted MIR": cache(config.MIR_EMBEDDINGS_DIR)}
    Xvgg, Xw2v = reps["VGGish"], reps["wav2vec 2.0"]

    # ---- 1 + 2: genre arms, film level, paired film bootstrap ------------------ #
    arms = {PRED_OOF: ("pred_oof", Xvgg, E), W2V_OOF: ("pred_oof", Xw2v, E),
            VGG: ("plain", Xvgg, None), W2V: ("plain", Xw2v, None),
            PCA8: ("pca8", Xvgg, None), PCA8_W2V: ("pca8", Xw2v, None)}
    pairs = [(W2V_OOF, PRED_OOF), (PCA8_W2V, W2V), (W2V_OOF, PCA8_W2V), (PCA8_W2V, PCA8),
             (W2V_OOF, W2V), (PRED_OOF, VGG)]
    if args.reuse_genre and OUT.exists():
        b = json.load(open(OUT, encoding="utf-8"))["genre"]
        repro = b.pop("reproduces_film_level_json")
        print("genre arms reused from", OUT.name, flush=True)
    else:
        print("genre arms (10x5 folds) ...", flush=True)
        b, _, _ = block(Y, films, genres, arms, pairs, args)
        print(f"  [{time.time() - t0:.0f}s]", flush=True)
        ref = json.load(open(FILM_LEVEL, encoding="utf-8"))["subset5"]["arms"]
        repro = {a: abs(b["arms"][a]["mean"] - ref[a]["mean"]) < 1e-9 for a in arms if a in ref}
        assert all(repro.values()), f"does not reproduce film_level.json: {repro}"

    # ---- non-linearity control: a random forest classifier on the embedding --- #
    exp_film_level.fit_fold = fit_fold_with_rf
    arms_nl = {RF_W2V: ("rf_direct", Xw2v, None), RF_VGG: ("rf_direct", Xvgg, None),
               W2V: ("plain", Xw2v, None), VGG: ("plain", Xvgg, None)}
    print("random-forest genre classifiers (10x5 folds) ...", flush=True)
    bn, _, _ = block(Y, films, genres, arms_nl, [(RF_W2V, W2V), (RF_VGG, VGG), (RF_W2V, RF_VGG)], args)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    # ---- 3: what the bottleneck keeps and removes ----------------------------- #
    folds = list(RepeatedGroupKFold(N_SPLITS, N_REPEATS, random_state=config.SEED)
                 .split(np.zeros(len(Y)), None, films))
    stage1, sig, preds = {}, {}, {}
    for name, X in (("VGGish", Xvgg), ("wav2vec 2.0", Xw2v)):
        print(f"out-of-fold emotions from {name} ...", flush=True)
        preds[name] = oof_predictions(X, E, folds, args.jobs)
        stage1[name] = stage1_profile(preds[name], E, emos, films)
        d_rated, _, sig[name] = signature_profile(preds[name], E, Y, genres, emos)
    agree = {e: round(float(np.mean([np.corrcoef(pv[:, j], pw[:, j])[0, 1]
                                     for pv, pw in zip(preds["VGGish"], preds["wav2vec 2.0"])])), 3)
             for j, e in enumerate(emos)}

    ident = {name: film_identity(X, films) for name, X in reps.items()}
    ident["human ratings (8 emotions)"] = film_identity(E, films)
    for name in ("VGGish", "wav2vec 2.0"):
        ident[f"predicted emotions from {name}"] = film_identity(preds[name].mean(0), films)

    out = {
        "what": "why wav2vec 2.0 is the weakest representation used directly but a strong "
                "input of the emotion pipeline (five-genre subset, film level, the 10x5 folds "
                "and seed of film_level.json)",
        "genre": {"arms": {k: {kk: v[kk] for kk in ("mean", "ci_lo", "ci_hi", "clip_level_mean",
                                                    "per_genre_f1", "per_genre_precision",
                                                    "per_genre_recall")}
                           for k, v in b["arms"].items()},
                  "comparisons": b["comparisons"], "genres": genres,
                  "per_repeat_film": b["per_repeat_film"],
                  "reproduces_film_level_json": repro},
        "nonlinear_control": {
            "what": "random-forest genre classifier (300 trees per genre, balanced class "
                    "weights, no tuning) on the embedding directly, same folds and scoring",
            "arms": {k: {kk: v[kk] for kk in ("mean", "ci_lo", "ci_hi", "clip_level_mean",
                                              "per_genre_f1")}
                     for k, v in bn["arms"].items()},
            "comparisons": bn["comparisons"]},
        "stage1": stage1,
        "prediction_agreement_vggish_vs_w2v_r": agree,
        "signatures": {"d_rated": {g: dict(zip(emos, np.round(d_rated[j], 3).tolist()))
                                   for j, g in enumerate(genres)}, **sig},
        "film_identity": ident,
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=2) + "\n")

    print("\nfilm-level macro-F1 (clip level):")
    for k, v in sorted(b["arms"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {k:48} {v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]  ({v['clip_level_mean']:.3f})")
    for c in b["comparisons"] + bn["comparisons"]:
        print(f"  {c['a'][:40]:40} vs {c['b'][:36]:36} {c['diff']:+.3f} p={c['p_two_sided']:.3f}"
              f" won {c['repeat_win_rate']:.1f}  clip {c['clip_level_diff']:+.3f} NB p={c['clip_level_nb_p']:.3f}")
    for k, v in bn["arms"].items():
        print(f"  {k:48} {v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]  ({v['clip_level_mean']:.3f})")
    for name in stage1:
        s = stage1[name]
        print(f"\n{name}: mean R2 {s['mean_r2']}, mean r {s['mean_r']}, sd ratio {s['mean_sd_ratio']},"
              f" mean r of film means {s['mean_r_film_means']}")
        print("  r film means :", s["pearson_r_film_means"])
        print("  r per emotion:", s["pearson_r"])
        print("  sd ratio     :", s["sd_pred_over_sd_rated"])
        print(f"  signatures: corr with rated d {sig[name]['corr_with_rated_d']}, "
              f"mean |d| {sig[name]['mean_abs_d']} (rated {sig[name]['mean_abs_d_rated']})")
    print("\nagreement of the two predictions (r):", agree)
    print("\nfilm identity:")
    for k, v in ident.items():
        print(f"  {k:36} kNN same film {v['knn_same_film_share']:.3f} (chance {v['chance_same_film_share']:.3f})"
              f"  between-film variance {v['between_film_variance_share']:.3f}")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
