"""Is macro-F1 the right metric, and how stable are the cross-validated numbers?

Two supervisor questions from the fourth meeting, answered on the corrected protocol of
``exp_cv_corrected.py`` (5-genre subset, 10x5 RepeatedGroupKFold by film, nested C,
training emotions predicted out-of-fold):

  1. "Explain the metric, and check whether there is a better approach than F1."
     Every arm is scored under the family of multi-label metrics a reader might ask
     about, from the SAME out-of-fold predictions:
       - F1 averaged four ways: macro (the headline), micro, weighted, samples;
       - macro-F1 with a per-genre decision threshold tuned on the training fold, since
         F1 depends on where the probability is cut and 0.5 is only a convention;
       - two threshold-free ranking metrics that need no cut at all: macro average
         precision (area under the precision-recall curve) and macro ROC-AUC;
       - film-level macro-F1: clip probabilities averaged per film, because genre is a
         property of the film and the clips of one film are not independent;
       - macro precision and recall, Hamming loss, exact match, Jaccard.
     The question that matters is not which number is highest but whether the
     ranking of the three routes (emotion, direct audio, PCA-8) depends on the metric.

  2. "Get the standard deviation or variance from the cross-validation, to know if the
     results are stable and reproducible." Three sources of variation are separated:
       - split variance: the spread over the 10 repeats (different film -> fold
         assignments) and over the 50 individual folds;
       - sampling variance: the film bootstrap, i.e. how much the score depends on
         which 41 films the dataset happens to contain;
       - seed variance: the whole 10x5 protocol re-run under four further master seeds,
         which changes both the fold assignment and the random forest.
     Reproducibility in the strict sense is checked too: under seed 42 this script must
     reproduce the pooled per-repeat macro-F1 of ``cv_corrected.json`` exactly.

Run:  python experiments/evaluation/exp_metrics_stability.py [--jobs 12] [--boot 1000]
      [--seeds 42 1 2 3 4]
Writes results/metrics_stability.json (a reduced run writes *.partial.json instead).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

RESULTS = config.RESULTS_DIR / "metrics_stability.json"
CV_CORRECTED = config.RESULTS_DIR / "cv_corrected.json"
N_REPEATS, N_SPLITS = 10, 5
DEFAULT_SEEDS = (42, 1, 2, 3, 4)
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Arm names match cv_corrected.json so the reproduction check is a plain lookup.
CEIL = "ground-truth emotion(11) [ceiling]"
EMO = "VGGish -> predicted emotion(11), OOF-trained"
VGG = "VGGish-128 (direct)"
AST = "AST-768"
PCA8 = "PCA-8(VGGish) [control]"
DUMMY = "dummy"
SHORT = {CEIL: "ratings (ceiling)", EMO: "emotion route", VGG: "direct audio (VGGish)",
         AST: "direct audio (AST)", PCA8: "PCA-8 control", DUMMY: "most frequent"}
PAIRS = [(EMO, VGG), (EMO, PCA8), (EMO, AST), (PCA8, VGG), (CEIL, VGG), (CEIL, EMO)]


# --------------------------------------------------------------------------- #
# One fold of one arm: probabilities, the 0.5 decision and the tuned decision
# --------------------------------------------------------------------------- #
def oof_emotions(X, E, groups, seed):
    """Out-of-fold emotions for the TRAINING clips (inner GroupKFold over films)."""
    out = np.zeros_like(E)
    for tr, te in GroupKFold(n_splits=5).split(X, E, groups):
        reg = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
        out[te] = reg.fit(X[tr], E[tr]).predict(X[te])
    return out


def proba(clf, Z):
    """(n, genres) P(genre) from the binary-relevance pipeline."""
    return np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(Z))
                            for p in clf.predict_proba(Z)])


def tune_thresholds(Z, Y, groups, C):
    """Per-genre threshold maximising F1 on inner out-of-fold probabilities (train only)."""
    P = np.zeros(Y.shape)
    for tr, te in GroupKFold(n_splits=3).split(Z, Y, groups):
        P[te] = proba(build_classifier("logreg", C=C).fit(Z[tr], Y[tr]), Z[te])
    best = []
    for j in range(Y.shape[1]):
        f = [f1_score(Y[:, j], P[:, j] >= t, zero_division=0) for t in THRESHOLDS]
        best.append(THRESHOLDS[int(np.argmax(f))])   # ties -> the lowest threshold
    return np.array(best)


def fit_fold(kind, X, Y, tr, te, groups, emo, seed, tune):
    """-> (probabilities, 0.5 predictions, tuned-threshold predictions or None)."""
    if kind == "dummy":
        clf = build_classifier("dummy").fit(X[tr], Y[tr])
        pred = np.asarray(clf.predict(X[te]))
        return pred.astype(float), pred, (pred if tune else None)
    if kind == "plain":
        Ztr, Zte = X[tr], X[te]
    elif kind == "pca8":
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=8, random_state=seed).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
    elif kind == "pred_oof":
        reg = RandomForestRegressor(n_estimators=300, random_state=seed,
                                    n_jobs=1).fit(X[tr], emo[tr])
        Zte = build_emotion_features(reg.predict(X[te]))
        Ztr = build_emotion_features(oof_emotions(X[tr], emo[tr], groups[tr], seed))
    else:
        raise ValueError(kind)
    C = select_logreg_C(Ztr, Y[tr], groups[tr])
    clf = build_classifier("logreg", C=C).fit(Ztr, Y[tr])
    P = proba(clf, Zte)
    pred = np.asarray(clf.predict(Zte))
    tuned = None
    if tune:
        tuned = (P >= tune_thresholds(Ztr, Y[tr], groups[tr], C)).astype(int)
    return P, pred, tuned


def run(arms, Y, groups, seed, jobs, tune):
    """All arms on the 10x5 folds of one master seed -> pooled per-repeat matrices."""
    folds = list(RepeatedGroupKFold(N_SPLITS, N_REPEATS, random_state=seed)
                 .split(np.zeros(len(Y)), None, groups))
    tasks = [(a, i) for a in arms for i in range(len(folds))]
    res = Parallel(n_jobs=jobs)(
        delayed(fit_fold)(arms[a][0], arms[a][1], Y, folds[i][0], folds[i][1], groups,
                          arms[a][2], seed, tune) for a, i in tasks)
    out = {a: {"P": np.zeros((N_REPEATS,) + Y.shape),
               "hard": np.zeros((N_REPEATS,) + Y.shape, dtype=int),
               "tuned": np.zeros((N_REPEATS,) + Y.shape, dtype=int),
               "fold_f1": np.zeros(len(folds))} for a in arms}
    for (a, i), (P, hard, tuned) in zip(tasks, res):
        r, te = i // N_SPLITS, folds[i][1]
        out[a]["P"][r][te], out[a]["hard"][r][te] = P, hard
        if tuned is not None:
            out[a]["tuned"][r][te] = tuned
        out[a]["fold_f1"][i] = f1_score(Y[te], hard, average="macro", zero_division=0)
    return out


# --------------------------------------------------------------------------- #
# The metrics. Each takes (Y, P, hard, tuned, films) for ONE pooled repeat.
# --------------------------------------------------------------------------- #
def film_level(Y, P, films):
    """Average clip probabilities per film; a film's labels are those of its clips."""
    uniq, inv = np.unique(films, return_inverse=True)
    M = np.zeros((len(uniq), len(films)))
    M[inv, np.arange(len(films))] = 1
    M /= M.sum(1, keepdims=True)
    return (M @ Y > 0.5).astype(int), M @ P


def film_macro_f1(Y, P, films):
    yf, pf = film_level(Y, P, films)
    return f1_score(yf, (pf >= 0.5).astype(int), average="macro", zero_division=0)


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except ValueError:          # a genre without positives in a bootstrap draw
        return np.nan


METRICS = {
    "macro_f1": lambda Y, P, h, t, f: f1_score(Y, h, average="macro", zero_division=0),
    "micro_f1": lambda Y, P, h, t, f: f1_score(Y, h, average="micro", zero_division=0),
    "weighted_f1": lambda Y, P, h, t, f: f1_score(Y, h, average="weighted", zero_division=0),
    "samples_f1": lambda Y, P, h, t, f: f1_score(Y, h, average="samples", zero_division=0),
    "macro_f1_tuned_threshold": lambda Y, P, h, t, f: f1_score(Y, t, average="macro",
                                                              zero_division=0),
    "macro_average_precision": lambda Y, P, h, t, f: _safe(average_precision_score, Y, P,
                                                           average="macro"),
    "macro_roc_auc": lambda Y, P, h, t, f: _safe(roc_auc_score, Y, P, average="macro"),
    "film_level_macro_f1": lambda Y, P, h, t, f: film_macro_f1(Y, P, f),
    "macro_precision": lambda Y, P, h, t, f: precision_score(Y, h, average="macro",
                                                             zero_division=0),
    "macro_recall": lambda Y, P, h, t, f: recall_score(Y, h, average="macro",
                                                       zero_division=0),
    "hamming_loss": lambda Y, P, h, t, f: hamming_loss(Y, h),
    "exact_match": lambda Y, P, h, t, f: float((Y == h).all(1).mean()),
    "jaccard_samples": lambda Y, P, h, t, f: jaccard_score(Y, h, average="samples",
                                                           zero_division=0),
}
LOWER_IS_BETTER = {"hamming_loss"}
DESCRIBE = {
    "macro_f1": "F1 per genre, then the unweighted mean over genres (the headline)",
    "micro_f1": "F1 over all clip-genre decisions pooled; dominated by frequent genres",
    "weighted_f1": "F1 per genre, averaged with weights proportional to genre frequency",
    "samples_f1": "F1 per clip over its genre vector, averaged over clips",
    "macro_f1_tuned_threshold": "macro-F1 with each genre's cut tuned on the training fold",
    "macro_average_precision": "area under each genre's precision-recall curve, averaged; "
                               "threshold-free",
    "macro_roc_auc": "probability a random positive clip outranks a random negative one, "
                     "averaged over genres; threshold-free, 0.5 is chance",
    "film_level_macro_f1": "macro-F1 over films, from clip probabilities averaged per film",
    "macro_precision": "of the clips predicted as a genre, the share that are",
    "macro_recall": "of the clips that are a genre, the share found",
    "hamming_loss": "share of individual clip-genre decisions that are wrong (lower is better)",
    "exact_match": "share of clips whose whole genre vector is right",
    "jaccard_samples": "per clip, the genres both predicted and true divided by those "
                       "predicted or true, averaged",
}


def score_repeats(Y, runs, films, arms, with_tuned):
    """{arm: {metric: [value per repeat]}}"""
    out = {}
    for a in arms:
        r = runs[a]
        out[a] = {m: [fn(Y, r["P"][k], r["hard"][k], r["tuned"][k], films)
                      for k in range(N_REPEATS)]
                  for m, fn in METRICS.items()
                  if with_tuned or m != "macro_f1_tuned_threshold"}
    return out


def random_guess_reference(Y, films, n_sim=500, seed=config.SEED):
    """Expected value of each metric for a base-rate random guess (per-genre Bernoulli)."""
    rng = np.random.default_rng(seed)
    prev = Y.mean(0)
    vals = {m: [] for m in METRICS}
    for _ in range(n_sim):
        P = rng.random(Y.shape)                        # random scores (ranking metrics)
        h = (rng.random(Y.shape) < prev).astype(int)   # base-rate random decisions
        for m, fn in METRICS.items():
            if m == "film_level_macro_f1":
                yf, _ = film_level(Y, P, films)
                hf = (rng.random(yf.shape) < yf.mean(0)).astype(int)
                vals[m].append(f1_score(yf, hf, average="macro", zero_division=0))
            else:
                vals[m].append(fn(Y, P, h, h, films))
    return {m: float(np.nanmean(v)) for m, v in vals.items()}


# --------------------------------------------------------------------------- #
# Paired film bootstrap for an arbitrary metric
# --------------------------------------------------------------------------- #
def film_bootstrap_diff(Y, runs, films, a, b, metric, n_boot, seed=config.SEED):
    """Mean difference a-b, 95% interval and two-sided p over films, averaged over repeats.

    Films are resampled with replacement and the SAME draws are applied to both arms, as
    in exp_cv_corrected; draws in which a genre has no positive clip are dropped.
    """
    fn = METRICS[metric]
    uniq = np.unique(films)
    members = [np.flatnonzero(films == f) for f in uniq]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in draw])
        Yb = Y[idx]
        if Yb.sum(0).min() == 0 or Yb.sum(0).max() == len(idx):
            continue
        # a film drawn twice counts as two films for the film-level metric
        fb = np.concatenate([np.full(len(members[i]), j) for j, i in enumerate(draw)])
        d = np.mean([fn(Yb, runs[a]["P"][k][idx], runs[a]["hard"][k][idx],
                        runs[a]["tuned"][k][idx], fb)
                     - fn(Yb, runs[b]["P"][k][idx], runs[b]["hard"][k][idx],
                          runs[b]["tuned"][k][idx], fb)
                     for k in range(N_REPEATS)])
        if np.isfinite(d):
            diffs.append(d)
    d = np.array(diffs)
    if metric in LOWER_IS_BETTER:
        d = -d
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return float(d.mean()), float(lo), float(hi), p, int(len(d))


def summary(values):
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max())}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = ap.parse_args()
    set_seed()
    t0 = time.time()
    full = args.boot >= 1000 and tuple(args.seeds) == DEFAULT_SEEDS
    out_path = RESULTS if full else RESULTS.with_suffix(".partial.json")

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    films = dfk["soundtrack"].to_numpy().astype(str)
    E = dfk[config.EMOTIONS].to_numpy(float)
    Xvgg = assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Xast = assemble_from_cache(dfk, "set1", cache_dir=config.EMBEDDINGS_DIR)
    arms = {CEIL: ("plain", dfk[config.FEATURE_COLS].to_numpy(float), None),
            EMO: ("pred_oof", Xvgg, E),
            VGG: ("plain", Xvgg, None),
            AST: ("plain", Xast, None),
            PCA8: ("pca8", Xvgg, None),
            DUMMY: ("dummy", Xvgg, None)}

    out = {"design": {
        "dataset": f"Eerola 5-genre subset, {len(dfk)} clips / {len(set(films))} films",
        "genres": config.GENRE_SUBSET,
        "protocol": "10x5 RepeatedGroupKFold by film, nested C (inner GroupKFold(3)), "
                    "training emotions predicted out-of-fold; as exp_cv_corrected.py",
        "pooling": "every metric computed once per repeat on the pooled out-of-fold "
                   "predictions (all clips, all genres present)",
        "threshold_tuning": "per genre, the F1-maximising cut in "
                            f"{THRESHOLDS[0]}..{THRESHOLDS[-1]} on inner GroupKFold(3) "
                            "out-of-fold probabilities of the training fold",
        "bootstrap": f"paired over films, {args.boot} draws, averaged over the repeats",
        "seeds": args.seeds, "arm_labels": SHORT, "metric_definitions": DESCRIBE}}

    # ---- 1. seed 42: every metric, with thresholds tuned ----------------------- #
    print(f"seed 42: {len(arms)} arms x 50 folds, all metrics ...", flush=True)
    runs = run(arms, Y, films, config.SEED, args.jobs, tune=True)
    reps = score_repeats(Y, runs, films, arms, with_tuned=True)
    chance = random_guess_reference(Y, films)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    metrics = {}
    for m in METRICS:
        metrics[m] = {"definition": DESCRIBE[m],
                      "lower_is_better": m in LOWER_IS_BETTER,
                      "base_rate_random_guess": chance[m],
                      "arms": {a: summary(reps[a][m]) for a in arms}}
    out["metrics"] = metrics

    # rank of the three routes under every metric
    routes = [EMO, VGG, PCA8, AST]
    ranking = {}
    for m in METRICS:
        sign = 1 if m in LOWER_IS_BETTER else -1
        order = sorted(routes, key=lambda a: sign * metrics[m]["arms"][a]["mean"])
        ranking[m] = [SHORT[a] for a in order]
    out["route_ranking_by_metric"] = ranking

    # ---- paired comparisons under the main alternative metrics ----------------- #
    comp_metrics = ["macro_f1", "micro_f1", "weighted_f1", "samples_f1",
                    "macro_f1_tuned_threshold", "macro_average_precision", "macro_roc_auc",
                    "film_level_macro_f1"]
    comps = []
    print("paired film bootstrap per metric ...", flush=True)
    boot_tasks = [(a, b, m) for a, b in PAIRS for m in comp_metrics]
    boot_res = Parallel(n_jobs=args.jobs)(
        delayed(film_bootstrap_diff)(Y, runs, films, a, b, m, args.boot)
        for a, b, m in boot_tasks)
    for (a, b, m), (d, lo, hi, p, n) in zip(boot_tasks, boot_res):
        ra, rb = np.array(reps[a][m]), np.array(reps[b][m])
        better = (ra < rb) if m in LOWER_IS_BETTER else (ra > rb)
        comps.append({"a": a, "b": b, "metric": m, "diff": d, "ci_lo": lo, "ci_hi": hi,
                      "p_two_sided": p, "repeat_win_rate": float(better.mean()),
                      "draws_kept": n})
    out["comparisons"] = comps
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    # ---- 2. stability ----------------------------------------------------------- #
    stab = {}
    for a in arms:
        fold = runs[a]["fold_f1"]
        rep = np.array(reps[a]["macro_f1"])
        stab[a] = {"repeat_sd": float(rep.std(ddof=1)),
                   "repeat_range": float(rep.max() - rep.min()),
                   "fold_sd": float(fold.std(ddof=1)),
                   "fold_mean": float(fold.mean()),
                   "coefficient_of_variation_repeats": float(rep.std(ddof=1) / rep.mean())}
    cvc = json.loads(CV_CORRECTED.read_text(encoding="utf-8"))["subset5"]["arms"]
    for a in arms:
        lo, hi = cvc[a]["ci_lo"], cvc[a]["ci_hi"]
        stab[a]["film_bootstrap_ci_width"] = float(hi - lo)
        stab[a]["film_bootstrap_se_approx"] = float((hi - lo) / (2 * 1.96))

    # strict reproducibility: seed 42 must equal cv_corrected.json repeat by repeat
    ref = json.loads(CV_CORRECTED.read_text(encoding="utf-8"))["subset5"]["per_repeat_pooled"]
    repro = {a: bool(np.allclose(reps[a]["macro_f1"], ref[a], atol=1e-12)) for a in arms}
    out["reproduction_check"] = {"all_match": all(repro.values()), "arms": repro,
                                 "what": "pooled per-repeat macro-F1 under seed 42 vs "
                                         "cv_corrected.json subset5.per_repeat_pooled"}
    print(f"REPRODUCTION vs cv_corrected.json: "
          f"{'identical' if all(repro.values()) else 'MISMATCH ' + str(repro)}", flush=True)

    # other master seeds: fold assignment AND random forest change together
    per_seed = {config.SEED: {a: summary(reps[a]["macro_f1"]) for a in arms}}
    per_seed_reps = {config.SEED: {a: reps[a]["macro_f1"] for a in arms}}
    for s in args.seeds:
        if s == config.SEED:
            continue
        print(f"seed {s} ...", flush=True)
        r = run(arms, Y, films, s, args.jobs, tune=False)
        rr = {a: [f1_score(Y, r[a]["hard"][k], average="macro", zero_division=0)
                  for k in range(N_REPEATS)] for a in arms}
        per_seed[s] = {a: summary(rr[a]) for a in arms}
        per_seed_reps[s] = rr
        print(f"  [{time.time() - t0:.0f}s]", flush=True)

    seeds_block = {"per_seed_mean": {str(s): {a: v[a]["mean"] for a in arms}
                                     for s, v in per_seed.items()}}
    seeds_block["across_seeds"] = {
        a: {"mean_of_means": float(np.mean([per_seed[s][a]["mean"] for s in per_seed])),
            "sd_of_means": float(np.std([per_seed[s][a]["mean"] for s in per_seed], ddof=1)),
            "range_of_means": float(np.ptp([per_seed[s][a]["mean"] for s in per_seed])),
            "sd_all_repeats": float(np.std(np.concatenate(
                [per_seed_reps[s][a] for s in per_seed]), ddof=1))}
        for a in arms}
    seeds_block["differences_per_seed"] = {
        f"{SHORT[a]} - {SHORT[b]}": {
            str(s): float(per_seed[s][a]["mean"] - per_seed[s][b]["mean"]) for s in per_seed}
        for a, b in PAIRS}
    seeds_block["difference_sign_stable"] = {
        k: bool(len({np.sign(v) for v in d.values()}) == 1)
        for k, d in seeds_block["differences_per_seed"].items()}
    stab_out = {"seed_42": stab, "seeds": seeds_block}
    out["stability"] = stab_out

    # ---- print ------------------------------------------------------------------ #
    w = 24
    print(f"\n{'=' * 100}\nMETRICS (seed 42, mean over 10 pooled repeats; "
          f"'random' = base-rate random guess)\n{'=' * 100}")
    head = "".join(f"{SHORT[a][:14]:>15}" for a in arms)
    print(f"{'metric':{w}}{head}{'random':>9}")
    for m in METRICS:
        row = "".join(f"{metrics[m]['arms'][a]['mean']:>15.3f}" for a in arms)
        print(f"{m:{w}}{row}{chance[m]:>9.3f}")
    print("\nroute ranking by metric (best first):")
    for m, r in ranking.items():
        print(f"  {m:{w}} {' > '.join(r)}")
    print(f"\n{'comparison':46}{'metric':26}{'diff':>8}{'p':>8}{'win':>6}")
    for c in comps:
        print(f"{SHORT[c['a']] + ' vs ' + SHORT[c['b']]:46}{c['metric']:26}"
              f"{c['diff']:>+8.3f}{c['p_two_sided']:>8.3f}{c['repeat_win_rate']:>6.0%}")
    print(f"\n{'STABILITY (macro-F1)':{w}}{'repeat sd':>10}{'fold sd':>9}{'boot se':>9}"
          f"{'seed sd':>9}{'seed range':>11}")
    for a in arms:
        s, g = stab[a], seeds_block["across_seeds"][a]
        print(f"{SHORT[a]:{w}}{s['repeat_sd']:>10.3f}{s['fold_sd']:>9.3f}"
              f"{s['film_bootstrap_se_approx']:>9.3f}{g['sd_of_means']:>9.3f}"
              f"{g['range_of_means']:>11.3f}")
    print("\ndifference sign stable across seeds:", seeds_block["difference_sign_stable"])

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {out_path}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
