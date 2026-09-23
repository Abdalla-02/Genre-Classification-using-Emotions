"""Eerola genre classification, re-run with two cross-validation corrections.

Why this exists
---------------
A check of the folds behind every in-domain Eerola result found two problems. Neither
lets test information leak into training, but both distort the numbers.

  (a) Rare genres are missing from many test folds. With 10x5 grouped folds, Comedy
      (4 films) and Horror (5 films) have no clip at all in 17 and 15 of the 50 test
      folds of the 5-genre protocol; on 8 genres, Documentary is absent from 31 of 50.
      Macro-F1 computed PER FOLD scores such a genre 0 in that fold whatever the model
      does (zero_division=0), so a fifth of the average is fixed at zero in about a
      third of the folds. Comparisons stay fair -- every arm sees the same folds -- but
      the absolute numbers are understated and the fold scores carry extra noise.
      Stratifying the folds cannot cure it: Comedy has four films for five folds.
      Fix: within one repeat the five test folds partition every clip exactly once, so
      the out-of-fold predictions are POOLED and macro-F1 is computed once per repeat,
      over all clips, where every genre is present.

  (b) The in-domain predicted-emotion arm trained the genre classifier on the random
      forest's predictions for its OWN training clips -- which a forest fits almost
      perfectly -- and then tested it on predictions for unseen films, which are much
      noisier. Train and test inputs came from different distributions.
      Fix: the training clips' emotions are predicted out-of-fold by an inner
      GroupKFold over films, exactly as the transfer experiments already do
      (``exp_zero_shot.oof_emotions``).

What is reported
----------------
Every arm is scored under BOTH metrics from the same predictions, and each
predicted-emotion arm is run BOTH ways, so the effect of each fix can be read off on its
own. The arms and folds are identical to ``exp_statistical_power.py``, so the old-metric
figures for the arms the two scripts share must reproduce ``statistical_power.json``
exactly; the script checks this and records the result. If they did not match, nothing
else in this file could be trusted.

The Nadeau-Bengio test needs one score per fold, so it cannot be used on the pooled
metric. Significance there comes from a paired cluster bootstrap over FILMS on the
pooled predictions, averaged over the ten repeats. That is the test the zero-shot
experiment already uses, with films as the resampling unit because clips of one film are
not independent. Resamples in which a genre has no positive clip are dropped, as there.
The per-arm interval is the same bootstrap and so reflects the finite set of films, which
the old per-repeat t-interval did not.

All six audio representations, including MusiCNN, MIR and wav2vec 2.0, which were
previously scored only in a separate 5x5 run, go through the same 10x5 folds here. The
headline table therefore no longer mixes two protocols. The emotion ablation is re-run on
the same folds as well, since fix (a) applies to it too.

Protocol: 10x5 RepeatedGroupKFold by film (seed 42), C tuned by nested inner GroupKFold,
binary-relevance logistic regression with class weighting.

Run:  python experiments/evaluation/exp_cv_corrected.py [--jobs 12] [--boot 2000]
Writes results/cv_corrected.json; statistical_power.json is left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.repeated import (  # noqa: E402
    RepeatedGroupKFold,
    corrected_paired_t,
    nb_p_limit,
    repeat_ci,
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

RESULTS = config.RESULTS_DIR / "cv_corrected.json"
OLD = config.RESULTS_DIR / "statistical_power.json"
N_REPEATS, N_SPLITS = 10, 5

# Arm names shared with statistical_power.json keep their exact names there, so the
# reproduction check is a plain lookup.
CEIL = "ground-truth emotion(11) [ceiling]"
PRED_IN = "VGGish -> PREDICTED emotion(11)"
PRED_OOF = "VGGish -> predicted emotion(11), OOF-trained"
W2V_IN = "wav2vec2 -> predicted emotion(11)"
W2V_OOF = "wav2vec2 -> predicted emotion(11), OOF-trained"
PCA8 = "PCA-8(VGGish) [control]"
VGG = "VGGish-128 (direct)"
AST = "AST-768"
CLAP = "CLAP-512"
MUSICNN = "MusiCNN-200"
MIR = "MIR-103"
W2V = "wav2vec2-768"
DUMMY = "dummy"
CEIL8 = "ground-truth emotion(11)"


# --------------------------------------------------------------------------- #
# One fold of one arm. Everything an arm learns is fitted on the training fold only.
# Random forests run single-threaded here because the folds run in parallel; a
# forest's output does not depend on n_jobs, so results are unchanged.
# --------------------------------------------------------------------------- #
def oof_emotions(X, E, groups):
    """Out-of-fold emotions for the TRAINING clips: inner GroupKFold over films."""
    out = np.zeros_like(E)
    for tr, te in GroupKFold(n_splits=5).split(X, E, groups):
        reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=1)
        out[te] = reg.fit(X[tr], E[tr]).predict(X[te])
    return out


def fit_predict(kind, X, Y, tr, te, groups, emo=None):
    if kind == "dummy":
        return np.asarray(build_classifier("dummy").fit(X[tr], Y[tr]).predict(X[te])), None
    if kind == "plain":
        Ztr, Zte = X[tr], X[te]
    elif kind == "pca8":
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=8, random_state=config.SEED).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
    elif kind in ("pred_in", "pred_oof"):
        reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED,
                                    n_jobs=1).fit(X[tr], emo[tr])
        Zte = build_emotion_features(reg.predict(X[te]))
        Etr = reg.predict(X[tr]) if kind == "pred_in" else oof_emotions(X[tr], emo[tr], groups[tr])
        Ztr = build_emotion_features(Etr)
    else:
        raise ValueError(kind)
    C = select_logreg_C(Ztr, Y[tr], groups[tr])
    return np.asarray(build_classifier("logreg", C=C).fit(Ztr, Y[tr]).predict(Zte)), C


def run_arms(arms, Y, folds, groups, jobs):
    """{name: (kind, X, emo)} -> {name: [(pred, C) per fold]} on the SAME folds."""
    tasks = [(name, i) for name in arms for i in range(len(folds))]
    res = Parallel(n_jobs=jobs)(
        delayed(fit_predict)(arms[n][0], arms[n][1], Y, folds[i][0], folds[i][1], groups,
                             arms[n][2])
        for n, i in tasks)
    out = {name: [None] * len(folds) for name in arms}
    for (name, i), r in zip(tasks, res):
        out[name][i] = r
    return out


# --------------------------------------------------------------------------- #
# The two metrics, and the film-level bootstrap for the pooled one
# --------------------------------------------------------------------------- #
def per_fold_f1(Y, preds, folds):
    """OLD metric: macro-F1 per test fold (a genre absent from the fold scores 0)."""
    return np.array([f1_score(Y[te], p, average="macro", zero_division=0)
                     for (_, te), (p, _) in zip(folds, preds)])


def pooled_predictions(Y, preds, folds, n_splits):
    """Stitch each repeat's five test folds back into one full prediction matrix."""
    n_rep = len(folds) // n_splits
    P = np.zeros((n_rep,) + Y.shape, dtype=int)
    for i, ((_, te), (p, _)) in enumerate(zip(folds, preds)):
        P[i // n_splits][te] = p
    return P


def film_counts(Y, P, M):
    """Per-film TP/FP/FN for one prediction matrix. M is the (films x clips) one-hot."""
    Y, P = Y.astype(int), P.astype(int)
    return np.stack([M @ (Y * P), M @ ((1 - Y) * P), M @ (Y * (1 - P))])


def macro_from_counts(W, C):
    """Macro-F1 for every bootstrap draw at once: W (B x films), C (3 x films x genres)."""
    tp, fp, fn = (W @ C[k] for k in range(3))
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)
    return f1.mean(axis=1)


class FilmBootstrap:
    """Paired cluster bootstrap over films, shared by every arm so comparisons pair."""

    def __init__(self, Y, films, n_boot, seed=config.SEED):
        uniq, film_of = np.unique(films, return_inverse=True)
        self.M = np.zeros((len(uniq), len(films)))
        self.M[film_of, np.arange(len(films))] = 1
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(uniq), (n_boot, len(uniq)))
        self.W = np.zeros((n_boot, len(uniq)))
        np.add.at(self.W, (np.repeat(np.arange(n_boot), len(uniq)), draws.ravel()), 1)
        self.W_full = np.ones((1, len(uniq)))
        pos = self.W @ (self.M @ Y)                   # positives per genre per draw
        n = self.W @ self.M.sum(axis=1)               # clips per draw
        # as in exp_zero_shot: drop a draw in which a genre vanished or is universal
        self.valid = (pos.min(axis=1) > 0) & (pos.max(axis=1) < n)
        self.Y = Y

    def scores(self, P):
        """(boot draws, point estimate), each averaged over the repeats in P."""
        C = [film_counts(self.Y, p, self.M) for p in P]
        boot = np.mean([macro_from_counts(self.W, c) for c in C], axis=0)
        point = float(np.mean([macro_from_counts(self.W_full, c)[0] for c in C]))
        return boot[self.valid], point


def summarise(Y, runs, folds, films, n_splits, boot):
    """Both metrics, per arm, from one set of predictions."""
    n_test = float(np.mean([len(te) for _, te in folds]))
    arms, per_fold, per_rep, draws = {}, {}, {}, {}
    for name, preds in runs.items():
        pf = per_fold_f1(Y, preds, folds)
        P = pooled_predictions(Y, preds, folds, n_splits)
        rep = np.array([f1_score(Y, p, average="macro", zero_division=0) for p in P])
        b, point = boot.scores(P)
        lo, hi = np.percentile(b, [2.5, 97.5])
        old = repeat_ci(pf, n_splits)
        Cs = [c for _, c in preds if c is not None]
        arms[name] = {
            "mean": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "pooled_repeat_min": float(rep.min()), "pooled_repeat_max": float(rep.max()),
            "old_metric_mean": old.mean, "old_metric_ci_lo": old.lo,
            "old_metric_ci_hi": old.hi,
            "C_selected": sorted(Counter(Cs).items()) if Cs else None,
        }
        per_fold[name], per_rep[name], draws[name] = pf, rep, b
    return arms, per_fold, per_rep, draws, n_test


def compare(pairs, per_fold, per_rep, draws, n_train, n_test):
    rows = []
    for a, b in pairs:
        d = draws[a] - draws[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        _, p_old = corrected_paired_t(per_fold[a], per_fold[b], n_train, n_test)
        rows.append({
            "a": a, "b": b, "diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "repeat_win_rate": float((per_rep[a] > per_rep[b]).mean()),
            "old_metric_diff": float(per_fold[a].mean() - per_fold[b].mean()),
            "old_metric_p_corrected": p_old,
            # the best p Nadeau-Bengio could reach with unlimited repeats on this corpus
            "old_metric_p_limit": nb_p_limit(per_fold[a], per_fold[b], n_train, n_test),
        })
    return rows


def print_block(title, arms, comps):
    w = max(len(k) for k in arms) + 2
    print(f"\n{'=' * (w + 58)}\n{title}\n{'=' * (w + 58)}")
    print(f"{'arm':{w}}{'OLD per-fold':>13}{'NEW pooled':>12}{'change':>9}   95% CI (films)")
    for k, v in sorted(arms.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{k:{w}}{v['old_metric_mean']:>13.3f}{v['mean']:>12.3f}"
              f"{v['mean'] - v['old_metric_mean']:>+9.3f}   [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    if comps:
        lw = max(len(f"{c['a']} vs {c['b']}") for c in comps) + 2
        print(f"\n{'comparison':{lw}}{'NEW diff':>9}{'p(boot)':>9}{'win':>6}"
              f"{'OLD diff':>10}{'p(NB)':>8}")
        for c in comps:
            s = "*" if c["p_two_sided"] < 0.05 else " "
            print(f"{c['a'] + ' vs ' + c['b']:{lw}}{c['diff']:>+9.3f}{c['p_two_sided']:>8.3f}{s}"
                  f"{c['repeat_win_rate']:>6.0%}{c['old_metric_diff']:>+10.3f}"
                  f"{c['old_metric_p_corrected']:>8.3f}")


def pack(arms, comps, per_fold, per_rep, n_clips, n_films, genres):
    return {
        "n_clips": int(n_clips), "n_films": int(n_films), "genres": genres,
        "arms": arms, "comparisons": comps,
        "old_vs_new": {k: {"old (per-fold mean)": v["old_metric_mean"],
                           "new (pooled per repeat)": v["mean"],
                           "change": v["mean"] - v["old_metric_mean"]}
                       for k, v in sorted(arms.items(), key=lambda kv: -kv[1]["mean"])},
        "per_repeat_pooled": {k: v.tolist() for k, v in per_rep.items()},
        "per_fold": {k: v.tolist() for k, v in per_fold.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    set_seed()
    t0 = time.time()

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    g8 = df["soundtrack"].to_numpy()
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y5 = Y8[keep][:, idx]
    g5 = dfk["soundtrack"].to_numpy()

    def cache(d, frame):
        return assemble_from_cache(frame, "set1", cache_dir=d)

    cv = RepeatedGroupKFold(N_SPLITS, N_REPEATS)
    folds5 = list(cv.split(np.zeros(len(dfk)), None, g5))
    folds8 = list(cv.split(np.zeros(len(df)), None, g8))
    out = {"design": {"n_repeats": N_REPEATS, "n_splits": N_SPLITS, "seed": config.SEED,
                      "scheme": "RepeatedGroupKFold by soundtrack (same folds as "
                                "statistical_power.json)",
                      "metric": "macro_f1 on out-of-fold predictions pooled per repeat",
                      "old_metric": "macro_f1 per fold, averaged",
                      "significance": "paired cluster bootstrap over films, "
                                      f"{args.boot} draws, averaged over repeats",
                      "emotion_training": "OOF-trained arms: training-clip emotions "
                                          "predicted by an inner GroupKFold(5) over films"}}

    # ---- 5-genre subset: every representation, both emotion-training modes -------- #
    E5 = dfk[config.EMOTIONS].to_numpy(float)
    Xvgg = cache(config.VGGISH_EMBEDDINGS_DIR, dfk)
    Xw2v = cache(config.W2V_EMBEDDINGS_DIR, dfk)
    arms5 = {
        CEIL: ("plain", dfk[config.FEATURE_COLS].to_numpy(float), None),
        PRED_IN: ("pred_in", Xvgg, E5),
        PRED_OOF: ("pred_oof", Xvgg, E5),
        W2V_IN: ("pred_in", Xw2v, E5),
        W2V_OOF: ("pred_oof", Xw2v, E5),
        VGG: ("plain", Xvgg, None),
        AST: ("plain", cache(config.EMBEDDINGS_DIR, dfk), None),
        CLAP: ("plain", cache(config.CLAP_EMBEDDINGS_DIR, dfk), None),
        MUSICNN: ("plain", cache(config.MUSICNN_EMBEDDINGS_DIR, dfk), None),
        MIR: ("plain", cache(config.MIR_EMBEDDINGS_DIR, dfk), None),
        W2V: ("plain", Xw2v, None),
        PCA8: ("pca8", Xvgg, None),
        DUMMY: ("dummy", Xvgg, None),
    }
    print(f"5-genre: {len(dfk)} clips / {len(set(g5))} films, {len(arms5)} arms x "
          f"{len(folds5)} folds ...", flush=True)
    runs5 = run_arms(arms5, Y5, folds5, g5, args.jobs)
    boot5 = FilmBootstrap(Y5, g5, args.boot)
    a5, pf5, pr5, dr5, nt5 = summarise(Y5, runs5, folds5, g5, N_SPLITS, boot5)
    pairs5 = [(PRED_OOF, x) for x in (VGG, AST, CLAP, MUSICNN, MIR, W2V, PCA8, CEIL)] + [
        (PRED_IN, PRED_OOF), (W2V_OOF, W2V), (W2V_OOF, W2V_IN),
        (CEIL, VGG), (CEIL, AST), (CEIL, CLAP), (AST, CLAP),
        (PRED_IN, PCA8), (PRED_IN, VGG), (PRED_IN, AST), (PRED_IN, CEIL),
        # for the waveform section: is wav2vec 2.0 behind the spectrogram models, and is
        # its emotion route level with the ratings?
        (VGG, W2V), (AST, W2V), (CLAP, W2V), (MUSICNN, W2V), (MIR, W2V), (W2V_OOF, CEIL),
        # the control against the embedding it compresses
        (PCA8, VGG)]
    c5 = compare(pairs5, pf5, pr5, dr5, len(dfk) - nt5, nt5)
    print_block("5 GENRES -- old per-fold metric vs new pooled metric", a5, c5)
    out["subset5"] = pack(a5, c5, pf5, pr5, len(dfk), len(set(g5)), config.GENRE_SUBSET)
    out["subset5"]["bootstrap_draws_kept"] = int(boot5.valid.sum())
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    # ---- 8 genres --------------------------------------------------------------- #
    Xvgg8 = cache(config.VGGISH_EMBEDDINGS_DIR, df)
    arms8 = {
        CEIL8: ("plain", df[config.FEATURE_COLS].to_numpy(float), None),
        PRED_OOF: ("pred_oof", Xvgg8, df[config.EMOTIONS].to_numpy(float)),
        VGG: ("plain", Xvgg8, None),
        AST: ("plain", cache(config.EMBEDDINGS_DIR, df), None),
        DUMMY: ("dummy", Xvgg8, None),
    }
    print(f"\n8-genre: {len(df)} clips / {len(set(g8))} films ...", flush=True)
    runs8 = run_arms(arms8, Y8, folds8, g8, args.jobs)
    boot8 = FilmBootstrap(Y8, g8, args.boot)
    a8, pf8, pr8, dr8, nt8 = summarise(Y8, runs8, folds8, g8, N_SPLITS, boot8)
    c8 = compare([(CEIL8, AST), (CEIL8, VGG), (PRED_OOF, AST), (PRED_OOF, VGG)],
                 pf8, pr8, dr8, len(df) - nt8, nt8)
    print_block("8 GENRES -- old per-fold metric vs new pooled metric", a8, c8)
    out["full8"] = pack(a8, c8, pf8, pr8, len(df), len(set(g8)), config.PRIMARY_GENRES)
    out["full8"]["bootstrap_draws_kept"] = int(boot8.valid.sum())

    # ---- emotion ablation, 5 genres, same folds (fix (a) only: ground truth) ------ #
    E = config.EMOTIONS
    subsets = {"all 8 emotions [reference]": E,
               **{f"without {e}": [x for x in E if x != e] for e in E},
               "valence + energy (2-d circumplex)": ["valence", "energy"],
               "valence + energy + tension (3-d)": ["valence", "energy", "tension"],
               "5 discrete only": ["anger", "fear", "happy", "sad", "tender"],
               "fear only": ["fear"],
               "fear + valence": ["fear", "valence"]}
    armsA = {k: ("plain", dfk[v].to_numpy(float), None) for k, v in subsets.items()}
    print("\nablation ...", flush=True)
    runsA = run_arms(armsA, Y5, folds5, g5, args.jobs)
    aA, pfA, prA, drA, _ = summarise(Y5, runsA, folds5, g5, N_SPLITS, boot5)
    ref = "all 8 emotions [reference]"
    cA = compare([(k, ref) for k in subsets if k != ref], pfA, prA, drA,
                 len(dfk) - nt5, nt5)
    print_block("ABLATION (ground-truth ratings, 5 genres)", aA, cA)
    out["ablation"] = pack(aA, cA, pfA, prA, len(dfk), len(set(g5)), config.GENRE_SUBSET)

    # ---- reproduction check against statistical_power.json ---------------------- #
    old = json.loads(OLD.read_text(encoding="utf-8"))
    checks = []
    for label, arms, ref_arms in (("subset5/tuned", a5, old["subset5"]["tuned"]["arms"]),
                                  ("full8", a8, old["full8"]["arms"])):
        for name, v in ref_arms.items():
            if name in arms:
                got, want = arms[name]["old_metric_mean"], v["mean"]
                checks.append({"arm": f"{label}: {name}", "statistical_power.json": want,
                               "this run, old metric": got,
                               "match": bool(abs(got - want) < 1e-9)})
    ok = all(c["match"] for c in checks)
    out["reproduction_check"] = {"all_match": ok, "arms": checks}
    print(f"\nREPRODUCTION CHECK vs statistical_power.json: "
          f"{'all ' + str(len(checks)) + ' arms identical' if ok else 'MISMATCH'}")
    for c in checks:
        if not c["match"]:
            print(f"  {c['arm']}: expected {c['statistical_power.json']:.6f}, "
                  f"got {c['this run, old metric']:.6f}")

    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {RESULTS}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
