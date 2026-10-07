"""Eerola genre classification scored per FILM: the headline metric of the thesis.

Why film level
--------------
Genre is a property of the film: every clip of a film carries the film's IMDb genres,
identical across its clips. Scoring clips therefore counts a film with twelve clips three
times as heavily as a film with four, and treats the clips of one film as independent
decisions when they are not. The Blockbuster dataset is scored per film already (its rows
are films), and so is the zero-shot transfer to it; scoring Eerola per film puts every
genre evaluation of the thesis on the same unit.

Film-level prediction: the clip probabilities of a film are averaged and each genre is
predicted when the average is at least 0.5 (as ``exp_metrics_stability.film_macro_f1``).
Macro-F1 is then computed over the films, once per repeat on the pooled out-of-fold
predictions, exactly like the clip-level score of ``exp_cv_corrected.py``.

What is reported
----------------
The same arms, folds and seeds as ``exp_cv_corrected.py`` (5 genres: all six
representations, both emotion routes, the PCA-8 control, the ceiling; 8 genres; the
emotion ablation), each scored at film level and, from the same predictions, at clip
level. The clip-level scores must reproduce ``cv_corrected.json`` exactly; the script
checks this. The stability analysis of ``exp_metrics_stability.py`` (spread over the
10 repeats, film bootstrap, five master seeds) is repeated at film level for its six
arms, and the clip-level seed means must reproduce ``metrics_stability.json``.

Significance: the paired film bootstrap (2000 draws, the draws of ``exp_cv_corrected``),
now resampling films that are scored as films. The Nadeau-Bengio test needs a score per
test fold; a fold holds about eight films, so at film level most genres have no positive
film in it and a per-fold score is meaningless. The clip-level Nadeau-Bengio p (and its
limit) is kept beside every comparison as the stricter robustness check.

Caveat: macro-F1 over 41 films gives each genre one fifth of the score, and Comedy has
4 films and Horror 5 (8 genres: Documentary 2, Biography 3). One film more or less
moves such a genre's F1 a lot, which the bootstrap intervals reflect.

Run:  python experiments/evaluation/exp_film_level.py [--jobs 12] [--boot 2000]
Writes results/film_level.json (a reduced run writes *.partial.json instead).
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
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.evaluation.repeated import (  # noqa: E402
    RepeatedGroupKFold,
    corrected_paired_t,
    nb_p_limit,
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
from exp_cv_corrected import (  # noqa: E402
    AST, CEIL, CEIL8, CLAP, DUMMY, MIR, MUSICNN, PCA8, PRED_IN, PRED_OOF, VGG, W2V,
    W2V_IN, W2V_OOF, FilmBootstrap,
)
from exp_metrics_stability import oof_emotions, proba, random_guess_reference  # noqa: E402

RESULTS = config.RESULTS_DIR / "film_level.json"
CV_CORRECTED = config.RESULTS_DIR / "cv_corrected.json"
METRICS_STABILITY = config.RESULTS_DIR / "metrics_stability.json"
N_REPEATS, N_SPLITS = 10, 5
SEEDS = (42, 1, 2, 3, 4)
STABILITY_ARMS = (CEIL, PRED_OOF, VGG, AST, PCA8, DUMMY)


# --------------------------------------------------------------------------- #
# One fold of one arm -> (clip probabilities, clip 0.5 decisions)
# --------------------------------------------------------------------------- #
def fit_fold(kind, X, Y, tr, te, groups, emo, seed):
    if kind == "dummy":
        pred = np.asarray(build_classifier("dummy").fit(X[tr], Y[tr]).predict(X[te]))
        return pred.astype(float), pred
    if kind == "plain":
        Ztr, Zte = X[tr], X[te]
    elif kind == "pca8":
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=8, random_state=seed).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
    elif kind in ("pred_in", "pred_oof"):
        reg = RandomForestRegressor(n_estimators=300, random_state=seed,
                                    n_jobs=1).fit(X[tr], emo[tr])
        Zte = build_emotion_features(reg.predict(X[te]))
        Etr = (reg.predict(X[tr]) if kind == "pred_in"
               else oof_emotions(X[tr], emo[tr], groups[tr], seed))
        Ztr = build_emotion_features(Etr)
    else:
        raise ValueError(kind)
    C = select_logreg_C(Ztr, Y[tr], groups[tr])
    clf = build_classifier("logreg", C=C).fit(Ztr, Y[tr])
    return proba(clf, Zte), np.asarray(clf.predict(Zte))


def run(arms, Y, groups, seed, jobs):
    """All arms on the 10x5 folds of one seed -> pooled per-repeat matrices + fold scores."""
    folds = list(RepeatedGroupKFold(N_SPLITS, N_REPEATS, random_state=seed)
                 .split(np.zeros(len(Y)), None, groups))
    tasks = [(a, i) for a in arms for i in range(len(folds))]
    res = Parallel(n_jobs=jobs)(
        delayed(fit_fold)(arms[a][0], arms[a][1], Y, folds[i][0], folds[i][1], groups,
                          arms[a][2], seed) for a, i in tasks)
    out = {a: {"P": np.zeros((N_REPEATS,) + Y.shape),
               "hard": np.zeros((N_REPEATS,) + Y.shape, dtype=int),
               "fold_f1": np.zeros(len(folds))} for a in arms}
    for (a, i), (P, hard) in zip(tasks, res):
        r, te = i // N_SPLITS, folds[i][1]
        out[a]["P"][r][te], out[a]["hard"][r][te] = P, hard
        out[a]["fold_f1"][i] = f1_score(Y[te], hard, average="macro", zero_division=0)
    n_test = float(np.mean([len(te) for _, te in folds]))
    return out, n_test


# --------------------------------------------------------------------------- #
# Film level
# --------------------------------------------------------------------------- #
class Films:
    """Averaging matrix clips -> films, and the film labels."""

    def __init__(self, Y, films):
        self.ids, inv = np.unique(films, return_inverse=True)
        M = np.zeros((len(self.ids), len(films)))
        M[inv, np.arange(len(films))] = 1
        self.M = M / M.sum(1, keepdims=True)
        self.Y = (self.M @ Y > 0.5).astype(int)   # identical within a film

    def decide(self, P):
        """(repeats, clips, genres) probabilities -> (repeats, films, genres) decisions."""
        return np.stack([(self.M @ p >= 0.5).astype(int) for p in P])


def summarise(Y, runs, fl: Films, boot_film, boot_clip, n_splits):
    arms, rep_film, rep_clip, draws_f, draws_c, fold = {}, {}, {}, {}, {}, {}
    for name, r in runs.items():
        Fh = fl.decide(r["P"])
        rf = np.array([f1_score(fl.Y, f, average="macro", zero_division=0) for f in Fh])
        rc = np.array([f1_score(Y, h, average="macro", zero_division=0) for h in r["hard"]])
        bf, pf = boot_film.scores(Fh)
        bc, pc = boot_clip.scores(r["hard"])
        lo, hi = np.percentile(bf, [2.5, 97.5])
        clo, chi = np.percentile(bc, [2.5, 97.5])
        per_genre = np.mean([f1_score(fl.Y, f, average=None, zero_division=0) for f in Fh], 0)
        arms[name] = {
            "mean": pf, "ci_lo": float(lo), "ci_hi": float(hi),
            "repeat_sd": float(rf.std(ddof=1)), "repeat_min": float(rf.min()),
            "repeat_max": float(rf.max()),
            "per_genre_f1": [float(x) for x in per_genre],
            "clip_level_mean": pc, "clip_level_ci_lo": float(clo), "clip_level_ci_hi": float(chi),
        }
        rep_film[name], rep_clip[name], draws_f[name], draws_c[name] = rf, rc, bf, bc
        fold[name] = r["fold_f1"]
    return arms, rep_film, rep_clip, draws_f, draws_c, fold


def compare(pairs, rep_film, draws_f, draws_c, fold, n_train, n_test):
    rows = []
    for a, b in pairs:
        d = draws_f[a] - draws_f[b]
        dc = draws_c[a] - draws_c[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        _, p_nb = corrected_paired_t(fold[a], fold[b], n_train, n_test)
        rows.append({
            "a": a, "b": b, "diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "repeat_win_rate": float((rep_film[a] > rep_film[b]).mean()),
            "clip_level_diff": float(dc.mean()),
            "clip_level_p_two_sided": float(min(1.0, 2 * min((dc <= 0).mean(), (dc >= 0).mean()))),
            "clip_level_nb_p": p_nb,
            "clip_level_nb_p_limit": nb_p_limit(fold[a], fold[b], n_train, n_test),
        })
    return rows


def block(Y, groups, genres, arms_def, pairs, args, seed=config.SEED):
    runs, n_test = run(arms_def, Y, groups, seed, args.jobs)
    fl = Films(Y, groups)
    bf = FilmBootstrap(fl.Y, np.arange(len(fl.ids)), args.boot)    # one row per film
    bc = FilmBootstrap(Y, groups, args.boot)                         # clips, cv_corrected
    arms, rf, rc, df_, dc, fold = summarise(Y, runs, fl, bf, bc, N_SPLITS)
    comps = compare(pairs, rf, df_, dc, fold, len(Y) - n_test, n_test)
    films_per_genre = dict(zip(genres, fl.Y.sum(0).astype(int).tolist()))
    out = {"n_clips": int(len(Y)), "n_films": int(len(fl.ids)), "genres": genres,
           "films_per_genre": films_per_genre, "arms": arms, "comparisons": comps,
           "per_repeat_film": {k: v.tolist() for k, v in rf.items()},
           "per_repeat_clip": {k: v.tolist() for k, v in rc.items()},
           "bootstrap_draws_kept": int(bf.valid.sum())}
    return out, runs, fl


def print_block(title, b):
    arms = b["arms"]
    w = max(len(k) for k in arms) + 2
    print(f"\n{title}\n{'-' * (w + 50)}")
    print(f"{'arm':{w}}{'film':>8}{'95% CI (films)':>18}{'clip':>8}")
    for k, v in sorted(arms.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{k:{w}}{v['mean']:>8.3f}   [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
              f"{v['clip_level_mean']:>8.3f}")
    for c in b["comparisons"]:
        s = "*" if c["p_two_sided"] < 0.05 else " "
        print(f"  {c['a'][:34]:34} vs {c['b'][:30]:30} {c['diff']:+.3f} p={c['p_two_sided']:.3f}{s}"
              f"  (clip {c['clip_level_diff']:+.3f}, p={c['clip_level_p_two_sided']:.3f};"
              f" NB {c['clip_level_nb_p']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-seeds", action="store_true", help="skip the four extra seeds")
    args = ap.parse_args()
    set_seed()
    t0 = time.time()
    full = args.boot >= 2000 and not args.no_seeds
    out_path = RESULTS if full else RESULTS.with_suffix(".partial.json")

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    g8 = df["soundtrack"].to_numpy().astype(str)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y5 = Y8[keep][:, idx]
    g5 = dfk["soundtrack"].to_numpy().astype(str)

    def cache(d, frame):
        return assemble_from_cache(frame, "set1", cache_dir=d)

    out = {"design": {
        "unit": "film: clip probabilities averaged per film, genre predicted if >= 0.5; "
                "a film's labels are its clips' (identical within a film)",
        "protocol": "10x5 RepeatedGroupKFold by film, nested C, OOF-trained emotions; "
                    "the arms and folds of exp_cv_corrected.py",
        "metric": "macro-F1 over films, once per repeat on the pooled out-of-fold "
                  "predictions; clip-level macro-F1 from the same predictions alongside",
        "significance": f"paired film bootstrap, {args.boot} draws (the draws of "
                        "cv_corrected.json), averaged over repeats; clip-level "
                        "Nadeau-Bengio kept beside it (not defined per film: a test "
                        "fold holds about 8 films)",
        "stability_seeds": list(SEEDS)}}

    # ---- 5 genres ------------------------------------------------------------- #
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
    pairs5 = [(PRED_OOF, x) for x in (VGG, AST, CLAP, MUSICNN, MIR, W2V, PCA8, CEIL)] + [
        (PRED_IN, PRED_OOF), (W2V_OOF, W2V), (W2V_OOF, W2V_IN),
        (CEIL, VGG), (CEIL, AST), (CEIL, CLAP), (AST, CLAP),
        (VGG, W2V), (AST, W2V), (CLAP, W2V), (MUSICNN, W2V), (MIR, W2V), (W2V_OOF, CEIL),
        (PCA8, VGG)]
    print(f"5 genres: {len(dfk)} clips / {len(set(g5))} films, {len(arms5)} arms ...", flush=True)
    b5, runs5, fl5 = block(Y5, g5, config.GENRE_SUBSET, arms5, pairs5, args)
    b5["random_guess_film"] = random_guess_reference(Y5, g5)["film_level_macro_f1"]
    print_block("5 GENRES (film level; clip level beside it)", b5)
    out["subset5"] = b5
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    # ---- 8 genres ------------------------------------------------------------- #
    Xvgg8 = cache(config.VGGISH_EMBEDDINGS_DIR, df)
    arms8 = {
        CEIL8: ("plain", df[config.FEATURE_COLS].to_numpy(float), None),
        PRED_OOF: ("pred_oof", Xvgg8, df[config.EMOTIONS].to_numpy(float)),
        VGG: ("plain", Xvgg8, None),
        AST: ("plain", cache(config.EMBEDDINGS_DIR, df), None),
        DUMMY: ("dummy", Xvgg8, None),
    }
    print(f"\n8 genres: {len(df)} clips / {len(set(g8))} films ...", flush=True)
    b8, _, _ = block(Y8, g8, config.PRIMARY_GENRES, arms8,
                     [(CEIL8, AST), (CEIL8, VGG), (PRED_OOF, AST), (PRED_OOF, VGG)], args)
    b8["random_guess_film"] = random_guess_reference(Y8, g8)["film_level_macro_f1"]
    print_block("8 GENRES (film level)", b8)
    out["full8"] = b8
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    # ---- ablation (ground-truth ratings, 5 genres) ---------------------------- #
    E = config.EMOTIONS
    subsets = {"all 8 emotions [reference]": E,
               **{f"without {e}": [x for x in E if x != e] for e in E},
               "valence + energy (2-d circumplex)": ["valence", "energy"],
               "valence + energy + tension (3-d)": ["valence", "energy", "tension"],
               "5 discrete only": ["anger", "fear", "happy", "sad", "tender"],
               "fear only": ["fear"],
               "fear + valence": ["fear", "valence"]}
    armsA = {k: ("plain", dfk[v].to_numpy(float), None) for k, v in subsets.items()}
    ref = "all 8 emotions [reference]"
    print("\nablation ...", flush=True)
    bA, _, _ = block(Y5, g5, config.GENRE_SUBSET, armsA,
                     [(k, ref) for k in subsets if k != ref], args)
    print_block("ABLATION (film level)", bA)
    out["ablation"] = bA

    # ---- stability over master seeds (the six arms of exp_metrics_stability) --- #
    seeds = {}
    if not args.no_seeds:
        sub = {k: arms5[k] for k in STABILITY_ARMS}
        seeds["42"] = {k: {"film": float(np.mean(b5["per_repeat_film"][k])),
                           "clip": float(np.mean(b5["per_repeat_clip"][k]))}
                       for k in STABILITY_ARMS}
        for s in SEEDS[1:]:
            print(f"\nseed {s} ...", flush=True)
            r, _ = run(sub, Y5, g5, s, args.jobs)
            seeds[str(s)] = {}
            for k in STABILITY_ARMS:
                Fh = fl5.decide(r[k]["P"])
                seeds[str(s)][k] = {
                    "film": float(np.mean([f1_score(fl5.Y, f, average="macro",
                                                    zero_division=0) for f in Fh])),
                    "clip": float(np.mean([f1_score(Y5, h, average="macro", zero_division=0)
                                           for h in r[k]["hard"]]))}
            print(f"  [{time.time() - t0:.0f}s]", flush=True)
    stab = {}
    for k in STABILITY_ARMS:
        a = b5["arms"][k]
        st = {"repeat_sd": a["repeat_sd"],
              "repeat_range": a["repeat_max"] - a["repeat_min"],
              "film_bootstrap_se_approx": (a["ci_hi"] - a["ci_lo"]) / (2 * 1.96)}
        if seeds:
            m = np.array([seeds[s][k]["film"] for s in seeds])
            st.update({"seed_means": m.tolist(), "sd_of_seed_means": float(m.std(ddof=1)),
                       "range_of_seed_means": float(m.max() - m.min())})
        stab[k] = st
    if seeds:
        stab["emotion_minus_vggish_per_seed"] = [
            seeds[s][PRED_OOF]["film"] - seeds[s][VGG]["film"] for s in seeds]
        stab["emotion_minus_pca8_per_seed"] = [
            seeds[s][PRED_OOF]["film"] - seeds[s][PCA8]["film"] for s in seeds]
    out["stability"] = {"per_seed": seeds, "summary": stab}

    # ---- reproduction checks --------------------------------------------------- #
    cvc = json.loads(CV_CORRECTED.read_text(encoding="utf-8"))
    checks = []
    for key, b in (("subset5", b5), ("full8", b8), ("ablation", bA)):
        for name, reps in b["per_repeat_clip"].items():
            want = cvc[key]["per_repeat_pooled"][name]
            checks.append({"what": f"cv_corrected {key}: {name}",
                           "match": bool(np.allclose(reps, want, atol=1e-12, rtol=0))})
        for c in b["comparisons"]:
            ref_c = [x for x in cvc[key]["comparisons"] if x["a"] == c["a"] and x["b"] == c["b"]]
            if ref_c:
                checks.append({"what": f"cv_corrected {key}: p {c['a']} vs {c['b']}",
                               "match": bool(abs(ref_c[0]["p_two_sided"]
                                                 - c["clip_level_p_two_sided"]) < 1e-12)})
    if seeds:
        ms = json.loads(METRICS_STABILITY.read_text(encoding="utf-8"))
        for s in seeds:
            for k in STABILITY_ARMS:
                want = ms["stability"]["seeds"]["per_seed_mean"][s][k]
                checks.append({"what": f"metrics_stability seed {s}: {k}",
                               "match": bool(abs(want - seeds[s][k]["clip"]) < 1e-12)})
        want_f = ms["metrics"]["film_level_macro_f1"]["arms"]
        for k in STABILITY_ARMS:
            checks.append({"what": f"metrics_stability film-level mean: {k}",
                           "match": bool(abs(want_f[k]["mean"] - seeds["42"][k]["film"]) < 1e-12)})
    ok = all(c["match"] for c in checks)
    out["reproduction_check"] = {"all_match": ok, "n_checks": len(checks),
                                 "mismatches": [c["what"] for c in checks if not c["match"]]}
    print(f"\nREPRODUCTION CHECK: {'all ' + str(len(checks)) + ' identical' if ok else 'MISMATCH'}")
    for c in checks:
        if not c["match"]:
            print("  mismatch:", c["what"])

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {out_path}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
