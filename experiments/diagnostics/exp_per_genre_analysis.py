"""What lies behind the per-genre figure (film-level F1 per genre and pipeline)?

The figure `per_genre` shows, for the five-genre subset, the film-level F1 of every pipeline
for each genre, with the ratings (the ceiling) and the most-frequent baseline as reference
rows. This script adds what the figure cannot show by itself.

1. **The right floor per genre.** The most-frequent baseline scores 0 on every genre except
   Drama, so it is no floor for a single minority genre. The trivial classifier that assigns
   every genre to every film has recall 1 and precision equal to the share of films p with
   the genre, so its F1 is 2p / (1 + p): 0.46 for Action and Crime, 0.84 for Drama, 0.18
   for Comedy, 0.22 for Horror, and its macro-F1 is their mean. It is run as an arm on the
   same folds, so it enters the paired film bootstrap like any other pipeline. It is the
   extreme case of the operating-point effect: F1 rewards recall, and this classifier has
   all of it.
2. **Uncertainty per genre.** The paired film bootstrap (the draws of `film_level.json`)
   applied to each genre separately: the emotion pipeline against every other arm, per genre.
3. **How often each pipeline assigns each genre**, against how often the genre occurs.
4. **Ranking, independent of the threshold.** The film-level area under the ROC curve per
   genre (film probabilities = clip probabilities averaged per film, as for F1), averaged
   over the ten repeats, with the same paired film bootstrap. The all-positive classifier
   scores 0.5 by construction. If the emotion pipeline's F1 lead were only a matter of
   predicting a genre more often, its AUC would not be higher.
5. **Which genres make up a macro-F1 difference**: (F1_a - F1_b) / 5 per genre.
6. **The all-positive floor on the eight-genre task and over clips**, from the labels alone.

The arms, folds, seeds and fitting code are those of `exp_film_level.py`; every arm shared
with it must reproduce `film_level.json` (macro-F1 and per-genre F1), which the script checks.

Run:  python experiments/diagnostics/exp_per_genre_analysis.py [--jobs 12]
      (about 20 minutes; the two emotion arms dominate)
Writes results/per_genre_analysis.json.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from src import config  # noqa: E402
from src.features import add_derived_features, assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.utils import set_seed  # noqa: E402
from exp_cv_corrected import (  # noqa: E402
    AST, CEIL, CLAP, DUMMY, MIR, MUSICNN, PCA8, PRED_OOF, VGG, W2V, W2V_OOF,
    FilmBootstrap, film_counts,
)
import exp_film_level  # noqa: E402
from exp_film_level import block  # noqa: E402

OUT = config.RESULTS_DIR / "per_genre_analysis.json"
FILM_LEVEL = config.RESULTS_DIR / "film_level.json"
ALLPOS = "every genre for every film [reference]"
_fit_fold = exp_film_level.fit_fold


def fit_fold_with_allpos(kind, X, Y, tr, te, groups, emo, seed):
    if kind == "all_positive":
        ones = np.ones((len(te), Y.shape[1]), dtype=int)
        return ones.astype(float), ones
    return _fit_fold(kind, X, Y, tr, te, groups, emo, seed)


def all_positive_f1(Y: np.ndarray) -> list[float]:
    p = Y.mean(0)
    return [float(x) for x in 2 * p / (1 + p)]


def per_genre_draws(bf: FilmBootstrap, Fh: np.ndarray) -> np.ndarray:
    """Per-genre F1 for every valid bootstrap draw, averaged over the repeats: (draws, genres)."""
    out = []
    for f in Fh:
        tp, fp, fn = (bf.W @ c for c in film_counts(bf.Y, f, bf.M))
        den = 2 * tp + fp + fn
        out.append(np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0))
    return np.mean(out, 0)[bf.valid]


def auc_draws(W: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Weighted ROC AUC for every draw: W (draws x films) film weights, y labels, s scores."""
    pos, neg = y == 1, y == 0
    S = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
    Wp, Wn = W[:, pos], W[:, neg]
    den = Wp.sum(1) * Wn.sum(1)
    num = np.einsum("bi,ij,bj->b", Wp, S, Wn)
    return np.divide(num, den, out=np.full(len(W), np.nan), where=den > 0)


def summary(d: np.ndarray) -> dict:
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    set_seed()
    t0 = time.time()

    df = add_derived_features(load_set1())
    Y8 = genre_matrix(df)
    g8 = df["soundtrack"].to_numpy().astype(str)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    films = dfk["soundtrack"].to_numpy().astype(str)
    E = dfk[config.EMOTIONS].to_numpy(float)
    genres = list(config.GENRE_SUBSET)

    def cache(d):
        return assemble_from_cache(dfk, "set1", cache_dir=d)

    Xvgg, Xw2v = cache(config.VGGISH_EMBEDDINGS_DIR), cache(config.W2V_EMBEDDINGS_DIR)
    arms = {
        CEIL: ("plain", dfk[config.FEATURE_COLS].to_numpy(float), None),
        PRED_OOF: ("pred_oof", Xvgg, E),
        W2V_OOF: ("pred_oof", Xw2v, E),
        VGG: ("plain", Xvgg, None),
        AST: ("plain", cache(config.EMBEDDINGS_DIR), None),
        CLAP: ("plain", cache(config.CLAP_EMBEDDINGS_DIR), None),
        MUSICNN: ("plain", cache(config.MUSICNN_EMBEDDINGS_DIR), None),
        MIR: ("plain", cache(config.MIR_EMBEDDINGS_DIR), None),
        W2V: ("plain", Xw2v, None),
        PCA8: ("pca8", Xvgg, None),
        DUMMY: ("dummy", Xvgg, None),
        ALLPOS: ("all_positive", Xvgg, None),
    }
    exp_film_level.fit_fold = fit_fold_with_allpos
    pairs = [(a, ALLPOS) for a in arms if a != ALLPOS]
    print(f"{len(arms)} arms on the 10x5 folds ...", flush=True)
    b, runs, fl = block(Y, films, genres, arms, pairs, args)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    ref = json.load(open(FILM_LEVEL, encoding="utf-8"))["subset5"]["arms"]
    repro = {a: bool(abs(b["arms"][a]["mean"] - ref[a]["mean"]) < 1e-9
                     and np.allclose(b["arms"][a]["per_genre_f1"], ref[a]["per_genre_f1"]))
             for a in arms if a in ref}
    assert all(repro.values()), f"does not reproduce film_level.json: {repro}"
    expected = all_positive_f1(fl.Y)
    assert np.allclose(b["arms"][ALLPOS]["per_genre_f1"], expected)
    # ---- 2 + 4: per-genre paired bootstrap, F1 and film-level AUC ------------- #
    bf = FilmBootstrap(fl.Y, np.arange(len(fl.ids)), args.boot)
    Wv = bf.W[bf.valid]
    Fh = {a: fl.decide(r["P"]) for a, r in runs.items()}
    # repeats x films x genres; rounded so that a constant score stays a tie (averaging
    # leaves 1 - 1e-16 for some films, which would rank them and move a constant arm off 0.5)
    Pf = {a: np.stack([np.round(fl.M @ p, 10) for p in r["P"]]) for a, r in runs.items()}
    f1d = {a: per_genre_draws(bf, Fh[a]) for a in arms}
    aucd, auc_point = {}, {}
    for a in arms:
        aucd[a] = np.stack([np.mean([auc_draws(Wv, fl.Y[:, g], Pf[a][r][:, g])
                                     for r in range(len(Pf[a]))], 0)
                            for g in range(len(genres))], 1)                   # draws x genres
        auc_point[a] = [float(np.mean([auc_draws(np.ones((1, len(fl.ids))), fl.Y[:, g],
                                                 Pf[a][r][:, g])[0] for r in range(len(Pf[a]))]))
                        for g in range(len(genres))]
    assert np.allclose(auc_point[ALLPOS], 0.5) and np.allclose(auc_point[DUMMY], 0.5)
    per_genre_cmp, auc_cmp, macro_auc_cmp = {}, {}, {}
    for a in arms:
        if a == PRED_OOF:
            continue
        per_genre_cmp[a] = {g: summary(f1d[PRED_OOF][:, k] - f1d[a][:, k]) for k, g in enumerate(genres)}
        auc_cmp[a] = {g: summary(aucd[PRED_OOF][:, k] - aucd[a][:, k]) for k, g in enumerate(genres)}
        macro_auc_cmp[a] = summary(aucd[PRED_OOF].mean(1) - aucd[a].mean(1))
    print(f"  bootstrap per genre [{time.time() - t0:.0f}s]", flush=True)

    # ---- 3: how often each genre is assigned --------------------------------- #
    assign = {a: {"share_of_films": [float(x) for x in Fh[a].mean((0, 1))],
                  "genres_per_film": float(Fh[a].sum(2).mean())} for a in arms}
    truth = {"share_of_films": [float(x) for x in fl.Y.mean(0)],
             "genres_per_film": float(fl.Y.sum(1).mean())}

    # ---- 5: which genres make up a macro-F1 difference ----------------------- #
    pg = {a: np.array(b["arms"][a]["per_genre_f1"]) for a in arms}
    decomp = {f"{PRED_OOF} vs {x}": dict(zip(genres, ((pg[PRED_OOF] - pg[x]) / len(genres)).round(4).tolist()))
              for x in (VGG, PCA8, MUSICNN, CLAP, CEIL, ALLPOS)}

    # ---- 6: the all-positive floor on eight genres and over clips ------------ #
    keep8 = Y8.sum(1) >= 1
    fl8 = exp_film_level.Films(Y8[keep8], g8[keep8])
    floors = {
        "subset5": {"film_per_genre": dict(zip(genres, expected)),
                    "film_macro": float(np.mean(expected)),
                    "clip_per_genre": dict(zip(genres, all_positive_f1(Y))),
                    "clip_macro": float(np.mean(all_positive_f1(Y)))},
        "full8": {"n_films": int(len(fl8.ids)),
                  "film_per_genre": dict(zip(config.PRIMARY_GENRES, all_positive_f1(fl8.Y))),
                  "film_macro": float(np.mean(all_positive_f1(fl8.Y))),
                  "clip_macro": float(np.mean(all_positive_f1(Y8[keep8])))},
    }

    out = {
        "design": {"source": "exp_film_level.py arms, folds and seed; film-level F1 at 0.5 "
                             "and film-level ROC AUC (clip probabilities averaged per film)",
                   "bootstrap": f"paired film bootstrap, {args.boot} draws (those of film_level.json), "
                                f"{int(bf.valid.sum())} valid; per genre and per repeat, averaged over repeats",
                   "all_positive": "every genre assigned to every film; F1 = 2p/(1+p), AUC = 0.5"},
        "genres": genres, "films_per_genre": b["films_per_genre"],
        "arms": {a: {k: b["arms"][a][k] for k in ("mean", "ci_lo", "ci_hi", "clip_level_mean",
                                                    "per_genre_f1", "per_genre_precision",
                                                    "per_genre_recall", "per_genre_f1_repeat_sd")}
                 | {"per_genre_auc": auc_point[a], "macro_auc": float(np.mean(auc_point[a]))}
                 for a in arms},
        "vs_all_positive": [c for c in b["comparisons"]],
        "emotion_vs_arm_per_genre_f1": per_genre_cmp,
        "emotion_vs_arm_per_genre_auc": auc_cmp,
        "emotion_vs_arm_macro_auc": macro_auc_cmp,
        "assignment": {"truth": truth, "arms": assign},
        "macro_f1_difference_by_genre": decomp,
        "all_positive_floor": floors,
        "reproduces_film_level_json": repro,
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    # ---- print ---------------------------------------------------------------- #
    print(f"\nall-positive floor: film {floors['subset5']['film_macro']:.3f} "
          f"{[round(x, 3) for x in expected]}, clip {floors['subset5']['clip_macro']:.3f}; "
          f"8 genres film {floors['full8']['film_macro']:.3f}, clip {floors['full8']['clip_macro']:.3f}")
    print(f"\n{'arm':48}{'F1':>7}{'AUC':>7}   per-genre F1 / AUC / share assigned")
    for a in arms:
        v = out["arms"][a]
        print(f"{a[:47]:48}{v['mean']:7.3f}{v['macro_auc']:7.3f}   "
              f"{[round(x, 2) for x in v['per_genre_f1']]} {[round(x, 2) for x in v['per_genre_auc']]} "
              f"{[round(x, 2) for x in assign[a]['share_of_films']]}")
    print(f"{'truth':48}{'':14}   share {[round(x, 2) for x in truth['share_of_films']]}, "
          f"genres per film {truth['genres_per_film']:.2f}")
    print("\nvs all-positive (macro, film bootstrap):")
    for c in b["comparisons"]:
        print(f"  {c['a'][:45]:45} {c['diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] "
              f"p={c['p_two_sided']:.3f} won {c['repeat_win_rate']:.1f}  clip {c['clip_level_diff']:+.3f}")
    print("\nemotion pipeline minus arm, per genre: F1 diff (p) | AUC diff (p); macro AUC")
    for a in per_genre_cmp:
        cells = "  ".join(f"{g[:6]} {per_genre_cmp[a][g]['diff']:+.2f}({per_genre_cmp[a][g]['p_two_sided']:.2f})"
                          f"|{auc_cmp[a][g]['diff']:+.2f}({auc_cmp[a][g]['p_two_sided']:.2f})" for g in genres)
        m = macro_auc_cmp[a]
        print(f"  {a[:30]:30} {cells}   macro AUC {m['diff']:+.3f} p={m['p_two_sided']:.3f}")
    print(f"\n[{time.time() - t0:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
