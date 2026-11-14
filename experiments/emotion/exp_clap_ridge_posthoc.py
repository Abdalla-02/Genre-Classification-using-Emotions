"""Post-hoc test: is CLAP with ridge regression better than the VGGish pipeline?

``exp_emotion_pipeline_grid.py`` found that the emotion pipeline on CLAP with ridge
regression is ahead of the thesis pipeline (VGGish, random forest) under all five master
seeds, 0.502 against 0.469 on average, but it never compared the two directly. The author
asked for that test on 4 Nov. It is post hoc: the pairing was picked because it came out
first among 18 cells, so an unadjusted p-value flatters it. The Bonferroni value (p times
the 17 other cells it could have been compared for) is written beside it as the bound that
allows for the selection.

The two arms run with the code of the grid on the 10x5 folds of each master seed, and must
reproduce its seed means. The master seeds change the folds and the models but not the 40
films, so five seeds are not five independent samples and agreeing seeds are no test.

  * per seed: the paired film bootstrap (film level; clip level beside it) and the
    Nadeau-Bengio test on the clip-level folds, as in every other comparison;
  * pooled: the same film bootstrap over the 50 repeats of all five seeds together. The
    bootstrap draws do not depend on the master seed, so this tests the five-seed mean.

Run:  python experiments/emotion/exp_clap_ridge_posthoc.py [--jobs -1]
Writes results/clap_ridge_posthoc.json.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.features import (  # noqa: E402
    add_derived_features, assemble_from_cache, genre_matrix, load_set1,
)
from src.utils import set_seed  # noqa: E402
from exp_cv_corrected import FilmBootstrap, PRED_OOF  # noqa: E402
import exp_film_level  # noqa: E402
from exp_film_level import Films, block  # noqa: E402
from exp_emotion_pipeline_grid import EXTRA_SEEDS, cell, fit_fold_grid  # noqa: E402

OUT = config.RESULTS_DIR / "clap_ridge_posthoc.json"
GRID = config.RESULTS_DIR / "emotion_pipeline_grid.json"
CLAP_RIDGE = cell("CLAP", "ridge")
N_OTHER_CELLS = 17


def pooled(fl, Y, films, runs_by_seed, n_boot) -> dict:
    """Paired film bootstrap of the difference over the repeats of all seeds together."""
    bf = FilmBootstrap(fl.Y, np.arange(len(fl.ids)), n_boot)
    bc = FilmBootstrap(Y, films, n_boot)

    def draws(arm):
        F = np.concatenate([fl.decide(r[arm]["P"]) for r in runs_by_seed])
        H = np.concatenate([r[arm]["hard"] for r in runs_by_seed])
        return bf.scores(F), bc.scores(H)

    (fa, pa), (ca, qa) = draws(CLAP_RIDGE)
    (fb, pb), (cb, qb) = draws(PRED_OOF)

    def test(d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        return {"diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "p_two_sided": p,
                "p_bonferroni_17": float(min(1.0, N_OTHER_CELLS * p))}

    return {"film": {"clap_ridge": pa, "vggish_forest": pb} | test(fa - fb),
            "clip": {"clap_ridge": qa, "vggish_forest": qb} | test(ca - cb)}


def main() -> None:
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
    films = dfk["soundtrack"].to_numpy().astype(str)
    E = dfk[config.EMOTIONS].to_numpy(float)
    genres = list(config.GENRE_SUBSET)
    arms = {
        PRED_OOF: ("emo:random forest",
                   assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR), E),
        CLAP_RIDGE: ("emo:ridge",
                     assemble_from_cache(dfk, "set1", cache_dir=config.CLAP_EMBEDDINGS_DIR), E),
    }
    exp_film_level.fit_fold = fit_fold_grid
    grid = json.load(open(GRID, encoding="utf-8"))["stability"]["per_arm"]

    per_seed, runs_by_seed, fl = {}, [], None
    for i, s in enumerate((config.SEED,) + tuple(EXTRA_SEEDS)):
        b, runs, fl = block(Y, films, genres, arms, [(CLAP_RIDGE, PRED_OOF)], args, seed=s)
        for a in arms:
            got = float(np.mean(b["per_repeat_film"][a]))
            assert abs(got - grid[a]["seed_means"][i]) < 1e-9, (s, a, got)
        c = b["comparisons"][0]
        per_seed[str(s)] = {"clap_ridge": float(np.mean(b["per_repeat_film"][CLAP_RIDGE])),
                            "vggish_forest": float(np.mean(b["per_repeat_film"][PRED_OOF]))} | {
            k: c[k] for k in ("diff", "ci_lo", "ci_hi", "p_two_sided", "repeat_win_rate",
                              "clip_level_diff", "clip_level_p_two_sided", "clip_level_nb_p",
                              "clip_level_nb_p_limit")}
        runs_by_seed.append({a: {"P": runs[a]["P"], "hard": runs[a]["hard"]} for a in arms})
        print(f"seed {s}: {per_seed[str(s)]['clap_ridge']:.3f} vs "
              f"{per_seed[str(s)]['vggish_forest']:.3f}  diff {c['diff']:+.3f} "
              f"p={c['p_two_sided']:.3f} won {c['repeat_win_rate']:.1f}  clip "
              f"{c['clip_level_diff']:+.3f} p={c['clip_level_p_two_sided']:.3f} "
              f"NB {c['clip_level_nb_p']:.3f}  [{time.time() - t0:.0f}s]", flush=True)

    pool = pooled(fl, Y, films, runs_by_seed, args.boot)
    print("pooled:", json.dumps(pool, indent=1), flush=True)
    out = {"design": {"question": "CLAP with ridge regression against VGGish with the random "
                                  "forest, both emotion pipelines, five genres, 40 films",
                      "status": "post hoc: the pairing was picked after the results, as the "
                                "first of 18 cells",
                      "bonferroni": f"p times the {N_OTHER_CELLS} other cells",
                      "pooled": "film bootstrap over the 50 repeats of the five master seeds; "
                                "the seeds share the same 40 films"},
           "per_seed": per_seed, "pooled": pool}
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"[{time.time() - t0:.0f}s] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
