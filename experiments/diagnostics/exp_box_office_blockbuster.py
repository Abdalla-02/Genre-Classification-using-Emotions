"""Box office vs. the soundtrack, on the 110 Blockbuster films (supervisor, 4th meeting).

"Box-office gross correlated with the Blockbuster dataset, and checked in all values of
the different metrics." The Eerola version (``exp_box_office.py``) has 37 films spread
over four decades. Blockbuster is the better test bed for this question: 110 studio films
released 2014-2019, a worldwide gross from one source (Box Office Mojo) for every one of
them, and every cue of every film. It still has no audio and no
emotion ratings, so every emotion below is PREDICTED, per cue, by the Eerola-trained
VGGish -> emotion regressor, then pooled per film (the representation of log section 17).

Four questions, each answered under three correlation measures -- Pearson on log10 gross,
Spearman, Kendall -- with a permutation p-value and a Benjamini-Hochberg correction
across each family of tests:

  A. Is a higher-grossing film's genre EASIER TO PREDICT from its music? Per-film
     classification quality under every per-film metric (F1, Jaccard, Hamming accuracy,
     exact match, label-ranking average precision, and the probability margin between
     the film's true and false genres), for every route: emotion, direct VGGish, the
     PCA-8 control and the full MIR set, in-domain (10x5 repeated KFold over films) and
     zero-shot (trained on Eerola only, strict regime, as in exp_zero_shot.py).
  B. Do the soundtrack's EMOTIONS track gross? The eight predicted emotions and the three
     combinations, the film mean over cues and the spread between cues, and the number
     of cues.
  C. Does GENRE track gross? Films carrying each genre against those that do not.
  D. Can gross be PREDICTED from the soundtrack at all? Cross-validated ridge regression
     of log gross on each representation, against a permutation null. This is the
     multivariate version of B and is not fooled by testing many single features.
  E. All 140 hand-crafted MIR descriptors, one by one, with the correction applied: the
     literal reading of "all values".

Controls: release year (partial Spearman) and, where Wikidata gives one, the production
budget, because gross largely follows budget and budget follows genre.

This is EXPLORATORY. A correlation here says nothing about cause: a large budget buys
both a bigger orchestra and a bigger marketing campaign.

Run:  python experiments/diagnostics/exp_box_office_blockbuster.py
      (needs data/processed/Blockbuster/box_office.csv from
       experiments/features/fetch_box_office_blockbuster.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score, jaccard_score, label_ranking_average_precision_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation import macro_prf  # noqa: E402
from src.features import (  # noqa: E402
    assemble_from_cache,
    build_emotion_features,
    load_set1,
    load_set1_shared,
    shared_genre_matrix,
)
from src.features.blockbuster import (  # noqa: E402
    aggregate_cues,
    load_blockbuster,
    load_blockbuster_cues,
    mir_feature_names,
)
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

RESULTS = config.RESULTS_DIR / "box_office_blockbuster.json"
BOX = config.DATA_ROOT / "processed" / "Blockbuster" / "box_office.csv"
PER_FILM = config.DATA_ROOT / "processed" / "Blockbuster" / "box_office_per_film.csv"
ZERO_SHOT = config.RESULTS_DIR / "zero_shot.json"
N_REPEATS, N_SPLITS = 10, 5
EMO11 = config.EMOTIONS + ["valence_x_energy", "negative_mean", "positive_mean"]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def perm_p(x, y, stat, n_perm, rng):
    obs = stat(x, y)
    null = np.array([stat(x, rng.permutation(y)) for _ in range(n_perm)])
    return float(obs), float((np.abs(null) >= abs(obs) - 1e-12).mean())


def spearman(x, y):
    return stats.spearmanr(x, y).correlation


def pearson(x, y):
    return stats.pearsonr(x, y)[0]


def kendall(x, y):
    return stats.kendalltau(x, y).correlation


def partial_spearman(x, y, z):
    """Spearman of x and y after removing the rank-linear effect of each column of z."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([np.ones(len(x))] + [stats.rankdata(c) for c in np.atleast_2d(z)])
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]   # noqa: E731
    return float(stats.pearsonr(res(rx), res(ry))[0])


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def correlate_family(target, cols: dict, controls, n_perm, rng):
    """Every column against log gross, three measures, BH within the family (Spearman)."""
    rows = {}
    for name, v in cols.items():
        v = np.asarray(v, float)
        if np.ptp(v) == 0:          # a constant column has no defined correlation
            continue
        rs, ps = perm_p(target, v, spearman, n_perm, rng)
        rp, pp = perm_p(target, v, pearson, n_perm, rng)
        rk = kendall(target, v)
        rows[name] = {"spearman": rs, "p_spearman": ps, "pearson_log": rp, "p_pearson": pp,
                      "kendall": float(rk),
                      **{f"spearman_partial_{k}": partial_spearman(target, v, z)
                         for k, z in controls.items()}}
    adj = bh([r["p_spearman"] for r in rows.values()])
    for (name, r), a in zip(rows.items(), adj):
        r["p_spearman_bh"] = float(a)
    return rows


# --------------------------------------------------------------------------- #
# Per-film classification quality
# --------------------------------------------------------------------------- #
def per_film_quality(Y, P, pred):
    """Several per-film metrics from one film's true vector, probabilities and decision."""
    out = {}
    out["f1"] = np.array([f1_score(y, p, zero_division=0) for y, p in zip(Y, pred)])
    out["jaccard"] = np.array([jaccard_score(y, p, zero_division=0) for y, p in zip(Y, pred)])
    out["hamming_accuracy"] = (Y == pred).mean(1)
    out["exact_match"] = (Y == pred).all(1).astype(float)
    out["ranking_ap"] = np.array([label_ranking_average_precision_score(y[None], p[None])
                                  for y, p in zip(Y, P)])
    out["prob_margin"] = np.array([P[i][Y[i] == 1].mean() - P[i][Y[i] == 0].mean()
                                   if 0 < Y[i].sum() < Y.shape[1] else np.nan
                                   for i in range(len(Y))])
    return out


def proba(clf, Z):
    return np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(Z))
                            for p in clf.predict_proba(Z)])


def in_domain_oof(F_by_route, Y):
    """10x5 repeated KFold over films (the folds of exp_blockbuster_deep); per route,
    out-of-fold probabilities and decisions for every film, per repeat."""
    out = {r: {"P": np.zeros((N_REPEATS,) + Y.shape),
               "pred": np.zeros((N_REPEATS,) + Y.shape, dtype=int)} for r in F_by_route}
    for rep in range(N_REPEATS):
        kf = KFold(N_SPLITS, shuffle=True, random_state=config.SEED + rep)
        for tr, te in kf.split(np.arange(len(Y))):
            for r, (kind, X) in F_by_route.items():
                if kind == "pca8":
                    sc = StandardScaler().fit(X[tr])
                    pca = PCA(n_components=8, random_state=config.SEED).fit(sc.transform(X[tr]))
                    Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
                else:
                    Ztr, Zte = X[tr], X[te]
                C = select_logreg_C(Ztr, Y[tr], np.arange(len(tr)))
                clf = build_classifier("logreg", C=C).fit(Ztr, Y[tr])
                out[r]["P"][rep][te] = proba(clf, Zte)
                out[r]["pred"][rep][te] = np.asarray(clf.predict(Zte))
    return out


def zero_shot_predictions(Yb, Xb_raw, cue):
    """Train on Eerola only (strict, source scaler), predict every Blockbuster film.

    Mirrors exp_zero_shot.py, including its one difference from the in-domain features:
    there the emotion regressor is fitted on the 319 clips of the shared genre space, not
    on all 360 rated clips. Its macro-F1 is checked against zero_shot.json below.
    """
    df = load_set1_shared()
    Ye = shared_genre_matrix(df)
    Xe = assemble_from_cache(df, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Ee = df[config.EMOTIONS].to_numpy(float)
    g = df["soundtrack"].to_numpy()
    oof = np.zeros_like(Ee)
    for tr, te in GroupKFold(n_splits=5).split(Xe, Ee, g):
        reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
        oof[te] = reg.fit(Xe[tr], Ee[tr]).predict(Xe[te])
    reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED,
                                n_jobs=-1).fit(Xe, Ee)
    Fb_cue = build_emotion_features(aggregate_cues(
        reg.predict(cue["X_vggish"]), cue["cue_film_vggish"], len(Yb)))
    sc = StandardScaler().fit(Xe)
    Xe_s, Xb_s = sc.transform(Xe), sc.transform(Xb_raw)
    pca = PCA(n_components=8, random_state=config.SEED).fit(Xe_s)
    routes = {"emotion route": (build_emotion_features(oof), Fb_cue),
              "direct audio (VGGish)": (Xe_s, Xb_s),
              "PCA-8 control": (pca.transform(Xe_s), pca.transform(Xb_s))}
    out = {}
    for r, (A, B) in routes.items():
        C = select_logreg_C(A, Ye, g)
        clf = build_classifier("logreg", C=C).fit(A, Ye)
        out[r] = {"P": proba(clf, B), "pred": np.asarray(clf.predict(B))}
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=5000)
    args = ap.parse_args()
    full = args.perm >= 5000
    out_path = RESULTS if full else RESULTS.with_suffix(".partial.json")
    set_seed()
    rng = np.random.default_rng(config.SEED)

    film = load_blockbuster()
    cue = load_blockbuster_cues()
    Y, films, genres = film["Y"], film["films"], film["genres"]
    n = len(films)

    box = pd.read_csv(BOX).set_index("slug").reindex(films)
    have = box["gross"].notna().to_numpy()
    print("=" * 78)
    print("BOX OFFICE vs THE SOUNDTRACK -- Blockbuster, 110 films  (exploratory)")
    print("=" * 78)
    print(f"films with a gross: {have.sum()} / {n}  "
          f"({(box['scope'] == 'worldwide').sum()} worldwide from Box Office Mojo, "
          f"{(box['source'] == 'wikidata').sum()} from Wikidata); "
          f"with a budget: {box['budget_usd'].notna().sum()}; "
          f"years {int(box['year'].min())}-{int(box['year'].max())}")

    # ---- per-film representations ------------------------------------------ #
    df_e = load_set1(clean=False)
    Xe = assemble_from_cache(df_e, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED,
                                n_jobs=-1).fit(Xe, df_e[config.EMOTIONS].to_numpy(float))
    cf = cue["cue_film_vggish"]
    Ecue = build_emotion_features(reg.predict(cue["X_vggish"]))       # (cues, 11)
    F_cue = aggregate_cues(Ecue, cf, n)                                # film mean
    spread = np.array([Ecue[cf == i].std(0) for i in range(n)])        # between-cue SD
    n_cues = np.bincount(cf, minlength=n)
    Xv = film["X_vggish"]
    Xmir = aggregate_cues(cue["X_mir"], cue["cue_film_mir"], n)

    # ---- A. per-film classification quality -------------------------------- #
    print("\nin-domain out-of-fold predictions (10x5 repeated KFold over films) ...",
          flush=True)
    routes = {"emotion route": ("plain", F_cue), "direct audio (VGGish)": ("plain", Xv),
              "PCA-8 control": ("pca8", Xv), "MIR-140": ("plain", Xmir)}
    oof = in_domain_oof(routes, Y)
    print("zero-shot predictions (trained on Eerola only) ...", flush=True)
    zs = zero_shot_predictions(Y, Xv, cue)
    zref = json.loads(ZERO_SHOT.read_text(encoding="utf-8"))["zero_shot"]["strict (source scaler)"]
    ref_names = {"emotion route": "VGGish -> predicted emotion(11), per-cue",
                 "direct audio (VGGish)": "VGGish-128 direct",
                 "PCA-8 control": "PCA-8(VGGish) [control]"}
    zcheck = {r: {"here": macro_prf(Y, zs[r]["pred"])["macro_f1"],
                  "zero_shot.json": zref[ref_names[r]]["macro_f1"]} for r in zs}
    for v in zcheck.values():
        v["match"] = bool(abs(v["here"] - v["zero_shot.json"]) < 1e-9)
    print("  zero-shot reproduction:",
          {r: f"{v['here']:.3f} ({'ok' if v['match'] else 'MISMATCH'})" for r, v in zcheck.items()})

    quality = {}
    for r in routes:
        per_rep = [per_film_quality(Y, oof[r]["P"][k], oof[r]["pred"][k])
                   for k in range(N_REPEATS)]
        for m in per_rep[0]:
            quality[f"in-domain | {r} | {m}"] = np.nanmean([q[m] for q in per_rep], axis=0)
    for r in zs:
        for m, v in per_film_quality(Y, zs[r]["P"], zs[r]["pred"]).items():
            quality[f"zero-shot | {r} | {m}"] = v

    logg = np.log10(box["gross"].to_numpy(float))
    year = box["year"].to_numpy(float)
    budget = box["budget_usd"].to_numpy(float)
    m = have
    controls = {"year": year[m]}
    has_budget = m & np.isfinite(budget)
    out = {"n_films": int(n), "n_with_gross": int(m.sum()),
           "n_with_budget": int(has_budget.sum()),
           "years": [int(np.nanmin(year)), int(np.nanmax(year))],
           "gross_usd": {"median": float(np.nanmedian(box["gross"])),
                         "min": float(np.nanmin(box["gross"])),
                         "max": float(np.nanmax(box["gross"]))},
           "zero_shot_reproduction": zcheck}

    print("\nA. Is a higher-grossing film's genre easier to predict from its music?")
    famA = correlate_family(logg[m], {k: v[m] for k, v in quality.items()}, controls,
                            args.perm, rng)
    out["A_quality_vs_gross"] = famA
    print(f"   {'route | metric':52}{'rho':>7}{'p':>7}{'p_BH':>7}{'r(log)':>8}{'tau':>7}")
    for k, r in famA.items():
        s = " *" if r["p_spearman_bh"] < 0.05 else ""
        print(f"   {k:52}{r['spearman']:>+7.3f}{r['p_spearman']:>7.3f}"
              f"{r['p_spearman_bh']:>7.3f}{r['pearson_log']:>+8.3f}{r['kendall']:>+7.3f}{s}")

    # ---- B. emotions -------------------------------------------------------- #
    colsB = {**{f"mean {e}": F_cue[:, j] for j, e in enumerate(EMO11)},
             **{f"spread {e}": spread[:, j] for j, e in enumerate(config.EMOTIONS)},
             "number of cues": n_cues}
    famB = correlate_family(logg[m], {k: np.asarray(v)[m] for k, v in colsB.items()},
                            controls, args.perm, rng)
    # budget control on the films that have one
    for k, v in colsB.items():
        if k not in famB:
            continue
        vv = np.asarray(v, float)
        famB[k]["spearman_partial_budget"] = (
            partial_spearman(logg[has_budget], vv[has_budget], np.log10(budget[has_budget]))
            if has_budget.sum() >= 30 else None)
    out["B_emotion_vs_gross"] = famB
    print("\nB. Do the soundtrack's (predicted) emotions track gross?")
    print(f"   {'quantity':26}{'rho':>7}{'p':>7}{'p_BH':>7}{'r(log)':>8}{'tau':>7}"
          f"{'|year':>7}{'|budget':>8}")
    for k, r in famB.items():
        pb = r["spearman_partial_budget"]
        s = " *" if r["p_spearman_bh"] < 0.05 else ""
        print(f"   {k:26}{r['spearman']:>+7.3f}{r['p_spearman']:>7.3f}{r['p_spearman_bh']:>7.3f}"
              f"{r['pearson_log']:>+8.3f}{r['kendall']:>+7.3f}"
              f"{r['spearman_partial_year']:>+7.3f}"
              f"{(f'{pb:+.3f}' if pb is not None else '--'):>8}{s}")

    # ---- C. genre ----------------------------------------------------------- #
    print("\nC. Gross by genre (films with the genre vs without; Mann-Whitney)")
    famC = {}
    for j, g in enumerate(genres):
        a = box["gross"].to_numpy(float)[m & (Y[:, j] == 1)]
        b = box["gross"].to_numpy(float)[m & (Y[:, j] == 0)]
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        famC[g] = {"n_with": int(len(a)), "median_with": float(np.median(a)),
                   "median_without": float(np.median(b)), "p_mw": float(p)}
    for (g, r), a in zip(famC.items(), bh([r["p_mw"] for r in famC.values()])):
        r["p_mw_bh"] = float(a)
        print(f"   {g:10}{r['n_with']:>5}{r['median_with'] / 1e6:>10.0f}M"
              f"{r['median_without'] / 1e6:>10.0f}M  p={r['p_mw']:.3f}  p_BH={a:.3f}"
              f"{' *' if a < 0.05 else ''}")
    out["C_genre_vs_gross"] = famC

    # ---- D. can gross be predicted from the soundtrack at all? -------------- #
    print("\nD. Cross-validated prediction of log gross (ridge, 10x5 KFold; R^2 out of fold)")
    targets = logg[m]
    reps = {"emotion (11)": F_cue[m], "VGGish-128": Xv[m], "MIR-140": Xmir[m],
            "genre labels (6)": Y[m].astype(float),
            "emotion (11) + genre (6)": np.column_stack([F_cue[m], Y[m]])}

    def cv_r2(X, yv, seed_shift=0):
        pred = np.zeros((N_REPEATS, len(yv)))
        for rep in range(N_REPEATS):
            kf = KFold(N_SPLITS, shuffle=True, random_state=config.SEED + rep + seed_shift)
            for tr, te in kf.split(X):
                mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
                pred[rep][te] = mdl.fit(X[tr], yv[tr]).predict(X[te])
        ss = ((yv - yv.mean()) ** 2).sum()
        return float(np.mean([1 - ((yv - p) ** 2).sum() / ss for p in pred]))

    if has_budget.sum() >= 30:
        # gross largely follows budget: how much does the budget alone predict, and how
        # much do the emotions predict on exactly the same films?
        hb = has_budget[m]
        lb = np.log10(budget[m][hb])[:, None]
        reps[f"log budget ({hb.sum()} films)"] = (lb, hb)
        reps[f"emotion (11), same {hb.sum()} films"] = (F_cue[m][hb], hb)
        reps[f"log budget + emotion (11), same {hb.sum()} films"] =             (np.column_stack([lb, F_cue[m][hb]]), hb)
    famD = {}
    for k, X in reps.items():
        X, sub = X if isinstance(X, tuple) else (X, np.ones(len(targets), bool))
        yv = targets[sub]
        r2 = cv_r2(X, yv)
        null = np.array([cv_r2(X, rng.permutation(yv)) for _ in range(100)])
        famD[k] = {"cv_r2": r2, "null_r2_95th": float(np.percentile(null, 95)),
                   "p_perm": float((null >= r2).mean())}
        print(f"   {k:28} R^2 = {r2:+.3f}   (permutation null 95th pct {famD[k]['null_r2_95th']:+.3f},"
              f" p = {famD[k]['p_perm']:.2f})")
    out["D_predict_gross"] = famD

    # ---- E. every MIR descriptor --------------------------------------------- #
    names = mir_feature_names()
    famE = correlate_family(logg[m], {nm: Xmir[m][:, j] for j, nm in enumerate(names)},
                            controls, 1000, rng)
    sig = {k: v for k, v in famE.items() if v["p_spearman_bh"] < 0.05}
    n_const = len(names) - len(famE)
    top = sorted(famE.items(), key=lambda kv: (kv[1]["p_spearman"], -abs(kv[1]["spearman"])))[:10]
    print(f"\nE. All {len(names)} MIR descriptors vs log gross ({n_const} constant, skipped): "
          f"{sum(v['p_spearman'] < 0.05 for v in famE.values())} with p < 0.05 uncorrected "
          f"(chance: about {0.05 * len(names):.0f}), {len(sig)} after Benjamini-Hochberg")
    for k, v in top:
        print(f"   {k:30}{v['spearman']:>+7.3f}  p={v['p_spearman']:.3f}  "
              f"p_BH={v['p_spearman_bh']:.3f}")
    out["E_mir_all"] = {"n_features": len(names), "n_constant_skipped": n_const,
                        "n_p_below_05_uncorrected": int(sum(v["p_spearman"] < 0.05
                                                           for v in famE.values())),
                        "n_significant_bh": len(sig),
                        "top10": {k: v for k, v in top}}

    # ---- confounds ------------------------------------------------------------- #
    rho_y, p_y = perm_p(logg[m], year[m], spearman, args.perm, rng)
    conf = {"year": {"rho": rho_y, "p_perm": p_y}}
    if has_budget.sum() >= 30:
        rho_b, p_b = perm_p(logg[has_budget], np.log10(budget[has_budget]), spearman,
                            args.perm, rng)
        conf["budget"] = {"rho": rho_b, "p_perm": p_b, "n": int(has_budget.sum())}
    out["confounds"] = conf
    print(f"\nconfounds: rho(log gross, year) = {rho_y:+.3f} (p={p_y:.3f})"
          + (f"; rho(log gross, log budget) = {conf['budget']['rho']:+.3f} "
             f"(n={conf['budget']['n']})" if "budget" in conf else ""))

    per_film = pd.DataFrame({"film": films, "gross": box["gross"].to_numpy(),
                             "year": year, "budget_usd": budget, "n_cues": n_cues,
                             **{f"emo_{e}": F_cue[:, j] for j, e in enumerate(EMO11)},
                             **{f"genre_{g}": Y[:, j] for j, g in enumerate(genres)},
                             **{k.replace(" | ", "__").replace(" ", "_"): v
                                for k, v in quality.items()}})
    per_film.to_csv(PER_FILM, index=False, lineterminator="\n")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {out_path} and {PER_FILM.name}")


if __name__ == "__main__":
    main()
