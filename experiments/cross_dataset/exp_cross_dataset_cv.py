"""Cross-dataset validation in every direction (supervisor: "make cross validations
through the datasets to be sure the results are accurate").

Three designs, same six-genre shared label space, same three arms in each:

  1. Eerola -> Blockbuster   train on all Eerola, test on all 110 Blockbuster films
                             (the zero-shot result of exp_zero_shot.py, repeated here so
                             the three designs sit in one table)
  2. Blockbuster -> Eerola   the reverse. The emotion regressor needs ratings, which
                             only Eerola has, so it is fitted on the Eerola TRAINING fold
                             of a GroupKFold and the genre classifier on Blockbuster; the
                             held-out Eerola fold is scored. No test clip touches either
                             fit.
  3. Pooled GroupKFold       both corpora in one table (Eerola clips + Blockbuster films
                             as rows, film as the group), 5x5 repeated. Tests whether the
                             two corpora can be LEARNED FROM jointly, which neither
                             transfer direction asks.

Arms: VGGish-128 direct, PCA-8 control, VGGish -> emotion (per cue) -> genre. Every
transform is fitted inside the training portion. Macro-F1 throughout; uncertainty from a
paired bootstrap over test films (designs 1-2) or repeated folds (design 3).

If the emotion bottleneck's advantage is real it should show in all three; if it appears
only in the direction reported so far, that is what this script is for finding out.

Run:  python experiments/cross_dataset/exp_cross_dataset_cv.py [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation import macro_prf  # noqa: E402
from src.evaluation.repeated import (  # noqa: E402
    RepeatedGroupKFold, corrected_paired_t, diff_ci, repeat_ci, win_rate,
)
from src.features import (  # noqa: E402
    assemble_from_cache, build_emotion_features, load_set1, load_set1_shared,
    shared_genre_matrix,
)
from src.features.blockbuster import aggregate_cues, load_blockbuster_cues  # noqa: E402
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

RESULTS = config.RESULTS_DIR / "cross_dataset_cv.json"
DIRECT, PCA8, EMO = "VGGish-128 direct", "PCA-8(VGGish) [control]", "VGGish -> emotion (per cue) -> genre"


# --------------------------------------------------------------------------- #
def fit_reg(X, E):
    return RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1).fit(X, E)


def logreg_fit_predict(Xtr, Ytr, gtr, Xte):
    C = select_logreg_C(Xtr, Ytr, gtr)
    return np.asarray(build_classifier("logreg", C=C).fit(Xtr, Ytr).predict(Xte))


def bootstrap(Yte, pa, pb, n=2000):
    rng = np.random.default_rng(config.SEED)
    d = []
    for _ in range(n):
        s = rng.integers(0, len(Yte), len(Yte))
        if Yte[s].sum(0).min() == 0:
            continue
        d.append(macro_prf(Yte[s], pa[s])["macro_f1"] - macro_prf(Yte[s], pb[s])["macro_f1"])
    d = np.array(d)
    return {"diff": float(d.mean()), "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def three_arms(Xtr, Ytr, gtr, Xte, Ftr, Fte):
    """Fit the three arms on (Xtr, Ftr) and predict Xte / Fte. Scaler+PCA fit on train."""
    sc = StandardScaler().fit(Xtr)
    pca = PCA(8, random_state=config.SEED).fit(sc.transform(Xtr))
    return {
        DIRECT: logreg_fit_predict(sc.transform(Xtr), Ytr, gtr, sc.transform(Xte)),
        PCA8: logreg_fit_predict(pca.transform(sc.transform(Xtr)), Ytr, gtr,
                                 pca.transform(sc.transform(Xte))),
        EMO: logreg_fit_predict(Ftr, Ytr, gtr, Fte),
    }


def print_block(title, res, Yte):
    print(f"\n{title}")
    print(f"  {'arm':44}{'prec':>7}{'recall':>8}{'F1':>7}")
    print("  " + "-" * 66)
    rows = {}
    for name, pred in res.items():
        m = macro_prf(Yte, pred)
        rows[name] = m
        print(f"  {name:44}{m['macro_precision']:>7.3f}{m['macro_recall']:>8.3f}{m['macro_f1']:>7.3f}")
    b1 = bootstrap(Yte, res[EMO], res[DIRECT]); b2 = bootstrap(Yte, res[EMO], res[PCA8])
    print(f"  emotion vs direct : {b1['diff']:+.3f} [{b1['ci_lo']:+.3f}, {b1['ci_hi']:+.3f}] p={b1['p']:.3f}")
    print(f"  emotion vs PCA-8  : {b2['diff']:+.3f} [{b2['ci_lo']:+.3f}, {b2['ci_hi']:+.3f}] p={b2['p']:.3f}")
    return {"arms": rows, "emotion_vs_direct": b1, "emotion_vs_pca8": b2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    # A reduced debug run writes to a scratch .partial.json (git-ignored) so it can
    # never replace the results file the documents quote.
    full = args.repeats >= 5
    out_path = RESULTS if full else RESULTS.with_suffix(".partial.json")
    set_seed()
    out = {}

    # ---- data ----------------------------------------------------------------
    dfe = load_set1_shared()
    Ye = shared_genre_matrix(dfe)
    Xe = assemble_from_cache(dfe, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Ee = dfe[config.EMOTIONS].to_numpy(float)
    ge = dfe["soundtrack"].to_numpy()
    # the regressor is trained on all 360 rated clips, not only the 319 in the shared space
    df_all = load_set1(clean=False)
    Xe_all = assemble_from_cache(df_all, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Ee_all = df_all[config.EMOTIONS].to_numpy(float)
    ge_all = df_all["soundtrack"].to_numpy()

    cue = load_blockbuster_cues()
    Yb, nb = cue["Y"], len(cue["films"])
    Xb_cue, cf = cue["X_vggish"], cue["cue_film_vggish"]
    Xb = aggregate_cues(Xb_cue, cf, nb)
    gb = np.array([f"BB::{f}" for f in cue["films"]])

    def cue_emotion_features(reg):
        return build_emotion_features(aggregate_cues(reg.predict(Xb_cue), cf, nb))

    def oof_eerola_features(X, E, g):
        F = np.zeros_like(E)
        for tr, te in GroupKFold(5).split(X, E, g):
            F[te] = fit_reg(X[tr], E[tr]).predict(X[te])
        return build_emotion_features(F)

    print("=" * 78)
    print("CROSS-DATASET VALIDATION -- shared 6-genre space, three designs, three arms")
    print("=" * 78)

    # ---- 1. Eerola -> Blockbuster --------------------------------------------
    reg_full = fit_reg(Xe_all, Ee_all)
    Fe_oof = oof_eerola_features(Xe, Ee, ge)
    res1 = three_arms(Xe, Ye, ge, Xb, Fe_oof, cue_emotion_features(reg_full))
    out["eerola_to_blockbuster"] = print_block(
        "1. EEROLA -> BLOCKBUSTER  (train 319 clips, test 110 films, zero-shot)", res1, Yb)

    # ---- 2. Blockbuster -> Eerola --------------------------------------------
    # 5-fold over Eerola films: regressor on the Eerola train fold, genre classifier on
    # ALL of Blockbuster, score the Eerola test fold. Pool the out-of-fold predictions.
    pred2 = {k: np.zeros_like(Ye) for k in (DIRECT, PCA8, EMO)}
    for tr, te in GroupKFold(5).split(Xe, Ye, ge):
        # regressor may use every rated clip whose FILM is in the training fold
        train_films = set(ge[tr])
        m = np.isin(ge_all, list(train_films))
        reg = fit_reg(Xe_all[m], Ee_all[m])
        Fb = cue_emotion_features(reg)
        Fe_te = build_emotion_features(reg.predict(Xe[te]))
        r = three_arms(Xb, Yb, gb, Xe[te], Fb, Fe_te)
        for k in pred2:
            pred2[k][te] = r[k]
    out["blockbuster_to_eerola"] = print_block(
        "2. BLOCKBUSTER -> EEROLA  (train 110 films, test Eerola clips out-of-fold)", pred2, Ye)

    # ---- 3. pooled repeated GroupKFold --------------------------------------
    print(f"\n3. POOLED  (both corpora, {args.repeats}x5 repeated GroupKFold by film)")
    Xp = np.vstack([Xe, Xb]); Yp = np.vstack([Ye, Yb])
    gp = np.concatenate([ge, gb])
    is_e = np.r_[np.ones(len(Xe), bool), np.zeros(nb, bool)]
    folds = list(RepeatedGroupKFold(5, args.repeats).split(np.zeros(len(Xp)), None, gp))
    scores = {k: np.zeros(len(folds)) for k in (DIRECT, PCA8, EMO)}
    # per-corpus scores are NaN on folds where a corpus contributes too few rows (or a genre
    # has no positive in that corpus's share of the fold) and are averaged with nanmean
    scores_by_corpus = {c: {k: np.full(len(folds), np.nan) for k in scores}
                        for c in ("eerola", "blockbuster")}
    for i, (tr, te) in enumerate(folds):
        e_tr = tr[is_e[tr]]                      # Eerola rows in the training fold
        m = np.isin(ge_all, list(set(gp[e_tr])))
        reg = fit_reg(Xe_all[m], Ee_all[m])
        # emotion features: Eerola rows predicted by the fold regressor, Blockbuster rows per cue
        F_e = build_emotion_features(reg.predict(Xe))
        F_b = cue_emotion_features(reg)
        Fp = np.vstack([F_e, F_b])
        r = three_arms(Xp[tr], Yp[tr], gp[tr], Xp[te], Fp[tr], Fp[te])
        for k, pred in r.items():
            scores[k][i] = f1_score(Yp[te], pred, average="macro", zero_division=0)
            for c, mask in (("eerola", is_e[te]), ("blockbuster", ~is_e[te])):
                if mask.sum() >= 10:
                    present = Yp[te][mask].sum(0) > 0      # score only genres present
                    scores_by_corpus[c][k][i] = f1_score(
                        Yp[te][mask][:, present], pred[mask][:, present],
                        average="macro", zero_division=0)
        print(f"  fold {i + 1}/{len(folds)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    n_test = float(np.mean([len(te) for _, te in folds])); n_train = len(Xp) - n_test
    print(f"  {'arm':44}{'all rows':>10}{'Eerola part':>13}{'Blockb. part':>14}")
    print("  " + "-" * 81)
    rows3 = {}
    for k in scores:
        ci = repeat_ci(scores[k], 5)
        ce = np.nanmean(scores_by_corpus["eerola"][k]); cb = np.nanmean(scores_by_corpus["blockbuster"][k])
        rows3[k] = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi,
                    "eerola_part": float(ce), "blockbuster_part": float(cb)}
        print(f"  {k:44}{ci.mean:>10.3f}{ce:>13.3f}{cb:>14.3f}")
    cmp = {}
    for a, b in ((EMO, DIRECT), (EMO, PCA8)):
        d = diff_ci(scores[a], scores[b], 5); _, p = corrected_paired_t(scores[a], scores[b], n_train, n_test)
        cmp[f"{a} vs {b}"] = {"diff": d.mean, "p_corrected": p, "win_rate": win_rate(scores[a], scores[b])}
        print(f"  {a} vs {b.split(' ')[0]}: {d.mean:+.3f}  p={p:.3f}  win={win_rate(scores[a], scores[b]):.0%}")
    out["pooled"] = {"arms": rows3, "comparisons": cmp,
                     "per_fold": {k: v.tolist() for k, v in scores.items()}}

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}" + ("" if full else "  (reduced run: scratch file)"))


if __name__ == "__main__":
    main()
