"""Zero-shot transfer: train on Eerola ONLY, test on Blockbuster -- and vice versa.

Answers two supervisor requests at once:

  #4  "the classifier has to have the same genre classes in both datasets to be fair."
      Both corpora are relabelled into ONE identical 6-genre space (config.SHARED_GENRES
      = Ma et al.'s six reduced genres). Their reduction rule was recovered from their
      published master list and is the identity rule -- a film gets genre g iff g is in
      its raw IMDb list -- which is exactly the rule already used on Eerola, so no
      dataset-specific heuristic is involved. Eerola: 319 clips / 41 films;
      Blockbuster: 110 films.

  #1  "test on Blockbuster (without training on it), train on Eerola, and compare with
      the results of the paper itself."
      Part C trains every arm on Eerola alone and predicts all 110 Blockbuster films.
      No Blockbuster label is ever seen in training. Part B provides the two reference
      points needed to read that number: our own in-domain Blockbuster score under Ma's
      own protocol (leave-one-out CV), and Ma et al.'s published Table 4.

VGGish is the only representation both corpora share (Eerola: our extraction; Blockbuster:
Ma et al.'s published features), and the two spaces are numerically compatible -- both are
the standard 0-255 post-processed VGGish, per-dimension means correlate r=0.97. So a model
fitted in one can be applied in the other.

Domain shift that remains: an Eerola row is one ~10 s clip, a Blockbuster row is a whole
film's cues mean-pooled, so the target features are smoothed (per-dim SD 50.4 vs 59.9).
Part C therefore reports two feature-scaling regimes -- strict (scaler fitted on the
training corpus) and target-standardised (each corpus z-scored with its own statistics,
an unsupervised adaptation that uses no labels, so it is still zero-shot).

Metrics follow Ma et al.: precision, recall and F1 computed per genre, then macro-averaged.
Their two baselines are reproduced as well, because "random guess" means something quite
different from our usual most-frequent dummy: theirs draws each genre independently at its
base rate (macro-F1 ~ the mean base rate, .32 in their corpus), ours predicts the single
most frequent label set (.19 in theirs). Comparing our dummy against their random-guess
would be an apples-to-oranges error.

Run:  python experiments/cross_dataset/exp_zero_shot.py
      python experiments/cross_dataset/exp_zero_shot.py --quick   # skip the LOO reference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation import macro_prf, per_genre_f1  # noqa: E402
from src.evaluation.repeated import RepeatedGroupKFold  # noqa: E402
from src.features import (  # noqa: E402
    assemble_from_cache,
    build_emotion_features,
    load_set1_shared,
    shared_genre_matrix,
)
from src.features.blockbuster import (  # noqa: E402
    aggregate_cues,
    load_blockbuster,
    load_blockbuster_cues,
)
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

G = config.SHARED_GENRES
RESULTS = config.RESULTS_DIR / "zero_shot.json"

# Ma et al. (2021), Table 4 -- cue-level features, leave-one-out CV on the 110 films,
# macro-averaged precision/recall/F1 over the same six genres. Quoted for comparison.
MA2021 = {
    "VGGish, kNN Simple-MI": (0.64, 0.59, 0.61),
    "VGGish, avg-pooling NN": (0.55, 0.78, 0.62),
    "VGGish, single-attention NN (their best)": (0.60, 0.73, 0.65),
    "MIR, SVM Simple-MI": (0.60, 0.52, 0.56),
    "MIR, avg-pooling NN": (0.55, 0.78, 0.61),
    "baseline: random guess (class frequencies)": (0.32, 0.32, 0.32),
    "baseline: plurality label set": (0.14, 0.34, 0.19),
}


# --------------------------------------------------------------------------- #
# Baselines, defined exactly as Ma et al. define them
# --------------------------------------------------------------------------- #
def random_guess_scores(Y_train: np.ndarray, Y_test: np.ndarray,
                        n_draws: int = 200) -> dict[str, float]:
    """Predict each genre independently at its training base rate (Ma's 'random guess')."""
    rng = np.random.default_rng(config.SEED)
    p = Y_train.mean(axis=0)
    out = [macro_prf(Y_test, (rng.random(Y_test.shape) < p).astype(int))
           for _ in range(n_draws)]
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def plurality_scores(Y_train: np.ndarray, Y_test: np.ndarray) -> dict[str, float]:
    """Always predict the most frequent label SET (Ma's 'plurality label' baseline)."""
    rows, counts = np.unique(Y_train, axis=0, return_counts=True)
    pred = np.repeat(rows[counts.argmax()][None, :], len(Y_test), axis=0)
    return macro_prf(Y_test, pred)


def bootstrap_diff(Yte, pred_a, pred_b, n_boot: int = 2000) -> dict:
    """Paired bootstrap over TEST ITEMS for macro-F1(a) - macro-F1(b).

    The transfer test is a single train/test split, so it has no fold structure to put a
    confidence interval on -- and this project has already been burnt once by reading a
    single-partition margin as a result (see the progress log, section 11). Resampling
    the 110 target films with replacement, and re-scoring BOTH arms on each resample,
    gives the sampling variability that is actually available here. It quantifies
    uncertainty from the finite test corpus only; it says nothing about how much the
    numbers would move under a different *training* set.
    """
    rng = np.random.default_rng(config.SEED)
    n = len(Yte)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        if Yte[s].sum(0).min() == 0 or Yte[s].sum(0).max() == len(s):
            diffs[i] = np.nan          # a genre vanished from the resample
            continue
        diffs[i] = (macro_prf(Yte[s], pred_a[s])["macro_f1"]
                    - macro_prf(Yte[s], pred_b[s])["macro_f1"])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_mean": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
            "n_boot": int(len(diffs))}


# --------------------------------------------------------------------------- #
# Feature builders. Every transform is fitted on the TRAINING corpus only.
# --------------------------------------------------------------------------- #
def oof_emotions(X: np.ndarray, E: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold predicted emotions for the training corpus (GroupKFold by film).

    The genre stage must be trained on the same *kind* of input it will see at test
    time: model-predicted emotions, not ground-truth ratings. Fitting it on in-sample
    predictions would train it on near-perfect emotions and then apply it to noisy ones.
    """
    out = np.zeros_like(E)
    for tr, te in GroupKFold(n_splits=5).split(X, E, groups):
        reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
        out[te] = reg.fit(X[tr], E[tr]).predict(X[te])
    return out


def fit_emotion_regressor(X: np.ndarray, E: np.ndarray) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300, random_state=config.SEED, n_jobs=-1).fit(X, E)


def transfer_score(Xtr, Ytr, Xte, Yte, groups_tr) -> dict:
    """Fit a tuned Binary-Relevance logistic regression on the source, score the target."""
    C = select_logreg_C(Xtr, Ytr, groups_tr)
    clf = build_classifier("logreg", C=C).fit(Xtr, Ytr)
    pred = np.asarray(clf.predict(Xte))
    res = macro_prf(Yte, pred)
    res["C"] = C
    res["per_genre_f1"] = per_genre_f1(Yte, pred).tolist()
    res["labels_per_item_pred"] = float(pred.sum(1).mean())
    res["labels_per_item_true"] = float(Yte.sum(1).mean())
    res["_pred"] = pred
    return res


def zscore_pair(Xa: np.ndarray, Xb: np.ndarray, target_standardise: bool):
    """Scale both corpora. Strict: one scaler fitted on the source. Adapted: each corpus
    z-scored with its OWN mean/SD (unsupervised -- uses no labels, still zero-shot)."""
    if target_standardise:
        return (StandardScaler().fit_transform(Xa), StandardScaler().fit_transform(Xb))
    sc = StandardScaler().fit(Xa)
    return sc.transform(Xa), sc.transform(Xb)


# --------------------------------------------------------------------------- #
def row(name: str, r: dict) -> str:
    return (f"  {name:<44}{r['macro_precision']:>7.3f}{r['macro_recall']:>8.3f}"
            f"{r['macro_f1']:>7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the 110-fold leave-one-out in-domain reference")
    args = ap.parse_args()
    set_seed()
    out: dict = {"shared_genres": G}

    # ---------------- data, in ONE shared label space (supervisor #4) ----------
    df = load_set1_shared()
    Ye = shared_genre_matrix(df)
    Xe_raw = assemble_from_cache(df, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Ee = df[config.EMOTIONS].to_numpy(float)
    groups_e = df["soundtrack"].to_numpy()

    d = load_blockbuster()
    Xb_raw, Yb = d["X_vggish"], d["Y"]

    print("=" * 78)
    print("A. THE SHARED LABEL SPACE (supervisor #4)")
    print("=" * 78)
    print(f"  {'genre':<12}{'Eerola clips':>14}{'Blockbuster films':>20}")
    for j, g in enumerate(G):
        print(f"  {g:<12}{int(Ye[:, j].sum()):>14}{int(Yb[:, j].sum()):>20}")
    print(f"  {'TOTAL':<12}{len(Ye):>14}{len(Yb):>20}")
    print(f"\n  Eerola: {len(Ye)} clips / {df['soundtrack'].nunique()} films, "
          f"{Ye.sum(1).mean():.2f} labels per clip")
    print(f"  Blockbuster: {len(Yb)} films, {Yb.sum(1).mean():.2f} labels per film")
    print("  Same six classes, same inclusion rule, same column order -> the same "
          "classifier\n  can be trained on one corpus and applied to the other.")
    out["counts"] = {"eerola": Ye.sum(0).tolist(), "blockbuster": Yb.sum(0).tolist()}

    # ---------------- B. reference points -------------------------------------
    print("\n" + "=" * 78)
    print("B. REFERENCE POINTS -- what a good score looks like on each corpus")
    print("=" * 78)
    print(f"  {'':<44}{'prec':>7}{'recall':>8}{'F1':>7}")
    print("  " + "-" * 66)
    print("  Ma et al. (2021), published, in-domain LOO on Blockbuster:")
    for name, (p, r, f) in MA2021.items():
        print(f"    {name:<42}{p:>7.2f}{r:>8.2f}{f:>7.2f}")
    out["ma2021"] = {k: list(v) for k, v in MA2021.items()}

    print("\n  Our pipeline, in-domain (no transfer):")
    b_rand = random_guess_scores(Yb, Yb)
    b_plur = plurality_scores(Yb, Yb)
    print(row("Blockbuster: random guess (freq) [Ma .32]", b_rand))
    print(row("Blockbuster: plurality label set [Ma .19]", b_plur))
    out["blockbuster_baselines"] = {"random_guess": b_rand, "plurality": b_plur}

    if not args.quick:
        # Ma's exact protocol: leave-one-film-out over the 110 films.
        Xb_z = StandardScaler().fit_transform(Xb_raw)
        C = select_logreg_C(Xb_z, Yb, np.arange(len(Yb)))
        pred = np.zeros_like(Yb)
        for tr, te in LeaveOneOut().split(Xb_z):
            pred[te] = build_classifier("logreg", C=C).fit(Xb_z[tr], Yb[tr]).predict(Xb_z[te])
        loo = macro_prf(Yb, pred)
        loo["C"] = C
        loo["per_genre_f1"] = per_genre_f1(Yb, pred).tolist()
        print(row("Blockbuster: VGGish -> LogReg, LOO (our reimpl.)", loo))
        out["blockbuster_loo_ours"] = loo

    # Eerola in-domain on the same six classes, GroupKFold by film.
    ein = {}
    Xe_z = StandardScaler().fit_transform(Xe_raw)
    C = select_logreg_C(Xe_z, Ye, groups_e)
    preds, truths = [], []
    for tr, te in RepeatedGroupKFold(n_splits=5, n_repeats=2,
                                     random_state=config.SEED).split(Xe_z, Ye, groups_e):
        preds.append(build_classifier("logreg", C=C).fit(Xe_z[tr], Ye[tr]).predict(Xe_z[te]))
        truths.append(Ye[te])
    ein = macro_prf(np.vstack(truths), np.vstack(preds))
    ein["C"] = C
    print(row("Eerola: VGGish -> LogReg, GroupKFold (in-domain)", ein))
    e_rand = random_guess_scores(Ye, Ye)
    print(row("Eerola: random guess (freq)", e_rand))
    out["eerola_in_domain"] = ein
    out["eerola_baselines"] = {"random_guess": e_rand}

    # ---------------- C. ZERO-SHOT Eerola -> Blockbuster ----------------------
    print("\n" + "=" * 78)
    print("C. ZERO-SHOT: trained on EEROLA ONLY, tested on all 110 Blockbuster films")
    print("   (no Blockbuster label is seen during training)")
    print("=" * 78)

    reg_full = fit_emotion_regressor(Xe_raw, Ee)
    Ee_oof = oof_emotions(Xe_raw, Ee, groups_e)
    Fe_pred = build_emotion_features(Ee_oof)          # train on OOF-predicted emotions
    Fe_true = build_emotion_features(Ee)              # ceiling variant: ground-truth
    Fb_pred = build_emotion_features(reg_full.predict(Xb_raw))

    # Cue-level bridging. The regressor was trained on ~17 s Eerola clips, and a
    # Blockbuster CUE is the same kind of object (median 19 s of film music), whereas the
    # film-level vector above is a mean over ~39 of them. Predicting emotion per cue and
    # pooling the EMOTIONS afterwards therefore keeps the model on its training unit;
    # pooling the embedding first and predicting once does not. In-domain this is worth
    # +0.059 Macro-F1 (p=0.020, section 17), so it is tested here too.
    cues = load_blockbuster_cues()
    Fb_cue = build_emotion_features(
        aggregate_cues(reg_full.predict(cues["X_vggish"]),
                       cues["cue_film_vggish"], len(Yb)))

    out["zero_shot"] = {}
    for adapt in (False, True):
        tag = "target-standardised" if adapt else "strict (source scaler)"
        Xe_s, Xb_s = zscore_pair(Xe_raw, Xb_raw, adapt)
        pca = PCA(n_components=8, random_state=config.SEED).fit(Xe_s)

        arms = {
            "VGGish-128 direct": (Xe_s, Xb_s),
            "VGGish -> predicted emotion(11), per-cue": (Fe_pred, Fb_cue),
            "VGGish -> predicted emotion(11), film-level": (Fe_pred, Fb_pred),
            "emotion(11), genre stage on ratings": (Fe_true, Fb_cue),
            "PCA-8(VGGish) [control]": (pca.transform(Xe_s), pca.transform(Xb_s)),
        }
        print(f"\n  scaling: {tag}")
        print(f"  {'':<44}{'prec':>7}{'recall':>8}{'F1':>7}")
        print("  " + "-" * 66)
        res = {}
        for name, (A, B) in arms.items():
            r = transfer_score(A, Ye, B, Yb, groups_e)
            res[name] = r
            print(row(name, r))
        print(row("random guess (freq) [Ma .32]", b_rand))
        out["zero_shot"][tag] = res

        if adapt:
            print(f"\n  per-genre F1 (target-standardised):")
            print(f"  {'':<38}" + "".join(f"{g:>9}" for g in G))
            for name, r in res.items():
                print(f"  {name:<38}"
                      + "".join(f"{v:>9.3f}" for v in r["per_genre_f1"]))
            print(f"  {'(support, films)':<38}"
                  + "".join(f"{int(v):>9}" for v in Yb.sum(0)))

        # The bootstrap is run in BOTH scaling regimes. Only the embedding arms move
        # between them (the emotion arms are in rating units and are identical), so a
        # bootstrap computed in one regime against a table printed for the other silently
        # compares against the weaker baseline. STRICT is the conservative regime -- the
        # direct embedding scores better there (0.407 vs 0.394) -- so the strict numbers
        # are the ones to quote.
        print(f"\n  paired bootstrap over the 110 target films (2000 resamples), "
              f"{tag}:")
        base = res["VGGish-128 direct"]["_pred"]
        boot: dict = {}
        for name in ("VGGish -> predicted emotion(11), per-cue",
                     "VGGish -> predicted emotion(11), film-level",
                     "PCA-8(VGGish) [control]"):
            bs = bootstrap_diff(Yb, res[name]["_pred"], base)
            boot[f"{name} vs VGGish direct"] = bs
            print(f"    {name + ' vs direct':<46}{bs['diff_mean']:>+7.3f}  "
                  f"[{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}]  p={bs['p_two_sided']:.3f}")
        bs = bootstrap_diff(Yb, res["VGGish -> predicted emotion(11), per-cue"]["_pred"],
                            res["PCA-8(VGGish) [control]"]["_pred"])
        boot["predicted-emotion vs PCA-8 control"] = bs
        print(f"    {'predicted-emotion vs PCA-8 control':<46}{bs['diff_mean']:>+7.3f}  "
              f"[{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}]  p={bs['p_two_sided']:.3f}")
        out.setdefault("bootstrap", {})[tag] = boot

    # ---------------- D. reverse direction ------------------------------------
    print("\n" + "=" * 78)
    print("D. REVERSE ZERO-SHOT: trained on BLOCKBUSTER ONLY, tested on Eerola clips")
    print("=" * 78)
    print(f"  {'':<44}{'prec':>7}{'recall':>8}{'F1':>7}")
    print("  " + "-" * 66)
    Xb_s, Xe_s = zscore_pair(Xb_raw, Xe_raw, True)
    films = np.arange(len(Yb))                       # each Blockbuster row is its own film
    rev = transfer_score(Xb_s, Yb, Xe_s, Ye, films)
    print(row("VGGish-128 direct (Blockbuster -> Eerola)", rev))
    print(row("random guess (freq) on Eerola", e_rand))
    out["reverse_zero_shot"] = rev

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "_pred"}
        return o

    out = strip(out)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
