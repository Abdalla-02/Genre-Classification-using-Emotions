"""Zero-shot transfer with decision thresholds tuned on the SOURCE dataset.

The open check of the Discussion: trained on Eerola and applied to the 110 Blockbuster
films without further fitting, the emotion pipeline beats direct VGGish by +0.106
(p = 0.013) and the PCA-8 control by +0.093 (p = 0.027) at the default threshold of 0.5
(`zero_shot.json`, strict regime). Its lead is partly recall-driven (recall 0.651 vs
0.435, precision 0.480 vs 0.705), so part of it could be an operating-point effect: the
balanced logistic regression on eleven emotion features may simply say "yes" more often.
In-domain, per-genre thresholds tuned on the training folds made the pipelines converge
(log §22.1). This script asks whether the same happens in transfer.

The test stays zero-shot. Every threshold is chosen on Eerola only; no Blockbuster label
is used for anything but scoring.

  * Each arm of the strict regime of `exp_zero_shot.py` is rebuilt with that script's own
    functions (same features, same C chosen on Eerola, same classifier fitted on all
    Eerola clips), and its default-threshold score is asserted to equal `zero_shot.json`.
  * Out-of-fold probabilities on Eerola: 10 x 5-fold grouped by film, the classifier with
    the arm's C refitted on each training part; each clip's probability is averaged over
    the ten repeats.
  * Per genre, the threshold (grid 0.10 to 0.90, step 0.025) that maximises the F1 of
    these source probabilities is applied to the target probabilities of the classifier
    fitted on all of Eerola.

Two tuning units, fixed before running:
  PRIMARY  clip level -- F1 over the 319 Eerola clips, the unit the classifier is
           trained on (stable: every genre has 19 to 239 clips);
  CHECK    film level -- F1 over the 40 Eerola films, clip probabilities averaged per
           film. This is the unit of the target, but it rests on few films (Sci-Fi and
           Horror have only a handful), so its thresholds are noisy.

Threshold-free, the film-level macro ROC AUC on Blockbuster is reported for every arm,
with a paired bootstrap over the 110 target films for the main comparisons. F1
comparisons use the bootstrap of `exp_zero_shot.py` (110 films, 2000 resamples).

Run:  python experiments/cross_dataset/exp_zero_shot_threshold.py      (a few minutes)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

from exp_zero_shot import (  # noqa: E402
    bootstrap_diff,
    fit_emotion_regressor,
    oof_emotions,
    zscore_pair,
)

G = config.SHARED_GENRES
RESULTS = config.RESULTS_DIR / "zero_shot_threshold.json"
ZERO_SHOT = config.RESULTS_DIR / "zero_shot.json"
STRICT = "strict (source scaler)"
GRID = np.round(np.arange(0.10, 0.9001, 0.025), 3)
N_REPEATS = 10
EMO = "VGGish -> predicted emotion(11), per-cue"
DIRECT = "VGGish-128 direct"
PCA8 = "PCA-8(VGGish) [control]"
COMPARISONS = ((EMO, DIRECT), (EMO, PCA8), ("VGGish -> predicted emotion(11), film-level", DIRECT),
               (PCA8, DIRECT))


def proba(clf, X) -> np.ndarray:
    """P(genre) for every genre from the fitted Binary-Relevance pipeline."""
    return np.column_stack([p[:, 1] for p in clf.predict_proba(X)])


def oof_source_proba(X, Y, groups, C) -> np.ndarray:
    """Out-of-fold probabilities on the source, averaged over the repeats."""
    P = np.zeros(Y.shape)
    for r, (tr, te) in enumerate(RepeatedGroupKFold(n_splits=5, n_repeats=N_REPEATS,
                                                    random_state=config.SEED)
                                 .split(X, Y, groups)):
        P[te] += proba(build_classifier("logreg", C=C).fit(X[tr], Y[tr]), X[te])
    return P / N_REPEATS


def tune(P, Y) -> np.ndarray:
    """Per-genre F1-maximising threshold; ties go to the threshold nearest 0.5."""
    out = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        f = np.array([f1_score(Y[:, j], (P[:, j] >= t).astype(int), zero_division=0)
                      for t in GRID])
        best = np.flatnonzero(f == f.max())
        out[j] = GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]]
    return out


def per_film(P, Y, groups):
    films = np.unique(groups)
    return (np.vstack([P[groups == f].mean(0) for f in films]),
            np.vstack([Y[groups == f][0] for f in films]))


def macro_auc(Y, P) -> float:
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))


def bootstrap_auc(Y, Pa, Pb, n_boot: int = 2000) -> dict:
    """Paired bootstrap over target films for macro AUC(a) - macro AUC(b)."""
    rng = np.random.default_rng(config.SEED)
    n = len(Y)
    d = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        if Y[s].sum(0).min() == 0 or Y[s].sum(0).max() == n:
            continue
        d.append(macro_auc(Y[s], Pa[s]) - macro_auc(Y[s], Pb[s]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff_mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "n_boot": int(len(d))}


def score(Y, pred) -> dict:
    r = macro_prf(Y, pred)
    r["per_genre_f1"] = per_genre_f1(Y, pred).tolist()
    r["labels_per_film_pred"] = float(pred.sum(1).mean())
    return r


def main() -> None:
    set_seed()
    ref = json.loads(ZERO_SHOT.read_text(encoding="utf-8"))["zero_shot"][STRICT]

    # ---- the strict-regime arms of exp_zero_shot.py, rebuilt with its own functions ----
    df = load_set1_shared()
    Ye = shared_genre_matrix(df)
    Xe_raw = assemble_from_cache(df, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Ee = df[config.EMOTIONS].to_numpy(float)
    groups = df["soundtrack"].to_numpy()
    d = load_blockbuster()
    Xb_raw, Yb = d["X_vggish"], d["Y"]

    reg_full = fit_emotion_regressor(Xe_raw, Ee)
    Fe_pred = build_emotion_features(oof_emotions(Xe_raw, Ee, groups))
    Fe_true = build_emotion_features(Ee)
    Fb_pred = build_emotion_features(reg_full.predict(Xb_raw))
    cues = load_blockbuster_cues()
    Fb_cue = build_emotion_features(
        aggregate_cues(reg_full.predict(cues["X_vggish"]), cues["cue_film_vggish"], len(Yb)))
    Xe_s, Xb_s = zscore_pair(Xe_raw, Xb_raw, False)
    pca = PCA(n_components=8, random_state=config.SEED).fit(Xe_s)
    arms = {
        DIRECT: (Xe_s, Xb_s),
        EMO: (Fe_pred, Fb_cue),
        "VGGish -> predicted emotion(11), film-level": (Fe_pred, Fb_pred),
        "emotion(11), genre stage on ratings": (Fe_true, Fb_cue),
        PCA8: (pca.transform(Xe_s), pca.transform(Xb_s)),
    }
    n_films = len(np.unique(groups))
    out: dict = {"design": {
        "question": "does the zero-shot lead survive per-genre thresholds tuned on the source?",
        "regime": STRICT, "source": f"Eerola, {len(Ye)} clips / {n_films} films, six shared genres",
        "target": f"Blockbuster, {len(Yb)} films", "threshold_grid": [float(GRID[0]), float(GRID[-1]), 0.025],
        "oof": f"RepeatedGroupKFold(5, {N_REPEATS}) by film, probabilities averaged over repeats",
        "primary": "clip-level tuning", "check": "film-level tuning",
        "true_labels_per_film_target": float(Yb.sum(1).mean())}, "arms": {}}
    print(f"source {len(Ye)} clips / {n_films} films, target {len(Yb)} films, "
          f"true labels per target film {Yb.sum(1).mean():.2f}\n")

    P_target = {}
    preds = {"default": {}, "clip-tuned": {}, "film-tuned": {}}
    hdr = f"  {'arm':<46}{'rule':>11}{'P':>7}{'R':>7}{'F1':>7}{'lab/film':>9}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for name, (A, B) in arms.items():
        C = select_logreg_C(A, Ye, groups)
        clf = build_classifier("logreg", C=C).fit(A, Ye)
        Pt = proba(clf, B)
        P_target[name] = Pt
        default = (Pt >= 0.5).astype(int)
        assert np.array_equal(default, np.asarray(clf.predict(B))), name
        assert abs(macro_prf(Yb, default)["macro_f1"] - ref[name]["macro_f1"]) < 1e-9, name
        assert C == ref[name]["C"], name
        Ps = oof_source_proba(A, Ye, groups, C)
        thr_clip = tune(Ps, Ye)
        thr_film = tune(*per_film(Ps, Ye, groups))
        rec = {"C": C, "macro_auc": macro_auc(Yb, Pt),
               "auc_per_genre": [float(roc_auc_score(Yb[:, j], Pt[:, j])) for j in range(len(G))],
               "source_oof_macro_f1_at_0.5": float(f1_score(Ye, (Ps >= 0.5).astype(int),
                                                            average="macro", zero_division=0))}
        for rule, thr in (("default", np.full(len(G), 0.5)), ("clip-tuned", thr_clip),
                          ("film-tuned", thr_film)):
            pred = (Pt >= thr).astype(int)
            preds[rule][name] = pred
            rec[rule] = score(Yb, pred) | {"thresholds": thr.tolist()}
            r = rec[rule]
            print(f"  {name:<46}{rule:>11}{r['macro_precision']:>7.3f}{r['macro_recall']:>7.3f}"
                  f"{r['macro_f1']:>7.3f}{r['labels_per_film_pred']:>9.2f}")
        print(f"  {'':<46}{'AUC':>11}{rec['macro_auc']:>21.3f}   thresholds clip "
              + " ".join(f"{t:.3f}" for t in thr_clip))
        out["arms"][name] = rec

    print("\n  paired bootstrap over the 110 target films (2000 resamples)")
    out["comparisons"] = []
    for a, b in COMPARISONS:
        c = {"a": a, "b": b}
        for rule in preds:
            c[rule] = bootstrap_diff(Yb, preds[rule][a], preds[rule][b])
        c["macro_auc"] = bootstrap_auc(Yb, P_target[a], P_target[b])
        out["comparisons"].append(c)
        print(f"  {a} vs {b}")
        for k in ("default", "clip-tuned", "film-tuned", "macro_auc"):
            s = c[k]
            print(f"    {k:<12}{s['diff_mean']:>+8.3f}  [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"
                  f"  p={s['p_two_sided']:.3f}")
    ref_boot = json.loads(ZERO_SHOT.read_text(encoding="utf-8"))["bootstrap"][STRICT]
    assert abs(out["comparisons"][0]["default"]["diff_mean"]
               - ref_boot[f"{EMO} vs VGGish direct"]["diff_mean"]) < 1e-9

    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
