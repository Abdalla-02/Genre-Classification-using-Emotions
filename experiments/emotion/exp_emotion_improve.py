"""Answers three questions with data:
  Q1  which regressor best predicts emotion (Ridge vs SVR-RBF vs RandomForest)?
  Q2  the emotion->genre pipeline evaluated on the agreed 5-genre subset (not 8);
  Q3  can emotion give better genre classification -- incl. emotion+VGGish fusion.
GroupKFold by film throughout.

Run:  python experiments/emotion/exp_emotion_improve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import assemble_from_cache, genre_matrix, load_set1  # noqa: E402
from src.models import build_classifier  # noqa: E402
from src.utils import set_seed  # noqa: E402

SUBSET = ["Action", "Crime", "Drama", "Comedy", "Horror"]


def build_11(E):
    d = {e: E[:, i] for i, e in enumerate(config.EMOTIONS)}
    return np.column_stack([E, d["valence"] * d["energy"],
                            np.mean([d["anger"], d["fear"], d["tension"], d["sad"]], 0),
                            np.mean([d["happy"], d["tender"], d["valence"]], 0)])


def gf1(X, Y, g, s):
    return evaluate(X, Y, g, s, "logreg", "group")["macro_f1"][0]


def main():
    set_seed()
    dfe = load_set1(clean=False)
    Xv_full = assemble_from_cache(dfe, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Emo_full = dfe[config.EMOTIONS].to_numpy(float)
    gfull = dfe["soundtrack"].to_numpy()

    # ---- Q1: emotion regressor comparison (VGGish -> 8 emotions) ----
    regs = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "SVR-RBF": make_pipeline(StandardScaler(),
                                 MultiOutputRegressor(SVR(C=10, gamma="scale"))),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=config.SEED,
                                              n_jobs=-1),
    }
    print("Q1  emotion regressor (VGGish->8 emotions, GroupKFold, mean R^2)")
    print("-" * 52)
    best_name, best_r2 = None, -1
    for name, reg in regs.items():
        pred = cross_val_predict(reg, Xv_full, Emo_full,
                                 cv=GroupKFold(5).split(Xv_full, Emo_full, gfull))
        r2 = np.mean([r2_score(Emo_full[:, j], pred[:, j]) for j in range(8)])
        print(f"  {name:14} mean R^2 = {r2:.3f}")
        if r2 > best_r2:
            best_name, best_r2, best_reg = name, r2, reg

    # ---- Q2/Q3: 5-genre subset on Eerola ----
    df = load_set1()
    idx = [config.PRIMARY_GENRES.index(x) for x in SUBSET]
    Y5 = genre_matrix(df)[:, idx]
    keep = Y5.sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y5[keep]
    g = dfk["soundtrack"].to_numpy()
    s = np.array([SUBSET[i] for i in Y.argmax(1)])
    Xv = assemble_from_cache(dfk, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    Emo = dfk[config.EMOTIONS].to_numpy(float)
    Xz = StandardScaler().fit_transform(Xv)
    from sklearn.decomposition import PCA
    pca8 = PCA(n_components=8, random_state=config.SEED).fit_transform(Xz)

    # out-of-fold predicted-emotion pipeline using the BEST regressor
    pred = np.zeros(Y.shape, dtype=int)
    for tr, te in GroupKFold(5).split(Xv, Y, g):
        from sklearn.base import clone
        r = clone(best_reg).fit(Xv[tr], Emo[tr])
        clf = build_classifier("logreg").fit(build_11(r.predict(Xv[tr])), Y[tr])
        pred[te] = np.asarray(clf.predict(build_11(r.predict(Xv[te]))))
    oof_pipe = f1_score(Y, pred, average="macro", zero_division=0)

    print(f"\nQ2/Q3  5-genre subset {SUBSET} (n={len(dfk)}, GroupKFold Macro-F1)")
    print("-" * 60)
    print(f"  {'ground-truth emotion(11) -> genre (ceiling)':45}{gf1(build_11(Emo), Y, g, s):.3f}")
    print(f"  {'VGGish -> PREDICTED emotion(11) -> genre ('+best_name+')':45}{oof_pipe:.3f}")
    print(f"  {'emotion(11) + VGGish(128) fusion [Q3]':45}"
          f"{gf1(np.hstack([build_11(Emo), Xv]), Y, g, s):.3f}")
    print(f"  {'VGGish-128 -> genre (direct)':45}{gf1(Xv, Y, g, s):.3f}")
    print(f"  {'PCA-8(VGGish) -> genre [control]':45}{gf1(pca8, Y, g, s):.3f}")
    print(f"  {'dummy':45}{evaluate(Xv, Y, g, s, 'dummy', 'group')['macro_f1'][0]:.3f}")


if __name__ == "__main__":
    main()
