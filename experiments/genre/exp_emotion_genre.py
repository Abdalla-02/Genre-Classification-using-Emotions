"""RQ3 / emotion-genre relationship: which emotions predict which genres?

Three complementary views on the WITH-emotion (ground-truth) data, cross-checked:
  * Logistic Regression coefficients (standardized features -> comparable, signed:
    + means the emotion raises the genre's probability)
  * Random Forest feature importances (non-linear, unsigned)
  * Cohen's d effect size (aggregate: emotion mean for clips with vs without the genre)

Fit on all 346 clips (this is an interpretability analysis, not a prediction estimate).

Run:  python experiments/genre/exp_emotion_genre.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import add_derived_features, genre_matrix, load_set1  # noqa: E402
from src.models import build_binary_logreg  # noqa: E402
from src.utils import set_seed  # noqa: E402

FEATS = config.FEATURE_COLS       # 11 (8 emotions + 3 engineered)
EMO = config.EMOTIONS             # 8 emotions (RQ3 focus)


RQ3_C = 0.003  # value selected by the nested CV in Section 11b


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def main():
    set_seed()
    df = add_derived_features(load_set1())
    Xz = StandardScaler().fit_transform(df[FEATS].to_numpy(float))
    Y = genre_matrix(df)
    G = config.PRIMARY_GENRES

    print("### RQ3: strongest emotion predictors per genre ###")
    print("LR coef (signed, standardized) | RF importance | Cohen's d (emotions only)\n")
    for j, g in enumerate(G):
        y = Y[:, j]
        # shared definition (src.models.build_binary_logreg) so these coefficients come
        # from the same estimator the reported results use. C matches the value the
        # nested CV selects in Section 11b -- at the sklearn default the coefficients are
        # inflated by under-regularisation, though the ranking is largely unchanged.
        lr_pipe = build_binary_logreg(C=RQ3_C, balanced=True).fit(Xz, y)
        lr = lr_pipe.named_steps["clf"]
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                    random_state=config.SEED, n_jobs=-1).fit(Xz, y)
        coef = dict(zip(FEATS, lr.coef_[0]))
        imp = dict(zip(FEATS, rf.feature_importances_))
        ds = {e: cohens_d(df.loc[y == 1, e].values, df.loc[y == 0, e].values) for e in EMO}

        top_lr = sorted(coef, key=lambda f: abs(coef[f]), reverse=True)[:3]
        top_rf = sorted(imp, key=lambda f: imp[f], reverse=True)[:3]
        top_d = max(ds, key=lambda e: abs(ds[e]))

        print(f"[{g}]  (n={int(y.sum())})")
        print("  LR : " + ", ".join(f"{f}{'+' if coef[f] >= 0 else '-'}{abs(coef[f]):.2f}"
                                     for f in top_lr))
        print("  RF : " + ", ".join(f"{f} {imp[f]:.2f}" for f in top_rf))
        print(f"  d  : {top_d} (d={ds[top_d]:+.2f})   all: " +
              ", ".join(f"{e}={ds[e]:+.2f}" for e in EMO))
        print()

    print("### Compact RQ3 answer (strongest signed emotion per genre) ###")
    for j, g in enumerate(G):
        y = Y[:, j]
        ds = {e: cohens_d(df.loc[y == 1, e].values, df.loc[y == 0, e].values) for e in EMO}
        e = max(ds, key=lambda k: abs(ds[k]))
        direction = "high" if ds[e] > 0 else "low"
        print(f"  {g:12} <- {direction} {e} (d={ds[e]:+.2f})")


if __name__ == "__main__":
    main()
