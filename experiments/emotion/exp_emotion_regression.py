"""Experiment 1: emotion regression -- audio embedding -> 8 emotion values.

Compares the thesis-primary AST (768-d) against VGGish (128-d, the cross-dataset bridge
feature) and librosa-MIR on predicting the 8 human emotion ratings. RandomForest regression
(the best regressor -- see experiments/emotion/exp_emotion_improve.py), GroupKFold by film, R^2 and
RMSE per emotion (scale 1-9).

Run:  python experiments/emotion/exp_emotion_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import assemble_from_cache, load_set1  # noqa: E402
from src.utils import set_seed  # noqa: E402

FEATURES = {
    "AST-768": config.EMBEDDINGS_DIR,
    "VGGish-128": config.VGGISH_EMBEDDINGS_DIR,
    "MIR-librosa": config.MIR_EMBEDDINGS_DIR,
}


def main():
    set_seed()
    df = load_set1(clean=False)  # all 360 clips have emotion ratings
    Y = df[config.EMOTIONS].to_numpy(float)
    groups = df["soundtrack"].to_numpy()
    cv = GroupKFold(5)

    for name, cache in FEATURES.items():
        X = assemble_from_cache(df, "set1", cache_dir=cache)
        model = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
        pred = cross_val_predict(model, X, Y, cv=cv.split(X, Y, groups))
        r2 = [r2_score(Y[:, j], pred[:, j]) for j in range(len(config.EMOTIONS))]
        rmse = [mean_squared_error(Y[:, j], pred[:, j]) ** 0.5
                for j in range(len(config.EMOTIONS))]
        print(f"\n{'=' * 68}\n{name}  (X={X.shape})   mean R^2={np.mean(r2):.3f}  "
              f"mean RMSE={np.mean(rmse):.2f}\n{'=' * 68}")
        print(f"{'emotion':10}{'R^2':>7}{'RMSE':>7}")
        for e, r, rm in zip(config.EMOTIONS, r2, rmse):
            print(f"{e:10}{r:>7.3f}{rm:>7.2f}")


if __name__ == "__main__":
    main()
