"""Cross-dataset emotion bridge: predict Blockbuster emotions from Eerola, then genre.

Pipeline: train a VGGish -> 8-emotion RandomForest regressor (the best regressor, see
exp_emotion_improve.py) on all Eerola, apply to Blockbuster's (film-level) VGGish ->
predicted emotions -> genre. Includes the key controls -- PCA-8 and a random-8 projection
of VGGish -> genre -- to test whether the *emotion* bottleneck beats a generic 8-d
compression, and a face-validity check of the predicted emotions per genre.

Run:  python experiments/cross_dataset/exp_blockbuster_emotion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import assemble_from_cache, load_set1  # noqa: E402
from src.features.blockbuster import load_blockbuster  # noqa: E402
from src.utils import set_seed  # noqa: E402


def build_11(E):
    d = {e: E[:, i] for i, e in enumerate(config.EMOTIONS)}
    return np.column_stack([E, d["valence"] * d["energy"],
                            np.mean([d["anger"], d["fear"], d["tension"], d["sad"]], 0),
                            np.mean([d["happy"], d["tender"], d["valence"]], 0)])


def kf1(X, Y):
    return evaluate(X, Y, None, None, "logreg", "kfold")["macro_f1"][0]


def main():
    set_seed()
    # train VGGish -> emotion (RandomForest) on all Eerola
    df = load_set1(clean=False)
    Xe = assemble_from_cache(df, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
    reg.fit(Xe, df[config.EMOTIONS].to_numpy(float))

    # apply to Blockbuster VGGish
    d = load_blockbuster()
    Xb, Yb, genres = d["X_vggish"], d["Y"], d["genres"]
    Eb = reg.predict(Xb)
    Xz = StandardScaler().fit_transform(Xb)
    pca8 = PCA(n_components=8, random_state=config.SEED).fit_transform(Xz)
    rand8 = GaussianRandomProjection(n_components=8, random_state=config.SEED).fit_transform(Xz)

    print(f"Blockbuster: {len(Xb)} films, {Yb.shape[1]} genres {genres}\n")
    print(f"{'approach (KFold, Macro-F1)':45}{'F1':>7}")
    print("-" * 52)
    print(f"{'VGGish-128 -> genre (direct)':45}{kf1(Xb, Yb):>7.3f}")
    print(f"{'VGGish -> predicted emotion(11) -> genre':45}{kf1(build_11(Eb), Yb):>7.3f}")
    print(f"{'PCA-8(VGGish) -> genre  [control]':45}{kf1(pca8, Yb):>7.3f}")
    print(f"{'random-8(VGGish) -> genre  [control]':45}{kf1(rand8, Yb):>7.3f}")
    print(f"{'dummy':45}"
          f"{evaluate(Xb, Yb, None, None, 'dummy', 'kfold')['macro_f1'][0]:>7.3f}")

    # face validity: predicted emotion deviation per genre
    print(f"\n{'face validity: mean PREDICTED emotion per genre (dev. from overall)':45}")
    print("-" * 60)
    overall = Eb.mean(axis=0)
    for j, gen in enumerate(genres):
        diff = Eb[Yb[:, j] == 1].mean(axis=0) - overall
        order = sorted(range(8), key=lambda i: diff[i], reverse=True)
        print(f"  {gen:9} (n={int(Yb[:, j].sum()):3d})  "
              f"highest: {config.EMOTIONS[order[0]]} ({diff[order[0]]:+.2f})   "
              f"lowest: {config.EMOTIONS[order[-1]]} ({diff[order[-1]]:+.2f})")


if __name__ == "__main__":
    main()
