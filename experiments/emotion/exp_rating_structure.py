"""How the eight rating scales relate to each other (all 360 rated clips of Set 1).

Figure 2.1 is a schematic: it draws valence, arousal, energy and tension at fixed angles and
places the discrete emotions where the literature puts them. The author asked on 4 Nov what
the ratings of this dataset actually look like. This script gives the correlation matrix of
the eight scales and a principal component analysis of that matrix. Because the analysis
runs on correlations, a scale's loading on a component is its correlation with that
component, so the scales can be drawn inside a unit circle (``make_figures.py
rating_structure``). The signs of the components are fixed so that valence and energy-side
happiness point right and up, as in Figure 2.1.

Run:  python experiments/emotion/exp_rating_structure.py
Writes results/rating_structure.json.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import load_set1  # noqa: E402

OUT = config.RESULTS_DIR / "rating_structure.json"


def main() -> None:
    df = load_set1(clean=False)
    scales = list(config.EMOTIONS)
    C = np.corrcoef(df[scales].to_numpy(float).T)
    values, vectors = np.linalg.eigh(C)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    loadings = vectors * np.sqrt(values)
    if loadings[scales.index("valence"), 0] < 0:
        loadings[:, 0] *= -1
    if loadings[scales.index("happy"), 1] < 0:
        loadings[:, 1] *= -1
    share = values / values.sum()

    out = {"n_clips": int(len(df)), "scales": scales,
           "correlation": {a: {b: float(C[i, j]) for j, b in enumerate(scales)}
                           for i, a in enumerate(scales)},
           "explained_variance": share.tolist(),
           "loadings": {a: loadings[i, :2].tolist() for i, a in enumerate(scales)}}
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"{len(df)} clips; first two components {share[0]:.3f} + {share[1]:.3f} = "
          f"{share[:2].sum():.3f}")
    for i, a in enumerate(scales):
        print(f"  {a:8} {loadings[i, 0]:+.2f} {loadings[i, 1]:+.2f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
