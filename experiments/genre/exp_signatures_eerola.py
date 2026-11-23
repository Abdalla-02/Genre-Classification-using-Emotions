"""Emotional signatures of the eight Eerola genres: Cohen's d of every rated emotion,
clips with the genre against all other clips, on the 346 clips of the eight-genre space.

The same numbers as the Cohen's d column of ``exp_emotion_genre.py`` (which only prints
them); this script saves the full genre x emotion matrix so that the heatmap of the thesis
(Figure "signatures", Section 5.5) is drawn from a results file. Human ratings only, no model.

Run:  python experiments/genre/exp_signatures_eerola.py
Out:  results/signatures_eerola.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import genre_matrix, load_set1  # noqa: E402


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def main() -> None:
    df = load_set1()
    Y = genre_matrix(df)
    genres, emos = config.PRIMARY_GENRES, config.EMOTIONS
    d = {}
    for j, g in enumerate(genres):
        y = Y[:, j] == 1
        d[g] = {e: round(cohens_d(df.loc[y, e].to_numpy(float), df.loc[~y, e].to_numpy(float)), 4)
                for e in emos}
    out = {
        "what": "Cohen's d of each rated emotion, clips with the genre vs all other clips "
                "(eight-genre Eerola space, human ratings, pooled SD)",
        "n_clips": int(len(df)),
        "n_films": int(df["soundtrack"].nunique()),
        "clips_per_genre": {g: int(Y[:, j].sum()) for j, g in enumerate(genres)},
        "emotions": emos,
        "genres": genres,
        "d": d,
        "dominant": {g: max(d[g], key=lambda e: abs(d[g][e])) for g in genres},
    }
    p = config.RESULTS_DIR / "signatures_eerola.json" if hasattr(config, "RESULTS_DIR") else \
        Path(__file__).resolve().parents[2] / "results" / "signatures_eerola.json"
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    for g in genres:
        k = out["dominant"][g]
        print(f"{g:12s} n={out['clips_per_genre'][g]:3d}  {k:8s} {d[g][k]:+.2f}")
    print(f"-> {p}  ({out['n_clips']} clips, {out['n_films']} films)")


if __name__ == "__main__":
    main()
