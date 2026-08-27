"""Ma et al. (2021) Blockbuster dataset -- features only (no audio, copyright).

Each of the 110 films is a *bag* of soundtrack cues (Multiple-Instance-Learning); each
cue is a per-frame feature time series. For a simple, fair baseline we mean-pool over
all frames of all cues into one fixed vector per film, for two feature sets:

  * VGGish (128-dim learned audio embedding)
  * MFCC   (78 of the 140 hand-crafted MIR features: mfcc + delta + deltadelta, mean/std)

Genre is multi-label over the 6 reduced genres used in the paper. This supports the
supervisor's "compare VGGish vs MFCC" task and mirrors Ma et al.'s finding that the
learned embedding does not clearly beat classic MFCC features.
"""

from __future__ import annotations

import csv

import numpy as np

from src import config

BLOCKBUSTER_GENRES = ["action", "drama", "comedy", "sci-fi", "romance", "horror"]
_MFCC_PREFIXES = ("mfcc", "deltamfcc", "deltadeltamfcc")


def _require_dir():
    if config.BLOCKBUSTER_DIR is None:
        raise FileNotFoundError(
            "Blockbuster dataset not found. Place the Ma et al. (2021) feature supplement "
            "(the files including 'mir_feature_names.csv') in data/raw/Blockbuster_DB/, "
            "or set the BLOCKBUSTER_DIR environment variable to its location."
        )
    return config.BLOCKBUSTER_DIR


def _mean_pool(bag: list[np.ndarray]) -> np.ndarray:
    """Mean over all frames of all cues -> one vector per film."""
    return np.vstack([np.asarray(cue, dtype=np.float64) for cue in bag]).mean(axis=0)


def load_blockbuster() -> dict:
    """Return dict: X_vggish (n,128), X_mfcc (n,78), Y (n,6), films, genres.

    Films are the intersection of the genre list and both feature dictionaries, in a
    stable sorted order.
    """
    bd = _require_dir()

    names = open(bd / "mir_feature_names.csv").read().strip().split("\t")
    mfcc_idx = [i for i, n in enumerate(names) if n.startswith(_MFCC_PREFIXES)]

    mir = np.load(bd / "mir_features_dictionary.npy", allow_pickle=True).item()
    vgg = np.load(bd / "vggish_features_dictionary.npy", allow_pickle=True).item()

    genre_of = {}
    with open(bd / "film_genre_master_list.csv", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            genre_of[row[0]] = [g.strip() for g in row[2].split(",") if g.strip()]

    films = sorted(set(genre_of) & set(mir) & set(vgg))

    X_mfcc = np.vstack([_mean_pool(mir[f])[mfcc_idx] for f in films]).astype(np.float32)
    X_vggish = np.vstack([_mean_pool(vgg[f]) for f in films]).astype(np.float32)
    Y = np.array(
        [[int(g in genre_of[f]) for g in BLOCKBUSTER_GENRES] for f in films], dtype=int
    )
    return {
        "X_vggish": X_vggish,
        "X_mfcc": X_mfcc,
        "Y": Y,
        "films": films,
        "genres": BLOCKBUSTER_GENRES,
    }
