"""Ma et al. (2021) Blockbuster dataset -- features only (no audio, copyright).

Each of the 110 films is a *bag* of soundtrack cues (Multiple-Instance-Learning); each
cue is a per-frame feature time series. For a simple, fair baseline we mean-pool over
all frames of all cues into one fixed vector per film, for two feature sets:

  * VGGish (128-dim learned audio embedding)
  * MFCC   (78 of the 140 hand-crafted MIR features: mfcc + delta + deltadelta, mean/std)

Genre is multi-label over the 6 reduced genres used in the paper. This supports the
supervisor's "compare VGGish vs MFCC" task and mirrors Ma et al.'s finding that the
learned embedding does not clearly beat classic MFCC features.

``load_blockbuster_cues`` keeps the bag structure instead of collapsing it. That matters
for two reasons. First, mean-pooling a whole film discards the MIL structure that Ma et
al. actually model. Second, and more important here: a cue is a single piece of film
music of median 19 s, i.e. the SAME unit as an Eerola clip (10-31 s, mean 17 s), whereas a
film-level mean over ~39 cues is a much smoother object. Any model trained on Eerola
clips and applied at film level therefore suffers a domain shift that disappears when it
is applied per cue and aggregated afterwards.
"""

from __future__ import annotations

import csv

import numpy as np

from src import config

# Ma et al.'s six reduced genres. Defined once in config as SHARED_GENRES because the
# Eerola target is relabelled into the *same* list, in the same order, for the
# cross-dataset experiments -- a reordering here would silently break that alignment.
BLOCKBUSTER_GENRES = config.SHARED_GENRES
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


# --------------------------------------------------------------------------- #
# Cue-level (bag-preserving) access
# --------------------------------------------------------------------------- #
# One film's VGGish and MIR bags normally hold the same cues, but they are stored
# independently and for one film ('ready_player_one') the two disagree (42 vs 71 cues).
# The two feature sets are therefore returned with their OWN film index rather than
# force-aligned, so no data is dropped and no cue is silently paired with the wrong one.
CUE_COUNT_MISMATCH = ("ready_player_one",)


def _cue_matrix(bag_of, films):
    """Mean-pool frames within each cue -> (n_cues, dim) plus the film index per cue."""
    rows, idx = [], []
    for i, f in enumerate(films):
        for cue in bag_of[f]:
            rows.append(np.asarray(cue, dtype=np.float64).mean(axis=0))
            idx.append(i)
    return np.vstack(rows).astype(np.float32), np.asarray(idx, dtype=int)


def load_blockbuster_cues() -> dict:
    """Return the corpus at CUE level, the unit comparable to an Eerola clip.

    Keys: ``X_vggish``/``cue_film_vggish`` (4664 cues), ``X_mir``/``cue_film_mir``
    (4693 cues, 140 features), ``Y`` (110, 6) film-level labels, ``films``, ``genres``.
    Each ``cue_film_*`` entry indexes into ``films``/``Y``, so a cue's bag label is
    ``Y[cue_film_x[j]]`` and predictions are aggregated back per film with that index.
    """
    bd = _require_dir()
    mir = np.load(bd / "mir_features_dictionary.npy", allow_pickle=True).item()
    vgg = np.load(bd / "vggish_features_dictionary.npy", allow_pickle=True).item()

    genre_of = {}
    with open(bd / "film_genre_master_list.csv", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            genre_of[row[0]] = [g.strip() for g in row[2].split(",") if g.strip()]
    films = sorted(set(genre_of) & set(mir) & set(vgg))

    Xv, iv = _cue_matrix(vgg, films)
    Xm, im = _cue_matrix(mir, films)
    Y = np.array(
        [[int(g in genre_of[f]) for g in BLOCKBUSTER_GENRES] for f in films], dtype=int
    )
    return {
        "X_vggish": Xv, "cue_film_vggish": iv,
        "X_mir": Xm, "cue_film_mir": im,
        "Y": Y, "films": films, "genres": BLOCKBUSTER_GENRES,
    }


def mir_feature_names() -> list[str]:
    """The 140 hand-crafted MIR feature names, in column order."""
    bd = _require_dir()
    return open(bd / "mir_feature_names.csv").read().strip().split("	")


def aggregate_cues(values: np.ndarray, cue_film: np.ndarray, n_films: int,
                   how: str = "mean") -> np.ndarray:
    """Pool a per-cue quantity back to one row per film (Ma et al.'s Simple-MI step)."""
    out = np.zeros((n_films, values.shape[1]), dtype=float)
    for i in range(n_films):
        sel = values[cue_film == i]
        out[i] = sel.mean(axis=0) if how == "mean" else sel.max(axis=0)
    return out
