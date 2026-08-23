"""Central configuration: paths, constants, genre/emotion definitions, expected counts.

The raw data (audio + enriched CSVs) lives only in the main checkout's ``data/``
folder and is intentionally NOT committed to git. Because code may run from a git
worktree (whose path does not contain ``data/``), :data:`DATA_ROOT` is resolved by
walking up the directory tree until a ``data/raw/Eerola_DB`` folder is found. Set the
``THESIS_DATA_ROOT`` environment variable to override.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
_EEROLA_REL = Path("raw") / "Eerola_DB"


def _resolve_data_root() -> Path:
    """Return the ``data/`` directory that contains ``raw/Eerola_DB``.

    Search order: ``$THESIS_DATA_ROOT`` -> every ancestor of this file
    (``<ancestor>/data/raw/Eerola_DB``). Raises if nothing matches so failures
    are loud rather than silent.
    """
    env = os.environ.get("THESIS_DATA_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "data")
    for cand in candidates:
        if (cand / _EEROLA_REL).is_dir():
            return cand.resolve()
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate the data directory (expected a "
        "'data/raw/Eerola_DB' folder). Set THESIS_DATA_ROOT to the absolute "
        "path of your 'data' folder. Searched:\n  " + searched
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = _resolve_data_root()
EEROLA_DIR = DATA_ROOT / _EEROLA_REL


def _resolve_blockbuster_dir():
    """Locate the Ma et al. (2021) Blockbuster supplement (features only, no audio).

    Primary location is ``data/raw/Blockbuster_DB`` (parallels ``data/raw/Eerola_DB``);
    ``$BLOCKBUSTER_DIR`` overrides, and a legacy fallback to a sibling 'Related Work' tree
    is kept for backward compatibility. Returns None if not found (the loader raises a
    clear error only when actually used).
    """
    env = os.environ.get("BLOCKBUSTER_DIR")
    legacy = Path("Related Work") / "Blockbuster-Dataset" / "journal.pone.0249957.s004"
    candidates = [
        Path(env) if env else None,
        DATA_ROOT / "raw" / "Blockbuster_DB",     # in-project, parallels Eerola_DB
        DATA_ROOT.parent.parent / legacy,          # legacy: sibling 'Related Work' tree
    ]
    for cand in candidates:
        if cand and (cand / "mir_feature_names.csv").is_file():
            return cand.resolve()
    return None


BLOCKBUSTER_DIR = _resolve_blockbuster_dir()

SET1_CSV = EEROLA_DIR / "mean_ratings_set1_enriched.csv"
SET2_CSV = EEROLA_DIR / "mean_ratings_set2_enriched.csv"
AUDIO_SET1 = EEROLA_DIR / "audio" / "set1"
AUDIO_SET2 = EEROLA_DIR / "audio" / "set2"
AUDIO_1MIN = EEROLA_DIR / "audio" / "1min"

# Derived/output locations. Processed data mirrors the raw layout
# (data/processed/Eerola_DB/...) so each dataset's derivatives sit together; embedding
# caches can be large -> kept next to the (git-ignored) data. Results live in the repo.
PROCESSED_DIR = DATA_ROOT / "processed" / "Eerola_DB"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"  # AST per-clip caches: embeddings/<set>/<n>.npy
CLAP_EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings_clap"  # CLAP baseline, same layout
# VGGish (128-d, postprocessed to match Blockbuster's VGGish space) -- the shared
# feature used to bridge Eerola and Blockbuster for cross-dataset emotion prediction.
VGGISH_EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings_vggish"
# librosa hand-crafted MIR features (MFCC/chroma/spectral/...) -- a classic-MER baseline
# for emotion regression (NOT Blockbuster's MATLAB MIR, so not a cross-dataset bridge).
MIR_EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings_mir"
# AST per-window embeddings: embeddings_ast_windows/<set>/<n>.npy holds an
# (n_windows, 768) stack (10.24 s windows, 50% overlap) so full-clip pooling
# strategies (first/center/mean/max) can be compared without re-running AST.
WINDOW_EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings_ast_windows"
RESULTS_DIR = REPO_ROOT / "results"

# --------------------------------------------------------------------------- #
# Emotions (8 dimensions, scale 1-9) and engineered features
# --------------------------------------------------------------------------- #
EMOTIONS = ["valence", "energy", "tension", "anger", "fear", "happy", "sad", "tender"]

DERIVED_FEATURES = ["valence_x_energy", "neg_composite", "pos_composite"]
FEATURE_COLS = EMOTIONS + DERIVED_FEATURES  # 11 features into the genre stage

# --------------------------------------------------------------------------- #
# Genres (multi-label). Order is fixed and used for the target-matrix columns.
# --------------------------------------------------------------------------- #
PRIMARY_GENRES = [
    "Action",
    "Crime",
    "Drama",
    "Adventure",
    "Comedy",
    "Biography",
    "Documentary",
    "Horror",
]

# Genres present in the raw IMDb annotations but excluded from modelling because
# they are too rare (documented in the thesis). A clip is dropped only if it has
# NO primary genre at all (see loader: "any-primary-present" inclusion rule).
RARE_GENRES = ["Fantasy", "Mystery"]

# --------------------------------------------------------------------------- #
# Expected counts after cleaning Set 1 (any-primary-present rule -> 346 clips).
# These are asserted in the loader so silent data drift fails loudly.
# --------------------------------------------------------------------------- #
SET1_EXPECTED_CLIPS = 346
SET1_EXPECTED_SOUNDTRACKS = 43

# AUTHORITATIVE genre distribution: multi-label positive counts (column sums of the
# target matrix). Use THIS for the thesis dataset table.
SET1_LABEL_COUNTS = {
    "Action": 103,
    "Crime": 98,
    "Drama": 239,
    "Adventure": 97,
    "Comedy": 40,
    "Biography": 24,
    "Documentary": 17,
    "Horror": 28,
}

# Distribution of the alphabetically-FIRST modelled genre per clip. IMPORTANT: IMDb
# stores genres ~alphabetically (342/348 Set 1 clips are in strict alphabetical
# order), NOT by relevance, so this is a biased view (early-alphabet genres are
# over-represented) and must NOT be presented as a "dominant/primary genre"
# distribution. Retained only as a deterministic data-integrity check in the loader.
SET1_FIRST_GENRE_COUNTS = {
    "Action": 103,
    "Crime": 73,
    "Drama": 45,
    "Adventure": 35,
    "Comedy": 32,
    "Biography": 18,
    "Documentary": 17,
    "Horror": 23,
}

SET2_EXPECTED_CLIPS = 102  # 110 total: -6 no genre, -2 with no primary genre
