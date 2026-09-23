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

# All cached embeddings live under one root, one sub-directory per model, each holding
# per-clip caches as <model>/<set>/<number>.npy:
#
#   embeddings/ast/set1/001.npy          768-d AST, the primary representation
#   embeddings/clap/set1/001.npy         512-d CLAP baseline
#   embeddings/vggish/set1/001.npy       128-d VGGish (the cross-dataset bridge)
#   embeddings/mir/set1/001.npy          103-d hand-crafted librosa MIR
#   embeddings/wav2vec2/set1/001.npy     768-d wav2vec 2.0, the waveform-domain baseline
#   embeddings/musicnn/set1/001.npy      200-d MusiCNN (music-tagging pretraining)
#   embeddings/wav2vec2/layers_set1/001.npy  (13, 768) all layers, git-ignored (large);
#                                        used only by the layer-sweep control
#   embeddings/ast_windows/set1/001.npy  (n_windows, 768) stack, git-ignored (large)
#
# These per-clip files are the SINGLE SOURCE OF TRUTH: every experiment reads them via
# ``assemble_from_cache``. No concatenated matrix is written alongside them -- an earlier
# layout did, and the copy silently drifted out of sync with the caches it was derived
# from while nothing actually read it.
EMBEDDINGS_ROOT = PROCESSED_DIR / "embeddings"
EMBEDDINGS_DIR = EMBEDDINGS_ROOT / "ast"           # primary (AST, 768-d)
CLAP_EMBEDDINGS_DIR = EMBEDDINGS_ROOT / "clap"     # CLAP baseline (512-d)
# VGGish (128-d, postprocessed to match Blockbuster's VGGish space) -- the shared
# feature used to bridge Eerola and Blockbuster for cross-dataset emotion prediction.
VGGISH_EMBEDDINGS_DIR = EMBEDDINGS_ROOT / "vggish"
# librosa hand-crafted MIR features (MFCC/chroma/spectral/...) -- a classic-MER baseline
# for emotion regression (NOT Blockbuster's MATLAB MIR, so not a cross-dataset bridge).
MIR_EMBEDDINGS_DIR = EMBEDDINGS_ROOT / "mir"
# wav2vec 2.0 (768-d) -- the WAVEFORM-domain baseline. Every other representation here
# is computed from a spectrogram; wav2vec 2.0 convolves the raw sample sequence, so it
# separates "learned embeddings do not beat emotion features" from "mel spectrograms do
# not beat emotion features" (supervisor request #3).
W2V_EMBEDDINGS_DIR = EMBEDDINGS_ROOT / "wav2vec2"
# MusiCNN (200-d penultimate layer) -- the only representation pretrained on MUSIC
# tagging rather than general audio events or speech. Extracted by a SEPARATE TensorFlow
# environment (experiments/features/extract_musicnn.py); nothing in src/ imports
# TensorFlow, the two sides meet only at this cache directory.
MUSICNN_EMBEDDINGS_DIR = EMBEDDINGS_ROOT / "musicnn"
# AST per-window embeddings: an (n_windows, 768) stack per clip (10.24 s windows, 50%
# overlap) so full-clip pooling strategies (first/center/mean/max) can be compared
# without re-running AST.
WINDOW_EMBEDDINGS_DIR = EMBEDDINGS_ROOT / "ast_windows"
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

# The curated 5-genre reporting subset: the genres that are both established at this
# corpus scale in the literature (Austin 2010: 4 genres; Ma 2021: 6 from 110 films) and
# carry a measurable emotional signature here. Adventure, Biography and Documentary are
# excluded -- the first has no emotional signature (|d|<=0.25), the other two have n<25.
# A clip joins the subset if it carries at least one of these five ("any-present" rule,
# mirroring the 8-genre inclusion rule) -> 329 clips from 41 films.
GENRE_SUBSET = ["Action", "Crime", "Drama", "Comedy", "Horror"]

# The SHARED genre space with the Blockbuster corpus (Ma et al. 2021) -- the label space
# used for every cross-dataset comparison, so that "train on Eerola, test on Blockbuster"
# is a fair test of the same classifier on the same classes (supervisor request #4).
#
# These are exactly Ma et al.'s six reduced genres. Their reduction rule was recovered
# from their published film_genre_master_list.csv and is the *identity* rule: a film
# carries reduced genre g iff g appears in its raw IMDb genre list (verified on 110/110
# films, zero mismatches). That is the same "any-present" rule already used for the
# 8-genre Eerola target, so the two corpora can be relabelled into one space without any
# dataset-specific heuristic.
#
# Order is fixed and shared by both datasets' target matrices -- never reorder, or the
# columns of a model trained on one corpus stop meaning the same thing on the other.
SHARED_GENRES = ["action", "drama", "comedy", "sci-fi", "romance", "horror"]

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

# Expected counts for Set 1 relabelled into SHARED_GENRES (any-present rule).
# Asserted in the loader like the 8-genre counts, so drift fails loudly.
SET1_SHARED_EXPECTED_CLIPS = 319
SET1_SHARED_EXPECTED_SOUNDTRACKS = 41
SET1_SHARED_LABEL_COUNTS = {
    "action": 103,
    "drama": 239,
    "comedy": 40,
    "sci-fi": 19,
    "romance": 51,
    "horror": 28,
}
