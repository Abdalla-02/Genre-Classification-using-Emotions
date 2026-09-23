"""Feature extraction and dataset loading/cleaning.

(Data loading lives here rather than in a separate ``src/data`` package to match the
project structure documented in the README: src/features, src/models, src/evaluation.)
"""

from src.features.ast_extractor import (
    AST_CHECKPOINT,
    EMBED_DIM,
    AstEmbedder,
    assemble_from_cache,
    durations_from_audio,
    extract_embeddings,
    extract_window_embeddings,
    pool_windows,
)
from src.features.clap_extractor import CLAP_CHECKPOINT, CLAP_EMBED_DIM, ClapEmbedder
from src.features.mir_extractor import MirEmbedder
from src.features.vggish_extractor import VGGISH_EMBED_DIM, VggishEmbedder
from src.features.wav2vec_extractor import (
    W2V_CHECKPOINT,
    W2V_EMBED_DIM,
    Wav2VecEmbedder,
)
from src.features.loader import (
    add_derived_features,
    build_emotion_features,
    genre_matrix,
    load_set1,
    load_set1_shared,
    load_set2,
    shared_genre_matrix,
)

__all__ = [
    # dataset
    "load_set1",
    "load_set1_shared",
    "load_set2",
    "shared_genre_matrix",
    "add_derived_features",
    "build_emotion_features",
    "genre_matrix",
    # embeddings
    "AstEmbedder",
    "ClapEmbedder",
    "VggishEmbedder",
    "VGGISH_EMBED_DIM",
    "MirEmbedder",
    "Wav2VecEmbedder",
    "W2V_CHECKPOINT",
    "W2V_EMBED_DIM",
    "extract_embeddings",
    "assemble_from_cache",
    "extract_window_embeddings",
    "pool_windows",
    "durations_from_audio",
    "AST_CHECKPOINT",
    "EMBED_DIM",
    "CLAP_CHECKPOINT",
    "CLAP_EMBED_DIM",
]
