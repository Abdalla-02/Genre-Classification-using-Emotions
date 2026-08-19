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
)
from src.features.clap_extractor import CLAP_CHECKPOINT, CLAP_EMBED_DIM, ClapEmbedder
from src.features.loader import (
    add_derived_features,
    genre_matrix,
    load_set1,
    load_set2,
)

__all__ = [
    # dataset
    "load_set1",
    "load_set2",
    "add_derived_features",
    "genre_matrix",
    # embeddings
    "AstEmbedder",
    "ClapEmbedder",
    "extract_embeddings",
    "assemble_from_cache",
    "durations_from_audio",
    "AST_CHECKPOINT",
    "EMBED_DIM",
    "CLAP_CHECKPOINT",
    "CLAP_EMBED_DIM",
]
