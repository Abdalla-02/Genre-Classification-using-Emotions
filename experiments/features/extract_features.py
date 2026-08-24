"""Extract & cache audio embeddings for Set 1 (and optionally Set 2).

Run from anywhere:
  python experiments/features/extract_features.py                       # AST (primary), Set 1
  python experiments/features/extract_features.py --model clap --set2   # CLAP, both sets
  python experiments/features/extract_features.py --model vggish        # VGGish (cross-dataset bridge)
  python experiments/features/extract_features.py --model mir           # hand-crafted MIR

Outputs (under DATA_ROOT/processed/Eerola_DB/):
  embeddings[_clap|_vggish|_mir]/<set>/<number>.npy   per-clip embedding (cache)
  <set>_<model>.npy                      (n_clips, dim) matrix in cleaned-DataFrame order
  <set>_numbers.npy                      clip numbers aligned to the matrix rows
  <set>_durations.csv                    clip durations (seconds; model-independent)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import (  # noqa: E402
    AstEmbedder,
    ClapEmbedder,
    MirEmbedder,
    VggishEmbedder,
    extract_embeddings,
    load_set1,
    load_set2,
)
from src.utils import set_seed  # noqa: E402

# model tag -> (embedder class, per-clip cache dir, display label)
# All four extractors share the AstEmbedder interface (load_audio / embed_waveform /
# sampling_rate), so they all drop into extract_embeddings unchanged. VGGish and MIR were
# previously reachable only from ad-hoc code, which left their caches -- consumed by the
# emotion-regression and cross-dataset experiments -- unreproducible from the repository.
MODELS = {
    "ast": (AstEmbedder, config.EMBEDDINGS_DIR, "AST"),
    "clap": (ClapEmbedder, config.CLAP_EMBEDDINGS_DIR, "CLAP"),
    "vggish": (VggishEmbedder, config.VGGISH_EMBEDDINGS_DIR, "VGGish"),
    "mir": (MirEmbedder, config.MIR_EMBEDDINGS_DIR, "MIR"),
}


def _all_cached(df, set_name: str, cache_dir: Path) -> bool:
    clip_dir = cache_dir / set_name
    return all((clip_dir / f"{int(n):03d}.npy").is_file() for n in df["number"])


def _run(df, set_name: str, model: str, embedder) -> None:
    _cls, cache_dir, label = MODELS[model]
    t = time.time()
    X, durations = extract_embeddings(
        df, set_name, embedder=embedder, cache_dir=cache_dir, label=label
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.PROCESSED_DIR / f"{set_name}_{model}.npy", X)
    np.save(config.PROCESSED_DIR / f"{set_name}_numbers.npy",
            df["number"].to_numpy(dtype=int))
    durations.to_csv(config.PROCESSED_DIR / f"{set_name}_durations.csv", index=False)
    print(f"[{model}/{set_name}] embeddings {X.shape} in {time.time() - t:.1f}s "
          f"-> {config.PROCESSED_DIR / f'{set_name}_{model}.npy'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="ast")
    ap.add_argument("--set2", action="store_true", help="also extract Set 2")
    args = ap.parse_args()
    set_seed()

    cls, cache_dir, label = MODELS[args.model]
    # Set 1 uses all 360 clips (clean=False): genre experiments use the 346-clip subset,
    # but Experiment 5 (TARGET) needs the full balanced 360. Cache is keyed by clip
    # number, so both assemble from the same cache.
    sets = [(load_set1(clean=False), "set1")]
    if args.set2:
        sets.append((load_set2(), "set2"))

    # Load the (large) model only if something still needs extracting.
    need = any(not _all_cached(df, name, cache_dir) for df, name in sets)
    embedder = None
    if need:
        print(f"Loading {label} model ...")
        embedder = cls()

    for df, name in sets:
        _run(df, name, args.model, embedder)


if __name__ == "__main__":
    main()
