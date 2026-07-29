"""Extract & cache AST embeddings for Set 1 (and optionally Set 2).

Run from anywhere:  python notebooks/extract_features.py [--set2]

Outputs (under DATA_ROOT/processed/embeddings/):
  <set>/<number>.npy    per-clip 768-dim embedding (cache, resumable)
  <set>_ast.npy         (n_clips, 768) matrix in cleaned-DataFrame row order
  <set>_numbers.npy     clip numbers aligned to the matrix rows
  <set>_durations.csv   clip durations (seconds)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.features import load_set1, load_set2  # noqa: E402
from src.features import AstEmbedder, extract_embeddings  # noqa: E402
from src.utils import set_seed  # noqa: E402


def _run(df, set_name: str, embedder: AstEmbedder) -> None:
    t = time.time()
    X, durations = extract_embeddings(df, set_name, embedder=embedder)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.PROCESSED_DIR / f"{set_name}_ast.npy", X)
    np.save(config.PROCESSED_DIR / f"{set_name}_numbers.npy",
            df["number"].to_numpy(dtype=int))
    durations.to_csv(config.PROCESSED_DIR / f"{set_name}_durations.csv", index=False)
    print(f"[{set_name}] embeddings {X.shape} saved in {time.time() - t:.1f}s "
          f"-> {config.PROCESSED_DIR / f'{set_name}_ast.npy'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set2", action="store_true", help="also extract Set 2")
    args = ap.parse_args()

    set_seed()
    print("Loading AST model ...")
    embedder = AstEmbedder()

    # Extract ALL 360 Set 1 clips (clean=False): the genre experiments use the 346
    # cleaned subset, but Experiment 5 (TARGET) needs the full balanced 360. The
    # per-clip cache is keyed by clip number, so both assemble from the same cache.
    _run(load_set1(clean=False), "set1", embedder)
    if args.set2:
        _run(load_set2(), "set2", embedder)


if __name__ == "__main__":
    main()
