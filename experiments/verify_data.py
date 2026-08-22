"""Sanity-check the data pipeline: load + clean Set 1/2, verify counts, check audio.

Run from anywhere:  python experiments/verify_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.features import add_derived_features, genre_matrix, load_set1, load_set2  # noqa: E402


def _check_audio(df, label: str) -> None:
    missing = [p for p in df["audio_path"] if not Path(p).is_file()]
    status = "OK" if not missing else f"MISSING {len(missing)}"
    print(f"  audio files present: {status}")
    for p in missing[:5]:
        print(f"    - {p}")


def main() -> None:
    print(f"DATA_ROOT = {config.DATA_ROOT}")

    print("\n=== Set 1 (primary training data) ===")
    df1 = load_set1()  # asserts committed counts internally
    print(f"  clips={len(df1)}  soundtracks={df1['soundtrack'].nunique()}")
    Y = genre_matrix(df1)
    print(f"  target matrix shape={Y.shape}  positive labels={int(Y.sum())}"
          f"  avg labels/clip={Y.sum() / len(df1):.3f}")
    print("  multi-label positive counts:")
    for g in config.PRIMARY_GENRES:
        print(f"    {g:12} {int(df1[g].sum())}")
    df1 = add_derived_features(df1)
    assert all(c in df1.columns for c in config.FEATURE_COLS)
    print(f"  engineered features present ({len(config.FEATURE_COLS)}): {config.FEATURE_COLS}")
    _check_audio(df1, "set1")

    print("\n=== Set 2 (Set1-vs-Set2 diff analysis, Exp 4) ===")
    df2 = load_set2()
    print(f"  clips={len(df2)}  soundtracks={df2['soundtrack'].nunique()}")
    print("  multi-label positive counts:")
    for g in config.PRIMARY_GENRES:
        print(f"    {g:12} {int(df2[g].sum())}")
    _check_audio(df2, "set2")

    print("\nAll Set 1 assertions passed.")


if __name__ == "__main__":
    main()
