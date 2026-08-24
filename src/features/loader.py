"""Loading and cleaning of the Eerola & Vuoskoski (2011) film-soundtrack dataset.

Public API
----------
load_set1(clean=True)  -> cleaned Set 1 DataFrame (primary training data, 346 clips)
load_set2(clean=True)  -> cleaned Set 2 DataFrame (Set-1-vs-Set-2 diff analysis, Exp 4)
add_derived_features(df) -> df with the 3 engineered features (11 total)
genre_matrix(df)       -> (n_clips, 8) int ndarray of multi-label targets

Cleaning rules (see thesis / memory ``genre-labeling-scheme``):
  * drop clips with no IMDb genre;
  * keep a clip if ANY of its IMDb genres is one of the 8 primary genres
    ("any-primary-present" rule -> 346 clips for Set 1);
  * genre is MULTI-LABEL: the target is a binary vector over the 8 primary genres.

Note on genre ORDER: the IMDb ``genres`` strings are stored ~alphabetically
(342/348 Set 1 clips are in strict alphabetical order), NOT by relevance/dominance.
The multi-label target and the "any-primary-present" inclusion rule are both
order-independent, so this does not affect modelling. The only order-dependent field
is the descriptive ``first_genre`` column (the alphabetically-first primary genre),
which is kept ONLY as a data-integrity check and must NOT be read as a "dominant
genre" -- the authoritative genre distribution is the multi-label column sums.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Map the canonical lowercase emotion name -> the various header spellings seen
# across the two CSVs (Set 1 is lowercase, Set 2 is capitalised).
_EMOTION_ALIASES = {e: {e, e.capitalize()} for e in config.EMOTIONS}
_NUMBER_ALIASES = {"number", "Number"}


def _split_genres(cell: object) -> list[str]:
    """Split a ';'-separated IMDb genre string into a clean list."""
    if pd.isna(cell):
        return []
    return [g.strip() for g in str(cell).split(";") if g.strip()]


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ``Number``/emotion columns to their canonical lowercase names."""
    rename: dict[str, str] = {}
    for col in df.columns:
        if col in _NUMBER_ALIASES:
            rename[col] = "number"
        else:
            for canon, aliases in _EMOTION_ALIASES.items():
                if col in aliases:
                    rename[col] = canon
                    break
    return df.rename(columns=rename)


def _add_genre_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``genre_list``, ``primary_genres`` (the modelled genres present, a set-like
    order-independent list), ``first_genre`` (descriptive only) and the 8 binary
    target columns. Assumes ``genres`` is already non-null."""
    df = df.copy()
    df["genre_list"] = df["genres"].apply(_split_genres)
    # order-independent: which of the 8 modelled genres are present on this clip
    df["primary_genres"] = df["genre_list"].apply(
        lambda gl: [g for g in gl if g in config.PRIMARY_GENRES]
    )
    # descriptive ONLY: alphabetically-first modelled genre (IMDb order ~alphabetical,
    # NOT relevance) -- do not use as a "dominant genre".
    df["first_genre"] = df["primary_genres"].apply(lambda gl: gl[0] if gl else None)
    for genre in config.PRIMARY_GENRES:
        df[genre] = df["primary_genres"].apply(lambda gl, g=genre: int(g in gl))
    return df


def _audio_path(number: int, audio_dir) -> str:
    return str(audio_dir / f"{int(number):03d}.mp3")


def genre_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the (n_clips, 8) multi-label target matrix in ``PRIMARY_GENRES`` order."""
    return df[config.PRIMARY_GENRES].to_numpy(dtype=int)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 3 engineered features, yielding the 11-feature genre-stage input."""
    df = df.copy()
    df["valence_x_energy"] = df["valence"] * df["energy"]
    df["neg_composite"] = df[["anger", "fear", "tension", "sad"]].mean(axis=1)
    df["pos_composite"] = df[["happy", "tender", "valence"]].mean(axis=1)
    return df


def _load(csv_path, audio_dir, clean: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _canonicalise_columns(df)

    missing = [c for c in config.EMOTIONS + ["number", "genres", "soundtrack"]
               if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing expected columns {missing}")

    # Drop rows with no clip number (e.g. trailing blank rows that spreadsheet editors
    # append), then make the key a clean int. Guards the whole pipeline against such
    # junk rows.
    df = df[df["number"].notna()].copy()
    df["number"] = df["number"].astype(int)

    # normalise the discrete-emotion TARGET label (strip stray whitespace, e.g.
    # "HIGH TENSION " -> "HIGH TENSION") and expose the audio path for every clip.
    if "TARGET" in df.columns:
        df["TARGET"] = df["TARGET"].astype(str).str.strip()
    df["audio_path"] = df["number"].apply(lambda n: _audio_path(n, audio_dir))

    if not clean:
        return df.reset_index(drop=True)

    # 1) drop clips with no IMDb genre
    df = df[df["genres"].notna()].copy()
    df = _add_genre_columns(df)
    # 2) any-primary-present inclusion rule (order-independent)
    df = df[df["primary_genres"].map(len) > 0].copy()

    return df.reset_index(drop=True)


def _assert_counts(df: pd.DataFrame) -> None:
    """Fail loudly if Set 1 cleaning drifts from the committed numbers."""
    n = len(df)
    assert n == config.SET1_EXPECTED_CLIPS, (
        f"Set 1 clip count {n} != expected {config.SET1_EXPECTED_CLIPS}"
    )
    n_st = df["soundtrack"].nunique()
    assert n_st == config.SET1_EXPECTED_SOUNDTRACKS, (
        f"Set 1 soundtrack count {n_st} != expected {config.SET1_EXPECTED_SOUNDTRACKS}"
    )
    for genre, expected in config.SET1_LABEL_COUNTS.items():
        got = int(df[genre].sum())
        assert got == expected, f"Set 1 label count {genre}={got} != expected {expected}"
    first = df["first_genre"].value_counts().to_dict()
    for genre, expected in config.SET1_FIRST_GENRE_COUNTS.items():
        got = int(first.get(genre, 0))
        assert got == expected, (
            f"Set 1 first-genre count {genre}={got} != expected {expected}"
        )


def load_set1(clean: bool = True, verify: bool = True) -> pd.DataFrame:
    """Load (and by default clean + verify) Set 1 -- the primary training data."""
    df = _load(config.SET1_CSV, config.AUDIO_SET1, clean=clean)
    if clean and verify:
        _assert_counts(df)
    return df


def load_set2(clean: bool = True) -> pd.DataFrame:
    """Load (and by default clean) Set 2 -- used for the Set-1-vs-Set-2 diff (Exp 4).

    Set 2 additionally carries ``beauty`` and ``liking`` ratings; they are kept as
    columns but are not part of the 8-emotion feature set shared with Set 1.
    """
    return _load(config.SET2_CSV, config.AUDIO_SET2, clean=clean)


# --------------------------------------------------------------------------- #
# Set 1 <-> Set 2 correspondence (Experiment 4)
# --------------------------------------------------------------------------- #
# Set 2 re-rates excerpts drawn from Set 1, but the two CSVs use INDEPENDENT clip
# numbering: Set 2's ``number`` is unrelated to Set 1's. The correspondence lives in
# Set 2's ``link`` column, which holds the Set 1 ``number``.
#
# This was verified against the audio (not just the metadata): full-lag normalised
# cross-correlation between each Set 2 clip and its linked Set 1 clip gives a median of
# 0.964 (10th pct 0.792), against a median of 0.027 for randomly paired clips -- 109 of
# 110 rows sit far above the null distribution's 99th percentile. Joining on ``number``
# instead produces a table that looks valid but is meaningless (mean rating correlation
# 0.157; soundtrack labels agree for 2.7% of rows).
LINK_MISMATCH_SET2 = [17]  # only row whose audio does not match its link (see above)


def align_sets(set1: pd.DataFrame | None = None,
               set2: pd.DataFrame | None = None,
               drop_mismatches: bool = True,
               average_repeats: bool = True) -> pd.DataFrame:
    """Join Set 2 onto Set 1 via ``link``; return one row per matched excerpt.

    Emotion columns are suffixed ``_set1`` / ``_set2``. ``beauty``/``liking`` (Set 2 only)
    are carried through unsuffixed.

    Seven Set 1 clips are linked by TWO Set 2 rows each -- repeat trials presented twice
    to the same panel. ``average_repeats=True`` averages them (one row per excerpt, the
    right unit for a between-panel comparison); ``False`` keeps them, which is what
    :func:`repeat_pairs` needs for the within-panel estimate.
    """
    s1 = load_set1(clean=False) if set1 is None else set1
    s2 = load_set2(clean=False) if set2 is None else set2
    if "link" not in s2.columns:
        raise ValueError("Set 2 has no 'link' column - cannot align to Set 1")

    s2 = s2[s2["link"].notna()].copy()
    s2["link"] = s2["link"].astype(int)
    if drop_mismatches:
        s2 = s2[~s2["number"].isin(LINK_MISMATCH_SET2)]

    s1i = s1.set_index("number")
    missing = sorted(set(s2["link"]) - set(s1i.index))
    if missing:
        raise ValueError(f"Set 2 links not present in Set 1: {missing}")

    cols = config.EMOTIONS
    out = pd.DataFrame({"set1_number": s2["link"].to_numpy(),
                        "set2_number": s2["number"].to_numpy()})
    for c in cols:
        out[f"{c}_set1"] = s1i.loc[s2["link"], c].to_numpy(dtype=float)
        out[f"{c}_set2"] = s2[c].to_numpy(dtype=float)
    for extra in ("Beauty", "Liking", "beauty", "liking"):
        if extra in s2.columns:
            out[extra.lower()] = pd.to_numeric(s2[extra], errors="coerce").to_numpy()
    out["soundtrack"] = s1i.loc[s2["link"], "soundtrack"].to_numpy()

    if average_repeats:
        agg = {c: "mean" for c in out.columns
               if c not in ("set1_number", "set2_number", "soundtrack")}
        agg["soundtrack"] = "first"
        out = out.groupby("set1_number", as_index=False).agg(agg)
    return out.reset_index(drop=True)


def repeat_pairs(set2: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the Set 2 repeat trials: excerpts rated twice by the SAME panel.

    Seven Set 1 clips are linked by two Set 2 rows each (after dropping the one audio
    mismatch). Comparing those two ratings isolates *within-panel* measurement noise,
    whereas Set 1 vs Set 2 additionally contains the between-panel difference.
    Columns are suffixed ``_a`` / ``_b``.
    """
    s2 = load_set2(clean=False) if set2 is None else set2
    s2 = s2[s2["link"].notna()].copy()
    s2["link"] = s2["link"].astype(int)
    s2 = s2[~s2["number"].isin(LINK_MISMATCH_SET2)]
    counts = s2["link"].value_counts()
    rows = []
    for link in sorted(counts[counts == 2].index):
        a, b = (s2[s2["link"] == link].iloc[0], s2[s2["link"] == link].iloc[1])
        row = {"set1_number": link, "set2_a": int(a["number"]), "set2_b": int(b["number"]),
               "soundtrack": a["soundtrack"]}
        for c in config.EMOTIONS:
            row[f"{c}_a"] = float(a[c]); row[f"{c}_b"] = float(b[c])
        rows.append(row)
    return pd.DataFrame(rows)
