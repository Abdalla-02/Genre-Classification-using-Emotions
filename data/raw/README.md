# `data/raw/` — not in version control

**The contents of this directory are deliberately excluded from git** (see `.gitignore`:
`data/raw/*/*`). The audio corpus alone is ~208 MB, which is far past what belongs in a
repository, and the soundtrack excerpts are third-party material that is not ours to
redistribute. Only this note is tracked, so the directory itself survives a clone.

## Download

The data is available here:

**https://cloud.ovgu.de/s/FL7ogdmrJB5nKHQ**

Unpack it so that this directory ends up looking exactly like the layout below — the
loader asserts the expected counts on startup, so a wrong layout fails loudly rather than
silently producing different numbers.

## Expected layout

```
data/raw/
├── README.md                          <- this file (the only tracked item)
├── Eerola_DB/                         <- primary dataset (Eerola & Vuoskoski 2011)
│   ├── audio/
│   │   ├── set1/     360 .mp3         <- primary training data
│   │   ├── set2/     110 .mp3         <- re-rated subset, used for Experiment 4
│   │   └── 1min/      16 .mp3
│   ├── mean_ratings_set1.csv          <- original ratings, unmodified
│   ├── mean_ratings_set1_enriched.csv <- + IMDb id / genres / poster  (READ BY THE CODE)
│   ├── mean_ratings_set2.csv
│   └── mean_ratings_set2_enriched.csv
└── Blockbuster_DB/                    <- external baseline (Ma et al. 2021)
    └── ...                            <- pre-extracted VGGish + MIR features, no audio
```

Notes:

- The pipeline reads the **`_enriched`** CSVs; the plain ones are kept unchanged as
  provenance. Do not delete them.
- `Blockbuster_DB/` is the feature supplement published with Ma et al. (2021) — features
  only, no audio (the paper could not redistribute it for copyright reasons). It is
  recognised by the presence of `mir_feature_names.csv`.
- Do **not** nest a second `Eerola_DB/` inside `Eerola_DB/`. An extracted archive once
  produced exactly that, silently doubling the folder to 416 MB; the duplicate was
  byte-identical and unused, since the code only ever reads the outer copy.

## Paths

`src/config.py` locates this directory automatically by walking up from the source tree
until it finds `data/raw/Eerola_DB`. Two environment variables override that if your data
lives elsewhere:

- `THESIS_DATA_ROOT` — absolute path to the `data/` directory
- `BLOCKBUSTER_DIR` — absolute path to the Blockbuster feature supplement

## Verifying

After unpacking, confirm the dataset is intact:

```bash
python experiments/features/verify_data.py
```

It checks clip counts (Set 1 cleans to 346 clips from 43 soundtracks), the per-genre
multi-label totals, and that every referenced audio file exists.

Derived data — cached embeddings and feature matrices — is written to
`data/processed/` and *is* tracked, so the experiments run without re-extracting anything.
