# Bachelor Thesis: Film-Genre Classification from Emotional Features of Soundtracks

Implementation for the bachelor thesis *"Filmgenre-Klassifikation anhand emotionaler
Merkmale von Soundtracks"* (German; "Film-genre classification from emotional features of
soundtracks"). Only the title is German: **the thesis and everything in this repository --
code, comments, documentation and the LaTeX sections under `docs/latex/` -- are in
English.** The project investigates whether emotional features extracted
from film soundtracks can predict film genre, and whether modelling emotion as an
interpretable intermediate step offers an advantage over classifying genre directly from
audio.

## Goal

Analyse audio-based data for film-genre and emotion recognition. The structure keeps data,
code, experiments, and results separate, reproducible, and easy to extend.

## Core Principles

- Reproducibility: pinned dependencies, fixed random seeds, cached embeddings.
- Isolated environment: dependencies live in a local virtual environment, not global.
- Clean structure: data, source code, experiments, documentation, and results separated.
- Scientific traceability: experiments are self-contained scripts and are documented in `docs/`.

## Requirements

- Python 3.11 or 3.12
- pip (dependencies in `requirements.txt`)
- Optional: NVIDIA GPU for faster embedding extraction (CPU works; embeddings are cached)

## Quick Start

Create and activate a virtual environment, then install dependencies.

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Project Structure

```text
Implementation/
├── requirements.txt          # Python dependencies (pip)
├── README.md                 # This file
├── data/
│   ├── raw/
│   │   ├── Eerola_DB/        # Audio + enriched rating CSVs (primary dataset)
│   │   └── Blockbuster_DB/   # Ma et al. 2021 VGGish/MIR features (second corpus)
│   └── processed/Eerola_DB/  # embeddings/<model>/<set>/ per-clip caches (committed)
├── src/
│   ├── config.py             # Paths, constants, expected dataset counts
│   ├── utils.py              # Reproducibility (seeding)
│   ├── features/             # Data loading/cleaning + AST/CLAP/VGGish/wav2vec2/MIR/Blockbuster features
│   ├── models/               # Multi-label genre classifiers
│   └── evaluation/           # Metrics + cross-validated scoring
├── experiments/              # Runnable experiments, mirroring the src/ layout
│   ├── features/             # Data checks + embedding extraction
│   ├── emotion/              # Stage 1: audio -> emotion
│   ├── genre/                # Stage 2: emotion -> genre
│   ├── diagnostics/          # Error analysis, thresholds, learning curve, clip length
│   ├── cross_dataset/        # Transfer to the Blockbuster dataset
│   └── evaluation/           # Repeated CV, confidence intervals, significance tests
├── docs/                     # current_state.md (latest supervisor meeting),
│                             # briefing_full.md (complete briefing), README.md
│                             # (technical log), literature reviews, thesis LaTeX
└── results/                  # Machine-readable experiment output (JSON) + plots
```

## Data

- Raw data is **git-ignored** (not redistributed) and lives under `data/raw/`:
  - `Eerola_DB/` — the primary dataset: clip audio + enriched rating CSVs.
  - `Blockbuster_DB/` — Ma et al. (2021): 110 films with pre-extracted VGGish and MIR
    features per music cue, but no audio (copyright). It is the second corpus: the
    zero-shot test (train on Eerola, test here), cross-validation in both directions,
    the replication of the emotion-genre signatures, and the in-domain reproduction of
    the published result (`experiments/cross_dataset/`).
- Derived data (per-clip embedding caches) is written to
  `data/processed/Eerola_DB/embeddings/<model>/<set>/` and reused across runs — never
  re-extracted per run. It is committed, so experiments run straight after a clone.
  See `data/raw/README.md` for the raw-data download.
- `src/config.py` resolves the data root automatically; set `THESIS_DATA_ROOT` to
  override the Eerola root, or `BLOCKBUSTER_DIR` to override the Blockbuster location.

## Running

Experiments are plain Python scripts, runnable from the project root, e.g.:

```bash
python experiments/features/verify_data.py              # load + clean + assert dataset counts
python experiments/evaluation/exp_cv_corrected.py       # the headline Eerola numbers (~27 min)
python experiments/cross_dataset/exp_zero_shot.py       # train on Eerola, test on Blockbuster (zero-shot)
python experiments/diagnostics/exp_box_office.py        # box office vs the soundtrack (exploratory)
python experiments/show_results.py                      # list every saved result; render one as tables
```

**The command for every experiment**, grouped and with what each needs, is in
`results/README.md` ("Running the experiments").

Where to read:

- `docs/current_state.md` — the update for the latest supervisor meeting.
- `docs/briefing_full.md` — the complete supervisor briefing: every round of notes, the
  definitions and all results.
- `docs/README.md` — the full technical log, one section per experiment.
- `results/README.md` — what each machine-readable results file contains and how to
  render it.

## Reproducibility

1. Dependencies are pinned in `requirements.txt`.
2. Random seeds are fixed everywhere (`SEED = 42`, see `src/utils.py`).
3. Per-clip embeddings are cached to `data/processed/` and committed, so a clone
   reproduces every result without re-running any model over the audio.
4. Cross-validation is grouped by film (`GroupKFold` on `soundtrack`) to prevent leakage.

## Dependencies and Notes

The stack is PyTorch-only by design (AST, CLAP and wav2vec 2.0 via `transformers`, VGGish
via `torchvggish`; the hand-crafted MIR baseline uses `librosa` and needs no framework), so all
learned components share a single deep-learning framework. TensorFlow-based models are
deliberately kept out of the main environment to prevent dependency conflicts. The one
exception, the music-tagging model MusiCNN, runs in its own `.venv-musicnn` (Python 3.11
+ TensorFlow) and talks to the project only through the embedding cache: nothing in
`src/` imports TensorFlow.
