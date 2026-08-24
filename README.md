# Bachelor Thesis: Film-Genre Classification from Emotional Features of Soundtracks

Implementation for the bachelor thesis *"Filmgenre-Klassifikation anhand emotionaler
Merkmale von Soundtracks."* The project investigates whether emotional features extracted
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
│   │   └── Blockbuster_DB/   # Ma et al. 2021 VGGish/MIR features (external baseline)
│   └── processed/Eerola_DB/  # Cached embeddings (AST, CLAP), matrices, durations
├── src/
│   ├── config.py             # Paths, constants, expected dataset counts
│   ├── utils.py              # Reproducibility (seeding)
│   ├── features/             # Data loading/cleaning + AST/CLAP/Blockbuster feature extraction
│   ├── models/               # Multi-label genre classifiers
│   └── evaluation/           # Metrics + cross-validated scoring
├── experiments/              # Runnable experiments, mirroring the src/ layout
│   ├── features/             # Data checks + embedding extraction
│   ├── emotion/              # Stage 1: audio -> emotion
│   ├── genre/                # Stage 2: emotion -> genre
│   ├── diagnostics/          # Error analysis, thresholds, learning curve, clip length
│   ├── cross_dataset/        # Transfer to the Blockbuster dataset
│   └── evaluation/           # Repeated CV, confidence intervals, significance tests
├── docs/                     # Progress log, literature reviews, thesis LaTeX snippets
└── results/                  # Generated plots and outputs
```

## Data

- Raw data is **git-ignored** (not redistributed) and lives under `data/raw/`:
  - `Eerola_DB/` — the primary dataset: clip audio + enriched rating CSVs.
  - `Blockbuster_DB/` — Ma et al. (2021) pre-extracted VGGish/MIR features (no audio),
    used only for the external VGGish-vs-MFCC baseline (`experiments/cross_dataset/exp_blockbuster.py`).
- Derived data (cached embeddings, matrices, durations) is written to
  `data/processed/Eerola_DB/` and reused across runs — never re-extracted per run.
- `src/config.py` resolves the data root automatically; set `THESIS_DATA_ROOT` to
  override the Eerola root, or `BLOCKBUSTER_DIR` to override the Blockbuster location.

## Running

Experiments are plain Python scripts, runnable from the project root, e.g.:

```bash
python experiments/features/verify_data.py          # load + clean + assert dataset counts
python experiments/features/extract_features.py     # extract & cache AST embeddings (Set 1)
python experiments/genre/exp_genre.py               # genre classification: with vs without emotion
python experiments/evaluation/exp_statistical_power.py   # repeated CV + significance (headline numbers)
```

See `docs/README.md` (the progress log) for the full list of experiments and their results.

## Reproducibility

1. Dependencies are pinned in `requirements.txt`.
2. Random seeds are fixed everywhere (`SEED = 42`, see `src/utils.py`).
3. Embeddings and derived matrices are cached to `data/processed/` for exact re-runs.
4. Cross-validation is grouped by film (`GroupKFold` on `soundtrack`) to prevent leakage.

## Dependencies and Notes

The stack is PyTorch-only by design (AST and CLAP via `transformers`), so all components
share a single deep-learning framework. TensorFlow-based baselines (e.g. YAMNet) are
deliberately avoided to prevent dependency conflicts; if ever needed, they should live in
a separate environment.
