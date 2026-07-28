# Bachelor Thesis Project: Audio-Based Film Genre and Emotion Recognition

This repository forms the foundation for the implementation of the bachelor thesis and contains a reproducible Python environment together with a clear project structure for data processing, modeling, and evaluation.

## Goal

The project focuses on the analysis of audio-based data for the classification of film genres and emotions. The structure is designed to keep experiments organized, reproducible, and easy to extend.

## Core Principles

- Reproducibility: dependencies are explicitly defined.
- Isolated environment: dependencies are not installed globally.
- Clean structure: data, code, notebooks, and results are kept separate.
- Scientific traceability: experiments should be documented and easy to follow.

## Requirements

- Python 3.11
- pip or conda
- Optional: NVIDIA GPU for faster training runs

## Quick Start

### Option 1: Conda (recommended)

If Conda or Mambaforge is installed:

```bash
conda env create -f environment.yml
conda activate ba-genre-emotion
```

### Option 2: Virtual Environment with venv

On Windows:

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
├── environment.yml          # Conda environment
├── requirements.txt         # Python dependencies
├── README.md                # Project description and setup
├── data/
│   ├── raw/                 # Original data; leave unchanged
│   └── processed/           # Preprocessed or cleaned data
├── src/
│   ├── features/            # Feature extraction
│   ├── models/              # Model implementations
│   └── evaluation/          # Metrics, plots, analysis
├── notebooks/              # Exploratory analysis and visualizations
├── results/                 # Model outputs, plots, logs
└── thesis/                  # LaTeX or Word files for the written thesis
```

## Data Organization

- Raw data should be stored in [data/raw](data/raw).
- Processed data should be placed in [data/processed](data/processed).
- Results and plots should be saved in [results](results).
- Raw data should not be modified directly; intermediate outputs should be stored in processed.

## Reproducibility

For a robust scientific workflow, the following practices are recommended:

1. Pin versions rather than using loose version ranges.
2. Keep dependencies in the repository.
3. Set random seeds explicitly.
4. Document important configurations and parameters.
5. Store results in a traceable way.

Example for fixed seeds:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

## Dependencies and Notes

The current configuration relies mainly on PyTorch and related libraries. This choice is intentional so the main components can work within a single deep-learning framework.

If additional baselines using TensorFlow are required, they should ideally be placed in a separate environment to avoid version conflicts.

## Recommended Workflow

1. Create and activate the environment.
2. Place data in [data/raw](data/raw).
3. Implement preprocessing and feature extraction in [src/features](src/features).
4. Develop models in [src/models](src/models).
5. Generate evaluations and plots in [src/evaluation](src/evaluation) or [notebooks](notebooks).
6. Save outputs in [results](results).

## VS Code Recommendation

When using Visual Studio Code, point the interpreter to the created virtual environment so that Python and Jupyter tools work correctly.

## Note

This repository should be understood as a project starter. The actual implementation of the models and experiments can be added gradually into the designated folders.
