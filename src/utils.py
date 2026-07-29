"""Small shared utilities."""

from __future__ import annotations

import os
import random

import numpy as np

from src import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
