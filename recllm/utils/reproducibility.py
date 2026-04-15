"""Reproducibility utilities: seeding, config serialization."""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducible experiments.

    Controls randomness in Python, NumPy, and PyTorch (if available).
    When CUDA is available, also sets deterministic cuDNN behavior.

    Args:
        seed: Random seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
