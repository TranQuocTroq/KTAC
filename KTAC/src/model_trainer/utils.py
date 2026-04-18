"""Configuration loading and reproducibility utilities.

Example:
    >>> cfg = load_config("configs/model_config.yaml")
    >>> set_seed(cfg["seed"])
"""

import os
import random

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Fixes seeds for Python, NumPy, PyTorch (CPU and CUDA) and disables
    cuDNN non-deterministic algorithms.

    Args:
        seed (int): Random seed value. Defaults to ``42``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
