import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set random seed"""
    if not isinstance(seed, int):
        seed = int(seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def santize_str_for_filename(s: str) -> str:
    """
    Sanitize a string to make it safe for use as a filename.

    This function replaces or removes characters that are typically problematic
    in filenames. Specifically:
        - Spaces are replaced with underscores (`_`).
        - Slashes (`/`) are replaced with underscores (`_`).
        - Parentheses (`(` and `)`) are removed.

    Parameters
    ----------
    s : str
        The input string to sanitize.

    Returns
    -------
    str
        The sanitized string, suitable for use as a filename.

    Examples
    --------
    >>> sanitize_str_for_filename("example (file)/name")
    ... "example_file_name"
    """
    return s.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
