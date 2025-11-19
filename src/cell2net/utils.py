import anndata as ad
import numpy as np
import torch

from cell2net._logging import logger

def set_random_seed(seed: int = 42):
    """Set the random seed for reproducibility across various libraries."""
    from lightning.pytorch import seed_everything
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_everything(seed, verbose=False)

def santize_str_for_filename(s: str) -> str:
    """
    Sanitize a string to make it safe for use as a filename.

    Remove invalid characters and spaces.

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
