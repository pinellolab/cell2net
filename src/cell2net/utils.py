import os
import random
import anndata as ad
import numpy as np
import torch
from lightning.pytorch import seed_everything

from cell2net._logging import logger

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_everything(seed, verbose=False)

# def seed_everything(seed: int = 42):
#     """Set random seed"""
#     if not isinstance(seed, int):
#         seed = int(seed)

#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True


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


def guess_is_lognorm(
    adata: ad.AnnData,
    n_cells: int | float = 5e2,
    epsilon: float = 1e-2,
    layer: str | None = None,
) -> bool:
    """Guess if the input is integer counts or log-normalized.

    This is an _educated guess_ based on whether the fractional component of cell sums.
    This _will not be able_ to distinguish between normalized input and log-normalized input.

    Returns:
        bool: True if the input is lognorm, False otherwise
    """
    # Determine the number of cells to use for the guess
    n_cells = int(min(adata.shape[0], n_cells))

    # Pick a random subset of cells
    cell_mask = np.random.choice(adata.shape[0], n_cells, replace=False)

    # Sum the counts for each cell
    # if a layer is specified, use that layer; otherwise, use the main data matrix
    if layer is not None:
        if layer not in adata.layers:
            logger.error(f"Layer '{layer}' not found in AnnData object.")
        cell_sums = adata.layers[layer][cell_mask].sum(axis=1)
    else:
        cell_sums = adata.X[cell_mask].sum(axis=1)  # type: ignore (can be float but super unlikely)

    # Check if any cell sum's fractional part is greater than epsilon
    return bool(np.any(np.abs((cell_sums - cell_sums.round())) > epsilon))
