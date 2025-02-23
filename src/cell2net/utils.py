import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """Set random seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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
    ... 'example_file_name'
    """
    return s.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def random_seq(seq_len: int, bases: list[str] | None = None) -> str:
    """
    Generate a random nucleotide sequence of a specified length.

    This function creates a random sequence of nucleotides (or other bases)
    by selecting characters from a provided list of bases. If no base list
    is provided, the default bases are adenine (A), cytosine (C), guanine (G),
    and thymine (T).

    Parameters
    ----------
    seq_len : int
        The length of the random sequence to generate.
    bases : list[str], optional
        A list of characters to use as the base set for the sequence.
        Defaults to ["A", "C", "G", "T"].

    Returns
    -------
    str
        A randomly generated sequence of the specified length using the
        provided bases.

    Examples
    --------
    >>> random_seq(10)
    'ACGTGCTAGC'
    >>> random_seq(5, bases=["A", "T"])
    'TATTA'
    """
    if bases is None:
        bases = ["A", "C", "G", "T"]

    rand_seq = "".join([np.random.choice(bases) for i in range(seq_len)])
    return rand_seq


def one_hot_encode(seq) -> torch.Tensor:
    """
    Convert a sequence to one-hot encoding

    Only ACTGN allow.

    Parameters
    ----------
    seq : _type_
        _description_

    Returns
    -------
    torch.Tensor
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(
            f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}"
        )

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [0.0, 0.0, 0.0, 0.0],
    }

    # Create array from nucleotide sequence
    vec = torch.tensor([nuc_d[x] for x in seq], dtype=torch.float32)

    return vec


def encode_seq(seq_list):
    data = []
    for seq in seq_list:
        data.append(one_hot_encode(seq))

    data = torch.stack(data)

    return data
