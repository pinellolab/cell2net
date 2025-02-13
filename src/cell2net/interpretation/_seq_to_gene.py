import random
from collections.abc import Sequence

import numpy as np
import torch
from captum.attr import DeepLift

from cell2net._logging import logger
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net


def dinucleotide_shuffle(sequence: str) -> str:
    """
    Shuffle a DNA sequence while preserving its dinucleotide composition.

    This function takes a DNA sequence as input, splits it into overlapping
    dinucleotides, shuffles them, and reconstructs a sequence with the same
    dinucleotide composition but in a randomized order.

    Parameters
    ----------
    sequence :
        The DNA sequence to shuffle. Must be a string of nucleotides (e.g., "ATCG").
        Sequences with fewer than 2 characters are returned unchanged.

    Returns
    -------
        A shuffled version of the input sequence with the same dinucleotide composition.
        If the input sequence has fewer than 2 characters, it is returned as is.

    Notes
    -----
        - The function ensures that the dinucleotide composition of the shuffled sequence matches that of the input sequence, but the overall sequence order is randomized.
        - Randomization is achieved using the `random.shuffle` function.

    Examples
    --------
    >>> import random
    >>> random.seed(42)  # For reproducibility
    >>> dinucleotide_shuffle("ATCG")
    'TACG'
    >>> dinucleotide_shuffle("A")
    'A'
    >>> dinucleotide_shuffle("")
    ''
    """
    if len(sequence) < 2:
        return sequence

    # Create a list of dinucleotides
    dinucleotides = [sequence[i : i + 2] for i in range(len(sequence) - 1)]

    # Shuffle the dinucleotides
    random.shuffle(dinucleotides)

    # Reconstruct the sequence from shuffled dinucleotides
    shuffled_sequence = dinucleotides[0]
    for dinucleotide in dinucleotides[1:]:
        shuffled_sequence += dinucleotide[1]

    return shuffled_sequence


def dinucleotide_one_hot_shuffle(one_hot_sequence: np.ndarray) -> np.ndarray:
    """
    Shuffle a one-hot encoded DNA sequence while preserving its dinucleotide composition.

    This function converts a one-hot encoded DNA sequence into its nucleotide representation,
    shuffles it while maintaining the same dinucleotide composition, and then converts the
    shuffled sequence back into one-hot encoding.

    Parameters
    ----------
    one_hot_sequence:
        A 2D array of shape (L, 4), where L is the sequence length, and each row is a
        one-hot encoded nucleotide. Each row should contain exactly one 1 and three 0s,
        corresponding to the nucleotides "A", "C", "G", and "T".

    Returns
    -------
        A 2D array of shape (L, 4) representing the shuffled sequence in one-hot encoding.
        The dinucleotide composition of the original sequence is preserved.

    Notes
    -----
        - The function assumes the input sequence is valid one-hot encoding. Behavior is undefined if the input contains invalid rows.
        - Shuffling is performed on the nucleotide sequence derived from the one-hot input, and the shuffled sequence is converted back to one-hot encoding.
        - The function uses the dinucleotide_shuffle helper function to handle the shuffling of the nucleotide sequence.

    Examples
    --------
    >>> import numpy as np
    >>> import random
    >>> random.seed(42)
    >>> one_hot_sequence = np.array([
    ...     [1, 0, 0, 0],  # A
    ...     [0, 1, 0, 0],  # C
    ...     [0, 0, 1, 0],  # G
    ...     [0, 0, 0, 1]   # T
    ... ])
    >>> shuffled_one_hot = dinucleotide_one_hot_shuffle(one_hot_sequence)
    >>> shuffled_one_hot
    array([[0., 1., 0., 0.],  # "C"
           [1., 0., 0., 0.],  # "A"
           [0., 0., 0., 1.],  # "T"
           [0., 0., 1., 0.]]) # "G"
    """
    # Convert one-hot encoded sequence to nucleotide sequence
    nucleotides = ["A", "C", "G", "T"]
    sequence = "".join([nucleotides[np.argmax(base)] for base in one_hot_sequence])

    # Shuffle the nucleotide sequence
    shuffled_sequence = dinucleotide_shuffle(sequence)

    # Convert shuffled nucleotide sequence back to one-hot encoding
    shuffled_one_hot = np.zeros_like(one_hot_sequence)
    for i, nucleotide in enumerate(shuffled_sequence):
        shuffled_one_hot[i, nucleotides.index(nucleotide)] = 1

    return shuffled_one_hot


def compute_seq_attr(
    model: Cell2Net,
    idx: Sequence[int] | Sequence[str] | None = None,
    batch_size: int = 4,
    num_workers: int = 1,
) -> np.ndarray:
    # create a dataloader
    logger.info("Create dataloader")
    data_loader = get_dataloader(
        mdata=model.mdata,
        covariates=model.covariates,
        idx=idx,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
    )

    model.module.eval()
    for data in data_loader:
        peak_seq = data["peak_seq"].to(model.device)
        peak_acc = data["peak_acc"].to(model.device)
        peak_dist = data["peak_dist"].to(model.device)
        tf_exp = data["tf_exp"].to(model.device)
        covariates = data["covariates"].to(model.device)

        bs = peak_seq.shape[0]
        n_peaks = peak_seq.shape[1]
        peak_len = peak_seq.shape[2]
        shuffle_n = 30

        attr_all = np.zeros((bs, n_peaks, peak_len, 4))
        dl = DeepLift(model.module)
        for i in range(bs):
            for j in range(n_peaks):
                attr_list = []
                for _ in range(shuffle_n):
                    _peak_seq = peak_seq.clone().detach().cpu().numpy()
                    _peak_seq[i][j] = dinucleotide_one_hot_shuffle(_peak_seq[i][j])
                    _peak_seq = torch.from_numpy(_peak_seq).to(model.device)

                    attributions = dl.attribute(
                        inputs=(peak_seq, peak_acc, peak_dist, tf_exp),
                        baselines=(_peak_seq, peak_acc, peak_dist, tf_exp),
                        additional_forward_args=covariates,
                    )
                    seq_attr = attributions[0].detach().cpu().numpy()
                    attr_list.append(seq_attr)

                attr_list = np.stack(attr_list)
                attr_mean = np.mean(attr_list, axis=0)
                attr_all[i][j] = attr_mean[i][j]

        attr_multiply_ohe = attr_all * peak_seq.detach().cpu().numpy()
        attr_multiply_ohe = np.transpose(attr_multiply_ohe, (0, 1, 3, 2))
        attr_all = np.transpose(attr_all, (0, 1, 3, 2))

    return attr_all
