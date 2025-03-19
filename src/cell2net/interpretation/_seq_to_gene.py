import random
from collections.abc import Sequence

import numpy as np
import torch
from captum.attr import DeepLift

from cell2net._logging import logger
from cell2net.interpretation._utils import is_sequence_of_strings
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
    >>> import cell2net as cn
    >>> random.seed(42)  # For reproducibility
    >>> cn.ip.dinucleotide_shuffle("ATCG")
    'TACG'
    >>> cn.ip.dinucleotide_shuffle("A")
    'A'
    >>> cn.ip.dinucleotide_shuffle("")
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
    >>> import cell2net as cn
    >>> import random
    >>> random.seed(42)
    >>> one_hot_sequence = np.array([
    ...     [1, 0, 0, 0],  # A
    ...     [0, 1, 0, 0],  # C
    ...     [0, 0, 1, 0],  # G
    ...     [0, 0, 0, 1]   # T
    ... ])
    >>> shuffled_one_hot = cn.ip.dinucleotide_one_hot_shuffle(one_hot_sequence)
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
    peaks: int | str | Sequence[int] | Sequence[str] | None = None,
    idx: Sequence[int] | Sequence[str] | None = None,
    batch_size: int = 4,
    num_workers: int = 1,
    shuffle_n: int = 50,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
) -> np.ndarray:
    """
    Computes sequence attribution scores using the DeepLift algorithm for a given model

    This function takes a trained `Cell2Net` model and computes attribution scores
    for input sequences using the DeepLift method. It generates shuffled baselines
    for comparison and averages the attributions over multiple shuffles.

    Parameters
    ----------
    model :
        The trained model containing the sequence and other input features.
    peaks :
        Peaks used to compute attribution.
        This can be a single peak index, a list of peak indices, a single peak name, or a list of peak names.
        If None, all peaks are used.
    idx :
        Indices of the samples to compute attribution for. If None, all samples are used.
    batch_size :
        The number of samples per batch in the DataLoader.
    num_workers :
        The number of worker threads for data loading.
    shuffle_n :
        The number of times to shuffle the dinucleotide sequences for baseline attribution.

    Returns
    -------
        A NumPy array of shape `(batch_size, num_peaks, 4, peak_length)`,
        representing the attribution scores for each base in the input sequences.
    """
    # create a dataloader
    logger.info("Create dataloader")
    data_loader = get_dataloader(
        mdata=model.mdata,
        rna_mod=rna_mod,
        atac_mod=atac_mod,
        covariates=model.covariates,
        idx=idx,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
    )

    # set model to evaluation mode
    model.module.eval()

    # get peak index
    if peaks is None:
        peak_indices = range(len(model.mdata[atac_mod].var_names))
    elif isinstance(peaks, int):
        peak_indices = [peaks]
    elif isinstance(peaks, str):
        peak_indices = [model.mdata[atac_mod].var.index.get_loc(peaks)]
    elif is_sequence_of_strings(peaks):
        peak_indices = [model.mdata[atac_mod].var.index.get_loc(p) for p in peaks]

    for data in data_loader:
        peak_seq = data["peak_seq"].to(model.device)
        peak_acc = data["peak_acc"].to(model.device)
        peak_dist = data["peak_dist"].to(model.device)
        tf_exp = data["tf_exp"].to(model.device)
        covariates = data["covariates"].to(model.device)

        bs = peak_seq.shape[0]
        peak_len = peak_seq.shape[2]

        attr_all = np.zeros((bs, len(peak_indices), peak_len, 4))
        dl = DeepLift(model.module)
        for i in range(bs):
            for j, peak_index in enumerate(peak_indices):
                attr_list = []
                # shuffle the dinucleotide sequence for shuffle_n times
                # and compute the attribution scores, then average them
                for _ in range(shuffle_n):
                    _peak_seq = peak_seq.clone().detach().cpu().numpy()
                    _peak_seq[i][peak_index] = dinucleotide_one_hot_shuffle(
                        _peak_seq[i][peak_index]
                    )
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
