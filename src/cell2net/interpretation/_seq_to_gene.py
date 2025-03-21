from collections.abc import Sequence

import numpy as np
import torch
from captum.attr import DeepLift
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.interpretation._utils import is_sequence_of_ints, is_sequence_of_strings
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net
from cell2net.preprocessing import dinucleotide_shuffle_one_hot


def seq_attr(
    model: Cell2Net,
    peaks: int | str | Sequence[int] | Sequence[str] | None = None,
    idx: Sequence[int] | Sequence[str] | None = None,
    batch_size: int = 4,
    num_workers: int = 1,
    shuffle_n: int = 50,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
) -> np.ndarray | None:
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
        A NumPy array of shape `(n_cells, num_peaks, 4, peak_length)`,
        representing the attribution scores for each base in the input sequences.
    """
    # get peak index
    if peaks is None:
        peak_indices = range(len(model.mdata[atac_mod].var_names))
    elif isinstance(peaks, int):
        peak_indices = [peaks]
    elif isinstance(peaks, str):
        peak_indices = [model.mdata[atac_mod].var.index.get_loc(peaks)]
    elif is_sequence_of_strings(peaks):
        peak_indices = [model.mdata[atac_mod].var.index.get_loc(p) for p in peaks]
    elif is_sequence_of_ints(peaks):
        peak_indices = peaks
    else:
        logger.error(
            "Invalid peaks input, must be a single peak index, a list of peak indices, a single peak name, or a list of peak names"
        )
        return None

    logger.info(f"Compute attribution scores for {len(peak_indices)} peaks")

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

    # Use DeepLift to estimate feature importances
    dl = DeepLift(model.module, multiply_by_inputs=False)
    attr_samples_peaks = []
    for data in tqdm(data_loader):
        peak_seq = data["peak_seq"].to(model.device)
        peak_acc = data["peak_acc"].to(model.device)
        peak_dist = data["peak_dist"].to(model.device)
        tf_exp = data["tf_exp"].to(model.device)
        covariates = data["covariates"].to(model.device)

        bs = peak_seq.shape[0]
        peak_len = peak_seq.shape[2]

        attr_batch_peaks = np.zeros((len(peak_indices), bs, peak_len, 4))
        # compute attribution scores for selected peaks
        for i, peak_index in enumerate(peak_indices):
            # shuffle the dinucleotide sequence for shuffle_n times
            # and compute the attribution scores, then average them
            _attr_list = []
            for _ in range(shuffle_n):
                _peak_seq = peak_seq.clone().detach().cpu().numpy()

                # shuffle the sequence for each sample to get the baseline
                for j in range(bs):
                    _peak_seq[j, peak_index] = dinucleotide_shuffle_one_hot(
                        _peak_seq[j, peak_index]
                    )

                _peak_seq = torch.from_numpy(_peak_seq).to(model.device)

                attributions = dl.attribute(
                    inputs=(peak_seq, peak_acc, peak_dist, tf_exp),
                    baselines=(_peak_seq, peak_acc, peak_dist, tf_exp),
                    additional_forward_args=covariates,
                )

                seq_attr = (
                    attributions[0].detach().cpu().numpy()
                )  # (batch_size, n_peaks, peak_length, 4)
                seq_attr = seq_attr[:, peak_index]  # type: ignore # (batch_size, peak_length, 4)
                _attr_list.append(seq_attr)

            # average the attributions over the shuffles
            _attr_list = np.stack(_attr_list)  # (shuffle_n, batch_size, peak_length, 4)
            attr_batch_peaks[i] = np.mean(
                _attr_list, axis=0
            )  # (batch_size, peak_length, 4)

        attr_batch_peaks = np.transpose(
            attr_batch_peaks, (1, 0, 2, 3)
        )  # (batch_size, n_peaks, peak_length, 4)
        _peak_seq = peak_seq[:, peak_indices].detach().cpu().numpy()

        attr_multiply_ohe = attr_batch_peaks * _peak_seq
        attr_multiply_ohe = np.transpose(attr_multiply_ohe, (0, 1, 3, 2))

        attr_samples_peaks.append(attr_multiply_ohe)

    attr_samples_peaks = np.concatenate(attr_samples_peaks, axis=0)

    return attr_samples_peaks
