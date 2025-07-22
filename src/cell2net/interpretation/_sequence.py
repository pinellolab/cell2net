from collections.abc import Sequence

import numpy as np
import torch
from captum.attr import DeepLift, DeepLiftShap
from tqdm import tqdm
import itertools
from cell2net._logging import logger
from cell2net.interpretation._utils import is_sequence_of_ints, is_sequence_of_strings
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net
from cell2net.preprocessing import dinucleotide_shuffle_one_hot, seq_to_one_hot, one_hot_to_seq


def seq_attr(
    model: Cell2Net,
    peaks: int | str | Sequence[int] | Sequence[str] | None = None,
    idx: Sequence[int] | Sequence[str] | None = None,
    batch_size: int = 4,
    num_workers: int = 1,
    shuffle_n: int = 50,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    multiply_by_inputs: bool = True,
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
    dl = DeepLift(model.module, multiply_by_inputs=multiply_by_inputs)
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


def seq_attr_v2(
    model: Cell2Net,
    peaks: int | str | Sequence[int] | Sequence[str] | None = None,
    idx: Sequence[int] | Sequence[str] | None = None,
    batch_size: int = 4,
    num_workers: int = 1,
    shuffle_n: int = 50,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    multiply_by_inputs: bool = True,
) -> np.ndarray | None:

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
    dl = DeepLiftShap(model.module, multiply_by_inputs=multiply_by_inputs)
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


def _edit_distance_one(X, start, end):
    """An internal function for generating all sequences of edit distance 1

    This internal function, which is meant to be used for ISM, will take in a
    one-hot encoded sequence and return all sequences that have an edit distance
    of one.


    Parameters
    ----------
    X: torch.Tensor, shape=(len(alphabet), sequence_length)
        A single one-hot encoded sequence.

    start: int
        The first nucleotide to begin making edits on, inclusive.

    end: int
        The end of the span. Edits are not made on this nucleotide at this
        index. Can be negative indexes.


    Returns
    -------
    X_: torch.Tensor, shape=(length*len(alphabet), len(alphabet), length)
        All one-hot encoded sequences that have an edit distance of 1 from the
        original sequence.
    """
    start = 0,
    end = X.shape[-1] + 1

    X_ = X.repeat((end-start)*X.shape[0], 1, 1)

    coords = itertools.product(range(X.shape[0]), range(start, end))
    for i, (j, k) in enumerate(coords):
        X_[i, :, k] = 0
        X_[i, j, k] = 1

    return X_


def saturation_mutagenesis(
    model: Cell2Net,
    peak: int | str = None,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    batch_size: int = 32,
    num_workers: int = 1,
    multiply_by_inputs: bool = True,
    smoothing: bool = True,
    window_size: int = 3,
) -> np.ndarray | None:

    if isinstance(peak, int):
        peak_idx = peak
    elif isinstance(peak, str):
        peak_idx = model.mdata[atac_mod].var.index.get_loc(peak)
    else:
        logger.error(
            "Invalid peak input, must be a single peak index or a single peak name"
        )
        return None

    logger.info(f"Compute saturation mutagenesis for peak {peak_idx}")

    logger.info("Get reference prediction using original sequence")
    pred_ref = model.predict(model.mdata,
                             rna_mod=rna_mod,
                             atac_mod=atac_mod,
                             batch_size=batch_size,
                             num_workers=num_workers)


    # get reference sequence for the peak
    ref_seq = model.mdata[atac_mod].var["dna_sequence"].values.tolist()[peak_idx]

    logger.info("Compute predictions for all alternative bases")
    bases = ['A', 'C', 'G', 'T']
    effects = []
    for i in tqdm(range(len(ref_seq))):
        pred_alt = np.zeros(model.mdata.n_obs)

        # compute predictions for mutated sequence
        for alt in bases:
            if alt != ref_seq[i]:
                alt_seq = ref_seq[:i] + alt + ref_seq[i+1:]
                model.mdata[atac_mod].var["dna_sequence"][peak_idx] = alt_seq

                pred_alt += model.predict(model.mdata,
                                          rna_mod=rna_mod,
                                          atac_mod=atac_mod,
                                          batch_size=batch_size,
                                          num_workers=num_workers)

        # average predictions for the alternative base
        pred_alt /= (len(bases) - 1)

        # compute the effect size across all cells
        effects.append(np.mean(pred_ref - pred_alt))

    effects = np.array(effects)
    effects -= np.mean(effects)  # center the effects around 0

    # if smoothing is needed, we can use a simple moving average
    if smoothing:
        effects = np.convolve(effects, np.ones(window_size)/window_size, mode='same')

    effects = np.tile(effects, (4, 1))

    if multiply_by_inputs:
        ref_seq_encode = seq_to_one_hot(ref_seq).transpose()
        effects = effects * ref_seq_encode

    return effects
