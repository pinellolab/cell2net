from collections.abc import Sequence

import numpy as np
import torch
from captum.attr import DeepLift
from tqdm.auto import tqdm

from cell2net._logging import logger
from cell2net.interpretation._utils import is_sequence_of_ints, is_sequence_of_strings
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net
from cell2net.preprocessing import dinucleotide_shuffle_one_hot, seq_to_one_hot


def deep_lift(
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

def saturation_mutagenesis(
    model: Cell2Net,
    peak: int | str = None,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    batch_size: int = 32,
    num_workers: int = 1,
    multiply_by_inputs: bool = True,
    normalize: bool = False,
    logfc: bool = False,
    smoothing: bool = True,
    window_size: int = 3,
    return_seq: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """
    Computes saturation mutagenesis effects for a given peak in the model.

    This function performs in-silico saturation mutagenesis by systematically mutating
    each position in a peak's DNA sequence to all possible alternative bases and
    measuring the effect on gene expression predictions. The method computes the
    difference between predictions using the reference sequence versus mutated sequences
    to identify functionally important positions.

    Parameters
    ----------
    model : Cell2Net
        The trained Cell2Net model containing the sequence and other input features.
    peak : int or str, optional
        Peak to analyze. Can be either a peak index (int) or peak name (str).
        Must be provided for the function to run.
    rna_mod : str, default "rna"
        Key for the RNA modality in the model's mdata object.
    atac_mod : str, default "atac"
        Key for the ATAC modality in the model's mdata object.
    batch_size : int, default 32
        The number of samples per batch for model predictions.
    num_workers : int, default 1
        The number of worker threads for data loading during predictions.
    multiply_by_inputs : bool, default True
        Whether to multiply the computed effects by the one-hot encoded reference
        sequence. When True, only positions matching the reference base will have
        non-zero values, making interpretation more intuitive.
    normalize : bool, default False
        Whether to normalize effects by centering them around zero. This removes
        the global mean effect across all positions.
    smoothing : bool, default True
        Whether to apply smoothing to the computed effects using a moving average
        filter to reduce noise.
    window_size : int, default 3
        Size of the moving average window for smoothing. Only used when
        smoothing=True.
    return_seq : bool, default False
        Whether to return the one-hot encoded reference sequence along with
        the effects. If True, returns a tuple of (effects, sequence).

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray] or None
        If return_seq=False: Returns a 2D numpy array of shape (4, sequence_length)
        containing the mutagenesis effects for each base at each position.

        If return_seq=True: Returns a tuple containing:
        - effects: 2D numpy array of shape (4, sequence_length) with mutagenesis effects
        - ref_seq_encode: 2D numpy array of shape (4, sequence_length) with one-hot
          encoded reference sequence

        Returns None if the peak parameter is invalid.

    Notes
    -----
    The saturation mutagenesis procedure:
    1. Makes predictions using the original reference sequence
    2. For each position in the sequence:
       - Mutates the position to each of the 3 alternative bases
       - Makes predictions with each mutated sequence
       - Averages the predictions across the 3 alternatives
    3. Computes effect size as the difference between reference and mutated predictions
    4. Averages effects across all cells in the dataset
    5. Optionally applies normalization and smoothing
    6. Tiles the effects across all 4 bases and optionally multiplies by reference sequence

    The resulting effects matrix indicates which positions are most critical for
    gene expression, with larger absolute values indicating greater functional importance.

    Examples
    --------
    >>> # Basic usage with peak index
    >>> effects = saturation_mutagenesis(model, peak=0)
    >>>
    >>> # With peak name and additional options
    >>> effects = saturation_mutagenesis(
    ...     model,
    ...     peak="chr1:1000-2000",
    ...     normalize=True,
    ...     smoothing=False
    ... )
    >>>
    >>> # Return both effects and sequence
    >>> effects, sequence = saturation_mutagenesis(
    ...     model,
    ...     peak=0,
    ...     return_seq=True
    ... )
    """

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

    logger.info("Predicting expression using reference sequence")
    pred_ref = model.predict(model.mdata,
                             rna_mod=rna_mod,
                             atac_mod=atac_mod,
                             batch_size=batch_size,
                             num_workers=num_workers)


    # get reference sequence for the peak
    ref_seq = model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx]

    logger.info("Predicting expression using alternative sequences")
    bases = ['A', 'C', 'G', 'T']
    ism_scores = []
    for i in tqdm(range(len(ref_seq))):
        pred_alt = np.zeros(model.mdata.n_obs)

        # compute predictions for mutated sequence
        for alt in bases:
            if alt != ref_seq[i]:
                alt_seq = ref_seq[:i] + alt + ref_seq[i+1:]
                model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx] = alt_seq

                pred_alt += model.predict(model.mdata,
                                          rna_mod=rna_mod,
                                          atac_mod=atac_mod,
                                          batch_size=batch_size,
                                          num_workers=num_workers)

        # average predictions for the alternative base
        pred_alt /= (len(bases) - 1)

        # compute the effect size across all cells
        if logfc:
            ism_scores.append(np.log2(np.mean(pred_ref) / np.mean(pred_alt)))
        else:
            ism_scores.append(np.mean(pred_ref - pred_alt))

    model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx] = ref_seq  # restore the original sequence

    ism_scores = np.array(ism_scores)
    if normalize:
        logger.info("Normalizing ISM scores by centering around 0")
        ism_scores -= np.mean(ism_scores)  # center the effects around 0

    # if smoothing is needed, we can use a simple moving average
    if smoothing:
        ism_scores = np.convolve(ism_scores, np.ones(window_size)/window_size, mode='same')

    ism_scores = np.tile(ism_scores, (4, 1))

    ref_seq_encode = seq_to_one_hot(ref_seq).transpose()
    if multiply_by_inputs:
        ism_scores = ism_scores * ref_seq_encode

    if return_seq:
        return ism_scores, ref_seq_encode
    else:
        return ism_scores



def saturation_mutagenesis_v2(
    model: Cell2Net,
    peak: int | str = None,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    batch_size: int = 32,
    num_workers: int = 1,
    multiply_by_inputs: bool = True,
    normalize: bool = False,
    logfc: bool = False,
    smoothing: bool = True,
    window_size: int = 3,
    return_seq: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:

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

    logger.info("Predicting expression using reference sequence")
    pred_ref = model.predict(model.mdata,
                             rna_mod=rna_mod,
                             atac_mod=atac_mod,
                             batch_size=batch_size,
                             num_workers=num_workers)


    # get reference sequence for the peak
    ref_seq = model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx]

    logger.info("Predicting expression using alternative sequences")
    bases = ['A', 'C', 'G', 'T']
    ism_scores = []
    for i in tqdm(range(len(ref_seq))):
        pred_alt = np.zeros(model.mdata.n_obs)

        # compute predictions for mutated sequence
        for alt in bases:
            if alt != ref_seq[i]:
                alt_seq = ref_seq[:i] + alt + ref_seq[i+1:]
                model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx] = alt_seq

                pred_alt += model.predict(model.mdata,
                                          rna_mod=rna_mod,
                                          atac_mod=atac_mod,
                                          batch_size=batch_size,
                                          num_workers=num_workers)

        # average predictions for the alternative base
        pred_alt /= (len(bases) - 1)

        fc = np.divide(pred_ref, pred_alt, out=np.zeros_like(pred_ref), where=pred_alt!=0)
        logfc = np.log2(fc).sum()
        ism_scores.append(logfc)

    model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx] = ref_seq  # restore the original sequence

    ism_scores = np.array(ism_scores)
    if normalize:
        logger.info("Normalizing ISM scores by centering around 0")
        ism_scores -= np.mean(ism_scores)  # center the effects around 0

    # if smoothing is needed, we can use a simple moving average
    if smoothing:
        ism_scores = np.convolve(ism_scores, np.ones(window_size)/window_size, mode='same')

    ism_scores = np.tile(ism_scores, (4, 1))

    ref_seq_encode = seq_to_one_hot(ref_seq).transpose()
    if multiply_by_inputs:
        ism_scores = ism_scores * ref_seq_encode

    if return_seq:
        return ism_scores, ref_seq_encode
    else:
        return ism_scores
