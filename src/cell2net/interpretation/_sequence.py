from collections.abc import Sequence

import numpy as np
import torch
from captum.attr import DeepLift
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net, Cell2NetWithGenotype
from cell2net.preprocessing import dinucleotide_shuffle_one_hot, seq_to_one_hot


def deep_lift_shap(
    model: Cell2Net,
    peak: int | str = None,
    batch_size: int = 4,
    num_workers: int = 1,
    n_shuffles: int = 20,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    multiply_by_inputs: bool = True,
) -> np.ndarray | None:
    """
    Computes sequence attribution scores using the DeepLiftShap algorithm for a given model

    This function takes a trained `Cell2Net` model and computes attribution scores
    for input sequences using the DeepLiftShap method from captum package.
    It generates shuffled baselines for comparison and averages the attributions over multiple shuffles.

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
    if isinstance(peak, int):
        peak_idx = peak
    elif isinstance(peak, str):
        peak_idx = model.mdata[atac_mod].var.index.get_loc(peak)
    else:
        logger.error(
            "Invalid peak input, must be a single peak index or a single peak name"
        )
        return None

    # create a dataloader
    data_loader = get_dataloader(
        mdata=model.mdata,
        rna_mod=rna_mod,
        atac_mod=atac_mod,
        covariates=model.covariates,
        idx=None,
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
    attr_samples = []
    for data in tqdm(data_loader):
        peak_seq = data["peak_seq"].to(model.device)
        peak_acc = data["peak_acc"].to(model.device)
        peak_dist = data["peak_dist"].to(model.device)
        tf_exp = data["tf_exp"].to(model.device)
        covariates = data["covariates"].to(model.device)

        bs = peak_seq.shape[0]

        _attr_list = []
        for i in range(n_shuffles):
            _peak_seq = peak_seq.clone().detach().cpu().numpy()

            # shuffle the sequence for each sample to get the baseline
            # use different random seed for each iteration
            for j in range(bs):
                _peak_seq[j, peak_idx] = dinucleotide_shuffle_one_hot(
                    _peak_seq[j, peak_idx], random_state=i
                )

            _peak_seq = torch.from_numpy(_peak_seq).to(model.device)

            attributions, delta = dl.attribute(
                inputs=(peak_seq, peak_acc, peak_dist, tf_exp),
                baselines=(_peak_seq, peak_acc, peak_dist, tf_exp),
                additional_forward_args=covariates,
                return_convergence_delta=True
            )

            seq_attr = attributions[0].detach().cpu().numpy()  # (batch_size, n_peaks, peak_length, 4)
            seq_attr = seq_attr[:, peak_idx]  # type: ignore # (batch_size, peak_length, 4)
            _attr_list.append(seq_attr)

        # average the attributions over the shuffles
        _attr_list = np.stack(_attr_list)  # (shuffle_n, batch_size, peak_length, 4)
        attr_batch_peak = np.mean(_attr_list, axis=0)  # (batch_size, peak_length, 4)

        _peak_seq = peak_seq[:, peak_idx].detach().cpu().numpy()
        attr_multiply_ohe = attr_batch_peak * _peak_seq
        attr_samples.append(attr_multiply_ohe)

    attr_samples = np.concatenate(attr_samples, axis=0)  # (n_cells, peak_length, 4)
    avg_attr = attr_samples.sum(axis=0).transpose()  # (4, peak_length)

    return avg_attr

def saturation_mutagenesis(
    model: Cell2Net,
    peak: int | str = None,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    batch_size: int = 32,
    num_workers: int = 1,
    multiply_by_inputs: bool = True,
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


def saturation_mutagenesis_with_genotype(
    model: Cell2NetWithGenotype,
    peak: int | str = None,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    batch_size: int = 32,
    num_workers: int = 1,
    multiply_by_inputs: bool = True,
    logfc: bool = False,
    smoothing: bool = True,
    window_size: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
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
    df_seq = model.mdata[atac_mod].uns["personal_genome_seq"]
    df_seq = df_seq[df_seq['peak'] == peak]

    # ensure all seq_1 are the same
    assert len(df_seq['seq_1'].unique()) == 1, "All seq_1 should be the same"
    assert len(df_seq['seq_2'].unique()) == 1, "All seq_2 should be the same"

    ref_seq_1 = df_seq['seq_1'].values[0]
    ref_seq_2 = df_seq['seq_2'].values[0]

    # ensure seq_1 and seq_2 are the same length
    assert len(ref_seq_1) == len(ref_seq_2), "seq_1 and seq_2 should be the same length"

    logger.info("Predicting expression using alternative sequences")
    bases = ['A', 'C', 'G', 'T']
    ism_scores_1 = []
    for i in tqdm(range(len(ref_seq_1))):
        pred_alt = np.zeros(model.mdata.n_obs)

        # compute predictions for mutated sequence
        for alt in bases:
            if alt != ref_seq_1[i]:
                alt_seq = ref_seq_1[:i] + alt + ref_seq_1[i+1:]

                # replace the sequence in mdata
                model.mdata[atac_mod].uns['personal_genome_seq'].loc[df_seq.index, 'seq_1'] = alt_seq
                pred_alt += model.predict(model.mdata,
                                          rna_mod=rna_mod,
                                          atac_mod=atac_mod,
                                          batch_size=batch_size,
                                          num_workers=num_workers)

        # average predictions for the alternative base
        pred_alt /= (len(bases) - 1)

        # compute the effect size across all cells
        if logfc:
            ism_scores_1.append(np.log2(np.mean(pred_ref) / np.mean(pred_alt)))
        else:
            ism_scores_1.append(np.mean(pred_ref - pred_alt))

    # restore the original sequence
    model.mdata[atac_mod].uns['personal_genome_seq'].loc[df_seq.index, 'seq_1'] = ref_seq_1

    ism_scores_2 = []
    for i in tqdm(range(len(ref_seq_2))):
        pred_alt = np.zeros(model.mdata.n_obs)

        # compute predictions for mutated sequence
        for alt in bases:
            if alt != ref_seq_2[i]:
                alt_seq = ref_seq_2[:i] + alt + ref_seq_2[i+1:]

                # replace the sequence in mdata
                model.mdata[atac_mod].uns['personal_genome_seq'].loc[df_seq.index, 'seq_2'] = alt_seq
                pred_alt += model.predict(model.mdata,
                                          rna_mod=rna_mod,
                                          atac_mod=atac_mod,
                                          batch_size=batch_size,
                                          num_workers=num_workers)

        # average predictions for the alternative base
        pred_alt /= (len(bases) - 1)

        # compute the effect size across all cells
        if logfc:
            ism_scores_2.append(np.log2(np.mean(pred_ref) / np.mean(pred_alt)))
        else:
            ism_scores_2.append(np.mean(pred_ref - pred_alt))

    # restore the original sequence
    model.mdata[atac_mod].uns['personal_genome_seq'].loc[df_seq.index, 'seq_2'] = ref_seq_2

    ism_scores_1 = np.array(ism_scores_1)
    ism_scores_2 = np.array(ism_scores_2)

    # if smoothing is needed, we can use a simple moving average
    if smoothing:
        ism_scores_1 = np.convolve(ism_scores_1, np.ones(window_size)/window_size, mode='same')
        ism_scores_2 = np.convolve(ism_scores_2, np.ones(window_size)/window_size, mode='same')

    ism_scores_1 = np.tile(ism_scores_1, (4, 1))
    ism_scores_2 = np.tile(ism_scores_2, (4, 1))

    ref_seq_encode_1 = seq_to_one_hot(ref_seq_1).transpose()
    ref_seq_encode_2 = seq_to_one_hot(ref_seq_2).transpose()
    if multiply_by_inputs:
        ism_scores_1 = ism_scores_1 * ref_seq_encode_1
        ism_scores_2 = ism_scores_2 * ref_seq_encode_2

    return ism_scores_1, ref_seq_encode_1, ism_scores_2, ref_seq_encode_2


def saturation_mutagenesis_with_genotype_v2(
    model: Cell2NetWithGenotype,
    peak: int | str = None,
    snp_pos: int | None = None,
    window: int = 20,
    coord: str = "genomic",
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    batch_size: int = 32,
    num_workers: int = 1,
    multiply_by_inputs: bool = True,
    logfc: bool = False,
    smoothing: bool = True,
    window_size: int = 3,
    return_positions: bool = False,
):
    """In-silico saturation mutagenesis for one peak, on both haplotypes.

    Parameters
    ----------
    peak : peak index (int) or peak name (str) in `mdata[atac_mod].var`.
    snp_pos : anchor position. Only bases within `window` of the anchor are
        mutated, which is what makes a per-variant scan affordable. None
        scans the whole peak (the previous behaviour).
    window : half-width in bp. The scanned span is
        [anchor - window, anchor + window], i.e. 2 * window + 1 positions.
    coord : "genomic" if `snp_pos` is a genome coordinate, in which case the
        peak's `start` column is subtracted to get the offset into the
        sequence; "offset" if `snp_pos` is already 0-based within the sequence.
    return_positions : also return the array of scanned offsets.

    Returns
    -------
    (ism_scores_1, ref_seq_encode_1, ism_scores_2, ref_seq_encode_2), each of
    shape (4, L). Positions outside the scanned window are 0. When
    `return_positions` is True, the scanned offsets are appended.
    """
    import numpy as np
    from tqdm import tqdm

    # ---- resolve the peak identifier --------------------------------------
    var_index = model.mdata[atac_mod].var.index
    if isinstance(peak, (int, np.integer)):
        peak_idx = int(peak)
        peak_name = var_index[peak_idx]
    elif isinstance(peak, str):
        peak_name = peak
        peak_idx = var_index.get_loc(peak)
    else:
        logger.error("Invalid peak input, must be a single peak index or peak name")
        return None

    logger.info(f"Saturation mutagenesis for peak {peak_name} (index {peak_idx})")

    # ---- pull the sequences for this peak ---------------------------------
    # Filter on the resolved name: the original code filtered on `peak`, which
    # silently returned an empty frame whenever an integer index was passed.
    seq_table = model.mdata[atac_mod].uns["personal_genome_seq"]
    df_seq = seq_table[seq_table["peak"] == peak_name]
    if df_seq.empty:
        logger.error(f"No sequence entries found for peak {peak_name}")
        return None

    # assert df_seq["seq_1"].nunique() == 1, "All seq_1 for a peak should be identical"
    # assert df_seq["seq_2"].nunique() == 1, "All seq_2 for a peak should be identical"

    ref_seq_1 = df_seq["seq_1"].iat[0]
    ref_seq_2 = df_seq["seq_2"].iat[0]
    assert len(ref_seq_1) == len(ref_seq_2), "seq_1 and seq_2 must be the same length"
    seq_len = len(ref_seq_1)

    # ---- work out which positions to scan ---------------------------------
    if snp_pos is None:
        positions = np.arange(seq_len)
        anchor = None
    else:
        if coord == "genomic":
            peak_start = int(df_seq["start"].iat[0])
            anchor = int(snp_pos) - peak_start
        elif coord == "offset":
            anchor = int(snp_pos)
        else:
            raise ValueError("coord must be 'genomic' or 'offset'")

        if not 0 <= anchor < seq_len:
            raise ValueError(
                f"anchor {anchor} falls outside the peak sequence "
                f"(length {seq_len}); check snp_pos and the `coord` setting"
            )

        lo = max(0, anchor - window)
        hi = min(seq_len, anchor + window + 1)
        positions = np.arange(lo, hi)
        logger.info(
            f"Anchored at offset {anchor}; scanning [{lo}, {hi}) "
            f"= {len(positions)} of {seq_len} bp"
        )

    # ---- reference prediction, computed once ------------------------------
    logger.info("Predicting expression using reference sequences")
    pred_ref = model.predict(
        model.mdata, 
        rna_mod=rna_mod, 
        atac_mod=atac_mod,
        batch_size=batch_size, 
        num_workers=num_workers,
    )
    mean_ref = float(np.mean(pred_ref))

    bases = ("A", "C", "G", "T")

    def _scan(ref_seq: str, col: str) -> np.ndarray:
        """Mutate `col` in place position by position; always restore it."""
        scores = np.zeros(seq_len, dtype=float)
        try:
            for i in tqdm(positions, desc=f"ISM {col}"):
                ref_base = ref_seq[i]
                acc, n_alt = 0.0, 0
                for alt in bases:
                    if alt == ref_base:
                        continue
                    alt_seq = ref_seq[:i] + alt + ref_seq[i + 1:]
                    seq_table.loc[df_seq.index, col] = alt_seq
                    pred_alt = model.predict(
                        model.mdata, 
                        rna_mod=rna_mod, 
                        atac_mod=atac_mod,
                        batch_size=batch_size, 
                        num_workers=num_workers,
                    )
                    acc += float(np.mean(pred_alt))
                    n_alt += 1

                mean_alt = acc / n_alt
                if logfc:
                    # Guard against non-positive means, which log2 cannot take
                    if mean_ref <= 0 or mean_alt <= 0:
                        scores[i] = np.nan
                    else:
                        scores[i] = np.log2(mean_ref / mean_alt)
                else:
                    scores[i] = mean_ref - mean_alt
        finally:
            # Restore even if prediction raises, otherwise mdata stays corrupted
            seq_table.loc[df_seq.index, col] = ref_seq
        return scores

    logger.info("Predicting expression using alternative sequences")
    ism_scores_1 = _scan(ref_seq_1, "seq_1")
    ism_scores_2 = _scan(ref_seq_2, "seq_2")

    # ---- smoothing, restricted to the scanned span ------------------------
    if smoothing and window_size > 1:
        kernel = np.ones(window_size) / window_size
        lo, hi = int(positions[0]), int(positions[-1]) + 1
        for scores in (ism_scores_1, ism_scores_2):
            # Convolving the full array would drag the zero padding outside the
            # window into the edges of the real signal
            scores[lo:hi] = np.convolve(scores[lo:hi], kernel, mode="same")

    # ---- broadcast to (4, L) and mask to the reference base ---------------
    ism_scores_1 = np.tile(ism_scores_1, (4, 1))
    ism_scores_2 = np.tile(ism_scores_2, (4, 1))

    ref_seq_encode_1 = seq_to_one_hot(ref_seq_1).transpose()
    ref_seq_encode_2 = seq_to_one_hot(ref_seq_2).transpose()
    if multiply_by_inputs:
        ism_scores_1 = ism_scores_1 * ref_seq_encode_1
        ism_scores_2 = ism_scores_2 * ref_seq_encode_2

    out = (ism_scores_1, ref_seq_encode_1, ism_scores_2, ref_seq_encode_2)
    return out + (positions,) if return_positions else out

# def saturation_mutagenesis_with_genotype_v2(
#     model: Cell2NetWithGenotype,
#     peak: int | str = None,
#     rna_mod: str = "rna",
#     atac_mod: str = "atac",
#     batch_size: int = 32,
#     num_workers: int = 1,
#     multiply_by_inputs: bool = True,
#     logfc: bool = False,
#     smoothing: bool = True,
#     window_size: int = 3,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
#     if isinstance(peak, int):
#         peak_idx = peak
#     elif isinstance(peak, str):
#         peak_idx = model.mdata[atac_mod].var.index.get_loc(peak)
#     else:
#         logger.error(
#             "Invalid peak input, must be a single peak index or a single peak name"
#         )
#         return None

#     logger.info(f"Compute saturation mutagenesis for peak {peak_idx}")

#     logger.info("Predicting expression using reference sequence")
#     pred_ref = model.predict(model.mdata,
#                              rna_mod=rna_mod,
#                              atac_mod=atac_mod,
#                              batch_size=batch_size,
#                              num_workers=num_workers)

#     # get reference sequence for the peak
#     df_seq = model.mdata[atac_mod].uns["personal_genome_seq"]
#     df_seq = df_seq[df_seq['peak'] == peak]

#     # ensure all seq_1 are the same
#     assert len(df_seq['seq_1'].unique()) == 1, "All seq_1 should be the same"
#     assert len(df_seq['seq_2'].unique()) == 1, "All seq_2 should be the same"

#     ref_seq_1 = df_seq['seq_1'].values[0]
#     ref_seq_2 = df_seq['seq_2'].values[0]

#     # ensure seq_1 and seq_2 are the same length
#     assert len(ref_seq_1) == len(ref_seq_2), "seq_1 and seq_2 should be the same length"

#     logger.info("Predicting expression using alternative sequences")
#     bases = ['A', 'C', 'G', 'T']
#     ism_scores_1 = []
#     for i in tqdm(range(len(ref_seq_1))):
#         pred_alt = np.zeros(model.mdata.n_obs)

#         # compute predictions for mutated sequence
#         for alt in bases:
#             if alt != ref_seq_1[i]:
#                 alt_seq = ref_seq_1[:i] + alt + ref_seq_1[i+1:]

#                 # replace the sequence in mdata
#                 model.mdata[atac_mod].uns['personal_genome_seq'].loc[df_seq.index, 'seq_1'] = alt_seq
#                 pred_alt += model.predict(model.mdata,
#                                           rna_mod=rna_mod,
#                                           atac_mod=atac_mod,
#                                           batch_size=batch_size,
#                                           num_workers=num_workers)

#         # average predictions for the alternative base
#         pred_alt /= (len(bases) - 1)

#         # compute the effect size across all cells
#         if logfc:
#             ism_scores_1.append(np.log2(np.mean(pred_ref) / np.mean(pred_alt)))
#         else:
#             ism_scores_1.append(np.mean(pred_ref - pred_alt))

#     # restore the original sequence
#     model.mdata[atac_mod].uns['personal_genome_seq'].loc[df_seq.index, 'seq_1'] = ref_seq_1

#     ism_scores_2 = []
#     for i in tqdm(range(len(ref_seq_2))):
#         pred_alt = np.zeros(model.mdata.n_obs)

#         # compute predictions for mutated sequence
#         for alt in bases:
#             if alt != ref_seq_2[i]:
#                 alt_seq = ref_seq_2[:i] + alt + ref_seq_2[i+1:]

#                 # replace the sequence in mdata
#                 model.mdata[atac_mod].uns['personal_genome_seq'].loc[df_seq.index, 'seq_2'] = alt_seq
#                 pred_alt += model.predict(model.mdata,
#                                           rna_mod=rna_mod,
#                                           atac_mod=atac_mod,
#                                           batch_size=batch_size,
#                                           num_workers=num_workers)

#         # average predictions for the alternative base
#         pred_alt /= (len(bases) - 1)

#         # compute the effect size across all cells
#         if logfc:
#             ism_scores_2.append(np.log2(np.mean(pred_ref) / np.mean(pred_alt)))
#         else:
#             ism_scores_2.append(np.mean(pred_ref - pred_alt))

#     # restore the original sequence
#     model.mdata[atac_mod].uns['personal_genome_seq'].loc[df_seq.index, 'seq_2'] = ref_seq_2

#     ism_scores_1 = np.array(ism_scores_1)
#     ism_scores_2 = np.array(ism_scores_2)

#     # if smoothing is needed, we can use a simple moving average
#     if smoothing:
#         ism_scores_1 = np.convolve(ism_scores_1, np.ones(window_size)/window_size, mode='same')
#         ism_scores_2 = np.convolve(ism_scores_2, np.ones(window_size)/window_size, mode='same')

#     ism_scores_1 = np.tile(ism_scores_1, (4, 1))
#     ism_scores_2 = np.tile(ism_scores_2, (4, 1))

#     ref_seq_encode_1 = seq_to_one_hot(ref_seq_1).transpose()
#     ref_seq_encode_2 = seq_to_one_hot(ref_seq_2).transpose()
#     if multiply_by_inputs:
#         ism_scores_1 = ism_scores_1 * ref_seq_encode_1
#         ism_scores_2 = ism_scores_2 * ref_seq_encode_2

#     return ism_scores_1, ref_seq_encode_1, ism_scores_2, ref_seq_encode_2


# def saturation_mutagenesis_v2(
#     model: Cell2Net,
#     peak: int | str = None,
#     rna_mod: str = "rna",
#     atac_mod: str = "atac",
#     batch_size: int = 32,
#     num_workers: int = 1,
#     multiply_by_inputs: bool = True,
#     logfc: bool = False,
#     smoothing: bool = True,
#     window_size: int = 3,
#     return_seq: bool = False,
# ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:

#     if isinstance(peak, int):
#         peak_idx = peak
#     elif isinstance(peak, str):
#         peak_idx = model.mdata[atac_mod].var.index.get_loc(peak)
#     else:
#         logger.error(
#             "Invalid peak input, must be a single peak index or a single peak name"
#         )
#         return None

#     logger.info(f"Compute saturation mutagenesis for peak {peak_idx}")

#     logger.info("Predicting expression using reference sequence")
#     pred_ref = model.predict(model.mdata,
#                              rna_mod=rna_mod,
#                              atac_mod=atac_mod,
#                              batch_size=batch_size,
#                              num_workers=num_workers)


#     # get reference sequence for the peak
#     ref_seq = model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx]

#     logger.info("Predicting expression using alternative sequences")
#     bases = ['A', 'C', 'G', 'T']
#     ism_scores = []
#     for i in tqdm(range(len(ref_seq))):
#         pred_alt = np.zeros(model.mdata.n_obs)

#         # compute predictions for mutated sequence
#         for alt in bases:
#             if alt != ref_seq[i]:
#                 alt_seq = ref_seq[:i] + alt + ref_seq[i+1:]
#                 model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx] = alt_seq

#                 pred_alt += model.predict(model.mdata,
#                                           rna_mod=rna_mod,
#                                           atac_mod=atac_mod,
#                                           batch_size=batch_size,
#                                           num_workers=num_workers)

#         # average predictions for the alternative base
#         pred_alt /= (len(bases) - 1)

#         fc = np.divide(pred_ref, pred_alt, out=np.zeros_like(pred_ref), where=pred_alt!=0)
#         logfc = np.log2(fc).mean()
#         ism_scores.append(logfc)

#     model.mdata[atac_mod].uns["peaks"]["sequence"][peak_idx] = ref_seq  # restore the original sequence

#     ism_scores = np.array(ism_scores)

#     # if smoothing is needed, we can use a simple moving average
#     if smoothing:
#         ism_scores = np.convolve(ism_scores, np.ones(window_size)/window_size, mode='same')

#     ism_scores = np.tile(ism_scores, (4, 1))

#     ref_seq_encode = seq_to_one_hot(ref_seq).transpose()
#     if multiply_by_inputs:
#         ism_scores = ism_scores * ref_seq_encode

#     if return_seq:
#         return ism_scores, ref_seq_encode
#     else:
#         return ism_scores
