from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from mudata import MuData

from cell2net._logging import logger
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net


def compute_peak_attr(
    model: Cell2Net,
    idx: Sequence[int] | Sequence[str] | None = None,
    batch_size: int = 4,
    num_workers: int = 1,
    n_steps: int = 100,
    multiply_by_inputs: bool = True,
) -> np.ndarray:
    """
    Compute feature attributions for peak accessibility using Integrated Gradients.

    This function uses Integrated Gradients (IG) to compute the attributions for
    peak accessibility features in a Cell2Net model. Attributions indicate the
    importance of each feature for the model's predictions.

    Parameters
    ----------
    model :
        The trained Cell2Net model for which feature attributions are computed.
        The model should have an attribute `mdata` for data, and it must be
        set up for gradient computations.
    idx :
        Indices or identifiers for the specific observations to include in the
        attribution computation.
        If None, all observations in the dataset are used.
    batch_size :
        The number of samples per batch for the DataLoader.
    num_workers :
        The number of worker threads to use for data loading.
    n_steps :
        The number of interpolation steps for the Integrated Gradients computation.
    multiply_by_inputs :
        Whether to scale attributions by the input values as per the Integrated Gradients
        method.

    Returns
    -------
    An array of attributions for peak accessibility features, with shape corresponding to the input dataset.

    Notes
    -----
    - This function uses the `captum` library for Integrated Gradients.
    - The model is expected to have the following inputs:

        - `peak_seq`: One-hot encoded sequence of peaks.
        - `peak_acc`: Peak accessibility values (with gradients enabled).
        - `peak_dist`: Distances of peaks to transcription start sites.
        - `tf_exp`: Transcription factor expression values.
        - `covariates`: Additional covariates provided as arguments.

    - Baseline values are set to zero for `peak_acc`.

    Examples
    --------
    >>> model = Cell2Net(mdata, ...)
    >>> attributions = compute_peak_attr(
    ...     model=model,
    ...     idx=[0, 1, 2],
    ...     batch_size=8,
    ...     num_workers=2,
    ...     n_steps=50,
    ... )
    >>> print(attributions.shape)
    (number_of_samples, number_of_features)
    """
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

    model.module.train()

    # Use Integrated Gradients to estimate feature importances
    ig = IntegratedGradients(model.module, multiply_by_inputs=multiply_by_inputs)

    logger.info("Compute attribution for peak accessibility")
    attr = []
    for data in data_loader:
        peak_seq = data["peak_seq"].to(model.device)
        peak_acc = data["peak_acc"].to(model.device).requires_grad_()
        peak_dist = data["peak_dist"].to(model.device)
        tf_exp = data["tf_exp"].to(model.device)
        covariates = data["covariates"].to(model.device)

        _peak_acc = torch.zeros_like(peak_acc)

        attributions = ig.attribute(
            inputs=(peak_seq, peak_acc, peak_dist, tf_exp),
            baselines=(peak_seq, _peak_acc, peak_dist, tf_exp),
            additional_forward_args=covariates,
            return_convergence_delta=False,
            n_steps=n_steps,
        )

        attr.append(attributions[1].detach().cpu())
        del attributions

    attr = torch.cat(attr, dim=0).numpy()

    return attr


def peak_to_gene(
    mdata: MuData,
    attr: np.ndarray,
    groupby: str | None = None,
) -> pd.DataFrame:
    """
    Extracts peak-to-gene links based on the attribution of peak accessibility

    This function assigns peak-level attributions to their corresponding genes based on
    the `peak_to_gene` mapping in a MuData object. It computes the average attribution
    for each peak, either across all cells or grouped by a specified metadata column.

    Parameters
    ----------
    mdata :
        A MuData object containing multi-modal single-cell data. It must have:

        - `mdata["rna"]`: RNA modality with gene names in `var_names`.
        - `mdata.uns["peak_to_gene"]`: A mapping between peaks and genes with a column "peak".
        - `mdata.obs`: Cell metadata, required if `groupby` is specified.

    attr :
        A 2D array of peak-level attributions with shape `(n_cells, n_peaks)`.
        Rows correspond to cells, and columns correspond to peaks.

    groupby :
        The name of a column in `mdata.obs` to group cells by. If None, attributions
        are averaged across all cells.

    Returns
    -------
    A DataFrame summarizing peak-to-gene attributions with the following columns:

        - "peak": Peak identifiers.
        - "gene": The associated gene (from the first gene in `mdata["rna"].var_names`).
        - "avg_attr": Average attribution for each peak.
        - Additional column(s) for group labels if `groupby` is specified.

    Raises
    ------
    AssertionError
        If `groupby` is specified but not found in `mdata.obs`.
        If the length of the `groupby` column does not match the number of cells in `attr`.

    Notes
    -----
    - If `groupby` is None, the function computes average attributions across all cells.
    - If `groupby` is specified, the function computes group-specific average attributions.
    - The `mdata.uns["peak_to_gene"]["peak"]` must contain a mapping of peaks to genes.

    Examples
    --------
    >>> mdata = MuData(...)  # Load MuData object
    >>> attr = np.random.rand(100, 5000)  # Example attributions for 100 cells and 5000 peaks
    >>> # Compute average attribution across all cells
    >>> df = peak_to_gene(mdata, attr)
    >>> print(df.head())
         peak    gene   avg_attr
    0  peak_1  gene_1  0.123456
    1  peak_2  gene_1  0.234567

    >>> # Compute group-specific average attributions
    >>> df_grouped = peak_to_gene(mdata, attr, groupby="cell_type")
    >>> print(df_grouped.head())
         peak    gene    cell_type   avg_attr
    0  peak_1  gene_1  B_cells      0.123456
    1  peak_2  gene_1  T_cells      0.234567

    """
    gene = mdata["rna"].var_names[0]
    if groupby is None:
        # compute average attribution using all cells
        df = pd.DataFrame(
            data={
                "peak": mdata.uns["peak_to_gene"]["peak"],
                "gene": gene,
                "avg_attr": np.sum(attr, axis=0),
            }
        )
    else:
        # compute average attribution for each group
        assert groupby in mdata.obs.columns, f"Cannot find {groupby} in mdata.obs"

        groups = mdata.obs[groupby].values

        assert (
            len(groups) == attr.shape[0]
        ), f"Length of grouby {len(groups)} is different from number of cells {attr.shape[0]}"

        unique_groups, group_indices = np.unique(groups, return_inverse=True)  # type: ignore

        df_list = []
        # For each group, subset the attribution and perform t-test
        for i, unique_group in enumerate(unique_groups):
            _attr = attr[group_indices == i]
            df = pd.DataFrame(
                data={
                    "peak": mdata.uns["peak_to_gene"]["peak"],
                    "gene": gene,
                    groupby: unique_group,
                    "avg_attr": np.sum(_attr, axis=0),
                }
            )
            df_list.append(df)

        df = pd.concat(df_list, axis=0)

    return df
