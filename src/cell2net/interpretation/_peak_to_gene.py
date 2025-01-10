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
        If None, all observations in the dataset are used, by default None.
    batch_size :
        The number of samples per batch for the DataLoader, by default 4
    num_workers :
        The number of worker threads to use for data loading, by default 1
    n_steps :
        The number of interpolation steps for the Integrated Gradients computation, by default 100
    multiply_by_inputs :
        Whether to scale attributions by the input values as per the Integrated Gradients
        method, by default True

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


# def integrated_gradients(
#     model: Cell2Net,
#     idx: Sequence[int] | Sequence[str] | None = None,
#     batch_size: int = 4,
#     num_workers: int = 1,
#     n_steps: int = 100,
#     multiply_by_inputs: bool = True,
# ) -> np.ndarray:
#     """
#     Compute peak attribution using integrated gradients

#     Parameters
#     ----------
#     model : Cell2Net
#         Model that has been trained
#     idx : Sequence[int] | Sequence[str] | None, optional
#         A list of int or string to indicate. If set to None, use all data, by default None
#     batch_size : int, optional
#         Batch size, by default 4
#     num_workers : int, optional
#         Number of CPUs used to load data, by default 1
#     n_steps : int, optional
#         Number of steps used by the approximation method, by default 100
#     multiply_by_inputs : bool, optional
#         Whether to factor model inputs' multiplier in the final attribution scores,
#         by default True

#     Returns
#     -------
#     np.ndarray
#          An numpy array of peak attribution with a shape of (n_cells, n_peaks)
#     """
#     # create a dataloader
#     logger.info("Create dataloader")
#     data_loader = get_dataloader(
#         mdata=model.mdata,
#         covariates=model.covariates,
#         idx=idx,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=False,
#         shuffle=False,
#         drop_last=False,
#         persistent_workers=False,
#     )

#     model.module.train()

#     # Use Integrated Gradients to estimate feature importances
#     ig = IntegratedGradients(model.module, multiply_by_inputs=multiply_by_inputs)

#     logger.info("Compute attribution for peak accessibility")
#     attr = []
#     for data in data_loader:
#         peak_seq = data["peak_seq"].to(model.device)
#         peak_acc = data["peak_acc"].to(model.device).requires_grad_()
#         peak_dist = data["peak_dist"].to(model.device)
#         tf_exp = data["tf_exp"].to(model.device)
#         covariates = data["covariates"].to(model.device)

#         _peak_acc = torch.zeros_like(peak_acc)

#         attributions = ig.attribute(
#             inputs=(peak_seq, peak_acc, peak_dist, tf_exp),
#             baselines=(peak_seq, _peak_acc, peak_dist, tf_exp),
#             additional_forward_args=covariates,
#             return_convergence_delta=False,
#             n_steps=n_steps,
#         )

#         attr.append(attributions[1].detach().cpu())
#         del attributions

#     attr = torch.cat(attr, dim=0).numpy()

#     return attr


def peak_to_gene(
    mdata: MuData,
    attr: np.ndarray,
    groupby: str | None = None,
) -> pd.DataFrame:
    """
    Extracts peak-to-gene links based on the attribution of peak accessibility

    Parameters
    ----------
    mdata : MuData
        MuData object including RNA and ATAC modalities
    attr : np.ndarray
        A numpy array of peak accessibility attribution
    groupby : str | None, optional
        Name of one column in mdata.obs to group cells. by default None

    Returns
    -------
    pd.DataFrame
        Calculated t-statistic and p-value for each peak-to-gene link
    """
    gene = mdata["rna"].var_names[0]
    if groupby is None:
        # compute average attribution using all cells
        df = pd.DataFrame(
            data={
                "peak": mdata.uns["peak_to_gene"]["peak"],
                "gene": gene,
                "avg_attr": np.mean(attr, axis=0),
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
                    "avg_attr": np.mean(_attr, axis=0),
                }
            )
            df_list.append(df)

        df = pd.concat(df_list, axis=0)

    return df
