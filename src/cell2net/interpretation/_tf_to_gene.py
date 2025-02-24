import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from mudata import MuData

from cell2net._logging import logger
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net


def compute_tf_attr(
    model: Cell2Net,
    idx: list[int] | list[str] | None = None,
    batch_size=4,
    num_workers=1,
    n_steps=100,
    multiply_by_inputs=True,
) -> np.ndarray:
    """
    Compute transcription factor (TF) attribution using Integrated Gradients

    This function calculates the attribution of TF expression to the output of
    a `Cell2Net` model using the Integrated Gradients method. The attributions
    are computed over a specified dataset and returned as a NumPy array.

    Parameters
    ----------
    model :
        The trained `Cell2Net` model. It must have a `mdata` attribute for metadata
        and a `covariates` attribute for covariate information.
    idx :
        Indices or identifiers specifying the subset of the data to compute
        attributions for. If `None`, the entire dataset is used.
    batch_size :
        The batch size to use for data loading.
    num_workers :
        Number of worker processes for data loading.
    n_steps :
        The number of steps to use for the Integrated Gradients computation.
        Larger values provide more accurate estimates but increase computation time.
    multiply_by_inputs :
        Whether to multiply the attributions by the inputs.
        This is recommended to preserve implementation invariance.

    Returns
    -------
        A NumPy array containing the attributions for TF expression. The shape
        of the output depends on the dataset and the number of TFs modeled.

    Notes
    -----
    - This function uses the Integrated Gradients algorithm for attribution computation. The `captum` library is required to perform this calculation.
    - The attributions are computed for the `tf_exp` input (transcription factor expression) while keeping other inputs (peak sequence, accessibility, and distance) fixed.
    - The model is set to training mode (`model.module.train()`) during computation.

    Examples
    --------
    >>> model = Cell2Net(...)
    >>> idx = [0, 1, 2, 3]  # Indices of samples to compute attributions for
    >>> attributions = compute_tf_attr(
    ...     model=model,
    ...     idx=idx,
    ...     batch_size=4,
    ...     num_workers=2,
    ...     n_steps=50,
    ...     multiply_by_inputs=True,
    ... )
    >>> attributions.shape
    (4, num_tfs)
    """
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

    logger.info("Compute attribution for TF expression")
    attr = []
    for data in data_loader:
        peak_seq = data["peak_seq"].to(model.device)
        peak_acc = data["peak_acc"].to(model.device)
        peak_dist = data["peak_dist"].to(model.device)
        tf_exp = data["tf_exp"].to(model.device).requires_grad_()
        covariates = data["covariates"].to(model.device)

        _tf_exp = torch.zeros_like(tf_exp)

        attributions = ig.attribute(
            inputs=(peak_seq, peak_acc, peak_dist, tf_exp),
            baselines=(peak_seq, peak_acc, peak_dist, _tf_exp),
            additional_forward_args=covariates,
            return_convergence_delta=False,
            n_steps=n_steps,
        )

        attr.append(attributions[3].detach().cpu())
        del attributions

    attr = torch.cat(attr, dim=0).numpy()

    return attr


def tf_to_gene(
    mdata: MuData,
    attr: np.ndarray,
    groupby: str,
    n_tfs: int = 10,
) -> pd.DataFrame:
    """
    Aggregate transcription factor (TF) attributions and link them to genes for each group.

    This function computes the mean TF regulation values for each group specified in
    the metadata, links them to a single gene (default RNA dataset), and selects the
    top `n_tfs` most regulated TFs for each group. The results are returned as a
    pandas DataFrame.

    Parameters
    ----------
    mdata :
         A `MuData` object containing the metadata and RNA dataset. The object must have:

            - `obs` (metadata) with the column specified by `groupby`.
            - `uns["tfs"]` containing the list of transcription factors.
            - `["rna"].var_names` containing gene names.

    attr :
        A 2D NumPy array of shape `(n_cells, n_tfs)` containing TF attributions. Each
        row corresponds to a cell, and each column corresponds to a TF.
    groupby :
        The column name in `mdata.obs` to group cells by (e.g., cell type, cluster ID).
    n_tfs :
        The number of top transcription factors to select for each group based on their
        mean regulation values.

    Returns
    -------
        A pandas DataFrame with the following columns:

        - `tf`: The transcription factor name.
        - `gene`: The linked gene (from `mdata["rna"].var_names[0]`).
        - `groupby`: The group name (e.g., cell type or cluster).
        - `attribution`: The mean attribution value of the TF within the group.

        The DataFrame is grouped by the `groupby` column, with the top `n_tfs` TFs
        included for each group.

    Raises
    ------
    AssertionError
        If the `groupby` column is not present in `mdata.obs`, or if the length of the
        `groupby` column does not match the number of rows in `attr`.

    Notes
    -----
    - The function assumes that `mdata.uns["tfs"]` contains a list of transcription factor names, and `mdata["rna"].var_names[0]` provides the associated gene name.
    - Within each group, TF regulation values are aggregated by their mean, and the top `n_tfs` with the highest mean regulation are retained.

    Examples
    --------
    >>> mdata = MuData(...)  # MuData object with metadata and RNA data
    >>> attr = np.random.rand(100, 20)  # Example attribution array (100 cells, 20 TFs)
    >>> groupby = "cell_type"
    >>> df = tf_to_gene(mdata, attr, groupby, n_tfs=5)
    >>> df.head()
       tf      gene      cell_type  attribution
    0  TF1     Gene1    Type1      0.1234
    1  TF2     Gene1    Type1      0.1123
    2  TF3     Gene1    Type1      0.0987
    3  TF4     Gene1    Type1      0.0876
    4  TF5     Gene1    Type1      0.0765
    """
    assert groupby in mdata.obs.columns, print(f"Cannot find {groupby} in mdata.obs")

    groups = mdata.obs[groupby].values

    assert len(groups) == attr.shape[0], print(
        f"Length of grouby {len(groups)} is different from number of cells {attr.shape[0]}"
    )
    unique_groups, group_indices = np.unique(groups, return_inverse=True)  # type: ignore

    df_list = []
    for i, unique_group in enumerate(unique_groups):
        _attr = attr[group_indices == i]
        df = pd.DataFrame(
            data={
                "tf": mdata.uns["tfs"],
                "gene": mdata["rna"].var_names[0],
                groupby: unique_group,
                "attribution": np.mean(_attr, axis=0),
            }
        )
        df_list.append(df)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    # Group by 'group_column' and select the top 20 rows within each group
    df = (
        df.groupby(groupby)
        .apply(lambda x: x.nlargest(n_tfs, "attribution"))
        .reset_index(drop=True)
    )

    return df


def get_top_regulator():
    pass
