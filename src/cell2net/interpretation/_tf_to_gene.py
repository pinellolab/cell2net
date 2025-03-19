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

    model.module.eval()

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
    groupby: str | None = None,
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
    gene = mdata["rna"].var_names[0]
    if groupby is None:
        df = pd.DataFrame(
            data={
                "tf": mdata.uns["tfs"],
                "gene": gene,
                "mean_attr": np.mean(attr, axis=0),
                "std_attr": np.std(attr, axis=0),
            }
        )

        df = df.apply(lambda x: x.nlargest(n_tfs, "mean_attr")).reset_index(drop=True)

    else:
        assert groupby in mdata.obs.columns, print(
            f"Cannot find {groupby} in mdata.obs"
        )

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
                    "gene": gene,
                    groupby: unique_group,
                    "mean_attr": np.mean(_attr, axis=0),
                    "std_attr": np.std(_attr, axis=0),
                }
            )
            df_list.append(df)

        df = pd.concat(df_list, axis=0).reset_index(drop=True)

        # Group by 'group_column' and select the top 20 rows within each group
        df = (
            df.groupby(groupby)
            .apply(lambda x: x.nlargest(n_tfs, "mean_attr"))
            .reset_index(drop=True)
        )

    return df


def get_top_tfs(
    df: pd.DataFrame, n_top_tfs: int = 5, var_cutoff: float = 0.5
) -> pd.DataFrame:
    """
    Identifies the top transcription factors (TFs) with the highest variability and returns the top `n_top_tfs` TFs for each sample/column.

    The function follows these steps:
    1. Computes the variance of each row (TF) and filters the top `var_cutoff` fraction
       based on variance.
    2. Performs z-score normalization across each row (TF).
    3. Selects the `n_top_tfs` most highly expressed TFs for each column (sample).

    Parameters
    ----------
    df :
        A DataFrame where rows represent transcription factors (TFs) and columns represent samples.
    n_top_tfs :
        The number of top TFs to retrieve per sample/column, by default 5.
    var_cutoff :
        The fraction of TFs to retain based on variance ranking (0 to 1), by default 0.5.

    Returns
    -------
        A DataFrame where each column corresponds to a sample, and each row contains
        the top `n_top_tfs` TFs with the highest normalized expression.

    Notes
    -----
        - The function normalizes each TF across samples using z-score transformation.
        - TFs with the highest variability are prioritized based on the `var_cutoff` threshold.
        - If `var_cutoff=1.0`, all TFs are considered; if `var_cutoff=0.5`, only the top 50% most variable TFs are used for ranking.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> impot cell2net as cn
    >>> df = pd.DataFrame(np.random.rand(10, 5),
    ...                   index=[f'TF{i}' for i in range(10)],
    ...                   columns=[f'Sample{j}' for j in range(5)])
    >>> cn.ip.get_top_tfs(df, n_top_tfs=3, var_cutoff=0.5)
    """
    # copy the dataframe
    df_act = df.copy()

    df_act["tf_var"] = df_act.var(axis=1)
    df_act = df_act.sort_values(by="tf_var", ascending=False)
    df_act = df_act.head(int(len(df_act) * var_cutoff))
    df_act = df_act.drop(columns=["tf_var"])

    # z-score normalization
    df_norm = df_act.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
    df_norm = pd.DataFrame(df_norm, columns=df_act.columns, index=df_act.index)

    top_tfs = {}
    for col in df_norm.columns:
        df_top = df_norm.sort_values(by=col, ascending=False).head(n_top_tfs)
        top_tfs[col] = df_top.index.values.tolist()

    df = pd.DataFrame(data=top_tfs)

    return df
