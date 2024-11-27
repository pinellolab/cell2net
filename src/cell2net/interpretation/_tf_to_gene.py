import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from mudata import MuData
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net


def compute_tf_attr(
    model: Cell2Net,
    idx: list[int] | list[str] | None = None,
    method: str = "IntegratedGradients",
    **kwargs,
) -> np.ndarray:
    r"""
    Calculate the attribution of TF expression to target gene expression.

    Parameters
    ----------
    model : Cell2Net
        Model that has been trained
    idx : list[int] | list[str] | None, optional
        A list of int or string to indicate, by default None

    Returns
    -------
    np.ndarray
        An numpy array of TF attribution with a shape of (n_cells, n_tfs)
    """
    # create a dataloader
    match method:
        case "IntegratedGradients":
            attr = _integrated_gradients(model=model, idx=idx, **kwargs)

    return attr


def _integrated_gradients(
    model, idx, batch_size=4, num_workers=1, n_steps=100, multiply_by_inputs=True
) -> np.ndarray:
    """
    Compute TF attribution using integrated gradients

    Parameters
    ----------
    model : _type_
        _description_
    idx : _type_
        _description_
    batch_size : int, optional
        _description_, by default 4
    num_workers : int, optional
        _description_, by default 1
    n_steps : int, optional
        _description_, by default 100
    multiply_by_inputs : bool, optional
        _description_, by default True

    Returns
    -------
    np.ndarray
        _description_
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
    for data in tqdm(data_loader):
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
    Extracts TF-gene regulation base on the attribution of transcription factor expression

    Parameters
    ----------
    mdata : MuData
        MuData object including RNA and ATAC modalities
    attr : np.ndarray
        A numpy array of TF expression attribution
    groupby : str | None, optional
        Name of one column in mdata.obs to group cells. by default None

    Returns
    -------
    pd.DataFrame
        Calculated t-statistic and p-value for each tf-gene pair
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
                "regulation": np.mean(_attr, axis=0),
            }
        )
        df_list.append(df)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    # # filter by min_attr
    # if min_attr:
    #     df = df[df["avg_attribution"] > min_attr].reset_index(drop=True)

    # Group by 'group_column' and select the top 20 rows within each group
    df = (
        df.groupby(groupby)
        .apply(lambda x: x.nlargest(n_tfs, "regulation"))
        .reset_index(drop=True)
    )

    return df


def get_top_regulator():
    pass
