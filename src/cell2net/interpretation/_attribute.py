import numpy as np
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.prediction.data import get_dataloader
from cell2net.prediction.model import Cell2Net


def compute_attribution(
    model: Cell2Net,
    idx: list[int] | list[str] | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
):
    r"""
    Calculate the attribution of each input feature to gene expression.

    This is done using the Integrated Gradients algorithm from captum package.
    Note to calcualte attribution of peak sequences,
    peak accessibility and TF expression for each single cell independently.

    Parameters
    ----------
    model : Cell2Net
        Model that has been trained
    idx : list[int] | list[str] | None, optional
        A list of int or string to indicate, by default None
    batch_size : int, optional
        Batch size, by default 32
    num_workers : int, optional
        _description_, by default 4

    Returns
    -------
    _type_
        _description_
    """
    # create a dataloader
    data_loader = get_dataloader(
        mdata=model.mdata,
        covariates=model.covariates,
        idx=idx,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=True,
    )

    model.to_device(model.device)
    model.module.train()

    # Use Integrated Gradients to estimate feature importances
    ig = IntegratedGradients(model.module)

    # create baselines for computing integral gradients
    # for DNA sequences, we generate random sequences with same length
    # for peak accessibility and TF expression, we use zeros as base line
    # for covarinces, we can use the input as baselines
    # rand_seq_list = []
    # for _ in range(model.n_peaks):
    #     rand_seq_list.append(random_seq(seq_len=256))

    # _peak_seq = encode_seq(rand_seq_list).unsqueeze(0)

    atac_attr, dna_attr, tf_attr = [], [], []
    for data in data_loader:
        peak_seq = data["peak_seq"].to(model.device).requires_grad_()
        peak_acc = data["peak_acc"].to(model.device).requires_grad_()
        tf_exp = data["tf_exp"].to(model.device).requires_grad_()
        covariates = data["covariates"].to(model.device).requires_grad_()

        # _dna = torch.zeros_like(atac).to(model.device)
        # _atac = torch.zeros_like(atac).to(model.device)
        # _tf_exp = torch.zeros_like(tf_exp).to(model.device)

        attributions, delta = ig.attribute(
            inputs=(peak_seq, peak_acc, tf_exp, covariates),
            return_convergence_delta=True,
        )

        dna_attr.append(attributions[0].detach().cpu())
        atac_attr.append(attributions[1].detach().cpu())
        tf_attr.append(attributions[2].detach().cpu())

    peak_seq_attr = torch.cat(dna_attr, dim=0).numpy()
    peak_acc_attr = torch.cat(atac_attr, dim=0).numpy()
    tf_exp_attr = torch.cat(tf_attr, dim=0).numpy()

    return peak_seq_attr, peak_acc_attr, tf_exp_attr


def compute_peak_attr(
    model: Cell2Net,
    idx: list[int] | list[str] | None = None,
    batch_size: int = 8,
    num_workers: int = 1,
    n_steps: int = 50,
    multiply_by_inputs: bool = True,
) -> np.ndarray:
    """
    Calculate the attribution of peak accessibility to gene expression.

    Parameters
    ----------
    model : Cell2Net
        Model that has been trained
    idx : list[int] | list[str] | None, optional
        A list of int or string to indicate.
        If set to None, use all cells. Default: None
    batch_size : int, optional
        Batch size, by default 8
    num_workers : int, optional
        Number of CPUs for dataloader, by default 1
    baseline: str, optional
        How to create baseline to compute integrated gradients. Default: "zero"
    n_steps: int, optional
        Number of steps used by the approximation method. Default: 50.
    multiply_by_inputs: bool, optional
        Whether or multiply input features when estimating attribution. Default: True

    Returns
    -------
    np.ndarray
        An numpy array of peak attribution with a shape of (n_cells, n_peaks)
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


def compute_tf_attr(
    model: Cell2Net,
    idx: list[int] | list[str] | None = None,
    batch_size: int = 4,
    num_workers: int = 1,
    n_steps: int = 100,
    multiply_by_inputs: bool = True,
) -> np.ndarray:
    r"""
    Calculate the attribution of TF expression to target gene expression.

    Parameters
    ----------
    model : Cell2Net
        Model that has been trained
    idx : list[int] | list[str] | None, optional
        A list of int or string to indicate, by default None
    batch_size : int, optional
        Batch size, by default 32
    num_workers : int, optional
        Number of CPUs for dataloader, by default 4

    Returns
    -------
    np.ndarray
        An numpy array of TF attribution with a shape of (n_cells, n_tfs)
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


def compute_seq_attr():
    pass
