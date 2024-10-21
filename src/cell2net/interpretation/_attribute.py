import torch
from captum.attr import IntegratedGradients

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


def compute_peak_attribution(
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

    atac_attr, dna_attr, tf_attr = [], [], []
    for data in data_loader:
        peak_seq = data["peak_seq"].to(model.device).requires_grad_()
        peak_acc = data["peak_acc"].to(model.device).requires_grad_()
        tf_exp = data["tf_exp"].to(model.device).requires_grad_()
        covariates = data["covariates"].to(model.device).requires_grad_()

        _peak_acc = torch.zeros_like(peak_acc).to(model.device)

        attributions, delta = ig.attribute(
            inputs=(peak_seq, peak_acc, tf_exp, covariates),
            baselines=(peak_seq, _peak_acc, tf_exp, covariates),
            return_convergence_delta=True,
        )

        dna_attr.append(attributions[0].detach().cpu())
        atac_attr.append(attributions[1].detach().cpu())
        tf_attr.append(attributions[2].detach().cpu())

    peak_seq_attr = torch.cat(dna_attr, dim=0).numpy()
    peak_acc_attr = torch.cat(atac_attr, dim=0).numpy()
    tf_exp_attr = torch.cat(tf_attr, dim=0).numpy()

    return peak_seq_attr, peak_acc_attr, tf_exp_attr
