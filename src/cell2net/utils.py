import os
import random

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from adpbulk import ADPBulk


def set_seed(seed=42):
    """Set random seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def random_seq(seq_len: int, bases: list[str] | None = None) -> str:
    """
    Generate a random DNA sequence

    Parameters
    ----------
    seq_len : int
        Length of the sequence
    bases : list[str] | None, optional
        Bases of sequence, by default ACTG

    Returns
    -------
    str
        DNA sequence
    """
    if bases is None:
        bases = ["A", "C", "G", "T"]

    rand_seq = "".join([np.random.choice(bases) for i in range(seq_len)])
    return rand_seq


def one_hot_encode(seq) -> torch.Tensor:
    """
    Convert a sequence to one-hot encoding

    Only ACTGN allow.

    Parameters
    ----------
    seq : _type_
        _description_

    Returns
    -------
    torch.Tensor
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(
            f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}"
        )

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [0.0, 0.0, 0.0, 0.0],
    }

    # Create array from nucleotide sequence
    vec = torch.tensor([nuc_d[x] for x in seq], dtype=torch.float32)

    return vec


def encode_seq(seq_list):
    data = []
    for seq in seq_list:
        data.append(one_hot_encode(seq))

    data = torch.stack(data)

    return data


def create_bulk_adata(
    adata: ad.AnnData,
    groupby: str,
    normalize: bool = True,
    target_sum: int | float = 10000,
    log1p: bool = True,
) -> ad.AnnData:
    """
    Create a pseudo-bulk anndata file using groupby information

    Parameters
    ----------
    adata : ad.AnnData
        _description_
    groupby : str
        _description_

    Returns
    -------
    ad.AnnData
        _description_
    """
    adpb = ADPBulk(adata, [groupby])

    # perform the pseudobulking
    counts = adpb.fit_transform()

    sample_meta = adpb.get_meta().set_index("SampleName")
    adata_bulk = ad.AnnData(X=counts, obs=sample_meta)
    adata_bulk.layers["counts"] = adata_bulk.X  # type: ignore

    if normalize:
        sc.pp.normalize_total(adata_bulk, target_sum=target_sum, layer="counts")

    if log1p:
        sc.pp.log1p(adata_bulk)

    return adata_bulk
