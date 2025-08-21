from collections.abc import Sequence

from mudata import MuData
from torch.utils.data import DataLoader

from cell2net._logging import logger

from ._dataset import MuTorchDataset, MuTorchDatasetPersonalGenome


def get_dataloader(
    mdata: MuData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    use_personal_genome: bool = False,
    idx: Sequence[int] | Sequence[str] | None = None,
    covariates: Sequence[str] | None = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader from a `MuData` object.

    This function converts a `MuData` object into a PyTorch `DataLoader` for
    training or evaluation. It allows customization of data loading parameters,
    such as batch size, shuffling, and number of workers.

    Parameters
    ----------
    mdata :
        A `MuData` object containing multimodal data.
    rna_mod :
        The name of the RNA modality in the `MuData` object. Default is "rna".
    atac_mod :
        The name of the ATAC modality in the `MuData` object. Default is "atac".
    idx :
        Indices or keys to subset the `MuData` object. If `None`, the entire dataset is used.
        Default is `None`.
    covariates :
        Covariates to include from the `MuData` object. Default is `None`.
    batch_size :
        The number of samples per batch. Default is 128.
    num_workers :
        The number of worker processes for data loading. Default is 4.
    pin_memory :
        Whether to pin memory in DataLoader for faster GPU transfers. Default is `True`.
    shuffle :
        Whether to shuffle the dataset. Default is `True`.
    drop_last :
        Whether to drop the last incomplete batch. Default is `True`.
    persistent_workers :
        Whether to keep data loading workers alive between epochs. Default is `True`.
    **kwargs :
        Additional keyword arguments passed to `torch.utils.data.DataLoader`.

    Returns
    -------
    A PyTorch DataLoader for the specified `MuData` dataset.

    Examples
    --------
    >>> from mudata import MuData
    >>> from cell2net.pd.data import get_dataloader
    >>> mdata = MuData("data.h5mu")
    >>> dataloader = get_dataloader(mdata, batch_size=32, shuffle=True)
    >>> for batch in dataloader:
    >>>     print(batch)
    """
    if idx is not None:
        _mdata = mdata[idx]
    else:
        _mdata = mdata

    # check if the subseted mudata has the same feature names
    if mdata[atac_mod].var_names.tolist() != _mdata[atac_mod].var_names.tolist():
        logger.error(
            f"Subsetted MuData object does not have the same feature names as the original {atac_mod} modality."
        )

    if mdata[rna_mod].var_names.tolist() != _mdata[rna_mod].var_names.tolist():
        logger.error(
            f"Subsetted MuData object does not have the same feature names as the original {rna_mod} modality."
        )

    if use_personal_genome:
        dataset = MuTorchDatasetPersonalGenome(
            mdata=_mdata,  # type: ignore
            rna_mod=rna_mod,
            atac_mod=atac_mod,
            covariates=covariates,
        )
    else:
        dataset = MuTorchDataset(
            mdata=_mdata,  # type: ignore
            rna_mod=rna_mod,
            atac_mod=atac_mod,
            covariates=covariates,
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        **kwargs,
    )

    return dataloader
