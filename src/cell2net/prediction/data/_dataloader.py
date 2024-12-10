from collections.abc import Sequence

from mudata import MuData
from torch.utils.data import DataLoader

from cell2net._setting import settings

from ._dataset import MuTorchDataset


def get_dataloader(
    mdata: MuData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    idx: Sequence[int] | Sequence[str] | None = None,
    covariates: Sequence[str] | None = None,
    batch_size: int = settings.batch_size,
    num_workers: int = settings.dl_num_works,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a dataloader to iterate through the mudata object.

    Parameters
    ----------
    mdata : Mudata
        Mudata object. Must include RNA and ATAC modalities
    rna_mod: str, optional
        Name of RNA modality. Default: "rna"
    atac_mod: str, optional
        Name of ATAC modality. Default: "atac"
    idx : list[str] | None, optional
        List of cell barcodes used to subset the mdata
        If None, will use all cells. Default: None
    batch_size : int, optional
        Batch size of the dataloader. Default: 128
    num_workers : int, optional
        Number of cpus used to prepare data. Default: 4
    pin_memory : bool, optional
        _description_, by default True
    shuffle : bool, optional
        _description_, by default True
    drop_last : bool, optional
        _description_, by default True
    persistent_workers : bool, optional
        _description_, by default True
    **kwargs:
        Additional keyword arguments passed into :class:`~torch.utils.data.DataLoader`.

    Returns
    -------
    DataLoader
        A dataloader instance
    """
    if idx:
        _mdata = mdata[idx]
    else:
        _mdata = mdata

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
