import anndata as ad
import scanpy as sc
from adpbulk import ADPBulk


def create_bulk_adata(
    adata: ad.AnnData,
    groupby: str,
    normalize: bool = True,
    target_sum: int | float = 10000,
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

    return adata_bulk
