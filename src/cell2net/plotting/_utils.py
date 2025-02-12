from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import scanpy as sc
from adpbulk import ADPBulk

_VarNames = str | Sequence[str]

def check_if_adjustText():
    """
    Checks if the `adjustText` library is installed and imports it.

    This function attempts to import the `adjustText` module. If the module is not installed,
    it raises an `ImportError` with instructions on how to install it.

    Returns
    -------
    adjustText module
        The imported `adjustText` module if it is successfully installed.

    Raises
    ------
    ImportError
        If the `adjustText` module is not installed, an `ImportError` is raised
        with a message prompting the user to install it using `pip install adjustText`.

    Examples
    --------
    >>> at = check_if_adjustText()
    >>> # Now `at` can be used to access the `adjustText` module.
    """
    try:
        import adjustText as at  # type: ignore
    except Exception:  # noqa: BLE001
        raise ImportError(
            "adjustText is not installed. Please install it with: pip install adjustText"
        ) from None
    return at


def check_if_logomaker():
    """
    Checks if the `logomaker` library is installed and imports it.

    This function attempts to import the `logomaker` module. If the module is not installed,
    it raises an `ImportError` with instructions on how to install it.

    Returns
    -------
    logomaker module
        The imported `logomaker` module if it is successfully installed.

    Raises
    ------
    ImportError
        If the `logomaker` module is not installed, an `ImportError` is raised
        with a message prompting the user to install it using `pip install logomaker`.

    Examples
    --------
    >>> lm = check_if_logomaker()
    >>> # Now `lm` can be used to access the `logomaker` module.
    """
    try:
        import logomaker as lm  # type: ignore
    except Exception:  # noqa: BLE001
        raise ImportError(
            "logomaker is not installed. Please install it with: pip install logomaker"
        ) from None
    return lm

def create_bulk_adata(adata: ad.AnnData, groupby: str) -> ad.AnnData:
    """
    Create

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
    sc.pp.normalize_total(adata_bulk, target_sum=1e6, layer="counts")

    return adata_bulk
