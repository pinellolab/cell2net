from __future__ import annotations

from collections.abc import Mapping, Sequence

import anndata as ad
import scanpy as sc
from adpbulk import ADPBulk

_VarNames = str | Sequence[str]


def check_if_igraph():
    """
    Verify that the `igraph` library is installed and meets the version requirement.

    This function checks if the `igraph` library is installed. If not, it raises an
    ImportError with instructions for installation. Additionally, it ensures that
    the installed version of `igraph` is at least 0.10.0. If the version is lower,
    an ImportError is raised with instructions to install the correct version.

    Returns
    -------
    The imported `igraph` module if the library is installed and meets the version requirement.

    Raises
    ------
    If `igraph` is not installed or its version is less than 0.10.0.

    Notes
    -----
    - This function is useful for ensuring compatibility when using `igraph`-dependent code.
    - To install `igraph`, use the following command: `pip install igraph`
    - To install a specific version of `igraph`, use: `pip install igraph==0.10.0`
    """
    try:
        import igraph as ig
    except Exception:  # noqa: BLE001
        raise ImportError(  # noqa: B904
            "igraph is not installed. Please install it with: pip install igraph"
        )
    from packaging.version import Version

    if Version(ig.__version__) < Version("0.10.0"):
        raise ImportError(
            "igraph version needs to be at least 0.10.0. Please install it with: pip install igraph==0.10.0"
        )
    return ig


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


def save_fig(fig, dpi: int, path: str, exts: Sequence[str] = ["", ".png"]):
    """Save current figure to file.

    The `filename` is generated as follows:

        filename = settings.figdir / (writekey + settings.plot_suffix + '.' + settings.file_format_figs)
    """
    pass


def process_var_names(var_names: _VarNames | Mapping[str, _VarNames]):
    has_var_groups = False
    if isinstance(var_names, Mapping):
        var_group_labels = []
        _var_names = []
        var_group_positions = []
        start = 0
        for label, vars_list in var_names.items():
            if isinstance(vars_list, str):
                vars_list = [vars_list]

            # use list() in case var_list is a numpy array or pandas series
            _var_names.extend(list(vars_list))
            var_group_labels.append(label)
            var_group_positions.append((start, start + len(vars_list) - 1))
            start += len(vars_list)

        var_names = _var_names
        var_group_labels = var_group_labels
        var_group_positions = var_group_positions
        has_var_groups = True

    elif isinstance(var_names, str):
        var_names = [var_names]
        var_group_labels = None
        var_group_positions = None

    return var_names, var_group_labels, var_group_positions, has_var_groups


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
