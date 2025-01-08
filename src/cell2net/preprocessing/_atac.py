from anndata import AnnData
from mudata import MuData
from scipy.sparse import issparse


def binarize(
    data: AnnData | MuData, atac_mod: str = "atac", layer: str | None = None
) -> None:
    """
    Binarize the data matrix in an AnnData or MuData object

    This function converts non-zero values in the specified data matrix (either `X` or
    a specified `layer`) to 1, effectively binarizing the data.
    It supports both dense and sparse matrix formats.

    Parameters
    ----------
    data
        The input data object containing the matrix to be binarized.
        If a MuData object is provided, the `atac_mod` parameter specifies which modality to use.
    atac_mod
        The modality to use when `data` is a MuData object. Defaults to "atac".
    layer
        The specific layer of the AnnData object to binarize.
        If None, adata.X is binarized. Defaults to None.

    Returns
    -------
        The input object is modified in place, with the specified matrix binarized.

    Raises
    ------
    TypeError
        If the input `data` is not an AnnData or MuData object, or if the specified
        `atac_mod` is not found in the MuData object.

    Notes
    -----
        - For sparse matrices, this function modifies the `.data` attribute directly to ensure efficient processing without densifying the matrix.
        - For dense matrices, non-zero values are updated directly in place.

    Examples
    --------
    >>> from anndata import AnnData
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> import cell2net as cn

    >>> # Example with a dense matrix
    >>> X = np.array([[0, 2, 0], [3, 0, 1]])
    >>> adata = AnnData(X)
    >>> cn.pp.binarize(adata)
    >>> print(adata.X)

    >>> # Example with a sparse matrix
    >>> X_sparse = csr_matrix([[0, 2, 0], [3, 0, 1]])
    >>> adata = AnnData(X_sparse)
    >>> binarize(adata)
    >>> print(adata.X.toarray())

    >>> # Example with a specified layer
    >>> adata.layers["counts"] = X
    >>> binarize(adata, layer="counts")
    >>> print(adata.layers["counts"])
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and atac_mod in data.mod:
        adata = data.mod[atac_mod]
    else:
        raise TypeError(f"Expected AnnData or MuData object with '{atac_mod}' modality")

    if layer:
        if issparse(adata.layers[layer]):
            # Sparse matrix
            adata.layers[layer].data[adata.layers[layer].data != 0] = 1  # type: ignore
        else:
            adata.layers[layer][adata.layers[layer] != 0] = 1  # type: ignore
    else:
        if issparse(adata.X):
            # Sparse matrix
            adata.X.data[adata.X.data != 0] = 1  # type: ignore
        else:
            adata.X[adata.X != 0] = 1  # type: ignore


def peaks_to_bed(data: AnnData | MuData, bed_filename: str, atac_mod: str = "atac"):
    pass
