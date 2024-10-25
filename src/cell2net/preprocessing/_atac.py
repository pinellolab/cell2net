from anndata import AnnData
from mudata import MuData
from scipy.sparse import issparse


def binarize(data: AnnData | MuData, atac_mod: str = "atac", layer: str | None = None):
    """
    Transform peak counts to the binary matrix (all the non-zero values become 1).

    Parameters
    ----------
    data : AnnData | MuData
        AnnData object with peak counts or multimodal MuData object with ATAC modality.
    atac_mod : str, optional
        Name of ATAC modality, by default "atac"
    layer : str | None, optional
        Which layer to use. If None, will use adata.X, by default None

    Raises
    ------
    TypeError
        _description_
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
