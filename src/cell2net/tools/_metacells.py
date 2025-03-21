from typing import Literal

import anndata as ad
import numpy as np
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csc_matrix

from cell2net._logging import logger


def get_metacells(
    mdata: MuData,
    n_metacells: int,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
    groupby: str | None = None,
    sampling: Literal["random", "geosketch"] = "geosketch",
) -> MuData:
    """
    Generate metacells from a multimodal MuData object using random selection or geosketch sampling.

    This function selects `n_metacells` representative cells from the input `mdata` and
    aggregates gene expression (RNA) and chromatin accessibility (ATAC) data to form metacells.
    Optionally, metacells can be grouped based on a categorical annotation.

    Parameters
    ----------
    mdata :
        The input MuData object containing RNA and ATAC modalities.
    n_metacells :
        The number of metacells to generate.
    rna_mod :
        The key for the RNA modality in `mdata`.
    atac_mod :
        The key for the ATAC modality in `mdata`.
    n_neighbors :
        The number of neighbors to use for metacell construction.
    use_rep :
        The representation to use for metacell selection.
    groupby :
        If provided, cells are grouped by this categorical column in `mdata.obs` before
        generating metacells, by default `None`.
    sampling :
        The sampling method used to select metacells:

        - `"random"`: Randomly selects `n_metacells` from all cells.
        - `"geosketch"`: Uses geosketch to select representative cells.

        By default, `"geosketch"` is used.

    Returns
    -------
        A new MuData object containing metacells for RNA and ATAC modalities. The `obs` attribute
        retains metadata for the selected metacells.

    Raises
    ------
    ImportError
        If `sampling="geosketch"` is chosen but the `geosketch` package is not installed.
    ValueError
        If an invalid sampling method is provided.

    Notes
    -----
        - The metacell selection process is influenced by the choice of `use_rep`, which determines the feature space used for sampling.
        - When `groupby` is provided, metacells are generated separately for each group in `mdata.obs[groupby]`.
        - The `_get_metacells` function is used internally to aggregate data for the selected metacells.

    Examples
    --------
    >>> import muon as mu
    >>> import cell2net as cn
    >>> from mudata import MuData
    >>> mdata = mu.read("multimodal_data.h5mu")
    >>> mdata_metacells = cn.pp.get_metacells(mdata, n_metacells=500)
    >>> print(mdata_metacells)

    Grouping metacells by a metadata column:

    >>> mdata_metacells = get_metacells(mdata, n_metacells=500, groupby="cell_type")
    >>> print(mdata_metacells.obs["cell_type"].value_counts())

    Using random sampling instead of geosketch:

    >>> mdata_metacells = get_metacells(mdata, n_metacells=500, sampling="random")

    """
    if sampling == "random":
        logger.info(f"Randomly select {n_metacells} metacells")
        metacell_indices = np.random.choice(
            mdata.n_obs, size=n_metacells, replace=False
        ).tolist()
    elif sampling == "geosketch":
        # Check if geosketch is installed
        try:
            from geosketch import gs  # type: ignore
        except ImportError:
            logger.error("Please install geosketch: pip install geosketch")

        logger.info(f"Select {n_metacells} metacells using geosketch")
        metacell_indices = gs(mdata[rna_mod].obsm[use_rep], n_metacells, replace=False)
    else:
        logger.error(f"Unknown sampling method: {sampling}")

    metacell_names = mdata.obs_names[metacell_indices]

    if groupby:
        logger.info(f"Group cells by {groupby}")
        groups = mdata.obs[groupby].unique().tolist()

        _adata_rna_list, _adata_atac_list = [], []
        for group in groups:
            _mdata = mdata[mdata.obs[groupby] == group]
            _metacell_names = list(set(metacell_names) & set(_mdata.obs_names))
            logger.info(f"Create {len(_metacell_names)} metacells for {group}")

            # get metacell indices
            _metacell_indices = [
                i
                for i, obs_name in enumerate(_mdata.obs_names)
                if obs_name in _metacell_names
            ]

            _adata_rna, _adata_atac = _get_metacells(
                mdata=_mdata,
                metacell_indices=_metacell_indices,
                rna_mod=rna_mod,
                atac_mod=atac_mod,
                n_neighbors=n_neighbors,
                use_rep=use_rep,
            )

            _adata_rna_list.append(_adata_rna)
            _adata_atac_list.append(_adata_atac)

        adata_rna = ad.concat(_adata_rna_list)
        adata_atac = ad.concat(_adata_atac_list)
    else:
        # create metacells using all cells
        logger.info("No groupby provided, using all cells")
        adata_rna, adata_atac = _get_metacells(
            mdata=mdata,
            metacell_indices=metacell_indices,
            rna_mod=rna_mod,
            atac_mod=atac_mod,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
        )

    mdata_metacells = MuData({rna_mod: adata_rna, atac_mod: adata_atac})

    # copy obs from original MuData
    mdata_metacells.obs = mdata.obs.iloc[metacell_indices].copy()  # type: ignore

    # copy uns from original MuData
    mdata_metacells.uns = {}
    if len(mdata.uns.keys()) > 0:
        for key in mdata.uns.keys():
            mdata_metacells.uns[key] = mdata.uns[key].copy()

    logger.info("Done")

    return mdata_metacells


def _get_metacells(
    mdata,
    metacell_indices: list[int],
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
):

    adata_rna = mdata[rna_mod]
    adata_atac = mdata[atac_mod]

    # compute kNN and the distance from each point to its nearest neighbors
    logger.info(f"Create KNN graph with {use_rep} and {n_neighbors} neighbors")
    sc.pp.neighbors(
        mdata.mod[rna_mod],
        use_rep=use_rep,
        n_neighbors=n_neighbors,
        knn=True,
    )
    knn_graph = mdata.mod[rna_mod].obsp["connectivities"]

    rna_counts = np.zeros(
        shape=(len(metacell_indices), adata_rna.n_vars), dtype=np.float32
    )
    atac_counts = np.zeros(
        shape=(len(metacell_indices), adata_atac.n_vars), dtype=np.float32
    )
    for i, idx in enumerate(metacell_indices):
        neighbors_idx = knn_graph[idx].nonzero()[1]  # type: ignore

        # get cell names
        cells = mdata.obs_names[neighbors_idx]

        rna_counts[i, :] = np.ravel(adata_rna[cells, :].layers["counts"].sum(axis=0))
        atac_counts[i, :] = np.ravel(adata_atac[cells, :].layers["counts"].sum(axis=0))

    # create pseudo-bulk profiles for RNA and ATAC
    adata_rna = AnnData(
        X=csc_matrix(rna_counts),
        obs=mdata[rna_mod].obs.iloc[metacell_indices].copy(),
        var=mdata[rna_mod].var,
    )

    adata_atac = AnnData(
        X=csc_matrix(atac_counts),
        obs=mdata[atac_mod].obs.iloc[metacell_indices].copy(),
        var=mdata[atac_mod].var,
    )

    adata_rna.layers["counts"] = adata_rna.X.copy()  # type: ignore
    adata_atac.layers["counts"] = adata_atac.X.copy()  # type: ignore

    sc.pp.calculate_qc_metrics(adata=adata_rna)
    sc.pp.calculate_qc_metrics(adata=adata_atac)

    return adata_rna, adata_atac
