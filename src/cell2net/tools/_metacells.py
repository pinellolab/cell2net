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
) -> MuData:
    """
    Create meta cells for single-cell multi-modal data

    It has been known that single cell has the sparsity issue, which means that
    many of the elements in the count matrix are zeros which poise challenges for
    data interpretation. To address this issue, a number of algorithms were developed
    to group cells based on their similarities,
    for example SEACells (Persad, Sitara, et al.) and Metacell2 (Ben-Kiki, Oren, et al.)

    We here use knn-based approach to group cells and then aggregate the profiles for scRNA-seq
    and scATAC-seq to enhance the signal for model interpretation.

    Parameters
    ----------
    data : MuData | AnnData
        Input data, can be an MuData or AnnData object
    mod_key: str
        If the input data is an Mudata object, which modality to use
    n_metacells : int
        How many meta cells to create
    n_neighbors : int, optional
        _description_, by default 15
    n_pcs : int, optional
        _description_, by default 30
    use_rep : str | None, optional
        _description_, by default "X_pca"
    groupby : str | None, optional
        _description_, by default None

    Returns
    -------
    MuData
        An Mudata object with metacells, where each observation represents an aggregated metacell.
    """
    # randome select number of cells that will be used as metacells
    logger.info(f"Select {n_metacells} metacells")

    metacell_indices = np.random.choice(
        mdata.n_obs, size=n_metacells, replace=False
    ).tolist()

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
    mdata_metacells.obs = mdata.obs.iloc[metacell_indices].copy()

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
