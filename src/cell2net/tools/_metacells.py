import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from mudata import MuData
from tqdm import tqdm

from cell2net._logging import logger


def metacells(
    mdata: MuData,
    n_metacells: int,
    mod_key: str | None = None,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    use_rep: str | None = "X_pca",
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
    obs_indices = np.random.choice(mdata.n_obs, size=n_metacells, replace=False)

    # compute kNN and the distance from each point to its nearest neighbors
    sc.pp.neighbors(
        mdata.mod[mod_key],
        use_rep=use_rep,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        knn=True,
    )

    rna_counts = np.zeros(shape=(n_metacells, mdata["rna"].n_vars), dtype=np.float32)
    atac_counts = np.zeros(shape=(n_metacells, mdata["atac"].n_vars), dtype=np.float32)

    logger.info("Find neighbors for each metacell")
    df_list = []
    for i, idx in tqdm(enumerate(obs_indices)):
        metacell_name = f"metacell_{i}"
        neighbors_idx = mdata.mod[mod_key].obsp["connectivities"][idx].nonzero()[1]  # type: ignore

        # get cell names
        cells = mdata.obs_names[neighbors_idx]

        rna_counts[i, :] = np.ravel(mdata["rna"][cells, :].layers["counts"].sum(axis=0))  # type: ignore
        atac_counts[i, :] = np.ravel(
            mdata["atac"][cells, :].layers["counts"].sum(axis=0)  # type: ignore
        )

        df = pd.DataFrame(
            data={
                "metacells": metacell_name,
                "center_cell": mdata.obs_names[idx],
                "obs_names": cells,
            }
        )

        df_list.append(df)

    # create pseudo-bulk profiles for RNA and ATAC
    logger.info("Create pseudo-bulk profiles")
    adata_rna = ad.AnnData(
        X=rna_counts, var=mdata["rna"].var, obs=mdata.obs.iloc[obs_indices]
    )

    adata_atac = ad.AnnData(
        X=atac_counts, var=mdata["atac"].var, obs=mdata.obs.iloc[obs_indices]
    )

    adata_rna.layers["counts"] = adata_rna.X.copy()  # type: ignore
    adata_atac.layers["counts"] = adata_atac.X.copy()  # type: ignore

    mdata_metacells = MuData({"rna": adata_rna, "atac": adata_atac})
    mdata_metacells.uns["metacells"] = pd.concat(df_list)

    mdata_metacells.obs = mdata.obs.iloc[obs_indices].copy()

    return mdata_metacells
