"""Rank TFs according to differential regulation activity."""

from __future__ import annotations

import pandas as pd
from anndata import AnnData


def _select_top_n():
    pass


def rank_tfs(adata: AnnData) -> None:
    """
    Rank TFs for each observation in Anndata

    This function summarize the TF-gene regualtion for each cell type, and
    obtain the average reuglation activity and the number of target
    genes of each TF within a cell type.

    Parameters
    ----------
    adata : AnnData
        Input data
    """
    # compute the total regulation activity for each TF in each cell type
    df = adata.to_df().reset_index()
    df = pd.melt(
        df,
        id_vars=["cell_type"],
        var_name="tf_gene",
        value_name="regulation",
    )
    df[["tf", "gene"]] = df["tf_gene"].str.split("_", expand=True)
    df = df.groupby(["tf", "cell_type"])["regulation"].sum().reset_index()
    df = df.pivot_table(index="cell_type", columns="tf", values="regulation")
    df.fillna(0, inplace=True)

    return None
