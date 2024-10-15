"""Plot functions for interpretation module"""

import logging

import pandas as pd
from mudata import MuData
from sklearn.preprocessing import StandardScaler

from cell2net._utils import create_bulk_adata


def peak_to_gene_links(mdata: MuData):

    return NotImplemented


def peak_to_gene_heatmap(
    mdata: MuData,
    groupby: str,
    df_p2g: pd.DataFrame,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
):
    adata_rna_bulk = create_bulk_adata(
        mdata[rna_mod], groupby=groupby, normalize=True, target_sum=1e6  # type: ignore
    )

    adata_atac_bulk = create_bulk_adata(
        mdata[atac_mod], groupby=groupby, normalize=True, target_sum=1e6  # type: ignore
    )

    df_rna = adata_rna_bulk[:, df_p2g["gene"]].to_df()  # type: ignore
    df_atac = adata_atac_bulk[:, df_p2g["peak"]].to_df()  # type: ignore

    logging.info("Create pseudo bulk profiles")
    scaler = StandardScaler()

    df_rna = scaler.fit_transform(df_rna)
    df_atac = scaler.fit_transform(df_atac)

    return NotImplemented


# from https://github.com/snehamitra/SCARlink/blob/main/scarlink/src/plotExtra.py
def genome_annotation(crhom: str, start: int, end: int, ax, gtf_file: str) -> None:

    return NotImplemented


def coverage(mdata: MuData, groupby: str, rna_mod: str = "rna", atac_mod: str = "atac"):

    return NotImplemented
