"""Plot functions for interpretation module"""

import pandas as pd
from gtfparse import read_gtf
from mudata import MuData
from sklearn.preprocessing import StandardScaler

from cell2net._logging import logger
from cell2net.utils import create_bulk_adata


def peak_attr_boxplot(data):
    return NotImplemented


def peak_to_gene_links(mdata: MuData):

    return NotImplemented


def p2g_heatmap(
    mdata: MuData,
    groupby: str,
    df_p2g: pd.DataFrame,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
):
    adata_rna = create_bulk_adata(
        mdata[rna_mod], groupby=groupby, normalize=True, target_sum=1e6  # type: ignore
    )

    adata_atac = create_bulk_adata(
        mdata[atac_mod], groupby=groupby, normalize=True, target_sum=1e6  # type: ignore
    )

    df_rna = adata_rna[:, df_p2g["gene"]].to_df()  # type: ignore
    df_atac = adata_atac[:, df_p2g["peak"]].to_df()  # type: ignore

    logger.info("Create pseudo bulk profiles")
    scaler = StandardScaler()

    df_rna = scaler.fit_transform(df_rna)
    df_atac = scaler.fit_transform(df_atac)

    return NotImplemented


# modified from https://github.com/snehamitra/SCARlink/blob/main/scarlink/src/plotExtra.py
def genome_annotation(crhom: str, start: int, end: int, ax, gtf_file: str):

    # returns GTF with essential columns such as "feature", "seqname", "start", "end"
    # alongside the names of any optional keys which appeared in the attribute column
    df_genes = read_gtf(gtf_file)

    return NotImplemented


def coverage(mdata: MuData, groupby: str, rna_mod: str = "rna", atac_mod: str = "atac"):

    return NotImplemented
