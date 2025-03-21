"""Functions for heatmaps"""

import pandas as pd
from mudata import MuData


def prepare_dataframes_for_tf_heatmap(df: pd.DataFrame):
    pass


def tf_co_reg_heatmap():
    pass


def peak_to_gene_heatmap(
    mdata: MuData,
    groupby: str,
    df_peak_to_gene: pd.DataFrame,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
):
    pass
