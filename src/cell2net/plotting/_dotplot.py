import pandas as pd
from anndata import AnnData


def tf_dotplot(adata: AnnData, tf_names):
    # compute the total regulation activity for each TF in each cell type
    dot_color_df = adata.to_df().reset_index()
    dot_color_df = pd.melt(
        dot_color_df,
        id_vars=["cell_type_v2"],
        var_name="tf_gene",
        value_name="regulation",
    )
    dot_color_df[["tf", "gene"]] = dot_color_df["tf_gene"].str.split("_", expand=True)
    dot_color_df = (
        dot_color_df.groupby(["tf", "cell_type_v2"])["regulation"].sum().reset_index()
    )
    dot_color_df = dot_color_df.pivot_table(
        index="cell_type_v2", columns="tf", values="regulation"
    )
    dot_color_df.fillna(0, inplace=True)

    # compute the sum per group which in the boolean matrix this is the number
    # of values >expression_cutoff, and divide the result by the total number of
    # values in the group (given by `count()`)

    pass
