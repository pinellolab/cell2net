import pandas as pd


def get_top_tfs(
    df: pd.DataFrame, n_top_tfs: int = 5, var_cutoff: float = 0.5
) -> pd.DataFrame:
    """
    Find the top TFs for each cell type based on their regulation activity

    Parameters
    ----------
    df_act : pd.DataFrame
        Input datadat
    n_top_tfs : int, optional
        _description_, by default 5

    Returns
    -------
    pd.DataFrame
        Output dataframe
    """
    # copy the dataframe
    df_act = df.copy()

    df_act["tf_var"] = df_act.var(axis=1)
    df_act = df_act.sort_values(by="tf_var", ascending=False)
    df_act = df_act.head(int(len(df_act) * var_cutoff))
    df_act = df_act.drop(columns=["tf_var"])

    # z-score normalization
    df_norm = df_act.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
    df_norm = pd.DataFrame(df_norm, columns=df_act.columns, index=df_act.index)

    top_tfs = {}
    for col in df_norm.columns:
        df_top = df_norm.sort_values(by=col, ascending=False).head(n_top_tfs)
        top_tfs[col] = df_top.index.values.tolist()

    df = pd.DataFrame(data=top_tfs)

    return df
