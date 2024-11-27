import numpy as np
import pandas as pd
from mudata import MuData


def peak_to_gene(
    mdata: MuData, attr: np.ndarray, groupby: str, n_peaks: int = 50
) -> pd.DataFrame:
    """
    Extracts peak-to-gene links based on the attribution of peak accessibility

    Parameters
    ----------
    mdata : MuData
        MuData object including RNA and ATAC modalities
    attr : np.ndarray
        A numpy array of peak accessibility attribution
    groupby : str | None, optional
        Name of one column in mdata.obs to group cells. by default None

    Returns
    -------
    pd.DataFrame
        Calculated t-statistic and p-value for each peak-to-gene link
    """
    assert groupby in mdata.obs.columns, print(f"Cannot find {groupby} in mdata.obs")

    groups = mdata.obs[groupby].values

    assert len(groups) == attr.shape[0], print(
        f"Length of grouby {len(groups)} is different from number of cells {attr.shape[0]}"
    )

    gene = mdata["rna"].var_names[0]

    # logger.info(f"Extract assocaited peaks for gene: {gene}")
    # logger.info(f"Cells are grouped by {groupby}")
    unique_groups, group_indices = np.unique(groups, return_inverse=True)  # type: ignore

    df_list = []
    # For each group, subset the attribution and perform t-test
    for i, unique_group in enumerate(unique_groups):
        _attr = attr[group_indices == i]
        df = pd.DataFrame(
            data={
                "peak": mdata.uns["peak_to_gene"]["peak"],
                "gene": gene,
                groupby: unique_group,
                "avg_attr": np.mean(_attr, axis=0),
            }
        )
        df_list.append(df)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    # Group by 'group_column' and select the top 20 rows within each group
    df = (
        df.groupby(groupby)
        .apply(lambda x: x.nlargest(n_peaks, "avg_attr"))
        .reset_index(drop=True)
    )

    return df
