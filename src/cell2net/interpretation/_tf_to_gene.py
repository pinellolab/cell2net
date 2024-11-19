import numpy as np
import pandas as pd
from mudata import MuData
from scipy import stats


def tf_to_gene(
    mdata: MuData,
    attr: np.ndarray,
    groupby: str,
    min_attr: float | None = 0.0,
    n_tfs: int = 15,
) -> pd.DataFrame:
    """
    Extracts TF-gene regulation base on the attribution of transcription factor expression

    This is done by performing t-test to compare the attributions to zero.
    If the groupby is None, all cells are used, otherwise the test is done for each group.

    Parameters
    ----------
    mdata : MuData
        MuData object including RNA and ATAC modalities
    attr : np.ndarray
        A numpy array of TF expression attribution
    groupby : str | None, optional
        Name of one column in mdata.obs to group cells. by default None

    Returns
    -------
    pd.DataFrame
        Calculated t-statistic and p-value for each tf-gene pair
    """
    assert groupby in mdata.obs.columns, print(f"Cannot find {groupby} in mdata.obs")

    groups = mdata.obs[groupby].values

    assert len(groups) == attr.shape[0], print(
        f"Length of grouby {len(groups)} is different from number of cells {attr.shape[0]}"
    )

    # logger.info("Extract tf-to-gene links")
    # logger.info(f"Cells are grouped by {groupby}")
    unique_groups, group_indices = np.unique(groups, return_inverse=True)  # type: ignore

    df_list = []
    # For each group, subset the attribution and perform t-test by comparing with the reference
    for i, unique_group in enumerate(unique_groups):
        _attr = attr[group_indices == i]

        res = stats.ttest_1samp(_attr, popmean=0, axis=0, alternative="greater")

        df = pd.DataFrame(
            data={
                "tf": mdata.uns["tfs"],
                "gene": mdata["rna"].var_names[0],
                groupby: unique_group,
                "avg_attr": np.mean(_attr, axis=0),
                # "std_attr": np.std(_attr, axis=0),
                "t-statistic": res[0],
                "pvalue": res[1],
            }
        )
        df_list.append(df)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    # filter by min_attr
    if min_attr:
        df = df[df["avg_attr"] > min_attr].reset_index(drop=True)

    # Group by 'group_column' and select the top 20 rows within each group
    df = (
        df.groupby(groupby)
        .apply(lambda x: x.nlargest(n_tfs, "avg_attr"))
        .reset_index(drop=True)
    )

    return df
