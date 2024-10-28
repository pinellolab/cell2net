import numpy as np
import pandas as pd
from mudata import MuData
from scipy import stats

from cell2net._logging import logger


def peak_to_gene(
    mdata: MuData,
    peak_acc_attr: np.ndarray,
    groupby: str,
) -> pd.DataFrame:
    """
    Extracts peak-to-gene links based on the attribution of peak accessibility

    This is done by performing t-test to compare the attributions to zero.
    If the groupby is None, all cells are used, otherwise the test is done for each group.

    Parameters
    ----------
    mdata : MuData
        MuData object including RNA and ATAC modalities
    peak_acc_attr : np.ndarray
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

    assert len(groups) == peak_acc_attr.shape[0], print(
        f"Length of grouby {len(groups)} is different from number of cells {peak_acc_attr.shape[0]}"
    )

    logger.info(f"Extract peak-to-gene links for {groupby}")
    unique_groups, group_indices = np.unique(groups, return_inverse=True)  # type: ignore

    df_list = []
    # For each group, subset the attribution and perform t-test
    for i, unique_group in enumerate(unique_groups):
        _peak_acc_attr = peak_acc_attr[group_indices == i]
        res = stats.ttest_1samp(
            _peak_acc_attr, popmean=0, axis=0, alternative="two-sided"
        )

        df = pd.DataFrame(
            data={"t-statistic": res[0], "pvalue": res[1], "cell_type": unique_group}
        )
        df["peak"] = mdata.uns["peak_to_gene"]["peak"]
        df["gene"] = mdata.uns["peak_to_gene"]["gene"]

        df_list.append(df)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df = df[["gene", "peak", "cell_type", "t-statistic", "pvalue"]]

    return df
