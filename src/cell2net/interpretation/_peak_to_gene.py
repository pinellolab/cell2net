import numpy as np
from mudata import MuData

from ._utils import average_attribution


def peak_to_gene(
    mdata: MuData,
    peak_acc_attr: np.ndarray,
    groupby: str | None = None,
    min_attr: float = 0,
):
    """
    Extracts peak-to-gene links based on the attribution of peak accessibility

    This is done by performing t-test to test if the attribution is significantly
    greater than 0. If the groupby is None, all cells are used, otherwise the test
    is performed for each group.

    Parameters
    ----------
    mdata : MuData
        MuData object including RNA and ATAC modalities
    peak_acc_attr : np.ndarray
        A numpy array
    groupby : str | None, optional
        Name of one column in mdata.obs to group cells. by default None

    Returns
    -------
    pd.DataFrame
        A dataframe containing all associated peaks
    """
    if groupby is None:
        groups = None
    else:
        assert groupby in mdata.obs.columns, print(
            f"Cannot find {groupby} in mdata.obs"
        )
        groups = mdata.obs[groupby].values

        assert len(groups) == peak_acc_attr.shape[0], print(
            f"Length of grouby {len(groups)} is different from number of cells {peak_acc_attr.shape[0]}"
        )

    avg_attr = average_attribution(peak_acc_attr, groups=groups)
    avg_attr.columns = mdata["atac"].var_names  # type: ignore

    mdata.uns["peak_attr"] = avg_attr

    # mdata.uns[]
    # mdata.uns["peak_to_gene"].loc[:, "attribution"] = avg_attr
    # df_p2g = mdata.uns["peak_to_gene"].copy()

    # df_p2g = df_p2g[df_p2g["attribution"] > min_attr].reset_index(drop=True)

    return None
