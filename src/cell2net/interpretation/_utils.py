import numpy as np


def average_attribution(attribution: np.ndarray, groups) -> np.ndarray:
    """
    Compute average attribution using group information

    For peak sequences, the input attribution should have shape of (n_cells, n_peaks, peak_len, n_bases)
    For peak accessibility or TF expression, the shape should be (n_cells, n_peaks) or (n_cells, n_tfs)

    If groupby is None, the average will be computed using all cells

    Parameters
    ----------
    attribution : np.ndarray
        Estimated feature attribution for each single cell
    groups : list[str] | list[int] | None
        A list of cell barcodes or indices

    Returns
    -------
    np.ndarray
        Average attribution for each group
    """
    # average for all cells
    if groups is None:
        avg_attr = np.mean(attribution, axis=0)
        return avg_attr

    if len(attribution.shape) == 4:
        avg_attr = _avg_attr_peak_seq(attribution, groups=groups)

    if len(attribution.shape) == 2:
        avg_attr = _avg_attr_peak_acc_or_tf_exp(attribution, groups=groups)

    return avg_attr


def _avg_attr_peak_acc_or_tf_exp(attr: np.ndarray, groups) -> np.ndarray:
    # Get unique string groups and map them to numeric labels
    unique_groups, group_indices = np.unique(groups, return_inverse=True)

    # Initialize an array to hold sums per group for each column slice (H, W)
    sum_by_group = np.zeros((len(unique_groups), attr.shape[1]))

    # Sum the values for each group along dimension 0
    for i in range(len(unique_groups)):
        sum_by_group[i] = attr[group_indices == i].sum(axis=0)

    # Count occurrences of each group
    count_by_group = np.bincount(group_indices)

    # Compute the average for each group along dimension 0
    avg_attr = sum_by_group / count_by_group[:, None]  # Add two None for broadcasting

    return avg_attr


def _avg_attr_peak_seq(attr: np.ndarray, groups) -> np.ndarray:
    # Get unique string groups and map them to numeric labels
    unique_groups, group_indices = np.unique(groups, return_inverse=True)

    # Initialize an array to hold sums per group for each (C, H, W)
    sum_by_group = np.zeros(
        (len(unique_groups), attr.shape[1], attr.shape[2], attr.shape[3])
    )

    # Sum the values for each group along dimension 0
    for i in range(len(unique_groups)):
        sum_by_group[i] = attr[group_indices == i].sum(axis=0)

    # Count occurrences of each group
    count_by_group = np.bincount(group_indices)

    # Compute the average for each group along dimension 0
    avg_attr = (
        sum_by_group / count_by_group[:, None, None, None]
    )  # Broadcast across C, H, W

    return avg_attr
