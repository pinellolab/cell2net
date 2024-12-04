from __future__ import annotations

from collections.abc import Mapping, Sequence

import anndata as ad
import scanpy as sc
from adpbulk import ADPBulk

_VarNames = str | Sequence[str]


def process_var_names(var_names: _VarNames | Mapping[str, _VarNames]):
    has_var_groups = False
    if isinstance(var_names, Mapping):
        var_group_labels = []
        _var_names = []
        var_group_positions = []
        start = 0
        for label, vars_list in var_names.items():
            if isinstance(vars_list, str):
                vars_list = [vars_list]

            # use list() in case var_list is a numpy array or pandas series
            _var_names.extend(list(vars_list))
            var_group_labels.append(label)
            var_group_positions.append((start, start + len(vars_list) - 1))
            start += len(vars_list)

        var_names = _var_names
        var_group_labels = var_group_labels
        var_group_positions = var_group_positions
        has_var_groups = True

    elif isinstance(var_names, str):
        var_names = [var_names]
        var_group_labels = None
        var_group_positions = None

    return var_names, var_group_labels, var_group_positions, has_var_groups


def create_bulk_adata(adata: ad.AnnData, groupby: str) -> ad.AnnData:
    """
    Create

    Parameters
    ----------
    adata : ad.AnnData
        _description_
    groupby : str
        _description_

    Returns
    -------
    ad.AnnData
        _description_
    """
    adpb = ADPBulk(adata, [groupby])

    # perform the pseudobulking
    counts = adpb.fit_transform()

    sample_meta = adpb.get_meta().set_index("SampleName")
    adata_bulk = ad.AnnData(X=counts, obs=sample_meta)
    adata_bulk.layers["counts"] = adata_bulk.X  # type: ignore
    sc.pp.normalize_total(adata_bulk, target_sum=1e6, layer="counts")

    return adata_bulk
