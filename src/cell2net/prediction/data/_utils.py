from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from mudata import MuData
from scvi import REGISTRY_KEYS

from . import _constants


def encode_seq(seq_list):
    data = []
    for seq in seq_list:
        data.append(one_hot_encode(seq))

    data = torch.stack(data)

    return data


def one_hot_encode(seq):
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}")

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [0.0, 0.0, 0.0, 0.0],
    }

    # Create array from nucleotide sequence
    vec = torch.tensor([nuc_d[x] for x in seq], dtype=torch.float32)

    return vec


def registry_key_to_default_dtype(key: str) -> type:
    """Returns the default dtype for a given registry key."""
    if key in [
        REGISTRY_KEYS.BATCH_KEY,
        REGISTRY_KEYS.LABELS_KEY,
        REGISTRY_KEYS.CAT_COVS_KEY,
        REGISTRY_KEYS.INDICES_KEY,
    ]:
        return np.int64

    return np.float32


def _check_mudata_fully_paired(mdata: MuData):
    for mod_key in mdata.mod:
        if not mdata.obsm[mod_key].all():
            raise ValueError(
                f"Detected unpaired observations in modality {mod_key}. "
                "Please make sure that data is fully paired in all MuData inputs. "
                "Either pad the unpaired modalities or take the intersection with "
                "muon.pp.intersect_obs()."
            )


def _assign_mdata_uuid(mdata: MuData, overwrite: bool = False) -> None:
    """Assigns a UUID unique to the AnnData object.

    If already present, the UUID is left alone, unless ``overwrite == True``.
    """
    if _constants._CELL2NET_UUID_KEY not in mdata.uns or overwrite:
        mdata.uns[_constants._CELL2NET_UUID_KEY] = str(uuid4())


def get_mudata_attribute(
    mdata: MuData,
    attr_name: str,
    attr_key: str | None,
    mod_key: str | None = None,
) -> np.ndarray | pd.DataFrame:
    """Returns the requested data from a given MuData object."""
    if mod_key is not None:
        if isinstance(mdata, MuData):
            raise ValueError(f"Cannot access modality {mod_key} on an AnnData object.")
        if mod_key not in mdata.mod:
            raise ValueError(f"{mod_key} is not a valid modality in mdata.mod.")
        adata = mdata.mod[mod_key]

    mdata_attr = getattr(mdata, attr_name)
    if attr_key is None:
        field = mdata_attr
    elif isinstance(mdata_attr, pd.DataFrame):
        if attr_key not in mdata_attr.columns:
            raise ValueError(f"{attr_key} is not a valid column in adata.{attr_name}.")
        field = mdata_attr.loc[:, attr_key]
    else:
        if attr_key not in mdata_attr.keys():
            raise ValueError(f"{attr_key} is not a valid key in adata.{attr_name}.")
        field = mdata_attr[attr_key]

    if isinstance(field, pd.Series):
        field = field.to_numpy().reshape(-1, 1)

    return field
