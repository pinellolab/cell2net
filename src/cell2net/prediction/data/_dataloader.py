import copy

import numpy as np
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from ._manager import MuDataManager


class MuDataLoader(DataLoader):
    def __init__(
        self,
        mdata_manager: MuDataManager,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 4,
        drop_last: bool = True,
        indices: list[int] | list[bool] | None = None,
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        load_sparse_tensor: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ):
        if indices is None:
            indices = np.arange(mdata_manager.mdata.shape[0])
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)

        self.indices = indices
        self.dataset = mdata_manager.create_torch_dataset(
            indices=indices,
            data_and_attributes=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )

        sampler_cls = SequentialSampler if not shuffle else RandomSampler
        sampler = BatchSampler(
            sampler=sampler_cls(self.dataset),
            batch_size=batch_size,
            drop_last=drop_last,
        )

        if "num_workers" not in kwargs:
            # kwargs["num_workers"] = settings.dl_num_workers
            kwargs["num_workers"] = num_workers
        if "persistent_workers" not in kwargs:
            # kwargs["persistent_workers"] = settings.dl_persistent_workers
            kwargs["persistent_workers"] = True

        self.kwargs = copy.deepcopy(kwargs)

        self.kwargs.update({"sampler": sampler})

        super().__init__(self.dataset, **self.kwargs)
