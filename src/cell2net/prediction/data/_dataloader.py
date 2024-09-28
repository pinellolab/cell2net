import numpy as np
from torch.utils.data import DataLoader

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
    ):
        self.dataset = mdata_manager.create_torch_dataset(
            indices=indices,
            data_and_attributes=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )

        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=True,
        )

    # def create_dataset() -> MultiOmeDataSet:


# def get_dataloader(
#     seq_list: list,
#     atac: torch.Tensor,
#     rna: torch.Tensor,
#     batch_size: int = 128,
#     num_workers: int = 8,
#     drop_last: bool = False,
#     shuffle: bool = True,
#     train: bool = True,
# ):
#     dataset = MultiOmeDataSet(
#         seq_list=seq_list,
#         atac=atac,
#         rna=rna,
#         train=train,
#     )

#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=True,
#         shuffle=shuffle,
#         drop_last=drop_last,
#         persistent_workers=True,
#     )

#     return dataloader
