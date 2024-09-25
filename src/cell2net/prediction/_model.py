import re
from typing import Literal

import pandas as pd
import pysam
from anndata import AnnData

from ._base import BaseModelClass
from ._module import Peaks2GeneExpression


class Cell2Net(BaseModelClass):
    def __init__(
        self,
        adata_rna: AnnData,
        adata_atac: AnnData,
        peak_to_gene: pd.DataFrame,
        fasta: pysam.FastaFile,
        n_channels: int = 4,
        n_filters: int = 30,
        kernel_size: int = 5,
        n_dims: int = 16,
    ) -> None:
        # super().__init__()

        self.adata_rna = adata_rna
        self.adata_atac = adata_atac
        self.peak_to_gene = peak_to_gene
        self.fasta = fasta

        # All genes that will be predicted
        self.genes = peak_to_gene["gene"].unique().tolist()

        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_dims = n_dims

    def build(self, gene: str):
        assert gene in self.genes, f"Cannot find gene {gene}!"

        # get gene and peaks
        self.gene = gene
        self.peak_to_gene_subset = self.peak_to_gene[self.peak_to_gene["gene"] == gene]
        self.peaks = self.peak_to_gene_subset["peak"].values.tolist()

        self.peak_list = []
        self.peak_lengths = []

        for peak in self.peaks:
            peak = re.split("-", peak)
            chrom, start, end = peak[0], int(peak[1]), int(peak[2])
            seq = self.fasta.fetch(chrom, start, end).upper()
            self.peak_list.append(seq)
            self.peak_lengths.append(len(seq))

        self.n_peaks = len(self.peak_list)

        self.module = Peaks2GeneExpression(
            peak_list=self.peak_list,
            peak_lengths=self.peak_lengths,
            n_filters=self.n_filters,
            n_channels=self.n_channels,
            n_dims=self.n_dims,
        )

        self._model_summary_string = f"gene_name: {self.gene}, " f"n_peaks: {self.n_peaks} "

        self.is_train = False

        # split the data

    def _make_data_loader(self):
        return NotImplementedError

        # self.rna_data = torch.from_numpy(
        #     np.array(self.adata_rna[:, gene].X.todense()).reshape(-1)
        # )
        # self.atac_data = torch.from_numpy(
        #     np.array(self.adata_atac[:, self.peaks].X.todense())
        # )

        # # split train and validation
        # n_cells = self.atac_data.shape[0]
        # idx = list(range(n_cells))
        # np.random.shuffle(idx)
        # self.train_idx = idx[: int(len(idx) * self.train_size)]
        # self.valid_idx = idx[int(len(idx) * self.train_size) :]

        # self.train_dl = get_dataloader(
        #     seq_list=self.seq_list,
        #     atac=self.atac_data[self.train_idx],
        #     rna=self.rna_data[self.train_idx],
        #     batch_size=128,
        #     drop_last=True,
        #     shuffle=True,
        #     train=True,
        # )

        # self.valid_dl = get_dataloader(
        #     seq_list=self.seq_list,
        #     atac=self.atac_data[self.valid_idx],
        #     rna=self.rna_data[self.valid_idx],
        #     batch_size=128,
        #     drop_last=False,
        #     shuffle=False,
        #     train=True,
        # )

    def train(
        self,
        max_epochs: int = 20,
        optimizer: Literal["Adam", "AdamW"] = "Adam",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        reduce_lr_on_plateau: bool = True,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: str = "validation_loss",
        lr_min: float = 1e-5,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        train_indices: list | None = None,
        validation_indices: list | None = None,
        test_indices: list | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        eps: float = 1e-08,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        datasplitter_kwargs: dict | None = None,
        # plan_kwargs: dict | None = None,
    ):
        """Trains the model"""
        # training_plan = TrainingPlan(
        #     self.module,
        #     optimizer=optimizer,
        #     lr=lr,
        #     weight_decay=weight_decay,
        #     reduce_lr_on_plateau=reduce_lr_on_plateau,
        #     lr_factor=lr_factor,
        #     lr_patience=lr_patience,
        #     lr_threshold=lr_threshold,
        #     lr_scheduler_metric=lr_scheduler_metric,
        #     lr_min=lr_min,
        # )

        return NotImplementedError

        # runner = TrainRunner(
        #     self,
        #     max_epochs=max_epochs,
        #     training_plan=training_plan,
        #     data_splitter=datamodule,
        #     accelerator=accelerator,
        #     devices=devices,
        #     **trainer_kwargs,
        # )

        # return runner()
