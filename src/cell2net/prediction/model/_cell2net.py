import numpy as np
import pandas as pd
import torch
from mudata import MuData
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from cell2net.prediction.data import MuTorchDatasetSimple
from cell2net.prediction.module import PeaksTF2GeneExpression


class Cell2Net:
    def __init__(
        self,
        mdata: MuData,
        gene: str,
        rna_mod: str = "rna",
        atac_mod: str = "atac",
        n_channels: int = 4,
        n_filters: int = 30,
        kernel_size: int = 5,
        n_dims: int = 16,
    ):
        super().__init__()
        self.gene = gene

        self.peak_to_gene = mdata.uns["peak_to_gene"][
            mdata.uns["peak_to_gene"]["gene"] == gene
        ]
        self.n_peaks = len(self.peak_to_gene)
        assert self.n_peaks > 0, print("Cannot find any associated peaks!")

        # create anndata for RNA and ATAC
        self.adata_atac = mdata[atac_mod][:, self.peak_to_gene["peak"].values.tolist()]
        self.adata_rna = mdata[rna_mod][:, gene]

        # Check how many genes are TFs
        self.n_tfs = np.sum(mdata[rna_mod].var["is_tf"])
        if mdata[rna_mod].var.loc[gene]["is_tf"]:
            self.n_tfs -= 1

        # If gene is a tf, then exclude it from the predictors
        adata_rna = mdata[rna_mod][:, ~mdata[rna_mod].var_names.isin([gene])]
        self.adata_rna.obsm["tf"] = adata_rna[:, adata_rna.var["is_tf"]].layers["counts"].copy()  # type: ignore

        self.mdata = MuData({rna_mod: self.adata_rna, atac_mod: self.adata_atac})  # type: ignore

        # Parameters for sequence encoder
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_dims = n_dims

        self.module = PeaksTF2GeneExpression(
            n_peaks=self.n_peaks,
            peak_len=256,
            n_tfs=self.n_tfs,
            n_filters=self.n_filters,
            n_channels=self.n_channels,
            n_dims=self.n_dims,
        )

        self._model_summary_string = (
            f"gene_name: {self.gene}, "
            f"n_peaks: {self.n_peaks}, "
            f"n_tfs: {self.n_tfs}, "
            f"n_filters: {self.n_filters}, "
            f"n_channels: {self.n_channels}, "
            f"n_dims: {self.n_dims}"
        )

        self.is_train = False

    def summary(self):
        return self._model_summary_string

    def _train(self):
        self.module.train()

        train_loss = 0.0
        for data in self.train_dl:
            # get data
            atac = data["atac"].to(self.device)
            rna = data["rna"].to(self.device)
            dna = data["dna"].to(self.device)
            tf_exp = data["tf"].to(self.device)

            # get prediction
            pred = self.module(dna, atac, tf_exp)
            loss = self.criterion(pred.view(-1).float(), rna.view(-1).float())

            # optimize parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() / len(self.train_dl)

        return train_loss

    def _valid(self):
        self.module.eval()

        valid_loss = 0.0
        for data in self.valid_dl:
            # get data
            atac = data["atac"].to(self.device)
            rna = data["rna"].to(self.device)
            dna = data["dna"].to(self.device)
            tf_exp = data["tf"].to(self.device)

            # get prediction
            pred = self.module(dna, atac, tf_exp)
            loss = self.criterion(pred.view(-1).float(), rna.view(-1).float())

            valid_loss += loss.item() / len(self.train_dl)

        return valid_loss

    def train(
        self,
        train_size: float | None = 0.8,
        train_idx: list[int] | None = None,
        valid_idx: list[int] | None = None,
        batch_size: int = 128,
        num_workers: int = 4,
        max_epochs: int = 20,
        random_state: int = 42,
        device_name: str = "cuda",
        lr: float = 3e-04,
        weight_decay: float = 1e-04,
        **kwargs,
    ) -> None:
        if train_size is not None and (train_idx is not None or valid_idx is not None):
            raise ValueError("Only one of train_size or train_idx can be provided.")

        if train_size is not None:
            train_idx, valid_idx = train_test_split(
                self.mdata.obs_names,
                train_size=train_size,
                random_state=random_state,
            )

        self.train_ds = MuTorchDatasetSimple(mdata=self.mdata[train_idx])  # type: ignore
        self.valid_ds = MuTorchDatasetSimple(mdata=self.mdata[valid_idx])  # type: ignore

        self.train_dl = DataLoader(
            dataset=self.train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

        self.valid_dl = DataLoader(
            dataset=self.train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

        self.device = torch.device(device_name)
        self.module = self.module.to(self.device)

        # Setup loss and optimizer
        self.criterion = torch.nn.PoissonNLLLoss(log_input=True)
        self.optimizer = Adam(
            self.module.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_score = np.inf
        self.history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss"])

        epochs, train_losses, valid_losses = [], [], []
        for epoch in tqdm(range(max_epochs)):
            train_loss = self._train()
            valid_loss = self._valid()

            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # save model if find a better validation score
            if valid_loss < self.best_score:
                self.best_score = valid_loss
                self.best_module = {
                    "state_dict": self.module.state_dict(),
                }
                # torch.save(state, args.model_path)

        self.history = pd.DataFrame(
            data={
                "epochs": epochs,
                "train_loss": train_losses,
                "valid_loss": valid_losses,
            }
        )

        return None
