import os

import mudata as md
import numpy as np
import pandas as pd
import torch
from mudata import MuData
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from cell2net.prediction.data import MuTorchDataset
from cell2net.prediction.module import PeaksTF2GeneExpressionPoisson

from ._constants import SAVE_KEYS


class Cell2Net:
    def __init__(
        self,
        mdata: MuData,
        gene: str,
        rna_mod: str = "rna",
        atac_mod: str = "atac",
        covariates: list[str] | None = None,
        n_channels: int = 4,
        n_filters: int = 30,
        kernel_size: int = 5,
        n_dims: int = 4,
    ):
        super().__init__()
        self.gene = gene

        self.peak_to_gene = mdata.uns["peak_to_gene"][mdata.uns["peak_to_gene"]["gene"] == gene]
        self.n_peaks = len(self.peak_to_gene)
        assert self.n_peaks > 0, print("Cannot find any associated peaks!")

        self.covariates = covariates
        if covariates is not None:
            self.n_covariates = len(covariates)
        else:
            self.n_covariates = 0

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
        self.mdata.obs = mdata.obs.copy()

        # Parameters for sequence encoder
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_dims = n_dims

        self.module = PeaksTF2GeneExpressionPoisson(
            n_peaks=self.n_peaks,
            peak_len=256,
            n_tfs=self.n_tfs,
            n_covariates=self.n_covariates,
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
            f"kernel_size: {self.kernel_size}, "
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
            covariates = data["covariates"].to(self.device)

            # get prediction
            pred = self.module(dna, atac, tf_exp, covariates)
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
            covariates = data["covariates"].to(self.device)

            # get prediction
            pred = self.module(dna, atac, tf_exp, covariates)
            loss = self.criterion(pred.view(-1).float(), rna.view(-1).float())

            valid_loss += loss.item() / len(self.train_dl)

        return valid_loss

    def _get_dataloader(
        self,
        idx,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        shuffle: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = True,
    ) -> DataLoader:
        dataset = MuTorchDataset(
            mdata=self.mdata[idx],
            covariates=self.covariates,  # type: ignore
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )

        return dataloader

    def train(
        self,
        train_size: float | None = 0.8,
        train_idx: list[int] | list[str] | None = None,
        valid_idx: list[int] | list[str] | None = None,
        batch_size: int = 128,
        num_workers: int = 4,
        max_epochs: int = 20,
        random_state: int = 42,
        device_name: str = "cuda",
        lr: float = 3e-04,
        weight_decay: float = 1e-04,
        **kwargs,
    ) -> None:
        if train_idx and valid_idx:
            print("Using provided index for training and validation")

        elif train_size:
            print(f"Split dataset for training and validation; training size is {train_size}")
            train_idx, valid_idx = train_test_split(
                self.mdata.obs_names,
                train_size=train_size,
                random_state=random_state,
            )
        else:
            raise ValueError("Please provide train_size or index")

        self.train_dl = self._get_dataloader(
            train_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        self.valid_dl = self._get_dataloader(
            train_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        self.device = torch.device(device_name)
        self.module = self.module.to(self.device)

        # Setup loss and optimizer
        self.criterion = torch.nn.PoissonNLLLoss(log_input=True)
        self.optimizer = Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)

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
                self.best_model = {
                    SAVE_KEYS.MODEL_STATE_DICT_KEY: self.module.state_dict(),
                }

        self.history = pd.DataFrame(
            data={
                "epochs": epochs,
                "train_loss": train_losses,
                "valid_loss": valid_losses,
            }
        )

        self.is_train = True

        return None

    def test(
        self,
        test_idx: list[int] | list[str] | None = None,
        batch_size: int = 128,
        num_workers: int = 4,
        device_name: str = "cuda",
    ) -> np.float32:
        self.device = torch.device(device_name)
        self.module = self.module.to(self.device)
        self.module.eval()

        # if test_idx is None, use all cells for testing
        if test_idx is None:
            test_idx = self.mdata.obs_names  # type: ignore

        self.test_dl = self._get_dataloader(
            test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        rna_true, rna_pred = [], []
        for data in self.test_dl:
            # get data
            atac = data["atac"].to(self.device)
            dna = data["dna"].to(self.device)
            tf_exp = data["tf"].to(self.device)
            covariates = data["covariates"].to(self.device)

            pred = self.module(dna, atac, tf_exp, covariates).detach().cpu().view(-1)

            rna = data["rna"]
            rna_true.append(rna)
            rna_pred.append(pred)

        rna_true = torch.concat(rna_true).numpy()
        rna_pred = torch.concat(rna_pred).numpy()

        corr, _ = stats.spearmanr(rna_true, rna_pred)

        return corr  # type: ignore

    def save(self, dir_path: str, save_mdata: bool = False) -> None:
        """Save the state of the model"""
        model_save_path = os.path.join(dir_path, f"{self.gene}.pt")

        # save the model state dict
        torch.save(self.best_model, model_save_path)

        if save_mdata:
            self.mdata.write_h5mu(os.path.join(dir_path, f"{self.gene}.h5mu"))

        return None

    def load(self, dir_path: str, load_mdata: bool = False) -> None:
        """Instantiate a model from the saved output."""
        model_path = os.path.join(dir_path, f"{self.gene}.pt")
        state_dict = torch.load(model_path, weights_only=True)

        self.module = PeaksTF2GeneExpressionPoisson(
            n_peaks=self.n_peaks,
            peak_len=256,
            n_tfs=self.n_tfs,
            n_covariates=self.n_covariates,
            n_filters=self.n_filters,
            n_channels=self.n_channels,
            n_dims=self.n_dims,
        )

        self.module.load_state_dict(state_dict[SAVE_KEYS.MODEL_STATE_DICT_KEY])

        if load_mdata:
            self.mdata = md.read_h5mu(os.path.join(dir_path, f"{self.gene}.h5mu"))  # type: ignore

        return None
