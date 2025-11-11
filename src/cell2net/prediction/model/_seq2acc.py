import os
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import copy
import torch
from mudata import MuData
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from cell2net._logging import logger
from cell2net.prediction.data import SequenceDataset
from cell2net.prediction.module import Peaks2Accessibility

from ._base import BaseModel
from ._constants import SAVE_KEYS

warnings.filterwarnings("ignore")

def poisson_loss(log_lambda: torch.Tensor,
                 counts: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
    crit = nn.PoissonNLLLoss(log_input=True, full=False, reduction="none", eps=1e-8)
    nll = crit(log_lambda, counts)             # (B, n_cells)
    if mask is not None:
        nll = nll * mask                       # same shape; 1=keep, 0=ignore
        return nll.sum() / mask.sum().clamp_min(1.0)
    return nll.mean()

class Seq2Acc(BaseModel):
    def __init__(self,
                 mdata: MuData,
                 atac_mod: str = "atac",
                 atac_layer: str | None = "counts",
                 peaks_key: str = "peaks",
                 seq_col: str = "sequence",
                 peak_len: int = 256,
                 n_filters: Sequence[int] | None = None,
                 n_channels: int = 4,
                 kernel_size: int = 5,
                 n_dims: int = 16,
                 dropout_rate: float = 0.25):
        super().__init__()

        # parameters for input data
        self.atac_mod = atac_mod
        self.atac_layer = atac_layer
        self.peaks_key = peaks_key
        self.seq_col = seq_col

        # parameters for sequence encoder
        self.peak_len = peak_len
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_dims = n_dims
        self.dropout_rate = dropout_rate

        # check if atac_mod exists
        if atac_mod not in mdata.mod:
            logger.error(f"{atac_mod} not found in MuData object")

        self.adata = mdata[self.atac_mod].copy()
        self.n_cells = self.adata.n_obs
        self.n_vars = self.adata.n_vars


        self.module = Peaks2Accessibility(n_cells=self.n_cells,
                                          peak_len=self.peak_len,
                                          n_filters=self.n_filters,
                                          n_channels=self.n_channels,
                                          kernel_size=self.kernel_size,
                                          n_dims=self.n_dims,
                                          dropout_rate=self.dropout_rate)

        self._module_summary = {
            "n_cells": self.n_cells,
            "peak_len": self.peak_len,
            "n_filters": self.n_filters,
            "n_channels": self.n_channels,
            "n_dims": self.n_dims,
        }

        self.summary_ = (
            f"n_cells: {self.n_cells}, "
            f"peak_len: {self.peak_len}, "
            f"n_filters: {self.n_filters}, "
            f"n_channels: {self.n_channels}, "
            f"n_dims: {self.n_dims}"
        )

    def _train(self) -> float:
        self.module.train()

        train_loss = 0.0
        for data in self.train_dl:
            # get input features
            peak_seq = data["peak_seq"].to(self.device)

            # get target accessibility
            target_acc = data["peak_acc"].to(self.device)

            # get prediction
            pred_acc = self.module(peak_seq)

            loss = self.criterion(pred_acc, target_acc)

            # optimize parameters
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients before optimizer step
            clip_grad_norm_(self.module.parameters(), max_norm=1.0)  # 1.0 is common
            self.optimizer.step()
            train_loss += loss.item() / len(self.train_dl)

        return train_loss

    def _valid(self) -> float:
        self.module.eval()

        valid_loss = 0.0
        with torch.no_grad():
            for data in self.valid_dl:
                # get input features
                peak_seq = data["peak_seq"].to(self.device)

                # get target accessibility
                target_acc = data["peak_acc"].to(self.device)

                # get prediction
                pred_acc = self.module(peak_seq)
                loss = self.criterion(pred_acc, target_acc)

                # loss = poisson_loss(pred_acc, target_acc)

                valid_loss += loss.item() / len(self.valid_dl)

        return valid_loss

    def predict(self) -> pd.DataFrame:
        """Predict accessibility for all peaks in the AnnData object.

        Returns
        -------
        pd.DataFrame
            DataFrame with predicted accessibility for each peak and cell.
        """
        if not self.is_trained_:
            logger.error("Model is not trained yet. Cannot predict.")

        # create dataloader for all peaks
        dataset = SequenceDataset(self.adata.copy(),
                                  peaks_key=self.peaks_key,
                                  seq_col=self.seq_col,
                                  atac_layer=self.atac_layer)

        dataloader = DataLoader(dataset,
                                batch_size=128,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=False)

        all_preds = []
        self.module.eval()
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Predicting"):
                peak_seq = data["peak_seq"].to(self.device)
                pred_acc = self.module(peak_seq)
                all_preds.append(pred_acc.cpu())

        all_preds = torch.cat(all_preds, dim=0)  # (n_peaks, n_cells)
        all_preds = torch.sigmoid(all_preds).numpy().transpose()  # (n_cells, n_peaks)

        pred_df = pd.DataFrame(
            data=all_preds,
            index=self.adata.obs_names,
            columns=self.adata.var_names,
        )

        return pred_df

    def train(
        self,
        device_name: str = "cuda",
        train_size: float | None = 0.8,
        batch_size: int = 128,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        max_epochs: int = 20,
        random_state: int = 42,
        lr: float = 3e-04,
        min_lr: float = 1e-06,
        patience: int = 5,
        weight_decay: float = 1e-04,
        verbose: bool = True,
    ) -> None:

        # split data into training and validation
        train_idx, valid_idx = train_test_split(
                self.adata.var_names.tolist(),
                train_size=train_size,
                random_state=random_state,
        )

        # create dataloaders for training and validation
        logger.info("Creating dataloaders for training and validation")
        train_ds = SequenceDataset(self.adata[:, train_idx].copy(),
                                   peaks_key=self.peaks_key,
                                   seq_col=self.seq_col,
                                   atac_layer=self.atac_layer)

        valid_ds = SequenceDataset(self.adata[:, valid_idx].copy(),
                                   peaks_key=self.peaks_key,
                                   seq_col=self.seq_col,
                                   atac_layer=self.atac_layer)

        self.train_dl = DataLoader(train_ds,
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent_workers)

        self.valid_dl = DataLoader(valid_ds,
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent_workers)

        logger.info(f"Training peaks: {len(train_ds)}, Validation peaks: {len(valid_ds)}")

        # Move module to device
        self.to_device(device_name=device_name)

        # Setup loss and optimizer
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = Adam(
            self.module.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        lr_scheduler = ReduceLROnPlateau(self.optimizer, "min",
                                         min_lr=min_lr,
                                         patience=patience)

        iterator = (
            tqdm(range(max_epochs), desc="Training") if verbose else range(max_epochs)
        )
        self.best_valid_loss = np.inf
        epochs, train_losses, valid_losses = [], [], []

        logger.info("Start training")
        for epoch in iterator:
            train_loss = self._train()
            valid_loss = self._valid()

            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            logger.info(f"Epoch {epoch}:, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}")

            # Save model if find a better validation score
            if valid_loss < self.best_valid_loss:
                logger.info(f"New best valid loss: {valid_loss:.4f}")

                self.best_valid_loss = valid_loss

                # save parameters for sequence encoder in module
                self.check_point = copy.deepcopy(self.module.seq_encoder.state_dict())

                self.train_loss = train_loss
                self.valid_loss = valid_loss

            lr_scheduler.step(valid_loss)  # type: ignore

        self.history_ = pd.DataFrame(
            data={
                "epochs": epochs,
                "train_loss": train_losses,
                "valid_loss": valid_losses,
            }
        )

        logger.info("Training finished")

        self.is_trained_ = True

        return None

    def save_module(self, filepath: str) -> None:
        """Save the trained module to a file.

        Parameters
        ----------
        filepath : str
            Path to save the module.
        """
        if not self.is_trained_:
            logger.error("Model is not trained yet. Cannot save untrained module.")

        torch.save(self.check_point, filepath)
        logger.info(f"Trained module saved to {filepath}")
        return None
