import warnings
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.prediction.data import SequenceDataset
from cell2net.prediction.module import Peaks2Accessibility

from ._base import BaseModel

warnings.filterwarnings("ignore")

class Seq2Acc(BaseModel):
    def __init__(self,
                 peak_len: int = 256,
                 n_filters: Sequence[int] | None = None,
                 n_channels: int = 4,
                 kernel_size: int = 5,
                 n_dims: int = 16,
                 dropout_rate: float = 0.25):
        super().__init__()
        # parameters for sequence encoder
        self.peak_len = peak_len
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_dims = n_dims
        self.dropout_rate = dropout_rate

        self.module = Peaks2Accessibility(peak_len=self.peak_len,
                                          n_filters=self.n_filters,
                                          n_channels=self.n_channels,
                                          kernel_size=self.kernel_size,
                                          n_dims=self.n_dims,
                                          dropout_rate=self.dropout_rate)

        self._module_summary = {
            "peak_len": self.peak_len,
            "n_filters": self.n_filters,
            "n_channels": self.n_channels,
            "n_dims": self.n_dims,
        }

        self.summary_ = (
            f"peak_len: {self.peak_len}, "
            f"n_filters: {self.n_filters}, "
            f"n_channels: {self.n_channels}, "
            f"n_dims: {self.n_dims}"
        )

    def _train(self) -> float:
        self.module.train()

        train_loss = 0.0
        for data in self.train_dl:
            peak_seq = data["peak_seq"].to(self.device)
            target_acc = data["peak_acc"].to(self.device)

            # get prediction
            pred_acc = self.module(peak_seq)

            loss = self.criterion(pred_acc.view(-1).float(),
                                  target_acc.view(-1).float())

            # optimize parameters
            self.optimizer.zero_grad()
            loss.backward()

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
                loss = self.criterion(pred_acc.view(-1).float(),
                                      target_acc.view(-1).float())

                valid_loss += loss.item() / len(self.valid_dl)

        return valid_loss

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict accessibility for all peaks in the AnnData object.

        Returns
        -------
        pd.DataFrame
            DataFrame with predicted accessibility for each peak and cell.
        """
        if not self.is_trained_:
            logger.error("Model is not trained yet. Cannot predict.")

        # create dataloader for all peaks
        dataset = SequenceDataset(df)

        dataloader = DataLoader(dataset,
                                batch_size=128,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=False,
                                drop_last=False)

        all_preds = []
        self.module.eval()
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Predicting"):
                peak_seq = data["peak_seq"].to(self.device)
                pred_acc = self.module(peak_seq)
                all_preds.append(pred_acc.view(-1).cpu())

        all_preds = torch.cat(all_preds, dim=0)  # (n_peaks)
        all_preds = all_preds.squeeze().numpy()

        return all_preds

    def train(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        device_name: str = "cuda",
        batch_size: int = 128,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        max_epochs: int = 20,
        lr: float = 3e-04,
        min_lr: float = 1e-06,
        patience: int = 5,
        weight_decay: float = 1e-04,
        verbose: bool = True,
    ) -> None:

        # create dataloaders for training and validation
        logger.info("Creating dataloaders for training and validation")
        self.train_ds = SequenceDataset(df_train)
        self.valid_ds = SequenceDataset(df_valid)

        self.train_dl = DataLoader(self.train_ds,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   persistent_workers=persistent_workers,
                                   drop_last=True)

        self.valid_dl = DataLoader(self.valid_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=persistent_workers,
                              drop_last=False)

        logger.info(f"Training peaks: {len(self.train_ds)}, Validation peaks: {len(self.valid_ds)}")

        # Move module to device
        self.to_device(device_name=device_name)

        # Setup loss and optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(
            self.module.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        lr_scheduler = ReduceLROnPlateau(self.optimizer, "min",
                                         min_lr=min_lr,
                                         patience=patience)

        self.best_valid_loss = np.inf
        epochs, train_losses, valid_losses = [], [], []

        logger.info("Start training")
        for epoch in range(max_epochs):
            train_loss = self._train()
            valid_loss = self._valid()

            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # Save model if find a better validation score
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss

                # save parameters for sequence encoder in module
                self.check_point = deepcopy(self.module.seq_encoder.state_dict())

                self.train_loss = train_loss
                self.valid_loss = valid_loss

                patience = 10 # reset patience
            else:
                # early stop
                patience -= 1
                if patience == 0:
                    logger.info("Early stop!")
                    break

            logger.info(f"Epoch {epoch}:, Train: {train_loss:.4f}, Valid: {valid_loss:.4f}, Best: {self.best_valid_loss:.4f}")
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
