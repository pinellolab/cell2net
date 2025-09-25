import os
import warnings
from collections.abc import Sequence

import mudata as md
import numpy as np
import pandas as pd
import copy
import torch
from mudata import MuData
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from cell2net._logging import logger
from cell2net.prediction.data import get_dataloader_with_variants
from cell2net.prediction.module import PeaksTF2GeneExpressionWithVariants

from ._base import BaseModel
from ._constants import SAVE_KEYS

warnings.filterwarnings("ignore")


class Cell2NetWithVariants(BaseModel):
    def __init__(
        self,
        mdata: MuData,
        rna_mod: str = "rna",
        atac_mod: str = "atac",
        tf_exp_mod: str = "tf_exp",
        variant_mod: str = "variant",
        covariates: Sequence[str] | None = None,
        peak_len: int = 256,
        n_filters: Sequence[int] | None = None,
        n_channels: int = 4,
        kernel_size: int = 5,
        n_dims: int = 16,
        dropout_rate: float = 0.25,
    ):
        super().__init__()

        if n_filters is None:
            n_filters = [64, 32, 32, 16]

        # Make sure the modalities exist in the MuData object
        if rna_mod not in mdata.mod:
            logger.error(f"RNA modality '{rna_mod}' not found in MuData object.")
        if atac_mod not in mdata.mod:
            logger.error(f"ATAC modality '{atac_mod}' not found in MuData object.")
        if tf_exp_mod not in mdata.mod:
            logger.error(f"TF expression modality '{tf_exp_mod}' not found in MuData object.")
        if variant_mod not in mdata.mod:
            logger.error(f"Variant modality '{variant_mod}' not found in MuData object.")

        self.rna_mod = rna_mod
        self.atac_mod = atac_mod
        self.tf_exp_mod = tf_exp_mod
        self.variant_mod = variant_mod

        self.mdata = mdata.copy()  # type: ignore

        self.covariates = covariates

        # Get number of genes, peaks, tfs, variants and covariates
        self.n_peaks = self.mdata[atac_mod].n_vars
        self.n_tfs = self.mdata[tf_exp_mod].n_vars
        self.n_variants = self.mdata[variant_mod].n_vars
        self.n_covariates = len(covariates) if covariates is not None else 0

        # Parameters for sequence encoder
        self.peak_len = peak_len
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_dims = n_dims
        self.dropout_rate = dropout_rate

        self.module = PeaksTF2GeneExpressionWithVariants(
            n_peaks=self.n_peaks,
            peak_len=self.peak_len,
            kernel_size=self.kernel_size,
            n_tfs=self.n_tfs,
            n_variants=self.n_variants,
            n_covariates=self.n_covariates,
            n_filters=self.n_filters,
            n_channels=self.n_channels,
            n_dims=self.n_dims,
            dropout_rate=self.dropout_rate,
        )

        self._module_summary = {
            "n_peaks": self.n_peaks,
            "peak_len": self.peak_len,
            "n_tfs": self.n_tfs,
            "n_variants": self.n_variants,
            "n_covariates": self.n_covariates,
            "n_filters": self.n_filters,
            "n_channels": self.n_channels,
            "n_dims": self.n_dims,
        }

        self.summary_ = (
            f"n_peaks: {self.n_peaks}, "
            f"peak_len: {self.peak_len}, "
            f"n_tfs: {self.n_tfs}, "
            f"n_variants: {self.n_variants}, "
            f"n_covariates: {self.n_covariates}, "
            f"n_filters: {self.n_filters}, "
            f"n_channels: {self.n_channels}, "
            f"kernel_size: {self.kernel_size}, "
            f"n_dims: {self.n_dims}"
        )

    def _train(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        self.module.train()

        train_loss = 0.0
        rna_true, rna_pred = [], []
        for data in self.train_dl:
            # get input features
            peak_seq = data["peak_seq"].to(self.device)
            peak_acc = data["peak_acc"].to(self.device)
            peak_dist = data["peak_dist"].to(self.device)
            tf_exp = data["tf_exp"].to(self.device)
            variants = data["variant"].to(self.device)
            covariates = data["covariates"].to(self.device)

            # get target gene expression
            target_exp = data["target_exp"].to(self.device)

            # get prediction
            pred_exp = self.module(peak_seq, peak_acc, peak_dist, tf_exp, variants, covariates)
            loss = self.criterion(
                pred_exp.view(-1).float(), target_exp.view(-1).float()
            )

            # optimize parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() / len(self.train_dl)

            rna_true.append(target_exp.detach().cpu().view(-1))
            rna_pred.append(pred_exp.detach().cpu().view(-1))

        # compute spearman correlation beetween target and predicted expression
        rna_true = torch.concat(rna_true).numpy()
        rna_pred = torch.concat(rna_pred).numpy()
        res = stats.spearmanr(rna_true, rna_pred)

        return train_loss, res.statistic, rna_true, rna_pred  # type: ignore

    def _valid(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        self.module.eval()

        valid_loss = 0.0
        rna_true, rna_pred = [], []
        with torch.no_grad():
            for data in self.valid_dl:
                # get input features
                peak_seq = data["peak_seq"].to(self.device)
                peak_acc = data["peak_acc"].to(self.device)
                peak_dist = data["peak_dist"].to(self.device)
                tf_exp = data["tf_exp"].to(self.device)
                variants = data["variant"].to(self.device)
                covariates = data["covariates"].to(self.device)

                # get target gene expression
                target_exp = data["target_exp"].to(self.device)

                # get prediction
                pred_exp = self.module(
                    peak_seq, peak_acc, peak_dist, tf_exp, variants, covariates
                )
                loss = self.criterion(
                    pred_exp.view(-1).float(), target_exp.view(-1).float()
                )

                valid_loss += loss.item() / len(self.train_dl)

                rna_true.append(target_exp.detach().cpu().view(-1))
                rna_pred.append(pred_exp.detach().cpu().view(-1))

        # compute spearman correlation beetween target and predicted expression
        rna_true = torch.concat(rna_true).numpy()
        rna_pred = torch.concat(rna_pred).numpy()
        res = stats.spearmanr(rna_true, rna_pred)

        return valid_loss, res.statistic, rna_true, rna_pred  # type: ignore

    def train(
        self,
        device_name: str = "cuda",
        train_size: float | None = 0.8,
        train_idx: list[int] | list[str] | None = None,
        valid_idx: list[int] | list[str] | None = None,
        stratify: list[str] | None = None,
        batch_size: int = 128,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        max_epochs: int = 20,
        random_state: int = 42,
        lr: float = 3e-04,
        weight_decay: float = 1e-04,
        verbose: bool = True,
    ) -> None:
        if train_idx is not None and valid_idx is not None:
            if verbose:
                logger.info("Using provided index for training and validation")

        elif train_size:
            if verbose:
                logger.info(f"Training size is provided: {train_size}")
                logger.info("Split the data for training and validation")
            train_idx, valid_idx = train_test_split(
                self.mdata.obs_names.values.tolist(),
                train_size=train_size,
                random_state=random_state,
                stratify=stratify,
            )
        else:
            raise ValueError(
                "Please provide train_size or indices for trainging and validation"
            )

        if verbose:
            logger.info(f"Number of training: {len(train_idx)}")  # type: ignore
            logger.info(f"Number of validation: {len(valid_idx)}")  # type: ignore

        self.train_dl = get_dataloader_with_variants(
            mdata=self.mdata,
            rna_mod=self.rna_mod,
            atac_mod=self.atac_mod,
            tf_exp_mod=self.tf_exp_mod,
            variant_mod=self.variant_mod,
            covariates=self.covariates,
            idx=train_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            shuffle=True,
            drop_last=True,
        )

        self.valid_dl = get_dataloader_with_variants(
            mdata=self.mdata,
            rna_mod=self.rna_mod,
            atac_mod=self.atac_mod,
            tf_exp_mod=self.tf_exp_mod,
            variant_mod=self.variant_mod,
            covariates=self.covariates,
            idx=valid_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            shuffle=False,
            drop_last=False,
        )

        # Move module to device
        self.to_device(device_name=device_name)

        # Setup loss and optimizer
        self.criterion = torch.nn.PoissonNLLLoss(log_input=True)
        self.optimizer = Adam(
            self.module.parameters(), lr=lr, weight_decay=weight_decay
        )
        lr_scheduler = ReduceLROnPlateau(self.optimizer, "max", min_lr=1e-5, patience=5)

        self.best_valid_corr, self.best_epoch = -np.inf, 0
        epochs, train_losses, valid_losses = [], [], []
        train_corrs, valid_corrs = [], []

        iterator = (
            tqdm(range(max_epochs), desc="Training") if verbose else range(max_epochs)
        )

        for epoch in iterator:
            train_loss, train_corr, train_true, train_pred = self._train()
            valid_loss, valid_corr, valid_true, valid_pred = self._valid()

            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_corrs.append(train_corr)
            valid_corrs.append(valid_corr)

            # print validation results
            if verbose:
                logger.info(f"Epoch {epoch}:, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, Train correlation: {train_corr:.4f}, Valid correlation: {valid_corr:.4f}")

            # Save model if find a better validation score
            if valid_corr > self.best_valid_corr:

                if verbose:
                    logger.info(f"New best valid correlation: {valid_corr:.4f}")

                self.best_valid_corr = valid_corr
                self.best_epoch = epoch

                # Use deep copy to create a new checkpoint
                self.check_point = copy.deepcopy(self.module.state_dict())

                self.train_loss = train_loss
                self.train_corr = train_corr
                self.train_pred = np.exp(train_pred)
                self.train_true = train_true

                self.valid_loss = valid_loss
                self.valid_corr = valid_corr
                self.valid_pred = np.exp(valid_pred)
                self.valid_true = valid_true

                self.results = pd.DataFrame(
                    data={
                        "true": np.concatenate([self.train_true, self.valid_true]),
                        "pred": np.concatenate([self.train_pred, self.valid_pred]),
                        "data": ["train"] * len(self.train_true)
                        + ["valid"] * len(self.valid_true),
                    }
                )

            lr_scheduler.step(valid_corr)  # type: ignore

        self.history_ = pd.DataFrame(
            data={
                "epochs": epochs,
                "train_loss": train_losses,
                "valid_loss": valid_losses,
                "train_corr": train_corrs,
                "valid_corr": valid_corrs,
            }
        )

        logger.info("Training finished")
        logger.info(
            f"Find best model at epoch {self.best_epoch} with valid correation {self.best_valid_corr: .3f}"
        )

        self.is_trained_ = True

        return None

    def predict(self,
                mdata: MuData,
                rna_mod: str = "rna",
                atac_mod: str = "atac",
                tf_exp_mod: str = "tf_exp",
                variant_mod: str = "variant",
                batch_size: int = 128,
                num_workers: int = 4,
                pin_memory: bool = False) -> np.ndarray:

        self.module = self.module.to(self.device)
        self.module.eval()

        dataloader = get_dataloader_with_variants(
            mdata=mdata,
            rna_mod=rna_mod,
            atac_mod=atac_mod,
            tf_exp_mod=tf_exp_mod,
            variant_mod=variant_mod,
            covariates=self.covariates,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False,
        )

        rna_pred = []
        with torch.no_grad():
            for data in dataloader:
                # get input features
                peak_seq = data["peak_seq"].to(self.device)
                peak_acc = data["peak_acc"].to(self.device)
                peak_dist = data["peak_dist"].to(self.device)
                tf_exp = data["tf_exp"].to(self.device)
                variants = data["variant"].to(self.device)
                covariates = data["covariates"].to(self.device)

                pred = self.module(peak_seq, peak_acc, peak_dist, tf_exp, variants, covariates)
                pred = pred.detach().cpu().view(-1)
                rna_pred.append(pred)

        # convert log(lambda) to lambda
        rna_pred = torch.concat(rna_pred).exp().numpy()
        return rna_pred

    def test(
        self,
        test_idx: list[int] | list[str] | None = None,
        batch_size: int = 128,
        num_workers: int = 4,
        device_name: str = "cuda",
    ) -> np.float32:
        device = torch.device(device_name)
        self.module = self.module.to(device)
        self.module.eval()

        self.test_dl = get_dataloader_with_variants(
            mdata=self.mdata,
            covariates=self.covariates,
            idx=test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        rna_true, rna_pred = [], []
        with torch.no_grad():
            for data in self.test_dl:
                # get input features
                peak_seq = data["peak_seq"].to(self.device)
                peak_acc = data["peak_acc"].to(self.device)
                peak_dist = data["peak_dist"].to(self.device)
                tf_exp = data["tf_exp"].to(self.device)
                covariates = data["covariates"].to(self.device)

                pred = self.module(peak_seq, peak_acc, peak_dist, tf_exp, covariates)
                pred = pred.detach().cpu().view(-1)

                rna = data["target_exp"]
                rna_true.append(rna)
                rna_pred.append(pred)

        # convert log(lambda) to lambda
        self.rna_true = torch.concat(rna_true).numpy()
        self.rna_pred = torch.concat(rna_pred).exp()
        self.rna_pred = torch.clamp(
            self.rna_pred, min=self.min_gex, max=self.max_gex
        ).numpy()

        corr, _ = stats.spearmanr(self.rna_true, self.rna_pred)

        return corr  # type: ignore

    def save(
        self,
        model_path: str,
        save_best_model: bool = True,
    ) -> None:
        """
        Save state of the model

        Parameters
        ----------
        dir_path : str
            A string indicating a directory path used to save the model
        save_best_model : bool, optional
            Whether or not save the best model based on validation results. Default: True
        save_mdata : bool, optional
            Whether or not save the mdata. Default: False
        overwrite : bool, optional
            Whether or not rewrite existing file. Default: True

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        # whether save the best model
        if save_best_model:
            model_state_dict = self.check_point
        else:
            model_state_dict = self.module.state_dict()

        torch.save(
            {
                SAVE_KEYS.MODEL_HISTORY: self.history_.to_dict(),
                SAVE_KEYS.MODEL_STATE_DICT_KEY: model_state_dict,
                SAVE_KEYS.MODULE_SUMMARY_DICT_KEY: self._module_summary,
                "results": self.results.to_dict(),
            },
            model_path,
        )

        return None

    def load(
        self, model_path: str, weights_only: bool = True, load_mdata: bool = False
    ) -> None:
        """Instantiate a model from the saved output."""
        state_dict = torch.load(model_path, weights_only=weights_only)

        self.module = PeaksTF2GeneExpressionWithVariants(
            n_peaks=self.n_peaks,
            peak_len=self.peak_len,
            n_tfs=self.n_tfs,
            n_variants=self.n_variants,
            n_covariates=self.n_covariates,
            n_filters=self.n_filters,
            n_channels=self.n_channels,
            n_dims=self.n_dims,
        )

        self.module.load_state_dict(state_dict[SAVE_KEYS.MODEL_STATE_DICT_KEY])
        self.history_ = pd.DataFrame.from_dict(state_dict[SAVE_KEYS.MODEL_HISTORY])
        self.results = pd.DataFrame.from_dict(state_dict["results"])

        return None
