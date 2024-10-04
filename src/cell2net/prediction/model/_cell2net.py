from lightning import LightningModule
from mudata import MuData

from cell2net.prediction.module import Peaks2GeneExpression


class Cell2Net(LightningModule):
    def __init__(
        self,
        mdata: MuData,
        n_channels: int = 4,
        n_filters: int = 30,
        kernel_size: int = 5,
        n_dims: int = 16,
    ) -> None:
        super().__init__()

        self.mdata = mdata

        self.gene = mdata["rna"].var_names[0]
        self.n_peaks = self.mdata["atac"].n_vars

        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_dims = n_dims

        self.module = Peaks2GeneExpression(
            n_peaks=self.n_peaks,
            peak_len=256,
            n_filters=self.n_filters,
            n_channels=self.n_channels,
            n_dims=self.n_dims,
        )

        self._model_summary_string = (
            f"gene_name: {self.gene}, " f"n_peaks: {self.n_peaks} "
        )

        self.is_train = False

        # train_mdata = self.mdata[self.train_idx,]
        # valid_mdata = self.mdata[self.valid_idx]
        # test_mdata = self.mdata[self.test_idx]

        # self.train_dl = get_dataloader(
        #     train_mdata, num_workers=4, drop_last=True, train=True, shuffle=True
        # )
        # self.valid_dl = get_dataloader(
        #     train_mdata, num_workers=4, drop_last=True, train=True, shuffle=False
        # )
        # self.test_dl = get_dataloader(
        #     test_mdata, num_workers=4, drop_last=False, train=False, shuffle=False
        # )

    # def training_step(self, batch, batch_idx) -> torch.Tensor:

    #     return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.adam.Adam(self.module.parameters(), lr=3e-4)
    #     return optimizer

    # def train(
    #     self,
    #     max_epochs: int = 20,
    #     optimizer_name: Literal["Adam", "AdamW"] = "Adam",
    #     lr: float = 3e-4,
    #     weight_decay: float = 1e-4,
    #     reduce_lr_on_plateau: bool = True,
    #     lr_factor: float = 0.5,
    #     lr_patience: int = 5,
    #     lr_threshold: float = 0.0,
    #     lr_scheduler_metric: str = "validation_loss",
    #     lr_min: float = 1e-5,
    #     accelerator: str = "auto",
    #     devices: int | list[int] | str = "auto",
    #     train_size: float = 0.9,
    #     validation_size: float | None = None,
    #     train_indices: list | None = None,
    #     validation_indices: list | None = None,
    #     test_indices: list | None = None,
    #     shuffle_set_split: bool = True,
    #     batch_size: int = 128,
    #     eps: float = 1e-08,
    #     early_stopping: bool = True,
    #     early_stopping_patience: int = 10,
    #     save_best: bool = True,
    #     check_val_every_n_epoch: int | None = None,
    #     datasplitter_kwargs: dict | None = None,
    #     plan_kwargs: dict | None = None,
    #     **kwargs,
    # ):
    #     update_dict = {
    #         "lr": lr,
    #         "weight_decay": weight_decay,
    #         "eps": eps,
    #         "optimizer": "AdamW",
    #     }
    #     if plan_kwargs is not None:
    #         plan_kwargs.update(update_dict)
    #     else:
    #         plan_kwargs = update_dict

    # if save_best:
    #     warnings.warn(
    #         "`save_best` is deprecated in v1.2 and will be removed in v1.3. Please use "
    #         "`enable_checkpointing` instead. See "
    #         "https://github.com/scverse/scvi-tools/issues/2568 for more details.",
    #         DeprecationWarning,
    #         stacklevel=settings.warnings_stacklevel,
    #     )

    #     if "callbacks" not in kwargs.keys():
    #         kwargs["callbacks"] = []
    #     kwargs["callbacks"].append(
    #         SaveBestState(monitor="reconstruction_loss_validation")
    #     )

    # super().train(
    #     max_epochs=max_epochs,
    #     train_size=train_size,
    #     accelerator=accelerator,
    #     devices=devices,
    #     validation_size=validation_size,
    #     shuffle_set_split=shuffle_set_split,
    #     early_stopping=early_stopping,
    #     early_stopping_monitor="validation_loss",
    #     early_stopping_patience=early_stopping_patience,
    #     datasplitter_kwargs=datasplitter_kwargs,
    #     plan_kwargs=plan_kwargs,
    #     check_val_every_n_epoch=check_val_every_n_epoch,
    #     batch_size=batch_size,
    #     **kwargs,
    # )

    # @classmethod
    # def setup_mudata(
    #     cls,
    #     mdata: MuData,
    #     batch_key: str | None = None,
    #     labels_key: str | None = None,
    #     rna_layer: str | None = None,
    #     atac_layer: str | None = None,
    #     rna_mod: str = "rna",
    #     atac_mod: str = "atac",
    #     **kwargs,
    # ):
    #     setup_method_args = cls._get_setup_method_args(**locals())

    #     # which data is used for model training
    #     mudata_fields = [
    #         MuDataLayerField(
    #             registry_key="rna",
    #             mod_key=rna_mod,
    #             layer=rna_layer,
    #             is_count_data=True,
    #             mod_required=True,
    #         ),
    #         MuDataLayerField(
    #             registry_key="atac",
    #             mod_key=atac_mod,
    #             layer=atac_layer,
    #             is_count_data=True,
    #             mod_required=True,
    #         ),
    #         CategoricalObsField("batch", batch_key),
    #         CategoricalObsField("label", labels_key),
    #     ]

    #     mdata_manager = MuDataManager(
    #         fields=mudata_fields, setup_method_args=setup_method_args
    #     )

    #     mdata_manager.register_fields(mdata, **kwargs)
    #     cls.register_manager(mdata_manager)
