from typing import Literal

from mudata import MuData
from scvi.data.fields import CategoricalObsField, MuDataLayerField

from cell2net.prediction.data import MuDataLoader, MuDataManager
from cell2net.prediction.module import Peaks2GeneExpression

from ._base import BaseModelClass


class Cell2Net(BaseModelClass):
    def __init__(
        self,
        mdata: MuData,
        n_channels: int = 4,
        n_filters: int = 30,
        kernel_size: int = 5,
        n_dims: int = 16,
    ) -> None:
        super().__init__(mdata)

        self.mdata = mdata

        self.peak_to_gene = mdata.uns["peak_to_gene"]
        self.genes = self.peak_to_gene["gene"].unique().tolist()

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

        # for peak in self.peaks:
        #     peak = re.split("-", peak)
        #     chrom, start, end = peak[0], int(peak[1]), int(peak[2])
        #     seq = self.fasta.fetch(chrom, start, end).upper()
        #     self.peak_list.append(seq)
        #     self.peak_lengths.append(len(seq))

        self.n_peaks = len(self.peak_list)

        self.module = Peaks2GeneExpression(
            peak_list=self.peak_list,
            peak_lengths=self.peak_lengths,
            n_filters=self.n_filters,
            n_channels=self.n_channels,
            n_dims=self.n_dims,
        )

        self._model_summary_string = (
            f"gene_name: {self.gene}, " f"n_peaks: {self.n_peaks} "
        )

        self.is_train = False

        # split the data

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
        optimizer_name: Literal["Adam", "AdamW"] = "Adam",
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
        mdata_manager = MuDataManager()

        train_dl = MuDataLoader(mdata_manager=mdata_manager, indices=train_indices)
        valid_dl = MuDataLoader(mdata_manager=mdata_manager, indices=validation_indices)
        test_dl = MuDataLoader(mdata_manager=mdata_manager, indices=test_indices)

        # optimizer =

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

    @classmethod
    def setup_mudata(
        cls,
        mdata: MuData,
        batch_key: str | None = None,
        labels_key: str | None = None,
        rna_layer: str | None = None,
        atac_layer: str | None = None,
        rna_mod: str = "rna",
        atac_mod: str = "atac",
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())

        # which data is used in model training
        mudata_fields = [
            MuDataLayerField(
                registry_key="rna",
                mod_key=rna_mod,
                layer=rna_layer,
                is_count_data=True,
                mod_required=True,
            ),
            MuDataLayerField(
                registry_key="atac",
                mod_key=atac_mod,
                layer=atac_layer,
                is_count_data=True,
                mod_required=True,
            ),
            CategoricalObsField("batch", batch_key),
            CategoricalObsField("label", labels_key),
        ]

        mdata_manager = MuDataManager(
            fields=mudata_fields, setup_method_args=setup_method_args
        )

        mdata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(mdata_manager)

    # @classmethod
    # # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     cls,
    #     mdata: MuData,
    #     batch_key: str | None = None,
    #     labels_key: str | None = None,
    #     categorical_covariate_keys: list[str] | None = None,
    #     continuous_covariate_keys: list[str] | None = None,
    #     layer: str | None = None,
    #     **kwargs,
    # ):
    #     """%(summary)s.

    #     Parameters
    #     ----------
    #     %(param_adata)s
    #     %(param_batch_key)s
    #     %(param_labels_key)s
    #     %(param_cat_cov_keys)s
    #     %(param_cont_cov_keys)s
    #     %(param_layer)s
    #     """
    #     setup_method_args = cls._get_setup_method_args(**locals())
    #     anndata_fields = [
    #         LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
    #         CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
    #         CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
    #         CategoricalJointObsField(
    #             REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
    #         ),
    #         NumericalJointObsField(
    #             REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
    #         ),
    #     ]
    #     adata_manager = AnnDataManager(
    #         fields=anndata_fields, setup_method_args=setup_method_args
    #     )
    #     adata_manager.register_fields(adata, **kwargs)
    #     cls.register_manager(adata_manager)
