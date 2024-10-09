from captum.attr import IntegratedGradients
from mudata import MuData

from cell2net.prediction.model import Cell2Net


def get_peak_to_gene(model: Cell2Net, mdata: MuData | None = None) -> None:

    model.module.train()

    ig = IntegratedGradients(model.module)

    attr, delta = ig.attribute(
        (dna_seq, atac, tf, covariates), return_convergence_delta=True
    )

    return None
