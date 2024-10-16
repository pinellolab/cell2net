from captum.attr import IntegratedGradients

from cell2net.prediction.model import Cell2Net


def compute_attribution(model: Cell2Net) -> None:

    model.module.train()

    # create a dataloader
    data_loader = model.get_dataloader()

    ig = IntegratedGradients(model.module)

    attr, delta = ig.attribute(
        (dna_seq, atac, tf, covariates), return_convergence_delta=True
    )

    return None
