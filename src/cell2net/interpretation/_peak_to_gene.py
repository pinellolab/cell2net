# from captum.attr import IntegratedGradients, ShapleyValues
# from mudata import MuData

# from cell2net.prediction.model import Cell2Net


# def compute_peak_attr(model: Cell2Net, mdata: MuData | None = None) -> None:
#     model.module.train()

#     # create data loader
#     model.

#     ig = IntegratedGradients(model.module)

#     attr, delta = ig.attribute(
#         (dna_seq, atac, tf, covariates), return_convergence_delta=True
#     )

#     return None
