import pytest
import torch

import cell2net as cn


def test_package_has_version():
    assert cn.__version__ is not None


def test_peaks_tf_to_gene_expression_poisson():
    from cell2net.prediction.module import PeaksTF2GeneExpressionPoisson

    seq_input = torch.randn(8, 79, 256, 4)
    atac_input = torch.randn(8, 79)
    tf_input = torch.randn(8, 513)
    covariates_input = torch.randn(8, 2)
    model = PeaksTF2GeneExpressionPoisson(79, 256, 513, 2)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    out = model(seq_input, atac_input, tf_input, covariates_input)
    print(out)


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():
    assert 1 == 0  # This test is designed to fail.
