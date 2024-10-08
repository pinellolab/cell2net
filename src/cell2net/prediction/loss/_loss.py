import torch
from torch.distributions import NegativeBinomial
from torch.nn.modules.loss import _Loss


class NegativeBinomialNLLLoss(_Loss):
    """
    Negative Binomial Negative Log-Likelihood Loss using PyTorch's NegativeBinomial distribution.

    Parameters
    ----------
    _Loss : _type_
        _description_
    """

    def __init__(
        self,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.eps = eps

    def forward(
        self,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        dist = NegativeBinomial(total_count=alpha, probs=mu / (mu + alpha))

        nll = -dist.log_prob(target)

        # Return the loss with the specified reduction
        if self.reduction == "mean":
            return torch.mean(nll)
        elif self.reduction == "sum":
            return torch.sum(nll)
        else:
            return nll
