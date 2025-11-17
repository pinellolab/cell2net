from collections.abc import Sequence

import torch
from torch import nn

from cell2net.prediction.module import SeqEncoder

class Peaks2Accessibility(nn.Module):
    """
    A PyTorch module to predict peak accessibility using peak sequences.

    This module encodes peak sequences using convolutional layers and predicts
    the accessibility across multiple cells. It consists of a sequence encoder
    followed by fully connected layers. This model is used in the Cell2Net framework
    for pretraining the sequence encoder.

    Parameters
    ----------
    n_cells : int
        Number of input cells
    peak_len: int
        Length of each peak
    n_channels: int
        Number of input channels of peak sequence. Default: 4 (ACTG)
    kernel_size: int
        Kernel size for convolutional layer. Default: 5
    n_dims: int
        Embedding size for peak sequence. Default: 16
    dropout_rate: float
        Dropout rate. Default: 0.25
    """

    def __init__(
        self,
        peak_len: int,
        n_filters: Sequence[int] | None = None,
        n_channels: int = 4,
        kernel_size: int = 5,
        n_dims: int = 16,
        dropout_rate: float = 0.25,
    ) -> None:
        if n_filters is None:
            n_filters = [64, 32, 32, 16]
        super().__init__()

        self.peak_len = peak_len

        # parameters for sequence encoder
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_dims = n_dims
        self.dropout_rate = dropout_rate

        # build sequence encoders
        self.seq_encoder = SeqEncoder(
            base_size=self.n_channels,
            kernel_size=self.kernel_size,
            n_filters=self.n_filters,
        )
        self.embd_len = (self.peak_len // (2 ** len(self.n_filters))) * self.n_dims

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embd_len, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 1),
        )

    def forward(self, peak_seq):
        # Embed peak sequence
        seq_embd = self.seq_encoder(peak_seq)  # (batch, 1, L, 4)
        seq_embd = torch.flatten(seq_embd.permute(0, 2, 1, 3), start_dim=1)  # (batch, embd_len)
        x = self.fc(seq_embd)
        return x



if __name__ == "__main__":
    # unit test
    batch_size = 10
    peak_len = 128

    model = Peaks2Accessibility(
        peak_len=peak_len,
        n_filters=[64, 32, 32, 16],
        n_channels=4,
        kernel_size=5,
        n_dims=16,
        dropout_rate=0.25,
    )

    peak_seq = torch.randn(batch_size, 1, peak_len, 4) # (batch_size, 1, peak_len, n_channels)

    print(peak_seq.shape)

    output = model(peak_seq)
    print(output)

    # generate random binary target
    target = torch.poisson(torch.ones(batch_size, 1) * 10)

    print(target)

    # compute BCE loss
    criterion = nn.PoissonNLLLoss(log_input=True)
    loss = criterion(output, target)
    print("Loss:", loss.item())
