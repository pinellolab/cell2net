import torch
from torch import nn

from ._seq_ecoder import SeqEncoder


class PeaksTF2GeneExpression(nn.Module):
    def __init__(
        self,
        n_peaks: int,
        peak_len: int,
        n_tfs: int,
        n_filters: int = 32,
        n_channels: int = 4,
        kernel_size: int = 5,
        n_dims: int = 8,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()

        self.n_peaks = n_peaks
        self.n_tfs = n_tfs
        self.peak_len = peak_len

        # parameters for sequence encoder
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_dims = n_dims
        self.dropout_rate = dropout_rate

        # build sequence encoders
        self.seq_encoders = nn.ModuleList([])
        for _ in range(self.n_peaks):
            encoder = SeqEncoder(
                seq_length=self.peak_len,
                n_channels=self.n_channels,
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                n_dims=self.n_dims,
                dropout_rate=self.dropout_rate,
            )

            self.seq_encoders.append(encoder)

        self.fc = nn.Sequential(
            nn.Linear(self.n_peaks * (self.n_dims + 1) + self.n_tfs, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, peak_seq, peak_acc, tf_exp):
        assert (
            peak_seq.shape[1] == self.n_peaks
        ), f"Input size is incorrect, found {peak_seq.shape[1]} peaks, expected {self.n_peaks} peaks!"

        # embed peak sequence
        x = []
        for i in range(self.n_peaks):
            _peak_seq = peak_seq[:, i, :, :]
            x.append(self.seq_encoders[i](_peak_seq))

        # concat sequence embeddings
        x = torch.concat(x, dim=1)

        # concat peak accessibility and tf expression
        x = torch.concat([x, peak_acc, tf_exp], dim=1)

        x = self.fc(x)

        return x
