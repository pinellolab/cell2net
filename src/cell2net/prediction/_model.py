import torch
from lightning import LightningModule
from torch import nn


# https://stackoverflow.com/questions/62162576/calculating-shape-of-conv1d-layer-in-pytorch
def _compute_output_size(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class SeqEncoder(nn.Module):
    """
    A CNN-based sequence encoder.

    Parameters
    ----------
    seq_len : int
        sequence length
    """

    def __init__(
        self,
        seq_length: int = 512,
        n_channels: int = 4,
        n_filters: int = 30,
        kernel_size: int = 5,
        n_dims: int = 16,
    ) -> None:
        super().__init__()

        self.seq_length = seq_length
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                self.n_channels,
                self.n_filters,
                dilation=1,
                kernel_size=self.kernel_size,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.n_filters, self.n_filters, dilation=2, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.n_filters, self.n_filters, dilation=4, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )

        # compute final output length after convolution layers
        self.len1 = (
            _compute_output_size(
                length_in=self.seq_length,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                dilation=1,
            )
            // 2
        )
        self.len2 = (
            _compute_output_size(
                length_in=self.len1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                dilation=2,
            )
            // 2
        )
        self.len3 = (
            _compute_output_size(
                length_in=self.len2,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                dilation=4,
            )
            // 2
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(self.len3 * self.n_filters, n_dims))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)

        return x


class Cell2Net(LightningModule):
    def __init__(
        self,
        n_peaks: int,
        seq_lengths: list[int],
        n_channels: int = 4,
        n_filters: int = 30,
        kernel_size: int = 5,
        n_dims: int = 16,
    ) -> None:
        super().__init__()

        self.n_peaks = n_peaks
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_dims = n_dims
        self.seq_lengths = seq_lengths

        self.seq_encoders = nn.ModuleList([])
        for i in range(self.n_peaks):
            encoder = SeqEncoder(
                seq_length=seq_lengths[i],
                n_channels=self.n_channels,
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                n_dims=self.n_dims,
            )

            self.seq_encoders.append(encoder)

        self.fc = nn.Sequential(
            nn.Linear(self.n_peaks * (self.n_dims + 1), 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, peak_seq, atac):
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

        # concat peak accessibility
        x = torch.concat([x, atac], dim=1)
        x = self.fc(x)

        return x
