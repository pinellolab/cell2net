from torch import nn


# https://stackoverflow.com/questions/62162576/calculating-shape-of-conv1d-layer-in-pytorch
def _compute_output_size(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class SeqEncoder(nn.Module):
    """
    A CNN-based sequence encoder.

    Parameters
    ----------
    seq_length : int
        sequence length
    n_channels : int
        number of bases in DNA sequence. Default: 4
    n_filters: int
        number of filters
    kernel_size:
        kernel size of filter
    n_dims:
        number of dimensions for seqence embedding
    """

    def __init__(
        self,
        seq_length: int = 512,
        n_channels: int = 4,
        n_filters: int = 30,
        kernel_size: int = 5,
        n_dims: int = 8,
        dropout_rate: float = 0.25,
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
            nn.Dropout(dropout_rate),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                self.n_filters, self.n_filters, dilation=2, kernel_size=self.kernel_size
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                self.n_filters, self.n_filters, dilation=4, kernel_size=self.kernel_size
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
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

        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(self.len3 * self.n_filters, n_dims)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)

        return x
