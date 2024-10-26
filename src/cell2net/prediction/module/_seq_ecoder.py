from torch import nn

# # https://stackoverflow.com/questions/62162576/calculating-shape-of-conv1d-layer-in-pytorch
# def _compute_output_size(length_in, kernel_size, stride=1, padding=0, dilation=1):
#     return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# class SeqEncoder(nn.Module):
#     """
#     A CNN-based sequence encoder.

#     Parameters
#     ----------
#     seq_length : int
#         sequence length
#     n_channels : int
#         number of bases in DNA sequence. Default: 4
#     n_filters: int
#         number of filters
#     kernel_size:
#         kernel size of filter
#     n_dims:
#         number of dimensions for seqence embedding
#     """

#     def __init__(
#         self,
#         seq_length: int = 512,
#         n_channels: int = 4,
#         n_filters: int = 30,
#         kernel_size: int = 5,
#         n_dims: int = 8,
#         dropout_rate: float = 0.25,
#     ) -> None:
#         super().__init__()

#         self.seq_length = seq_length
#         self.n_channels = n_channels
#         self.n_filters = n_filters
#         self.kernel_size = kernel_size

#         self.conv1 = nn.Sequential(
#             nn.Conv1d(
#                 self.n_channels,
#                 self.n_filters,
#                 dilation=1,
#                 kernel_size=self.kernel_size,
#             ),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2),
#             nn.Dropout(dropout_rate),
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv1d(
#                 self.n_filters, self.n_filters, dilation=2, kernel_size=self.kernel_size
#             ),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2),
#             nn.Dropout(dropout_rate),
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv1d(
#                 self.n_filters, self.n_filters, dilation=4, kernel_size=self.kernel_size
#             ),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2),
#             nn.Dropout(dropout_rate),
#         )

#         # compute final output length after convolution layers
#         self.len1 = (
#             _compute_output_size(
#                 length_in=self.seq_length,
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=0,
#                 dilation=1,
#             )
#             // 2
#         )
#         self.len2 = (
#             _compute_output_size(
#                 length_in=self.len1,
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=0,
#                 dilation=2,
#             )
#             // 2
#         )
#         self.len3 = (
#             _compute_output_size(
#                 length_in=self.len2,
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=0,
#                 dilation=4,
#             )
#             // 2
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(), nn.Linear(self.len3 * self.n_filters, n_dims)
#         )

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.fc(x)

#         return x


class SeqEncoder(nn.Module):
    def __init__(
        self,
        base_size: int = 4,
        kernel_size: int = 7,
        n_filters: list[int] | None = None,
    ) -> None:
        if n_filters is None:
            n_filters = [128, 64, 32, 32]
        super().__init__()
        self.conv_dims = n_filters
        self.base_size = base_size
        self.kernal_size = kernel_size
        # cropped_len = 46
        self.stem_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=base_size,
                out_channels=self.conv_dims[0],
                kernel_size=(1, self.kernal_size),
                stride=1,
                padding="same",
                dilation=1,
            ),
            nn.ELU(),
        )
        self.conv_tower = nn.ModuleList([])
        conv_dim = self.conv_dims + [self.conv_dims[-1]]
        for i in range(len(self.conv_dims)):
            self.conv_tower.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=conv_dim[i],
                        out_channels=conv_dim[i + 1],
                        kernel_size=(1, 3),
                        padding=(0, 1),
                    ),
                    nn.BatchNorm2d(conv_dim[i + 1]),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                )
            )
            self.conv_tower.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=conv_dim[i + 1],
                        out_channels=conv_dim[i + 1],
                        kernel_size=(1, 1),
                    ),
                    nn.ELU(),
                )
            )

    def forward(self, seq_input):
        x = seq_input.permute(0, 3, 1, 2).contiguous()
        x = self.stem_conv(x)
        for i in range(0, len(self.conv_tower), 2):
            x = self.conv_tower[i](x)
            x = self.conv_tower[i + 1](x) + x
        return x
