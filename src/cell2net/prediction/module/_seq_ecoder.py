from collections.abc import Sequence

from torch import nn


class SeqEncoder(nn.Module):
    def __init__(
        self,
        base_size: int = 4,
        kernel_size: int = 7,
        n_filters: Sequence[int] | None = None,
    ) -> None:
        if n_filters is None:
            n_filters = [128, 64, 32, 32]
        super().__init__()
        self.conv_dims = list(n_filters)
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
