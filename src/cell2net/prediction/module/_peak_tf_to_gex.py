import copy

import torch
from torch import nn

from ._seq_ecoder import SeqEncoder


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class AttentionBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_embd: int,
        nhead: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, nhead, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlpf = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.Linear(4 * n_embd, n_embd),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def _sa_block(self, x, key_padding_mask=None, attn_mask=None):
        x, w = self.attn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return x, w

    def forward(self, x):
        att_x, att_w = self._sa_block(self.ln_1(x))
        x = x + att_x
        x = x + self.mlpf(self.ln_2(x))
        return x, att_w


class PeaksTF2GeneExpressionPoisson(nn.Module):
    """
    Predict gene expression using peak sequences, accessibility and TF expression

    Parameters
    ----------
    n_peaks : int
        Number of input peaks
    peak_len: int
        Length of each peak
    n_tfs: int
        Number of TFs used for prediction
    n_covariates: int
        Number of covariates
    n_channels: int
        Number of input channels of peak sequence. Default: 4 (ACTG)
    kernel_size: int
        Kernel size for convolutional layer. Default: 5
    n_dims: int
        Embedding size for peak sequence. Default: 16
    n_attn_blocks: int
        Number of attention blocks of transformer layers. Default: 1
    dropout_rate: float
        Dropout rate. Default: 0.25
    """

    def __init__(
        self,
        n_peaks: int,
        peak_len: int,
        n_tfs: int,
        n_covariates: int,
        n_filters: list[int] | None = None,
        n_channels: int = 4,
        kernel_size: int = 5,
        n_dims: int = 16,
        n_attn_blocks: int = 1,
        dropout_rate: float = 0.25,
    ) -> None:
        if n_filters is None:
            n_filters = [64, 32, 32, 16]
        super().__init__()

        self.n_peaks = n_peaks
        self.n_tfs = n_tfs
        self.peak_len = peak_len
        self.n_covariates = n_covariates

        # parameters for sequence encoder
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_dims = n_dims
        self.dropout_rate = dropout_rate
        self.n_attn_blocks = n_attn_blocks

        # build sequence encoders
        self.seq_encoder = SeqEncoder(
            base_size=self.n_channels,
            kernel_size=self.kernel_size,
            n_filters=self.n_filters,
        )
        self.embd_len = self.peak_len // (2 ** len(self.n_filters))
        self.attn_blocks = get_clones(
            AttentionBlock(n_embd=self.n_dims), self.n_attn_blocks
        )

        self.merge_seq_atac = nn.Sequential(
            nn.Conv1d(
                in_channels=self.embd_len * self.n_filters[-1] + 1,
                out_channels=126,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv1d(in_channels=126, out_channels=self.n_dims, kernel_size=1),
            nn.ReLU(),
        )
        self.embed_len = self.peak_len

        # fully connected layers to predict the log(lambda) of Poisson distribution
        self.fc = nn.Sequential(
            nn.Linear(
                self.n_peaks * (self.n_dims + 1) + self.n_tfs + self.n_covariates, 32
            ),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, peak_seq, peak_acc, peak_dist=None, tf_exp=None, covariates=None):
        assert (
            peak_seq.shape[1] == self.n_peaks
        ), f"Input size is incorrect, found {peak_seq.shape[1]} peaks, expected {self.n_peaks} peaks!"

        # Embed peak sequence
        seq_embd = self.seq_encoder(peak_seq)
        seq_embd = torch.flatten(seq_embd.permute(0, 2, 1, 3), start_dim=2)
        peak_acc = peak_acc.unsqueeze(-1)

        # Merge signal with sequence embedding
        seq_atac_embd = self.merge_seq_atac(
            torch.concat([seq_embd, peak_acc], axis=-1).permute(0, 2, 1)  # type: ignore
        ).permute(0, 2, 1)

        # Send merged embeding to Transformer encoder
        attn_list = []
        for i in range(self.n_attn_blocks):
            seq_atac_embd, attn = self.attn_blocks[i](seq_atac_embd)
            attn_list.append(attn.unsqueeze(0))
        seq_atac_embd = torch.flatten(seq_atac_embd, start_dim=1)

        x = torch.concat([seq_atac_embd, peak_dist, tf_exp, covariates], dim=1)  # type: ignore

        # Concat peak accessibility, tf expression, and covariates
        x = self.fc(x)
        return x


# Testing
# seq_input = torch.randn(8, 79, 256, 4)
# atac_input = torch.randn(8, 79)
# tf_input = torch.randn(8, 513)
# covariates_input = torch.randn(8, 2)
# m = PeaksTF2GeneExpressionPoisson_Jc(79, 256, 513, 2)
# total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print(f'Total trainable parameters: {total_params}')
# out = m(seq_input, atac_input, tf_input, covariates_input)
# print(out)
