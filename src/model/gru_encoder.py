"""GRU temporal encoder for ST-HGAT-DRIO.

Encodes the last 14 days of demand history per node into a fixed-size embedding.
Improvements over v1:
  - 2-layer bidirectional GRU for richer temporal context
  - Dropout between layers for regularisation
  - Layer normalisation on the output embedding
"""

import torch
import torch.nn as nn
from torch import Tensor


class GRUEncoder(nn.Module):
    """Two-layer bidirectional GRU encoder.

    Args:
        input_dim: Number of input features per timestep.
        d_hidden:  Dimension of the GRU hidden state per direction (default 64).
                   The output embedding has size d_hidden (projected from 2*d_hidden).
        num_layers: Number of stacked GRU layers (default 2).
        dropout:    Dropout probability between GRU layers (default 0.1).
    """

    SEQ_LEN = 14

    def __init__(
        self,
        input_dim: int,
        d_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project bidirectional output (2*d_hidden) back to d_hidden
        self.proj = nn.Linear(2 * d_hidden, d_hidden, bias=False)
        self.norm = nn.LayerNorm(d_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of node feature sequences.

        Args:
            x: ``[batch, seq_len, input_dim]``.
               Sequences shorter than SEQ_LEN are left-padded with zeros.

        Returns:
            Embedding of shape ``[batch, d_hidden]``.

        Raises:
            ValueError: If ``x`` contains any NaN values.
        """
        if torch.isnan(x).any():
            raise ValueError(
                "GRUEncoder received input tensor containing NaN values. "
                "Please ensure all NaN values are handled before encoding."
            )

        batch, seq_len, input_dim = x.shape

        # Left-pad sequences shorter than SEQ_LEN with zeros
        if seq_len < self.SEQ_LEN:
            pad = x.new_zeros(batch, self.SEQ_LEN - seq_len, input_dim)
            x = torch.cat([pad, x], dim=1)

        # h_n: [num_layers*2, batch, d_hidden]
        _, h_n = self.gru(x)

        # Take the last layer's forward + backward hidden states
        # h_n[-2] = last layer forward, h_n[-1] = last layer backward
        h_fwd = h_n[-2]  # [batch, d_hidden]
        h_bwd = h_n[-1]  # [batch, d_hidden]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # [batch, 2*d_hidden]

        out = self.norm(self.proj(self.drop(h_cat)))  # [batch, d_hidden]
        return out
