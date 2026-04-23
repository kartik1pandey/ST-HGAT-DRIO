"""ESN encoder — re-exports from sgp_encoder for run_submission.py compatibility."""
from src.model.sgp_encoder import SGPEncoder  # noqa: F401
import torch
import torch.nn as nn
from typing import Dict, Optional
from torch import Tensor


class EchoStateNetwork(nn.Module):
    """ESN with reservoir_size alias for run_submission.py compatibility."""

    def __init__(self, input_dim: int, reservoir_size: int = 256,
                 d_out: int = 64, spectral_radius: float = 0.9,
                 leaking_rate: float = 0.3, seed: int = 42,
                 sparsity: float = 0.9) -> None:
        super().__init__()
        from src.model.sgp_encoder import EchoStateNetwork as _ESN
        self._esn = _ESN(input_dim=input_dim, reservoir_dim=reservoir_size,
                         spectral_radius=spectral_radius, leaking_rate=leaking_rate,
                         sparsity=sparsity)
        self.reservoir_dim = reservoir_size
        self.d_out = d_out
        # Trainable readout: reservoir_dim → d_out
        self.readout = nn.Sequential(
            nn.Linear(reservoir_size, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [N, T, input_dim] → [N, d_out]"""
        h = self._esn(x)          # [N, reservoir_dim]
        return self.readout(h)    # [N, d_out]


def build_normalized_adjacency(
    edge_index_dict: Dict[str, Tensor], n_nodes: int
) -> Tensor:
    """Build D^{-1/2} A D^{-1/2} from all edge types. Returns dense [N, N] tensor."""
    rows, cols = [], []
    for ei in edge_index_dict.values():
        if ei.numel() > 0:
            rows.extend([ei[0], ei[1]])
            cols.extend([ei[1], ei[0]])
    if not rows:
        return torch.eye(n_nodes)
    row = torch.cat(rows)
    col = torch.cat(cols)
    idx = (row * n_nodes + col).unique()
    row = idx // n_nodes
    col = idx % n_nodes
    deg = torch.zeros(n_nodes)
    deg.scatter_add_(0, row, torch.ones(len(row)))
    d = deg.pow(-0.5).clamp(max=1e6)
    d[deg == 0] = 0.0
    vals = d[row] * d[col]
    A = torch.sparse_coo_tensor(torch.stack([row, col]), vals, (n_nodes, n_nodes)).to_dense()
    return A
