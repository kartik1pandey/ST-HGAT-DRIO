"""Scalable Graph Predictor (SGP) encoder.

Pre-computes spatiotemporal node embeddings using:
  1. Echo State Network (ESN) — randomized RNN, no backprop through time.
     Vectorized over the time dimension for speed.
  2. Graph diffusion — powers A^0..A^K applied to reservoir states.
     Adjacency is cached per (n_nodes, edge_hash) to avoid recomputation.

O(1) memory relative to graph size: decoder trains node-wise in mini-batches.

Reference: Cini et al., "Scalable Spatiotemporal Graph Neural Networks" (2023).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class EchoStateNetwork(nn.Module):
    """Randomized recurrent reservoir (Echo State Network).

    Fixed weights — no gradient flows through the reservoir.
    The time loop is kept sequential (required for recurrence) but
    all N nodes are processed in parallel at each step.

    Args:
        input_dim:       Number of input features per timestep.
        reservoir_dim:   Reservoir state size (default 256).
        spectral_radius: Target spectral radius of W_res (default 0.9).
        leaking_rate:    Leaking rate in [0,1] (default 0.3).
        sparsity:        Fraction of zero weights in W_res (default 0.9).
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 256,
        spectral_radius: float = 0.9,
        leaking_rate: float = 0.3,
        sparsity: float = 0.9,
    ) -> None:
        super().__init__()
        self.reservoir_dim = reservoir_dim
        self.leaking_rate = leaking_rate

        # Input weights — fixed random [reservoir_dim, input_dim]
        W_in = torch.randn(reservoir_dim, input_dim) * 0.1
        self.register_buffer("W_in", W_in)

        # Reservoir weights — sparse random, scaled to spectral_radius
        W_res = torch.randn(reservoir_dim, reservoir_dim)
        mask = torch.rand(reservoir_dim, reservoir_dim) < sparsity
        W_res[mask] = 0.0
        try:
            eigs = torch.linalg.eigvals(W_res)
            rho = eigs.abs().max().item()
            if rho > 1e-8:
                W_res = W_res * (spectral_radius / rho)
        except Exception:
            W_res = W_res * spectral_radius
        self.register_buffer("W_res", W_res)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Run reservoir dynamics over T timesteps.

        Args:
            x: [N, T, input_dim]

        Returns:
            [N, reservoir_dim]  final reservoir state
        """
        N, T, _ = x.shape
        h = x.new_zeros(N, self.reservoir_dim)

        # Pre-project input for all timesteps: [N, T, reservoir_dim]
        # x @ W_in.T  →  [N, T, reservoir_dim]
        x_proj = x @ self.W_in.T  # [N, T, reservoir_dim]

        for t in range(T):
            pre = x_proj[:, t, :] + h @ self.W_res.T   # [N, reservoir_dim]
            h_new = torch.tanh(pre)
            h = (1.0 - self.leaking_rate) * h + self.leaking_rate * h_new

        return h  # [N, reservoir_dim]


class SGPEncoder(nn.Module):
    """Scalable Graph Predictor encoder.

    Combines ESN reservoir states with graph diffusion (A^0..A^K).
    Adjacency matrix is cached to avoid recomputation across forward passes.

    Args:
        input_dim:     Input feature dimension.
        reservoir_dim: ESN reservoir size (default 256).
        d_out:         Output embedding dimension (default 64).
        K:             Number of diffusion hops (default 3).
        dropout:       Dropout on readout (default 0.1).
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 256,
        d_out: int = 64,
        K: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.K = K
        self.reservoir_dim = reservoir_dim
        self.d_out = d_out

        self.esn = EchoStateNetwork(input_dim, reservoir_dim)

        # Readout: (K+1) * reservoir_dim → d_out
        self.readout = nn.Sequential(
            nn.Linear((K + 1) * reservoir_dim, d_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )

        # Adjacency cache: (n_nodes, edge_hash) → A tensor
        self._adj_cache: Dict[Tuple[int, int], Tensor] = {}

    def _edge_hash(self, edge_index_dict: Dict[str, Tensor]) -> int:
        """Cheap hash of edge structure for cache keying."""
        total = 0
        for k, v in sorted(edge_index_dict.items()):
            if v.numel() > 0:
                total += hash(k) + int(v.sum().item()) + v.shape[1]
        return total

    def _build_norm_adj(
        self,
        edge_index_dict: Dict[str, Tensor],
        n_nodes: int,
        device: torch.device,
    ) -> Tensor:
        """Build symmetric D^{-1/2} A D^{-1/2} from all edge types.

        Uses cache to avoid recomputation when graph structure is unchanged.
        """
        cache_key = (n_nodes, self._edge_hash(edge_index_dict))
        if cache_key in self._adj_cache:
            return self._adj_cache[cache_key].to(device)

        rows, cols = [], []
        for ei in edge_index_dict.values():
            if ei.numel() > 0:
                rows.extend([ei[0], ei[1]])
                cols.extend([ei[1], ei[0]])

        if not rows:
            A = torch.eye(n_nodes, device=device)
            self._adj_cache[cache_key] = A
            return A

        row = torch.cat(rows).to(device)
        col = torch.cat(cols).to(device)

        # Deduplicate
        idx = (row * n_nodes + col).unique()
        row = idx // n_nodes
        col = idx % n_nodes

        # D^{-1/2} A D^{-1/2}
        deg = torch.zeros(n_nodes, device=device)
        deg.scatter_add_(0, row, torch.ones(len(row), device=device))
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e6)
        deg_inv_sqrt[deg == 0] = 0.0

        vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        A = torch.sparse_coo_tensor(
            torch.stack([row, col]), vals, (n_nodes, n_nodes), device=device
        ).to_dense()

        # Limit cache size to avoid memory leak
        if len(self._adj_cache) > 32:
            self._adj_cache.clear()
        self._adj_cache[cache_key] = A.cpu()
        return A

    def forward(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
    ) -> Tensor:
        """Compute SGP embeddings.

        Args:
            x:               [N, T, input_dim]
            edge_index_dict: edge type → [2, E]

        Returns:
            [N, d_out]
        """
        N = x.shape[0]
        device = x.device

        # Step 1: ESN reservoir states [N, reservoir_dim]
        h = self.esn(x)

        # Step 2: Graph diffusion A^0 h, A^1 h, ..., A^K h
        A = self._build_norm_adj(edge_index_dict, N, device)
        diffused = [h]
        h_k = h
        for _ in range(self.K):
            h_k = A @ h_k
            diffused.append(h_k)

        # Step 3: Concatenate and project
        cat = torch.cat(diffused, dim=-1)  # [N, (K+1)*reservoir_dim]
        return self.readout(cat)           # [N, d_out]
