"""SC-RIHN: Supply-Chain Resilience Inference Hypergraph Network.

Replaces binary SKU-to-SKU edges with hyperedges that capture higher-order
interactions: firm → shared_plant → firm, firm → product_group → firm.

A hyperedge connects a set of nodes (e.g., all SKUs sharing a plant).
Message passing:
  1. Node → Hyperedge: aggregate node features into hyperedge embedding
  2. Hyperedge → Node: redistribute hyperedge signal back to member nodes

This allows a disruption in one factory to propagate to ALL related SKUs
simultaneously, rather than only to direct graph neighbors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HypergraphConv(nn.Module):
    """One layer of hypergraph convolution.

    Given:
        X: node features [N, d_in]
        H: incidence matrix [N, M]  (H[i,j]=1 if node i belongs to hyperedge j)

    Computes:
        X' = D_v^{-1} H W D_e^{-1} H^T X  (spectral HGNN)

    where D_v = diag(H @ 1) is node degree, D_e = diag(H^T @ 1) is hyperedge degree,
    W = learnable diagonal weight matrix for hyperedges.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_node = nn.Linear(d_in, d_out, bias=False)
        self.W_edge = nn.Linear(d_in, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        # Residual
        self.residual = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        """
        Args:
            x: [N, d_in]  node features
            H: [N, M]     incidence matrix (float, 0/1)

        Returns:
            [N, d_out]  updated node features
        """
        N, M = H.shape

        # Node degree: [N]
        d_v = H.sum(dim=1).clamp(min=1.0)
        # Hyperedge degree: [M]
        d_e = H.sum(dim=0).clamp(min=1.0)

        # Step 1: Node → Hyperedge aggregation
        # e_j = (1/d_e_j) * sum_{i in e_j} x_i
        # [M, d_in] = H^T [N, d_in] / d_e [M, 1]
        x_proj = self.W_node(x)                          # [N, d_out]
        edge_feat = (H.T @ x_proj) / d_e.unsqueeze(-1)  # [M, d_out]

        # Step 2: Hyperedge → Node redistribution
        # x'_i = (1/d_v_i) * sum_{j: i in e_j} e_j
        # [N, d_out] = H [M, d_out] / d_v [N, 1]
        node_update = (H @ edge_feat) / d_v.unsqueeze(-1)  # [N, d_out]

        # Residual + norm
        out = self.norm(node_update + self.residual(x))
        return self.drop(out)


class SCRIHNEncoder(nn.Module):
    """Supply-Chain Resilience Inference Hypergraph Network.

    Builds hyperedges from the SupplyGraph edge structure:
    - Plant hyperedges: all SKUs sharing a plant
    - Product-group hyperedges: all SKUs in the same product group
    - Storage hyperedges: all SKUs at the same storage location

    Two HypergraphConv layers with residual connections.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.conv1 = HypergraphConv(d_in, d_out, dropout)
        self.conv2 = HypergraphConv(d_out, d_out, dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor, incidence_dict: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            x:               [N, d_in]  node features
            incidence_dict:  dict of edge_type → incidence matrix [N, M]

        Returns:
            [N, d_out]  hypergraph-enriched node features
        """
        # Combine all incidence matrices
        H_list = list(incidence_dict.values())
        if not H_list:
            # No hyperedges — identity pass-through
            if self.d_in != self.d_out:
                return self.conv1.residual(x)
            return x

        H = torch.cat(H_list, dim=1)  # [N, sum(M_k)]

        out = self.act(self.conv1(x, H))
        out = self.conv2(out, H)
        return out


def build_incidence_matrices(
    edge_index_dict: Dict[str, Tensor],
    n_nodes: int,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tensor]:
    """Build hyperedge incidence matrices from edge_index tensors.

    For each edge type, groups edges by their shared attribute (plant, product group, etc.)
    to form hyperedges. Each unique source node becomes a hyperedge containing
    all its destination nodes.

    Args:
        edge_index_dict: dict of edge_type_key → [2, E] edge index tensor
        n_nodes:         total number of nodes
        device:          target device

    Returns:
        dict of edge_type → incidence matrix [n_nodes, n_hyperedges]
    """
    incidence = {}

    for etype_key, edge_index in edge_index_dict.items():
        if edge_index.numel() == 0:
            continue

        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        # Group by source node: each unique source = one hyperedge
        # containing all its destination nodes
        unique_src = src.unique()
        M = len(unique_src)
        if M == 0:
            continue

        src_to_idx = {int(s): i for i, s in enumerate(unique_src.tolist())}

        # Build sparse incidence matrix [N, M]
        H = torch.zeros(n_nodes, M, device=device)
        for e_idx in range(edge_index.shape[1]):
            s = int(src[e_idx])
            d = int(dst[e_idx])
            if s in src_to_idx and d < n_nodes:
                H[d, src_to_idx[s]] = 1.0

        # Also include source nodes themselves in their hyperedge
        for s, idx in src_to_idx.items():
            if s < n_nodes:
                H[s, idx] = 1.0

        incidence[etype_key] = H

    return incidence
