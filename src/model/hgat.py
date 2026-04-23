"""Dual-branch Heterogeneous Graph Attention Network for ST-HGAT-DRIO.

Improvements over v1:
  - Multi-head attention (default 4 heads) for richer representations
  - Layer normalisation + residual connection on branch outputs
  - ELU activation instead of LeakyReLU (smoother gradients)
  - Dropout on attention coefficients
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Multi-head attention convolution
# ---------------------------------------------------------------------------

class _MultiHeadAttentionConv(nn.Module):
    """Multi-head attention message-passing for one edge type.

    For each head k:
        e_ij^k = ELU( a_k^T [W_k h_i || W_k h_j] )
        alpha_ij^k = softmax_j(e_ij^k)
        h'_i^k = sum_j alpha_ij^k * W_k h_j

    Outputs from all heads are concatenated then projected to d_out.
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.d_out = d_out
        self.dropout = dropout

        # Per-head linear transforms
        self.W = nn.Linear(d_in, d_out, bias=False)          # shared projection
        self.a = nn.Linear(2 * self.d_head, 1, bias=False)   # per-head attention
        self.attn_drop = nn.Dropout(dropout)

        # Populated during forward for inspection
        self.last_alpha: Optional[Tensor] = None
        self.last_edge_index: Optional[Tensor] = None

    def forward(
        self,
        src_feat: Tensor,    # [N_src, d_in]
        dst_feat: Tensor,    # [N_dst, d_in]
        edge_index: Tensor,  # [2, E]
        num_dst: int,
    ) -> Tensor:
        """Return updated dst embeddings of shape [N_dst, d_out]."""
        if edge_index.numel() == 0:
            self.last_alpha = None
            self.last_edge_index = edge_index
            return src_feat.new_zeros(num_dst, self.d_out)

        src_idx = edge_index[0]  # [E]
        dst_idx = edge_index[1]  # [E]
        E = src_idx.shape[0]

        # Project all nodes: [N, d_out] → reshape to [N, H, d_head]
        h_src = self.W(src_feat).view(-1, self.num_heads, self.d_head)  # [N_src, H, d_head]
        h_dst = self.W(dst_feat).view(-1, self.num_heads, self.d_head)  # [N_dst, H, d_head]

        # Gather edge features: [E, H, d_head]
        h_src_e = h_src[src_idx]
        h_dst_e = h_dst[dst_idx]

        # Attention logits per head: [E, H]
        cat_e = torch.cat([h_src_e, h_dst_e], dim=-1)  # [E, H, 2*d_head]
        # Reshape for linear: [E*H, 2*d_head] → [E*H, 1] → [E, H]
        e = F.elu(self.a(cat_e.view(E * self.num_heads, 2 * self.d_head)))
        e = e.view(E, self.num_heads)  # [E, H]

        # Softmax per destination node per head
        alpha = _softmax_per_node_multihead(e, dst_idx, num_dst, self.num_heads)  # [E, H]
        alpha = self.attn_drop(alpha)

        # Store mean alpha across heads for inspection
        self.last_alpha = alpha.mean(dim=-1).detach()
        self.last_edge_index = edge_index

        # Aggregate: [N_dst, H, d_head]
        out = src_feat.new_zeros(num_dst, self.num_heads, self.d_head)
        # alpha: [E, H, 1] * h_src_e: [E, H, d_head]
        weighted = alpha.unsqueeze(-1) * h_src_e  # [E, H, d_head]
        # scatter_add over dst dimension
        dst_expand = dst_idx.view(E, 1, 1).expand(E, self.num_heads, self.d_head)
        out.scatter_add_(0, dst_expand, weighted)

        # Flatten heads: [N_dst, d_out]
        return out.view(num_dst, self.d_out)


def _softmax_per_node_multihead(
    e: Tensor, dst_idx: Tensor, num_dst: int, num_heads: int
) -> Tensor:
    """Softmax of [E, H] logits grouped by destination node."""
    # e: [E, H]
    e_max = e.new_full((num_dst, num_heads), float("-inf"))
    e_max.scatter_reduce_(
        0,
        dst_idx.unsqueeze(-1).expand(-1, num_heads),
        e,
        reduce="amax",
        include_self=True,
    )
    e_shifted = e - e_max[dst_idx]  # [E, H]
    exp_e = torch.exp(e_shifted)

    denom = e.new_zeros(num_dst, num_heads)
    denom.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, num_heads),
                       exp_e.to(denom.dtype))

    alpha = exp_e / (denom[dst_idx] + 1e-12)
    return alpha  # [E, H]


# Keep the single-head version for backward compatibility with tests
class _AttentionConv(_MultiHeadAttentionConv):
    """Single-head attention conv (alias for test compatibility)."""
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__(d_in, d_out, num_heads=1, dropout=0.0)


def _softmax_per_node(e: Tensor, dst_idx: Tensor, num_dst: int) -> Tensor:
    """Single-head softmax (kept for test compatibility)."""
    return _softmax_per_node_multihead(e.unsqueeze(-1), dst_idx, num_dst, 1).squeeze(-1)


# ---------------------------------------------------------------------------
# DualBranchHGAT
# ---------------------------------------------------------------------------

_INTRA_EDGE_TYPES = {
    "sku__plant__sku",
    "sku__product_group__sku",
    "sku__subgroup__sku",
    "sku__storage__sku",
}

_CROSS_EDGE_TYPES = {
    "sku__ships_to__warehouse",
    "sku__supplied_by__supplier",
}


class DualBranchHGAT(nn.Module):
    """Dual-branch Heterogeneous Graph Attention Network with multi-head attention.

    Args:
        d_in:      Input embedding dimension.
        d_out:     Output embedding dimension (default 64).
        num_heads: Number of attention heads (default 4).
        dropout:   Dropout on attention weights (default 0.1).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self._branch_dim = d_out

        # Branch A: intra-type edges
        self.branch_a: nn.ModuleDict = nn.ModuleDict({
            etype: _MultiHeadAttentionConv(d_in, self._branch_dim, num_heads, dropout)
            for etype in _INTRA_EDGE_TYPES
        })

        # Branch B: cross-type edges
        self.branch_b: nn.ModuleDict = nn.ModuleDict({
            etype: _MultiHeadAttentionConv(d_in, self._branch_dim, num_heads, dropout)
            for etype in _CROSS_EDGE_TYPES
        })

        # Projection + residual
        self.proj = nn.Linear(2 * self._branch_dim, d_out)
        self.norm = nn.LayerNorm(d_out)
        # Residual adapter when d_in != d_out
        self.residual = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()

    def _run_branch(
        self,
        convs: nn.ModuleDict,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        dst_sizes: Dict[str, int] = {ntype: x.shape[0] for ntype, x in x_dict.items()}
        accum: Dict[str, Optional[Tensor]] = {ntype: None for ntype in dst_sizes}

        for etype_key, conv in convs.items():
            if etype_key not in edge_index_dict:
                continue
            src_type, _, dst_type = etype_key.split("__", 2)
            if src_type not in x_dict or dst_type not in x_dict:
                continue

            edge_index = edge_index_dict[etype_key]
            out = conv(x_dict[src_type], x_dict[dst_type], edge_index, dst_sizes[dst_type])

            if accum[dst_type] is None:
                accum[dst_type] = out
            else:
                accum[dst_type] = accum[dst_type] + out  # type: ignore[operator]

        result: Dict[str, Tensor] = {}
        for ntype, size in dst_sizes.items():
            if accum[ntype] is None:
                result[ntype] = x_dict[ntype].new_zeros(size, self._branch_dim)
            else:
                result[ntype] = accum[ntype]  # type: ignore[assignment]
        return result

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        out_a = self._run_branch(self.branch_a, x_dict, edge_index_dict)
        out_b = self._run_branch(self.branch_b, x_dict, edge_index_dict)

        result: Dict[str, Tensor] = {}
        for ntype, x in x_dict.items():
            cat = torch.cat([out_a[ntype], out_b[ntype]], dim=-1)  # [N, 2*branch_dim]
            projected = self.proj(cat)                              # [N, d_out]
            # Residual + layer norm
            result[ntype] = self.norm(projected + self.residual(x))
        return result

    def forward_with_attention(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        embeddings = self.forward(x_dict, edge_index_dict)
        attention: Dict[str, Tensor] = {}
        for etype_key, conv in {**dict(self.branch_a), **dict(self.branch_b)}.items():
            conv_obj: _MultiHeadAttentionConv = conv  # type: ignore[assignment]
            if conv_obj.last_alpha is not None:
                attention[etype_key] = conv_obj.last_alpha
        return embeddings, attention
