"""Heterogeneous graph builder for ST-HGAT-DRIO (Phase 0).

Reads SupplyGraph CSVs and assembles a torch_geometric.data.HeteroData object
with node types `sku`, `warehouse`, `supplier` and four SKU-to-SKU edge types.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# Mapping from edge CSV key → HeteroData edge type tuple
_EDGE_TYPE_MAP: dict[str, tuple[str, str, str]] = {
    "plant": ("sku", "plant", "sku"),
    "product_group": ("sku", "product_group", "sku"),
    "subgroup": ("sku", "subgroup", "sku"),
    "storage": ("sku", "storage", "sku"),
}

# Canonical edge type strings used in get_connectivity keys
_EDGE_TYPE_STRINGS: dict[str, str] = {
    "plant": "sku__plant__sku",
    "product_group": "sku__product_group__sku",
    "subgroup": "sku__subgroup__sku",
    "storage": "sku__storage__sku",
}


class GraphBuilder:
    """Builds a HeteroData graph from SupplyGraph node and edge CSVs."""

    def __init__(self) -> None:
        self._data: HeteroData | None = None
        self._node_to_idx: dict[str, int] = {}
        # edge_key → edge_index tensor [2, E]
        self._edge_indices: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, nodes_csv: Path, edge_csvs: dict[str, Path]) -> HeteroData:
        """Build and return a HeteroData object.

        Parameters
        ----------
        nodes_csv:
            Path to ``Nodes.csv``.  Each row is one SKU node; the first
            (and only) column contains the node identifier string.
        edge_csvs:
            Mapping of edge-type key → CSV path.  Recognised keys are
            ``plant``, ``product_group``, ``subgroup``, ``storage``.
            Each CSV must have columns ``node1`` and ``node2``.
        """
        nodes_csv = Path(nodes_csv)
        self._node_to_idx = self._read_nodes(nodes_csv)
        # num_sku is the number of *unique* node ids (dict deduplicates)
        num_sku = len(self._node_to_idx)

        data = HeteroData()

        # Node feature tensors (identity / placeholder — shape [N, 1])
        data["sku"].x = torch.zeros(num_sku, 1)
        data["sku"].node_id = torch.arange(num_sku, dtype=torch.long)
        # warehouse and supplier nodes: empty placeholders (0 nodes)
        data["warehouse"].x = torch.zeros(0, 1)
        data["supplier"].x = torch.zeros(0, 1)

        self._edge_indices = {}
        for key, csv_path in edge_csvs.items():
            if key not in _EDGE_TYPE_MAP:
                logger.warning("Unknown edge key '%s'; skipping.", key)
                continue
            edge_index = self._read_edges(key, Path(csv_path))
            src_type, rel, dst_type = _EDGE_TYPE_MAP[key]
            data[src_type, rel, dst_type].edge_index = edge_index
            self._edge_indices[key] = edge_index

        self._data = data
        return data

    def get_connectivity(self) -> dict[str, torch.Tensor]:
        """Return COO adjacency tensors keyed by canonical edge-type string.

        Each value is a ``[2, E]`` tensor identical to the ``edge_index``
        stored in the HeteroData object.  The non-zero count across all
        returned tensors equals the total number of edges in the graph.
        """
        if self._data is None:
            raise RuntimeError("Call build() before get_connectivity().")
        return {
            _EDGE_TYPE_STRINGS[key]: tensor
            for key, tensor in self._edge_indices.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_nodes(self, nodes_csv: Path) -> dict[str, int]:
        """Read Nodes.csv and return a node-id → integer-index mapping.

        Duplicate node names are deduplicated; indices are assigned
        sequentially (0, 1, 2, …) over the unique node list so that
        the maximum index always equals len(mapping) - 1.
        """
        df = pd.read_csv(nodes_csv, header=0, encoding="utf-8")
        node_col = df.columns[0]
        node_ids = df[node_col].astype(str).str.strip().tolist()
        # Deduplicate while preserving first-occurrence order
        seen: dict[str, int] = {}
        for nid in node_ids:
            if nid not in seen:
                seen[nid] = len(seen)
        return seen

    def _read_edges(self, key: str, csv_path: Path) -> torch.Tensor:
        """Read an edge CSV and return a valid [2, E] edge_index tensor.

        Rows referencing nodes absent from Nodes.csv are logged and skipped.
        """
        df = pd.read_csv(csv_path, header=0, encoding="utf-8")
        # Normalise column names to lower-case stripped strings
        df.columns = [c.strip().lower() for c in df.columns]

        if "node1" not in df.columns or "node2" not in df.columns:
            logger.warning(
                "Edge CSV '%s' missing node1/node2 columns; skipping.", csv_path
            )
            return torch.zeros(2, 0, dtype=torch.long)

        src_indices: list[int] = []
        dst_indices: list[int] = []
        skipped = 0

        for _, row in df.iterrows():
            n1 = str(row["node1"]).strip()
            n2 = str(row["node2"]).strip()
            if n1 not in self._node_to_idx:
                logger.warning(
                    "Edge CSV '%s': source node '%s' not in Nodes.csv; skipping edge.",
                    csv_path.name,
                    n1,
                )
                skipped += 1
                continue
            if n2 not in self._node_to_idx:
                logger.warning(
                    "Edge CSV '%s': destination node '%s' not in Nodes.csv; skipping edge.",
                    csv_path.name,
                    n2,
                )
                skipped += 1
                continue
            src_indices.append(self._node_to_idx[n1])
            dst_indices.append(self._node_to_idx[n2])

        if skipped:
            logger.info(
                "Edge CSV '%s' (%s): skipped %d edge(s) with unknown nodes.",
                csv_path.name,
                key,
                skipped,
            )

        if not src_indices:
            return torch.zeros(2, 0, dtype=torch.long)

        edge_index = torch.tensor(
            [src_indices, dst_indices], dtype=torch.long
        )
        return edge_index
