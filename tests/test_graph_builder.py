"""Tests for src/graph/builder.py.

Covers:
  - Property 1: Graph Index Validity          (Validates: Requirements 1.4)
  - Property 2: Edge Count Consistency        (Validates: Requirements 1.5)
  - Unit test: unknown-node edge skipping     (Requirements 1.6)
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.graph.builder import GraphBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_nodes_csv(tmp_dir: Path, node_ids: list[str]) -> Path:
    """Write a minimal Nodes.csv to *tmp_dir* and return its path."""
    path = tmp_dir / "Nodes.csv"
    path.write_text("Node\n" + "\n".join(node_ids) + "\n", encoding="utf-8")
    return path


def _write_edge_csv(tmp_dir: Path, name: str, edges: list[tuple[str, str]]) -> Path:
    """Write a minimal edge CSV (node1, node2) to *tmp_dir* and return its path."""
    path = tmp_dir / f"{name}.csv"
    lines = ["node1,node2"] + [f"{s},{d}" for s, d in edges]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_from_raw(
    node_ids: list[str],
    edge_groups: dict[str, list[tuple[str, str]]],
) -> tuple[GraphBuilder, object]:
    """Build a HeteroData from raw lists using a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        nodes_csv = _write_nodes_csv(tmp_path, node_ids)
        edge_csvs: dict[str, Path] = {}
        for key, edges in edge_groups.items():
            edge_csvs[key] = _write_edge_csv(tmp_path, key, edges)
        gb = GraphBuilder()
        data = gb.build(nodes_csv, edge_csvs)
        return gb, data


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# A strategy that generates a non-empty list of unique ASCII alphanumeric node id strings
_ASCII_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
node_list_st = st.lists(
    st.text(alphabet=_ASCII_ALNUM, min_size=1, max_size=10),
    min_size=1,
    max_size=30,
    unique=True,
)

# Edge key names we support
EDGE_KEYS = ["plant", "product_group", "subgroup", "storage"]


def valid_edges_st(node_ids: list[str]):
    """Strategy: list of (src, dst) pairs drawn from *node_ids*."""
    if len(node_ids) < 2:
        return st.just([])
    return st.lists(
        st.tuples(st.sampled_from(node_ids), st.sampled_from(node_ids)),
        min_size=0,
        max_size=20,
    )


# ---------------------------------------------------------------------------
# Property 1: Graph Index Validity
# Feature: st-hgat-drio, Property 1: graph index validity
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(node_ids=node_list_st)
def test_property1_graph_index_validity(node_ids: list[str]) -> None:
    """All edge indices in every edge_index tensor must be < num_nodes.

    **Validates: Requirements 1.4**
    """
    # Feature: st-hgat-drio, Property 1: graph index validity
    num_nodes = len(node_ids)

    # Build edges that are all valid (drawn from node_ids)
    if num_nodes >= 2:
        edges = [(node_ids[i % num_nodes], node_ids[(i + 1) % num_nodes])
                 for i in range(min(num_nodes, 10))]
    else:
        edges = []

    edge_groups = {key: edges for key in EDGE_KEYS}
    gb, data = _build_from_raw(node_ids, edge_groups)

    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index  # [2, E]
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        num_src = data[src_type].x.shape[0]
        num_dst = data[dst_type].x.shape[0]
        assert edge_index[0].max().item() < num_src, (
            f"Source index out of bounds for edge type {edge_type}"
        )
        assert edge_index[1].max().item() < num_dst, (
            f"Destination index out of bounds for edge type {edge_type}"
        )


# ---------------------------------------------------------------------------
# Property 2: Edge Count Consistency
# Feature: st-hgat-drio, Property 2: edge count consistency
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(node_ids=node_list_st)
def test_property2_edge_count_consistency(node_ids: list[str]) -> None:
    """sum(get_connectivity() non-zeros) == sum(edge_index sizes).

    **Validates: Requirements 1.5**
    """
    # Feature: st-hgat-drio, Property 2: edge count consistency
    num_nodes = len(node_ids)

    if num_nodes >= 2:
        edges = [(node_ids[i % num_nodes], node_ids[(i + 1) % num_nodes])
                 for i in range(min(num_nodes * 2, 15))]
    else:
        edges = []

    edge_groups = {key: edges for key in EDGE_KEYS}
    gb, data = _build_from_raw(node_ids, edge_groups)

    # Count edges from HeteroData edge_index tensors
    hetero_edge_count = sum(
        data[et].edge_index.shape[1]
        for et in data.edge_types
        if data[et].edge_index.numel() > 0
    )

    # Count non-zeros from get_connectivity()
    connectivity = gb.get_connectivity()
    connectivity_count = sum(
        tensor.shape[1] for tensor in connectivity.values() if tensor.numel() > 0
    )

    assert connectivity_count == hetero_edge_count, (
        f"Connectivity non-zero count ({connectivity_count}) != "
        f"HeteroData edge count ({hetero_edge_count})"
    )


# ---------------------------------------------------------------------------
# Unit test: unknown-node edge skipping
# Requirements: 1.6
# ---------------------------------------------------------------------------


def test_unit_unknown_node_edge_skipping(tmp_path: Path) -> None:
    """Edges referencing non-existent nodes are skipped without exception.

    Requirements: 1.6
    """
    # Known nodes
    node_ids = ["A", "B", "C"]
    nodes_csv = _write_nodes_csv(tmp_path, node_ids)

    # Edge CSV: one valid edge (A→B) and one with unknown node (A→UNKNOWN)
    edge_csv = _write_edge_csv(
        tmp_path,
        "plant",
        [("A", "B"), ("A", "UNKNOWN"), ("GHOST", "C")],
    )

    gb = GraphBuilder()
    # Must not raise
    data = gb.build(nodes_csv, {"plant": edge_csv})

    # Only the valid edge (A→B) should be present
    edge_index = data["sku", "plant", "sku"].edge_index
    assert edge_index.shape[1] == 1, (
        f"Expected 1 valid edge, got {edge_index.shape[1]}"
    )
    # Verify it's the A→B edge (indices 0→1)
    assert edge_index[0, 0].item() == 0  # A
    assert edge_index[1, 0].item() == 1  # B


def test_unit_all_unknown_nodes_produces_empty_edge_index(tmp_path: Path) -> None:
    """When all edges reference unknown nodes, edge_index is empty — no exception."""
    nodes_csv = _write_nodes_csv(tmp_path, ["X", "Y"])
    edge_csv = _write_edge_csv(tmp_path, "storage", [("GHOST1", "GHOST2")])

    gb = GraphBuilder()
    data = gb.build(nodes_csv, {"storage": edge_csv})

    edge_index = data["sku", "storage", "sku"].edge_index
    assert edge_index.shape[1] == 0
