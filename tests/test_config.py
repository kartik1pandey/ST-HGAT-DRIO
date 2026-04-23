"""Unit tests for the pipeline configuration loader.

Requirements: 9.1, 9.3
"""

import re
from pathlib import Path

import pytest

from src.pipeline.config import load_config, PipelineConfig

CONFIG_PATH = Path("configs/default.yaml")


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return load_config(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Key presence tests (Requirement 9.1)
# ---------------------------------------------------------------------------

def test_datasets_supply_graph_nodes_csv(cfg):
    assert cfg.datasets.supply_graph.nodes_csv != ""


def test_datasets_supply_graph_edge_csvs_keys(cfg):
    expected_keys = {"plant", "product_group", "subgroup", "storage"}
    assert expected_keys == set(cfg.datasets.supply_graph.edge_csvs.keys())


def test_datasets_dataco_csv(cfg):
    assert cfg.datasets.dataco_csv != ""


def test_datasets_benchmark_dir(cfg):
    assert cfg.datasets.benchmark_dir != ""


def test_model_keys(cfg):
    assert cfg.model.d_hidden == 64
    assert cfg.model.d_out == 64
    assert cfg.model.horizon == 28   # updated to M5 evaluation horizon
    assert cfg.model.seq_len == 14
    assert cfg.model.seed == 42


def test_training_keys(cfg):
    assert cfg.training.max_epochs == 50
    assert cfg.training.batch_size == 256   # node-wise mini-batch size
    assert cfg.training.learning_rate == pytest.approx(3e-4)


def test_optimization_keys(cfg):
    assert cfg.optimization.solver == "gurobi"
    assert cfg.optimization.epsilon == pytest.approx(0.1)
    assert cfg.optimization.gamma == pytest.approx(0.99)
    assert cfg.optimization.holding_cost == pytest.approx(1.0)
    assert cfg.optimization.stockout_penalty == pytest.approx(5.0)


def test_streaming_keys(cfg):
    assert cfg.streaming.backend == "flink"
    assert cfg.streaming.buffer_seconds == 60
    assert cfg.streaming.throughput_target == 1000


# ---------------------------------------------------------------------------
# Config hash tests (Requirement 9.3)
# ---------------------------------------------------------------------------

def test_config_hash_is_64_char_hex(cfg):
    """SHA-256 digest must be a 64-character lowercase hex string."""
    assert len(cfg.config_hash) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", cfg.config_hash) is not None


def test_config_hash_is_deterministic():
    """Loading the same file twice must produce the same hash."""
    cfg1 = load_config(CONFIG_PATH)
    cfg2 = load_config(CONFIG_PATH)
    assert cfg1.config_hash == cfg2.config_hash


def test_config_hash_changes_with_content(tmp_path):
    """A different file must produce a different hash."""
    alt = tmp_path / "alt.yaml"
    alt.write_text(CONFIG_PATH.read_text() + "\n# extra comment\n")
    cfg_orig = load_config(CONFIG_PATH)
    cfg_alt = load_config(alt)
    assert cfg_orig.config_hash != cfg_alt.config_hash
