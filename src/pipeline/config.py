"""Configuration loader for ST-HGAT-DRIO pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SupplyGraphDatasetConfig:
    nodes_csv: str
    edge_csvs: dict[str, str]


@dataclass
class DatasetsConfig:
    supply_graph: SupplyGraphDatasetConfig
    dataco_csv: str
    benchmark_dir: str
    m5_dir: str = "m5-forecasting-accuracy/"


@dataclass
class ModelConfig:
    d_hidden: int
    d_out: int
    horizon: int
    seq_len: int
    seed: int
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    max_epochs: int
    batch_size: int
    learning_rate: float
    joint_alpha: float = 0.8
    n_series: int = 500


@dataclass
class OptimizationConfig:
    solver: str
    epsilon: float
    gamma: float
    holding_cost: float
    stockout_penalty: float
    dynamic_epsilon: bool = True


@dataclass
class DisruptionConfig:
    duration: int = 4
    shock_fraction: float = 0.5
    epsilon_disruption: float = 0.5
    target_recovery_reduction_pct: float = 32.41


@dataclass
class StreamingConfig:
    backend: str
    buffer_seconds: int
    throughput_target: int


@dataclass
class PipelineConfig:
    datasets: DatasetsConfig
    model: ModelConfig
    training: TrainingConfig
    optimization: OptimizationConfig
    streaming: StreamingConfig
    disruption: DisruptionConfig = field(default_factory=DisruptionConfig)
    config_hash: str = field(default="", repr=False)


def _compute_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    sha256.update(path.read_bytes())
    return sha256.hexdigest()


def _parse_config(raw: dict[str, Any]) -> PipelineConfig:
    sg = raw["datasets"]["supply_graph"]
    datasets = DatasetsConfig(
        supply_graph=SupplyGraphDatasetConfig(
            nodes_csv=sg["nodes_csv"],
            edge_csvs=sg["edge_csvs"],
        ),
        dataco_csv=raw["datasets"]["dataco_csv"],
        benchmark_dir=raw["datasets"]["benchmark_dir"],
        m5_dir=raw["datasets"].get("m5_dir", "m5-forecasting-accuracy/"),
    )

    # Model — accept extra keys gracefully
    m = raw["model"]
    model = ModelConfig(
        d_hidden=m["d_hidden"],
        d_out=m["d_out"],
        horizon=m["horizon"],
        seq_len=m["seq_len"],
        seed=m["seed"],
        num_heads=m.get("num_heads", 4),
        dropout=m.get("dropout", 0.1),
    )

    # Training — accept extra keys gracefully
    t = raw["training"]
    training = TrainingConfig(
        max_epochs=t["max_epochs"],
        batch_size=t["batch_size"],
        learning_rate=t["learning_rate"],
        joint_alpha=t.get("joint_alpha", 0.8),
        n_series=t.get("n_series", 500),
    )

    # Optimization
    o = raw["optimization"]
    optimization = OptimizationConfig(
        solver=o["solver"],
        epsilon=o["epsilon"],
        gamma=o["gamma"],
        holding_cost=o["holding_cost"],
        stockout_penalty=o["stockout_penalty"],
        dynamic_epsilon=o.get("dynamic_epsilon", True),
    )

    streaming = StreamingConfig(**raw["streaming"])

    # Disruption config (optional section)
    d = raw.get("disruption", {})
    disruption = DisruptionConfig(
        duration=d.get("duration", 4),
        shock_fraction=d.get("shock_fraction", 0.5),
        epsilon_disruption=d.get("epsilon_disruption", 0.5),
        target_recovery_reduction_pct=d.get("target_recovery_reduction_pct", 32.41),
    )

    return PipelineConfig(
        datasets=datasets,
        model=model,
        training=training,
        optimization=optimization,
        streaming=streaming,
        disruption=disruption,
    )


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load YAML config, parse into typed dataclass, and attach SHA-256 hash."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text())
    cfg = _parse_config(raw)
    cfg.config_hash = _compute_hash(path)
    return cfg
