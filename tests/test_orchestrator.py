"""Tests for PipelineOrchestrator.

Covers:
  - Property 14: Pipeline Determinism (Requirements 9.5)
  - Property 15: Config hash in checkpoint (Requirements 9.3)
  - Unit test: stage failure halts pipeline and logs stage name + traceback (Requirements 9.4)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.pipeline.orchestrator import PipelineOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/default.yaml")


def _make_noop_factories() -> dict:
    """Return stage factories that do nothing (no real I/O or training)."""
    return {
        "graph_builder": lambda cfg: MagicMock(),
        "feature_engineer": lambda cfg: MagicMock(),
        "model": lambda cfg, config_hash: MagicMock(),
        "dro": lambda cfg: MagicMock(),
        "evaluator": lambda cfg: MagicMock(),
    }


def _make_orchestrator(config_path: Path, factories: dict | None = None) -> PipelineOrchestrator:
    return PipelineOrchestrator(
        config_path=config_path,
        stage_factories=factories or _make_noop_factories(),
    )


# ---------------------------------------------------------------------------
# Property 14: Pipeline Determinism
# Feature: st-hgat-drio, Property 14: pipeline determinism
# Validates: Requirements 9.5
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_pipeline_determinism(seed: int, tmp_path: Path) -> None:
    """Run orchestrator twice with same config and seed; assert checkpoint configs are identical.

    # Feature: st-hgat-drio, Property 14: pipeline determinism
    """
    # Write a config with the given seed
    base_cfg = yaml.safe_load(CONFIG_PATH.read_text())
    base_cfg["model"]["seed"] = seed
    cfg_file = tmp_path / f"cfg_{seed}.yaml"
    cfg_file.write_text(yaml.dump(base_cfg), encoding="utf-8")

    # First run
    orch1 = _make_orchestrator(cfg_file)
    orch1.run()
    hparams1 = dict(orch1.checkpoint_hparams)

    # Second run — same config, same seed
    orch2 = _make_orchestrator(cfg_file)
    orch2.run()
    hparams2 = dict(orch2.checkpoint_hparams)

    # Architecture and hyperparameter configuration must be identical
    assert hparams1 == hparams2, (
        f"Checkpoint hparams differ between runs:\n  run1={hparams1}\n  run2={hparams2}"
    )


# ---------------------------------------------------------------------------
# Property 15: Config hash in checkpoint
# Feature: st-hgat-drio, Property 15: config hash in checkpoint
# Validates: Requirements 9.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    extra_comment=st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters=" _-",
            max_codepoint=127,  # ASCII only — avoids encoding issues on Windows
        ),
        min_size=0,
        max_size=50,
    )
)
def test_config_hash_in_checkpoint(extra_comment: str, tmp_path: Path) -> None:
    """For any config file, checkpoint metadata contains its SHA-256 hash.

    # Feature: st-hgat-drio, Property 15: config hash in checkpoint
    """
    # Create a unique config file by appending a comment
    base_text = CONFIG_PATH.read_text(encoding="utf-8")
    cfg_text = base_text + f"\n# {extra_comment}\n"
    cfg_file = tmp_path / "cfg_variant.yaml"
    cfg_file.write_text(cfg_text, encoding="utf-8")

    # Compute expected hash
    expected_hash = hashlib.sha256(cfg_file.read_bytes()).hexdigest()

    orch = _make_orchestrator(cfg_file)
    orch.run()

    assert "config_hash" in orch.checkpoint_hparams, (
        "checkpoint_hparams must contain 'config_hash'"
    )
    assert orch.checkpoint_hparams["config_hash"] == expected_hash, (
        f"Expected hash {expected_hash!r}, got {orch.checkpoint_hparams['config_hash']!r}"
    )


# ---------------------------------------------------------------------------
# Unit test: stage failure halts pipeline and logs stage name + traceback
# Requirements: 9.4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failing_stage", [
    "graph_builder",
    "feature_engineer",
    "model",
    "dro",
    "evaluator",
])
def test_stage_failure_halts_pipeline(failing_stage: str, caplog: pytest.LogCaptureFixture) -> None:
    """Mock one stage to raise; assert orchestrator halts and logs stage name + traceback."""
    error_msg = f"Simulated failure in {failing_stage}"

    def _raise(cfg: Any, *args: Any, **kwargs: Any) -> Any:
        raise ValueError(error_msg)

    factories = _make_noop_factories()
    factories[failing_stage] = _raise

    orch = PipelineOrchestrator(config_path=CONFIG_PATH, stage_factories=factories)

    with caplog.at_level(logging.ERROR, logger="src.pipeline.orchestrator"):
        with pytest.raises(RuntimeError) as exc_info:
            orch.run()

    # Pipeline must halt with RuntimeError
    assert "halted" in str(exc_info.value).lower() or "stage" in str(exc_info.value).lower()

    # Log must contain the stage name and the original traceback
    stage_name_map = {
        "graph_builder": "GraphBuilder",
        "feature_engineer": "FeatureEngineer",
        "model": "STHGATModel",
        "dro": "DROModule",
        "evaluator": "Evaluator",
    }
    expected_stage_name = stage_name_map[failing_stage]

    log_text = "\n".join(caplog.messages)
    assert expected_stage_name in log_text, (
        f"Expected stage name '{expected_stage_name}' in log output:\n{log_text}"
    )
    # Traceback should appear (ValueError is logged)
    assert "ValueError" in log_text or error_msg in log_text, (
        f"Expected error details in log output:\n{log_text}"
    )
