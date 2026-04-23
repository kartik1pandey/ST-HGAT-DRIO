"""Pipeline orchestrator for ST-HGAT-DRIO.

Reads a YAML config, seeds all RNGs, and executes pipeline stages in order:
  GraphBuilder → FeatureEngineer → STHGATModel training → DROModule → Evaluator

Halts and logs on any stage failure (Requirement 9.4).
Includes config hash in model checkpoint hparams (Requirement 9.3).
Deterministic initialization via seeded RNG (Requirement 9.5).

Supports injectable stage factories for testing.
"""

from __future__ import annotations

import logging
import random
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from src.pipeline.config import load_config, PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default stage factories
# ---------------------------------------------------------------------------

def _default_graph_builder_factory(cfg: PipelineConfig) -> Any:
    from src.graph.builder import GraphBuilder
    return GraphBuilder()


def _default_feature_engineer_factory(cfg: PipelineConfig) -> Any:
    from src.data.feature_engineer import FeatureEngineer
    return FeatureEngineer()


def _default_model_factory(cfg: PipelineConfig, config_hash: str) -> Any:
    from src.model.st_hgat import STHGATModel
    return STHGATModel(cfg=cfg, config_hash=config_hash)


def _default_dro_factory(cfg: PipelineConfig) -> Any:
    from src.optimization.dro import DROModule
    return DROModule(cfg=cfg)


def _default_evaluator_factory(cfg: PipelineConfig) -> Any:
    from src.evaluation.evaluator import Evaluator
    return Evaluator()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Orchestrates the full ST-HGAT-DRIO pipeline.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    stage_factories:
        Optional dict of injectable factory callables for testing.
        Keys: "graph_builder", "feature_engineer", "model", "dro", "evaluator".
        Each factory receives ``(cfg, ...)`` and returns the stage object.
    """

    def __init__(
        self,
        config_path: str | Path,
        stage_factories: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.cfg: PipelineConfig = load_config(self.config_path)

        # Seed all RNGs for deterministic initialization (Requirement 9.5)
        seed = self.cfg.model.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Stage factories — allow injection for testing
        factories = stage_factories or {}
        self._graph_builder_factory: Callable = factories.get(
            "graph_builder", _default_graph_builder_factory
        )
        self._feature_engineer_factory: Callable = factories.get(
            "feature_engineer", _default_feature_engineer_factory
        )
        self._model_factory: Callable = factories.get(
            "model", _default_model_factory
        )
        self._dro_factory: Callable = factories.get(
            "dro", _default_dro_factory
        )
        self._evaluator_factory: Callable = factories.get(
            "evaluator", _default_evaluator_factory
        )

        # Checkpoint metadata — populated during run()
        self.checkpoint_hparams: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all pipeline stages in dependency order.

        Halts and logs on any stage failure (Requirement 9.4).
        """
        cfg = self.cfg
        config_hash = cfg.config_hash

        stages = [
            ("GraphBuilder",      self._run_graph_builder),
            ("FeatureEngineer",   self._run_feature_engineer),
            ("STHGATModel",       lambda: self._run_model(config_hash)),
            ("DROModule",         self._run_dro),
            ("Evaluator",         self._run_evaluator),
        ]

        for stage_name, stage_fn in stages:
            logger.info("Starting pipeline stage: %s", stage_name)
            try:
                stage_fn()
                logger.info("Completed pipeline stage: %s", stage_name)
            except Exception:
                tb = traceback.format_exc()
                logger.error(
                    "Pipeline stage '%s' failed:\n%s", stage_name, tb
                )
                raise RuntimeError(
                    f"Pipeline halted at stage '{stage_name}'. "
                    f"See logs for details."
                ) from None

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_graph_builder(self) -> None:
        self._graph_builder = self._graph_builder_factory(self.cfg)

    def _run_feature_engineer(self) -> None:
        self._feature_engineer = self._feature_engineer_factory(self.cfg)

    def _run_model(self, config_hash: str) -> None:
        self._model = self._model_factory(self.cfg, config_hash)
        # Record checkpoint hparams including config hash (Requirement 9.3)
        self.checkpoint_hparams = {
            "config_hash": config_hash,
            "seed": self.cfg.model.seed,
            "d_hidden": self.cfg.model.d_hidden,
            "d_out": self.cfg.model.d_out,
            "horizon": self.cfg.model.horizon,
            "seq_len": self.cfg.model.seq_len,
        }

    def _run_dro(self) -> None:
        self._dro = self._dro_factory(self.cfg)

    def _run_evaluator(self) -> None:
        self._evaluator = self._evaluator_factory(self.cfg)
