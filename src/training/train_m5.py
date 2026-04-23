"""ST-HGAT with joint loss — used by run_submission.py."""
from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn
from torch import Tensor
from src.model.advanced_model import AdvancedSTHGATModel, _mape, _dro_cost


class STHGATWithJointLoss(AdvancedSTHGATModel):
    """Thin wrapper that exposes huber_delta and joint_alpha as constructor args."""

    def __init__(self, cfg: Any, input_dim: int = 1, config_hash: str = "",
                 num_heads: int = 4, dropout: float = 0.1,
                 huber_delta: float = 1.0, joint_alpha: float = 0.3,
                 **kwargs) -> None:
        for k in ["use_sgp", "use_hypergraph", "alpha", "alpha_disruption",
                  "d_hidden", "d_out", "horizon", "seq_len", "lr", "max_epochs",
                  "num_heads", "dropout", "h_cost", "p_cost"]:
            kwargs.pop(k, None)
        super().__init__(
            cfg=cfg, input_dim=input_dim, config_hash=config_hash,
            use_sgp=False, use_hypergraph=False,
            alpha=joint_alpha, alpha_disruption=0.4,
        )
        self.huber_delta = huber_delta
        # Expose lr for subclass configure_optimizers
        from src.model.advanced_model import _get
        self.lr = _get(cfg, "training", "learning_rate", default=3e-4)
        self.max_epochs = _get(cfg, "training", "max_epochs", default=50)

    def _dro_cost(self, y_hat: Tensor) -> Tensor:
        return torch.zeros(1, device=y_hat.device)
