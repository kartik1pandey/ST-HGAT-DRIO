"""Advanced ST-HGAT-DRIO model implementing the full roadmap.

Architecture:
  - SGP encoder (Echo State Network + graph diffusion) for scalable O(1) inference
  - SC-RIHN tripartite hypergraph for disruption propagation
  - Joint loss: (1-alpha)*MAPE + alpha*DRO_cost
  - Dynamic alpha: increases to 0.4 during disruption windows
  - Holt-Winters warm-start for the readout head

Training stages:
  Stage 1 (pre-training): GRU encoder on full M5 dataset
  Stage 2 (disruption):   Joint loss with alpha=0.4 on factory-shock batches
  Stage 3 (QAT):          Quantization-aware training for INT8 export
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor

from src.model.sgp_encoder import SGPEncoder
from src.model.hypergraph import SCRIHNEncoder, build_incidence_matrices
from src.model.gru_encoder import GRUEncoder
from src.model.hgat import DualBranchHGAT

logger = logging.getLogger(__name__)


def _get(cfg: Any, *keys: str, default: Any = None) -> Any:
    obj = cfg
    for key in keys:
        try:
            obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
        except (KeyError, AttributeError):
            return default
    return obj


def _smape(y: Tensor, y_hat: Tensor) -> Tensor:
    eps = 1e-8
    return (2.0 * torch.abs(y - y_hat) / (torch.abs(y) + torch.abs(y_hat) + eps)).mean() * 100.0


def _mape(y: Tensor, y_hat: Tensor) -> Tensor:
    return (torch.abs(y - y_hat) / (torch.abs(y) + 1e-8)).mean() * 100.0


def _dro_cost(y: Tensor, y_hat: Tensor, h: float = 1.0, p: float = 5.0) -> Tensor:
    """Linearized DRO cost: h * max(y_hat - y, 0) + p * max(y - y_hat, 0).

    Normalized by mean(|y|) so it's on the same scale as MAPE (0-100 range).
    """
    over  = torch.clamp(y_hat - y, min=0.0)
    under = torch.clamp(y - y_hat, min=0.0)
    raw_cost = (h * over + p * under).mean()
    # Normalize to percentage scale: divide by mean absolute actual
    scale = (torch.abs(y) + 1e-8).mean()
    return raw_cost / scale * 100.0


class AdvancedSTHGATModel(pl.LightningModule):
    """Full advanced model: SGP + SC-RIHN + joint loss + disruption training.

    Args:
        cfg:            Config dict or object.
        input_dim:      Input feature dimension.
        config_hash:    SHA-256 of config file.
        use_sgp:        Use SGP encoder instead of GRU (default True).
        use_hypergraph: Use SC-RIHN hypergraph (default True).
        alpha:          Base joint loss weight for DRO cost (default 0.2).
        alpha_disruption: Alpha during disruption windows (default 0.4).
    """

    def __init__(
        self,
        cfg: Any,
        input_dim: int = 5,
        config_hash: str = "",
        use_sgp: bool = True,
        use_hypergraph: bool = True,
        alpha: float = 0.2,
        alpha_disruption: float = 0.4,
    ) -> None:
        super().__init__()

        d_hidden: int  = _get(cfg, "model", "d_hidden", default=64)
        d_out: int     = _get(cfg, "model", "d_out", default=64)
        horizon: int   = _get(cfg, "model", "horizon", default=7)
        seq_len: int   = _get(cfg, "model", "seq_len", default=14)
        lr: float      = _get(cfg, "training", "learning_rate", default=3e-4)
        max_epochs: int = _get(cfg, "training", "max_epochs", default=50)
        num_heads: int = _get(cfg, "model", "num_heads", default=4)
        dropout: float = _get(cfg, "model", "dropout", default=0.1)
        h_cost: float  = _get(cfg, "optimization", "holding_cost", default=1.0)
        p_cost: float  = _get(cfg, "optimization", "stockout_penalty", default=5.0)

        self.save_hyperparameters({
            "d_hidden": d_hidden, "d_out": d_out, "horizon": horizon,
            "seq_len": seq_len, "lr": lr, "max_epochs": max_epochs,
            "input_dim": input_dim, "num_heads": num_heads, "dropout": dropout,
            "use_sgp": use_sgp, "use_hypergraph": use_hypergraph,
            "alpha": alpha, "alpha_disruption": alpha_disruption,
            "config_hash": config_hash,
        })

        self.horizon = horizon
        self.lr = lr
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.alpha_disruption = alpha_disruption
        self.h_cost = h_cost
        self.p_cost = p_cost
        self.use_sgp = use_sgp
        self.use_hypergraph = use_hypergraph

        # ── Temporal encoder ─────────────────────────────────────────
        if use_sgp:
            self.encoder = SGPEncoder(
                input_dim=input_dim,
                reservoir_dim=max(d_hidden * 4, 256),
                d_out=d_hidden,
                K=3,
                dropout=dropout,
            )
        else:
            self.encoder = GRUEncoder(
                input_dim=input_dim,
                d_hidden=d_hidden,
                num_layers=2,
                dropout=dropout,
            )

        # ── Spatial encoder ──────────────────────────────────────────
        self.hgat = DualBranchHGAT(
            d_in=d_hidden, d_out=d_out, num_heads=num_heads, dropout=dropout
        )

        # ── Hypergraph encoder (SC-RIHN) ──────────────────────────────
        if use_hypergraph:
            self.hypergraph = SCRIHNEncoder(d_in=d_out, d_out=d_out, dropout=dropout)
        else:
            self.hypergraph = None

        # ── Forecast head ─────────────────────────────────────────────
        mid = max(d_out, horizon * 2)
        self.head = nn.Sequential(
            nn.Linear(d_out, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(mid // 2, horizon),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        x: Tensor = batch["x"]                          # [N, T, F]
        edge_index_dict = batch.get("edge_index_dict", {})

        # Temporal encoding
        if self.use_sgp:
            node_emb = self.encoder(x, edge_index_dict)  # [N, d_hidden]
        else:
            node_emb = self.encoder(x)                   # [N, d_hidden]

        # Spatial encoding
        hgat_out    = self.hgat({"sku": node_emb}, edge_index_dict)
        spatial_emb = hgat_out["sku"]                    # [N, d_out]

        # Hypergraph enrichment
        if self.hypergraph is not None and edge_index_dict:
            N = spatial_emb.shape[0]
            incidence = build_incidence_matrices(
                edge_index_dict, N, device=spatial_emb.device
            )
            if incidence:
                spatial_emb = self.hypergraph(spatial_emb, incidence)

        return self.head(spatial_emb)                    # [N, H]

    # ------------------------------------------------------------------
    # Joint loss
    # ------------------------------------------------------------------

    def _joint_loss(
        self, y: Tensor, y_hat: Tensor, is_disruption: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute joint loss = (1-alpha)*MAPE + alpha*DRO_cost.

        Both MAPE and DRO_cost are on the same 0-100 percentage scale.
        Returns (total_loss, mape, dro_cost).
        """
        alpha = self.alpha_disruption if is_disruption else self.alpha
        mape_val = _mape(y, y_hat)
        dro_val  = _dro_cost(y, y_hat, self.h_cost, self.p_cost)
        # Both terms are in percentage units — no division by 100 needed
        total = (1.0 - alpha) * mape_val + alpha * dro_val
        return total, mape_val, dro_val

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        y: Tensor = batch["y"]
        y_hat = self(batch)
        is_disruption = bool(batch.get("is_disruption", False))
        loss, mape_val, dro_val = self._joint_loss(y, y_hat, is_disruption)
        self.log("train_loss",    loss,     on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mape",    mape_val, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_dro",     dro_val,  on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        y: Tensor = batch["y"]
        y_hat = self(batch)
        smape_val = _smape(y, y_hat)
        mape_val  = _mape(y, y_hat)
        dro_val   = _dro_cost(y, y_hat, self.h_cost, self.p_cost)
        self.log("val_smape", smape_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mape",  mape_val,  on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_dro",   dro_val,   on_step=False, on_epoch=True, prog_bar=False)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-4, betas=(0.9, 0.95)
        )
        try:
            steps = self.trainer.estimated_stepping_batches
        except Exception:
            steps = self.max_epochs * 100
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lr * 10,
            total_steps=steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
