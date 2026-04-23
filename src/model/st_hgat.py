"""ST-HGAT joint spatiotemporal model with PyTorch Lightning training loop.

Phase 3 additions:
  - Optional SC-RIHN hypergraph encoder for resilience
  - Dynamic alpha (increases during disruption window)
  - INT8 quantization support via torch.ao.quantization
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor

from src.model.gru_encoder import GRUEncoder
from src.model.hgat import DualBranchHGAT
from src.model.hypergraph import SCRIHNEncoder, build_incidence_matrices


def _get(cfg: Any, *keys: str) -> Any:
    obj = cfg
    for key in keys:
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            obj = getattr(obj, key)
    return obj


def _smape(y: Tensor, y_hat: Tensor) -> Tensor:
    """Symmetric MAPE — handles near-zero actuals gracefully."""
    eps = 1e-8
    return (2.0 * torch.abs(y - y_hat) / (torch.abs(y) + torch.abs(y_hat) + eps)).mean() * 100.0


class STHGATModel(pl.LightningModule):
    """Joint spatiotemporal model: GRU encoder + dual-branch HGAT + MLP head.

    Args:
        cfg:          Config dict or object.
        input_dim:    Number of input features per timestep per node.
        config_hash:  SHA-256 hash of the config file.
        num_heads:    Number of HGAT attention heads (default 4).
        dropout:      Dropout probability (default 0.1).
        huber_delta:  Delta for Huber loss (default 1.0).
    """

    def __init__(
        self,
        cfg: Any,
        input_dim: int = 1,
        config_hash: str = "",
        num_heads: int = 4,
        dropout: float = 0.1,
        huber_delta: float = 1.0,
        use_hypergraph: bool = False,
    ) -> None:
        super().__init__()

        d_hidden: int = _get(cfg, "model", "d_hidden")
        d_out: int    = _get(cfg, "model", "d_out")
        horizon: int  = _get(cfg, "model", "horizon")
        seq_len: int  = _get(cfg, "model", "seq_len")
        lr: float     = _get(cfg, "training", "learning_rate")
        try:
            max_epochs: int = _get(cfg, "training", "max_epochs")
        except (KeyError, AttributeError):
            max_epochs = 50

        self.save_hyperparameters({
            "d_hidden": d_hidden, "d_out": d_out, "horizon": horizon,
            "seq_len": seq_len, "learning_rate": lr, "max_epochs": max_epochs,
            "input_dim": input_dim, "num_heads": num_heads, "dropout": dropout,
            "huber_delta": huber_delta, "use_hypergraph": use_hypergraph,
            "config_hash": config_hash,
        })

        self.horizon = horizon
        self.lr = lr
        self.max_epochs = max_epochs
        self.huber_delta = huber_delta
        self.use_hypergraph = use_hypergraph

        self.gru_encoder = GRUEncoder(input_dim=input_dim, d_hidden=d_hidden,
                                       num_layers=2, dropout=dropout)
        self.hgat = DualBranchHGAT(d_in=d_hidden, d_out=d_out,
                                    num_heads=num_heads, dropout=dropout)

        # Optional SC-RIHN hypergraph encoder (Phase 3)
        if use_hypergraph:
            self.hypergraph = SCRIHNEncoder(d_in=d_out, d_out=d_out, dropout=dropout)
        else:
            self.hypergraph = None

        mid = max(d_out // 2, horizon)
        self.head = nn.Sequential(
            nn.Linear(d_out, mid), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid, horizon),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        x: Tensor = batch["x"]
        edge_index_dict: Dict[str, Tensor] = batch["edge_index_dict"]

        node_emb    = self.gru_encoder(x)                        # [N, d_hidden]
        hgat_out    = self.hgat({"sku": node_emb}, edge_index_dict)
        spatial_emb = hgat_out["sku"]                            # [N, d_out]

        # Optional hypergraph enrichment (Phase 3 — SC-RIHN)
        if self.hypergraph is not None and edge_index_dict:
            N = spatial_emb.shape[0]
            incidence = build_incidence_matrices(
                edge_index_dict, N, device=spatial_emb.device
            )
            if incidence:
                spatial_emb = self.hypergraph(spatial_emb, incidence)

        return self.head(spatial_emb)                            # [N, H]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        y: Tensor = batch["y"]
        y_hat = self(batch)
        loss = nn.functional.huber_loss(y_hat, y, delta=self.huber_delta)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        y: Tensor = batch["y"]
        y_hat = self(batch)
        smape = _smape(y, y_hat)
        mape  = (torch.abs(y - y_hat) / (torch.abs(y) + 1e-8)).mean() * 100.0
        self.log("val_smape", smape, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mape",  mape,  on_step=False, on_epoch=True, prog_bar=False)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # OneCycleLR needs steps_per_epoch; fall back to a fixed total if unknown
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
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
