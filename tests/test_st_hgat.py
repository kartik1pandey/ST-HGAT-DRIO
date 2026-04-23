"""Tests for STHGATModel.

Covers:
- Property 9: ST-HGAT output shape (Requirements 5.1, 5.2)
- Unit test: checkpoint round trip (Requirement 5.5)
- Unit test: metric logging (Requirement 5.3)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from src.model.st_hgat import STHGATModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cfg(
    d_hidden: int = 16,
    d_out: int = 16,
    horizon: int = 7,
    seq_len: int = 14,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Build a minimal config dict accepted by STHGATModel."""
    return {
        "model": {
            "d_hidden": d_hidden,
            "d_out": d_out,
            "horizon": horizon,
            "seq_len": seq_len,
        },
        "training": {
            "learning_rate": lr,
        },
    }


def _make_batch(
    num_nodes: int,
    seq_len: int = 14,
    input_dim: int = 1,
    horizon: int = 7,
) -> Dict[str, Any]:
    """Create a minimal batch dict for STHGATModel."""
    return {
        "x": torch.randn(num_nodes, seq_len, input_dim),
        "edge_index_dict": {},
        "y": torch.randn(num_nodes, horizon),
    }


# ---------------------------------------------------------------------------
# Property 9: ST-HGAT output shape
# Validates: Requirements 5.1, 5.2
# ---------------------------------------------------------------------------

# Feature: st-hgat-drio, Property 9: ST-HGAT output shape
@given(
    num_nodes=st.integers(min_value=1, max_value=30),
    horizon=st.integers(min_value=1, max_value=14),
    seq_len=st.integers(min_value=1, max_value=14),
    input_dim=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_st_hgat_output_shape(
    num_nodes: int,
    horizon: int,
    seq_len: int,
    input_dim: int,
) -> None:
    """**Validates: Requirements 5.1, 5.2**

    For any valid graph snapshot with N nodes and horizon H, the ST-HGAT model
    must produce output of shape [N, H].
    """
    cfg = _make_cfg(horizon=horizon, seq_len=seq_len)
    model = STHGATModel(cfg=cfg, input_dim=input_dim)
    model.eval()

    batch = _make_batch(num_nodes=num_nodes, seq_len=seq_len, input_dim=input_dim, horizon=horizon)

    with torch.no_grad():
        out = model(batch)

    assert out.shape == (num_nodes, horizon), (
        f"Expected output shape ({num_nodes}, {horizon}), got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# Unit test: checkpoint round trip (Requirement 5.5)
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip() -> None:
    """Save model checkpoint; reload via torch.load; assert weights and config hash present.

    Validates: Requirement 5.5
    """
    config_hash = "abc123def456" * 5 + "abcd"  # 64-char hex-like string
    cfg = _make_cfg()
    model = STHGATModel(cfg=cfg, input_dim=2, config_hash=config_hash)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model.ckpt"

        # Save via PyTorch Lightning trainer (1 epoch, no data needed for save)
        # Use torch.save directly to save state dict + hparams as Lightning does
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": model.hparams,
        }
        torch.save(checkpoint, ckpt_path)

        # Reload
        loaded = torch.load(ckpt_path, weights_only=False)

    # Assert weights are present
    assert "state_dict" in loaded, "Checkpoint must contain 'state_dict'"
    state_dict = loaded["state_dict"]
    assert len(state_dict) > 0, "state_dict must not be empty"

    # Assert config hash is in hparams
    assert "hyper_parameters" in loaded, "Checkpoint must contain 'hyper_parameters'"
    hparams = loaded["hyper_parameters"]
    assert "config_hash" in hparams, "hparams must contain 'config_hash'"
    assert hparams["config_hash"] == config_hash, (
        f"Expected config_hash={config_hash!r}, got {hparams['config_hash']!r}"
    )

    # Assert key weight tensors are present and match
    for key in ["gru_encoder.gru.weight_ih_l0", "head.0.weight"]:
        assert key in state_dict, f"Expected key '{key}' in state_dict"
        assert torch.equal(state_dict[key], model.state_dict()[key]), (
            f"Weight mismatch for '{key}' after round trip"
        )


# ---------------------------------------------------------------------------
# Unit test: metric logging (Requirement 5.3)
# ---------------------------------------------------------------------------

class _SimpleDataset(Dataset):
    """Minimal dataset that returns identical batches."""

    def __init__(self, batch: Dict[str, Any], length: int = 4) -> None:
        self._batch = batch
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._batch


def _collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack batch items — for our fixed-size tensors just return the first."""
    # All samples are identical; stack x and y along a new batch dim then squeeze
    # Since x is [N, seq_len, F] and y is [N, H], we just return the first sample
    return samples[0]


def test_metric_logging() -> None:
    """Run one training epoch; assert train_loss and val_mape appear in logged metrics.

    Validates: Requirement 5.3
    """
    cfg = _make_cfg(d_hidden=8, d_out=8, horizon=3, seq_len=4)
    model = STHGATModel(cfg=cfg, input_dim=1)

    batch = _make_batch(num_nodes=5, seq_len=4, input_dim=1, horizon=3)
    dataset = _SimpleDataset(batch, length=4)
    train_loader = DataLoader(dataset, batch_size=1, collate_fn=_collate)
    val_loader = DataLoader(dataset, batch_size=1, collate_fn=_collate)

    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=True,  # default TensorBoard logger captures metrics
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logged = trainer.logged_metrics
    assert "train_loss" in logged, (
        f"Expected 'train_loss' in logged metrics, got: {list(logged.keys())}"
    )
    assert "val_mape" in logged, (
        f"Expected 'val_mape' in logged metrics, got: {list(logged.keys())}"
    )
    assert logged["train_loss"] >= 0, "train_loss must be non-negative"
    assert logged["val_mape"] >= 0, "val_mape must be non-negative"
