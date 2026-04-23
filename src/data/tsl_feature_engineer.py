"""Fast M5 dataset builder for run_submission.py."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class M5WindowDataset(Dataset):
    """Sliding-window dataset over M5 series.

    Normalization: log1p only (no z-score).
    Training in log1p space gives the model the same scale as the LSTM baseline,
    allowing direct MAPE comparison without inversion artifacts.
    """

    def __init__(self, values: np.ndarray, seq_len: int = 14, horizon: int = 7,
                 stride: int = 28, split: str = "train") -> None:
        self.seq_len = seq_len
        self.horizon = horizon
        self.split = split

        # log1p transform only — no z-score
        log_vals = np.log1p(np.maximum(values.astype(float), 0)).astype(np.float32)
        self.vals = log_vals  # [N, T]  in log1p space

        # Store raw values for original-scale MAPE
        self.raw_vals = values.astype(np.float32)

        # Dummy stats (kept for API compatibility)
        self.series_mean = np.zeros(len(values), dtype=np.float32)
        self.series_std  = np.ones(len(values), dtype=np.float32)

        N, T = self.vals.shape
        self.window = seq_len + horizon
        self.index: List[Tuple[int, int]] = [
            (n, t) for n in range(N)
            for t in range(0, T - self.window + 1, stride)
        ]
        self._edge_index_dict: Dict[str, torch.Tensor] = {}
        self._input_dim: int = 1

    def set_edge_index_dict(self, d: Dict[str, torch.Tensor]) -> None:
        self._edge_index_dict = d

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n, t = self.index[idx]
        x_raw = self.vals[n, t:t+self.seq_len].copy()
        y_raw = self.vals[n, t+self.seq_len:t+self.window].copy()

        x_1d = torch.from_numpy(x_raw).unsqueeze(-1)
        input_dim = self._input_dim
        x = x_1d.expand(-1, input_dim).clone() if input_dim > 1 else x_1d
        y = torch.from_numpy(y_raw)

        return {
            "x": x, "y": y,
            "edge_index_dict": {},
            "is_disruption": False,
            "series_mean": 0.0,
            "series_std":  1.0,
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([b["x"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    s_mean = torch.tensor([b["series_mean"] for b in batch], dtype=torch.float32)
    s_std  = torch.tensor([b["series_std"]  for b in batch], dtype=torch.float32)
    return {
        "x": x, "y": y,
        "edge_index_dict": {},
        "is_disruption": any(b["is_disruption"] for b in batch),
        "series_mean": s_mean,
        "series_std":  s_std,
    }


def build_m5_dataset_fast(
    m5_dir: str | Path,
    n_series: Optional[int] = None,
    seq_len: int = 14,
    horizon: int = 7,
    stride: int = 28,
) -> M5WindowDataset:
    import pandas as pd
    m5_dir = Path(m5_dir)
    p = m5_dir / "sales_train_evaluation.csv"
    if not p.exists():
        p = m5_dir / "sales_train_validation.csv"
    df = pd.read_csv(p)
    day_cols = sorted([c for c in df.columns if c.startswith("d_")],
                      key=lambda c: int(c.split("_")[1]))
    vals = df[day_cols].to_numpy(dtype=float)
    if n_series is not None:
        vals = vals[:n_series]
    # Use all but last horizon days for training
    return M5WindowDataset(vals[:, :-horizon], seq_len=seq_len, horizon=horizon, stride=stride)


def make_dataloader(ds: Dataset, batch_size: int = 256, shuffle: bool = False,
                    num_workers: int = 0, input_dim: int = 5) -> DataLoader:
    # Set _input_dim on the underlying dataset (handles Subset wrappers)
    target = getattr(ds, 'dataset', ds)
    target._input_dim = input_dim
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_collate, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


class GraphAwareM5Dataset(Dataset):
    """Graph-aware dataset: each item is one time window over ALL 40 SKU nodes.

    Each batch item contains:
      - x: [n_sku, seq_len, 1]   log1p demand for all nodes
      - y: [n_sku, horizon]      log1p targets for all nodes
      - edge_index_dict: real graph edges (populated via set_edge_index_dict)

    This allows the HGAT to propagate signals across the full graph topology
    in every training step, enabling cross-SKU demand transfer learning.
    """

    def __init__(self, sku_values: np.ndarray, seq_len: int = 14, horizon: int = 7,
                 stride: int = 7) -> None:
        """
        Args:
            sku_values: [n_sku, T] raw demand values (will be log1p transformed)
            seq_len:    input window length
            horizon:    forecast horizon
            stride:     step between windows
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_sku = sku_values.shape[0]

        # log1p transform
        self.vals = np.log1p(np.maximum(sku_values.astype(float), 0)).astype(np.float32)
        T = self.vals.shape[1]
        self.window = seq_len + horizon

        self.starts: List[int] = list(range(0, T - self.window + 1, stride))
        self._edge_index_dict: Dict[str, torch.Tensor] = {}

    def set_edge_index_dict(self, d: Dict[str, torch.Tensor]) -> None:
        self._edge_index_dict = d

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t = self.starts[idx]
        x = torch.from_numpy(self.vals[:, t:t+self.seq_len]).unsqueeze(-1)  # [N, seq, 1]
        y = torch.from_numpy(self.vals[:, t+self.seq_len:t+self.window])    # [N, horizon]
        return {"x": x, "y": y, "edge_index_dict": self._edge_index_dict,
                "is_disruption": False, "series_mean": 0.0, "series_std": 1.0}


def graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate graph-aware batches — each item already has all N nodes."""
    # Stack along a new batch dimension: [B, N, seq, 1] and [B, N, H]
    x = torch.stack([b["x"] for b in batch])   # [B, N, seq, 1]
    y = torch.stack([b["y"] for b in batch])   # [B, N, H]
    # Reshape to [B*N, seq, 1] and [B*N, H] for the model
    B, N, S, F = x.shape
    x = x.view(B * N, S, F)
    y = y.view(B * N, y.shape[-1])
    return {"x": x, "y": y,
            "edge_index_dict": batch[0]["edge_index_dict"],
            "is_disruption": False,
            "series_mean": 0.0, "series_std": 1.0}


def make_graph_dataloader(ds: "GraphAwareM5Dataset", batch_size: int = 8,
                          shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """DataLoader for graph-aware dataset. batch_size = number of time windows per batch."""
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=graph_collate_fn, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())
