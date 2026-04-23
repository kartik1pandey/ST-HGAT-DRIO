"""Feature engineering and data alignment for ST-HGAT-DRIO.

Implements FeatureEngineer which:
  - Applies log1p normalization to demand values (Requirement 2.1)
  - Builds a gapless daily SKU-Store time index (Requirement 2.2)
  - Forward-fills then zero-fills missing values (Requirement 2.3)
  - Joins external signals (CSI, CPI, sentiment) by date (Requirement 2.4)
  - Produces a feature tensor of shape [N, T, F] (Requirement 2.5)
  - Serializes/deserializes via numpy.savez for round-trip fidelity (Requirement 2.6)
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Aligns multi-source supply chain data into a [N, T, F] feature tensor.

    Features (F=5):
        0: log1p demand
        1: inventory level
        2: CSI
        3: CPI
        4: sentiment score
    """

    FEATURES: ClassVar[list[str]] = ["log1p_demand", "inventory", "csi", "cpi", "sentiment"]

    def __init__(self) -> None:
        self._tensor: np.ndarray | None = None
        self._nodes: list[str] | None = None
        self._dates: pd.DatetimeIndex | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        sales_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        external_df: pd.DataFrame,
    ) -> np.ndarray:
        """Align and engineer features from raw DataFrames.

        Parameters
        ----------
        sales_df:
            Columns: sku_store (str), date (datetime), demand (float)
        inventory_df:
            Columns: sku_store (str), date (datetime), inventory (float)
        external_df:
            Columns: date (datetime), csi (float), cpi (float), sentiment (float)

        Returns
        -------
        np.ndarray of shape [N, T, F] with dtype float32
        """
        # --- Normalise dates to midnight (no time component) ---
        sales_df = sales_df.copy()
        inventory_df = inventory_df.copy()
        external_df = external_df.copy()

        sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.normalize()
        inventory_df["date"] = pd.to_datetime(inventory_df["date"]).dt.normalize()
        external_df["date"] = pd.to_datetime(external_df["date"]).dt.normalize()

        # --- Apply log1p to demand (Requirement 2.1) ---
        sales_df["log1p_demand"] = np.log1p(sales_df["demand"].astype(float))

        # --- Build gapless daily time index (Requirement 2.2) ---
        all_dates = pd.concat([sales_df["date"], inventory_df["date"]])
        min_date = all_dates.min()
        max_date = all_dates.max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        # All unique SKU-Store nodes
        all_nodes = sorted(
            set(sales_df["sku_store"].unique()) | set(inventory_df["sku_store"].unique())
        )

        # Build a complete multi-index: every (sku_store, date) combination
        full_index = pd.MultiIndex.from_product(
            [all_nodes, full_date_range], names=["sku_store", "date"]
        )
        full_df = pd.DataFrame(index=full_index).reset_index()

        # --- Merge sales (log1p demand) ---
        sales_pivot = sales_df[["sku_store", "date", "log1p_demand"]].drop_duplicates(
            subset=["sku_store", "date"]
        )
        full_df = full_df.merge(sales_pivot, on=["sku_store", "date"], how="left")

        # --- Merge inventory ---
        inv_pivot = inventory_df[["sku_store", "date", "inventory"]].drop_duplicates(
            subset=["sku_store", "date"]
        )
        full_df = full_df.merge(inv_pivot, on=["sku_store", "date"], how="left")

        # --- Forward-fill then zero-fill per SKU-Store (Requirement 2.3) ---
        full_df = full_df.sort_values(["sku_store", "date"])
        full_df[["log1p_demand", "inventory"]] = (
            full_df.groupby("sku_store")[["log1p_demand", "inventory"]]
            .transform(lambda s: s.ffill().fillna(0.0))
        )

        # --- Join external signals by date (Requirement 2.4) ---
        ext_dedup = external_df.drop_duplicates(subset=["date"])
        full_df = full_df.merge(ext_dedup[["date", "csi", "cpi", "sentiment"]], on="date", how="left")
        # Fill any missing external values with 0
        full_df[["csi", "cpi", "sentiment"]] = full_df[["csi", "cpi", "sentiment"]].fillna(0.0)

        # --- Reshape to [N, T, F] (Requirement 2.5) ---
        N = len(all_nodes)
        T = len(full_date_range)
        F = len(self.FEATURES)

        # Pivot to (N, T) for each feature
        full_df = full_df.sort_values(["sku_store", "date"])
        tensor = np.zeros((N, T, F), dtype=np.float32)

        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        date_to_idx = {d: i for i, d in enumerate(full_date_range)}

        for _, row in full_df.iterrows():
            n = node_to_idx[row["sku_store"]]
            t = date_to_idx[row["date"]]
            tensor[n, t, 0] = row["log1p_demand"]
            tensor[n, t, 1] = row["inventory"]
            tensor[n, t, 2] = row["csi"]
            tensor[n, t, 3] = row["cpi"]
            tensor[n, t, 4] = row["sentiment"]

        self._tensor = tensor
        self._nodes = all_nodes
        self._dates = full_date_range

        return tensor

    def save(self, path: Path | str) -> None:
        """Serialize the feature tensor to disk using numpy.savez.

        Preserves dtype and shape for round-trip fidelity (Requirement 2.6).
        """
        if self._tensor is None:
            raise RuntimeError("No tensor to save — call fit_transform first.")
        path = Path(path)
        np.savez(path, tensor=self._tensor)

    @classmethod
    def load(cls, path: Path | str) -> "FeatureEngineer":
        """Deserialize a FeatureEngineer from a .npz file.

        Returns a new FeatureEngineer instance with the loaded tensor.
        """
        path = Path(path)
        # numpy.savez appends .npz if not present
        npz_path = path if path.suffix == ".npz" else Path(str(path) + ".npz")
        data = np.load(npz_path)
        fe = cls()
        fe._tensor = data["tensor"]
        return fe
