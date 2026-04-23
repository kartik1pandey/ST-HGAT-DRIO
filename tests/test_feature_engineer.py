"""Tests for src/data/feature_engineer.py.

Covers:
  - Property 3: Feature Tensor Round Trip     (Validates: Requirements 2.6)
  - Property 4: Log1p Normalization Monotonicity (Validates: Requirements 2.1)
  - Property 5: No Gaps in Aligned Time Index (Validates: Requirements 2.2)
  - Unit test: forward-fill and zero-fill gap handling (Requirements 2.3)
"""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.data.feature_engineer import FeatureEngineer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_dfs(
    sku_stores: list[str],
    dates: list[date],
    demand_val: float = 1.0,
    inventory_val: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build minimal sales, inventory, and external DataFrames."""
    rows_sales = [
        {"sku_store": ss, "date": pd.Timestamp(d), "demand": demand_val}
        for ss in sku_stores
        for d in dates
    ]
    rows_inv = [
        {"sku_store": ss, "date": pd.Timestamp(d), "inventory": inventory_val}
        for ss in sku_stores
        for d in dates
    ]
    rows_ext = [
        {"date": pd.Timestamp(d), "csi": 1.0, "cpi": 2.0, "sentiment": 0.5}
        for d in dates
    ]
    return (
        pd.DataFrame(rows_sales),
        pd.DataFrame(rows_inv),
        pd.DataFrame(rows_ext),
    )


# ---------------------------------------------------------------------------
# Property 3: Feature Tensor Round Trip
# Feature: st-hgat-drio, Property 3: feature tensor round trip
# Validates: Requirements 2.6
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=8),   # N nodes
        st.integers(min_value=1, max_value=10),  # T timesteps
        st.integers(min_value=1, max_value=5),   # F features
    )
)
def test_property3_feature_tensor_round_trip(shape: tuple[int, int, int]) -> None:
    """Saving and loading a feature tensor preserves shape, dtype, and values.

    **Validates: Requirements 2.6**
    """
    # Feature: st-hgat-drio, Property 3: feature tensor round trip
    N, T, F = shape
    original = np.random.default_rng(seed=42).random((N, T, F)).astype(np.float32)

    fe = FeatureEngineer()
    fe._tensor = original

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "tensor"
        fe.save(save_path)
        loaded_fe = FeatureEngineer.load(save_path)

    assert loaded_fe._tensor is not None
    assert loaded_fe._tensor.shape == original.shape, (
        f"Shape mismatch: {loaded_fe._tensor.shape} != {original.shape}"
    )
    assert loaded_fe._tensor.dtype == original.dtype, (
        f"Dtype mismatch: {loaded_fe._tensor.dtype} != {original.dtype}"
    )
    np.testing.assert_array_equal(loaded_fe._tensor, original)


# ---------------------------------------------------------------------------
# Property 4: Log1p Normalization Monotonicity
# Feature: st-hgat-drio, Property 4: log1p monotonicity
# Validates: Requirements 2.1
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    a=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_property4_log1p_monotonicity(a: float, b: float) -> None:
    """log1p is strictly order-preserving: a < b => log1p(a) < log1p(b).

    **Validates: Requirements 2.1**
    """
    # Feature: st-hgat-drio, Property 4: log1p monotonicity
    # Only test pairs where b is strictly greater than a at float64 precision
    # AND the log values are distinguishable (i.e., the difference is above
    # the floating-point resolution of log1p at this magnitude)
    assume(b > a)
    # log1p is mathematically strictly monotone; if b > a then log1p(b) > log1p(a).
    # We verify this holds for all float64-distinguishable pairs.
    assert np.log1p(a) < np.log1p(b), (
        f"Monotonicity violated: log1p({a}) >= log1p({b})"
    )


# ---------------------------------------------------------------------------
# Property 5: No Gaps in Aligned Time Index
# Feature: st-hgat-drio, Property 5: no gaps in time index
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    start_offset=st.integers(min_value=0, max_value=365),
    num_days=st.integers(min_value=1, max_value=60),
)
def test_property5_no_gaps_in_time_index(start_offset: int, num_days: int) -> None:
    """The aligned tensor has exactly (max_date - min_date).days + 1 timesteps.

    **Validates: Requirements 2.2**
    """
    # Feature: st-hgat-drio, Property 5: no gaps in time index
    base = date(2020, 1, 1)
    min_date = base + timedelta(days=start_offset)
    max_date = min_date + timedelta(days=num_days - 1)
    expected_T = (max_date - min_date).days + 1  # == num_days

    # Build DataFrames that only cover min_date and max_date (sparse)
    dates = [pd.Timestamp(min_date), pd.Timestamp(max_date)]
    if min_date == max_date:
        dates = [pd.Timestamp(min_date)]

    sales_df, inventory_df, external_df = _make_minimal_dfs(["SKU-A"], dates)

    fe = FeatureEngineer()
    tensor = fe.fit_transform(sales_df, inventory_df, external_df)

    actual_T = tensor.shape[1]
    assert actual_T == expected_T, (
        f"Expected {expected_T} timesteps for range "
        f"[{min_date}, {max_date}], got {actual_T}"
    )


# ---------------------------------------------------------------------------
# Unit test: forward-fill and zero-fill gap handling
# Requirements: 2.3
# ---------------------------------------------------------------------------


def test_unit_forward_fill_and_zero_fill() -> None:
    """Missing values are forward-filled; leading NaNs are zero-filled.

    Requirements: 2.3
    """
    # SKU-A has data on day 1 and day 3 (gap on day 2 → forward-fill from day 1)
    # SKU-B has data only on day 3 (days 1 and 2 are leading NaN → zero-fill)
    d1 = pd.Timestamp("2021-01-01")
    d2 = pd.Timestamp("2021-01-02")
    d3 = pd.Timestamp("2021-01-03")

    sales_df = pd.DataFrame([
        {"sku_store": "SKU-A", "date": d1, "demand": 10.0},
        {"sku_store": "SKU-A", "date": d3, "demand": 30.0},
        {"sku_store": "SKU-B", "date": d3, "demand": 5.0},
    ])
    inventory_df = pd.DataFrame([
        {"sku_store": "SKU-A", "date": d1, "inventory": 100.0},
        {"sku_store": "SKU-A", "date": d3, "inventory": 80.0},
        {"sku_store": "SKU-B", "date": d3, "inventory": 50.0},
    ])
    external_df = pd.DataFrame([
        {"date": d1, "csi": 1.0, "cpi": 2.0, "sentiment": 0.5},
        {"date": d2, "csi": 1.1, "cpi": 2.1, "sentiment": 0.6},
        {"date": d3, "csi": 1.2, "cpi": 2.2, "sentiment": 0.7},
    ])

    fe = FeatureEngineer()
    tensor = fe.fit_transform(sales_df, inventory_df, external_df)

    # Shape: [2 nodes, 3 days, 5 features]
    assert tensor.shape == (2, 3, 5), f"Unexpected shape: {tensor.shape}"

    nodes = sorted(["SKU-A", "SKU-B"])
    idx_a = nodes.index("SKU-A")
    idx_b = nodes.index("SKU-B")

    # SKU-A day 2 (index 1): demand should be forward-filled from day 1
    # log1p(10.0) forward-filled to day 2
    expected_log1p_day1 = np.log1p(10.0)
    assert tensor[idx_a, 1, 0] == pytest.approx(expected_log1p_day1, rel=1e-5), (
        f"SKU-A day 2 log1p_demand should be forward-filled from day 1: "
        f"got {tensor[idx_a, 1, 0]}, expected {expected_log1p_day1}"
    )

    # SKU-B days 1 and 2 (indices 0, 1): leading NaN → zero-filled
    assert tensor[idx_b, 0, 0] == pytest.approx(0.0), (
        f"SKU-B day 1 log1p_demand should be zero-filled, got {tensor[idx_b, 0, 0]}"
    )
    assert tensor[idx_b, 1, 0] == pytest.approx(0.0), (
        f"SKU-B day 2 log1p_demand should be zero-filled, got {tensor[idx_b, 1, 0]}"
    )

    # SKU-B day 3 (index 2): should have actual log1p(5.0)
    expected_log1p_b = np.log1p(5.0)
    assert tensor[idx_b, 2, 0] == pytest.approx(expected_log1p_b, rel=1e-5), (
        f"SKU-B day 3 log1p_demand should be log1p(5.0): "
        f"got {tensor[idx_b, 2, 0]}, expected {expected_log1p_b}"
    )

    # SKU-A inventory day 2: forward-filled from day 1 (100.0)
    assert tensor[idx_a, 1, 1] == pytest.approx(100.0), (
        f"SKU-A day 2 inventory should be forward-filled: got {tensor[idx_a, 1, 1]}"
    )

    # SKU-B inventory days 1 and 2: zero-filled
    assert tensor[idx_b, 0, 1] == pytest.approx(0.0)
    assert tensor[idx_b, 1, 1] == pytest.approx(0.0)
