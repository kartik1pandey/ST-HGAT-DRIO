"""Tests for src/evaluation/evaluator.py — Evaluator class.

Covers:
  - Property 10: Bullwhip Ratio Computation Correctness (Req 7.2)
  - Property 11: Cost Consistency Check Correctness (Req 7.4)
  - Property 12: Base-Stock Order Structure (Req 7.3)
  - Unit test: rationality check failure recording (Req 7.6)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.evaluator import Evaluator, RationalityCheckResult, TrajectoryRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

evaluator = Evaluator()


def make_trajectory(
    T: int = 20,
    base_stock_level: float = 50.0,
    holding_cost: float = 1.0,
    stockout_penalty: float = 5.0,
    sku_id: str = "SKU-001",
    lead_time: int = 0,
    demand: np.ndarray | None = None,
    orders: np.ndarray | None = None,
    inventory: np.ndarray | None = None,
) -> TrajectoryRecord:
    rng = np.random.default_rng(42)
    return TrajectoryRecord(
        sku_id=sku_id,
        lead_time=lead_time,
        demand=demand if demand is not None else rng.uniform(1, 10, T),
        orders=orders if orders is not None else rng.uniform(1, 10, T),
        inventory=inventory if inventory is not None else rng.uniform(-5, 20, T),
        base_stock_level=base_stock_level,
        holding_cost=holding_cost,
        stockout_penalty=stockout_penalty,
    )


# ---------------------------------------------------------------------------
# Property 10: Bullwhip Ratio Computation Correctness
# Feature: st-hgat-drio, Property 10: bullwhip ratio correctness
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    orders=st.lists(st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
    demand=st.lists(st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
)
def test_property_10_bullwhip_ratio_correctness(orders, demand):
    # Feature: st-hgat-drio, Property 10: bullwhip ratio correctness
    # Validates: Requirements 7.2
    o = np.array(orders)
    d = np.array(demand)

    # Skip degenerate cases where demand variance is zero
    if np.var(d) == 0.0:
        return

    result = evaluator.bullwhip_ratio(o, d)
    expected = np.var(o) / np.var(d)
    assert abs(result - expected) < 1e-9, (
        f"bullwhip_ratio returned {result}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Property 12: Base-Stock Order Structure
# Feature: st-hgat-drio, Property 12: base-stock order structure
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    base_stock=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    inv_positions=st.lists(
        st.floats(min_value=-50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    ),
    use_correct_formula=st.booleans(),
)
def test_property_12_base_stock_order_structure(base_stock, inv_positions, use_correct_formula):
    # Feature: st-hgat-drio, Property 12: base-stock order structure
    # Validates: Requirements 7.3
    inv = np.array(inv_positions)
    correct_orders = np.maximum(0.0, base_stock - inv)

    if use_correct_formula:
        orders = correct_orders
    else:
        # Perturb orders by at least 5% of base_stock + 10 to always exceed tolerance
        # Tolerance is max(0.01 * base_stock, 1.0), so perturbation must exceed that
        rng = np.random.default_rng(0)
        min_perturb = max(0.02 * base_stock, 2.0) + 5.0
        perturbation = rng.uniform(min_perturb, min_perturb + 10.0, size=len(inv))
        orders = correct_orders + perturbation

    traj = make_trajectory(
        T=len(inv),
        base_stock_level=base_stock,
        inventory=inv,
        orders=orders,
    )

    passed, _, _ = evaluator.check_base_stock(traj, orders)

    if use_correct_formula:
        assert passed, (
            f"Expected check to pass for correct base-stock orders "
            f"(base_stock={base_stock}, inv={inv_positions[:3]}...)"
        )
    else:
        assert not passed, (
            f"Expected check to fail for perturbed orders "
            f"(base_stock={base_stock}, inv={inv_positions[:3]}...)"
        )


# ---------------------------------------------------------------------------
# Property 11: Cost Consistency Check Correctness
# Feature: st-hgat-drio, Property 11: cost consistency correctness
# Validates: Requirements 7.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    inventory=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    ),
    holding_cost=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    stockout_penalty=st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_property_11_cost_consistency_correctness(inventory, holding_cost, stockout_penalty):
    # Feature: st-hgat-drio, Property 11: cost consistency correctness
    # Validates: Requirements 7.4
    inv = np.array(inventory)
    h = holding_cost
    p = stockout_penalty

    traj = make_trajectory(
        T=len(inv),
        inventory=inv,
        holding_cost=h,
        stockout_penalty=p,
    )

    passed, computed, expected = evaluator.check_cost_consistency(traj)

    # The check should always pass (computed == expected by construction)
    assert passed, "Cost consistency check should pass when cost is computed from the same formula"

    # Verify the computed value matches the formula directly
    direct = float(np.sum(h * np.maximum(0.0, inv) + p * np.maximum(0.0, -inv)))
    assert abs(computed - direct) < 1e-6, (
        f"Computed cost {computed} does not match direct formula {direct}"
    )


# ---------------------------------------------------------------------------
# Unit test: rationality check failure recording (Req 7.6)
# ---------------------------------------------------------------------------

def test_rationality_check_failure_recording():
    """Construct a trajectory that fails the bullwhip check; assert failure is
    recorded with correct fields and all five checks still run."""
    # Create orders with very high variance relative to demand → bullwhip > 1.5
    T = 30
    rng = np.random.default_rng(7)
    demand = np.ones(T) * 5.0  # constant demand → std=0, var=0
    # To get a finite bullwhip ratio > 1.5, demand must have non-zero variance
    demand = rng.uniform(4.9, 5.1, T)   # tiny variance
    orders = rng.uniform(0.0, 100.0, T)  # huge variance → bullwhip >> 1.5

    traj = make_trajectory(
        T=T,
        demand=demand,
        orders=orders,
        sku_id="FAIL-SKU",
    )

    results = evaluator.run_rationality_checks(traj, orders)

    # All five checks must have been run
    check_names = {r.check_name for r in results}
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    assert Evaluator.CHECK_BASE_STOCK in check_names
    assert Evaluator.CHECK_BULLWHIP in check_names
    assert Evaluator.CHECK_ALLOCATION in check_names
    assert Evaluator.CHECK_COST_CONSISTENCY in check_names
    assert Evaluator.CHECK_ORDER_SMOOTHING in check_names

    # Bullwhip check should have failed
    bullwhip_result = next(r for r in results if r.check_name == Evaluator.CHECK_BULLWHIP)
    assert not bullwhip_result.passed, "Bullwhip check should have failed"
    assert bullwhip_result.trajectory_id == "FAIL-SKU"
    assert bullwhip_result.computed_value > Evaluator.BULLWHIP_THRESHOLD
    assert bullwhip_result.expected_value == Evaluator.BULLWHIP_THRESHOLD

    # Verify RationalityCheckResult fields are populated correctly
    for r in results:
        assert isinstance(r, RationalityCheckResult)
        assert r.trajectory_id == "FAIL-SKU"
        assert isinstance(r.check_name, str)
        assert isinstance(r.passed, bool)
        assert isinstance(r.computed_value, float)
        # expected_value may be None (allocation placeholder)


def test_rationality_checks_all_pass_for_valid_trajectory():
    """A well-behaved trajectory should pass all checks."""
    T = 20
    base_stock = 50.0
    rng = np.random.default_rng(99)
    demand = rng.uniform(8.0, 12.0, T)
    inventory = rng.uniform(30.0, 50.0, T)  # all positive → no stockouts
    # Orders follow base-stock formula exactly
    orders = np.maximum(0.0, base_stock - inventory)

    traj = make_trajectory(
        T=T,
        demand=demand,
        orders=orders,
        inventory=inventory,
        base_stock_level=base_stock,
        holding_cost=1.0,
        stockout_penalty=5.0,
        sku_id="PASS-SKU",
    )

    results = evaluator.run_rationality_checks(traj, orders)
    assert len(results) == 5

    # Base-stock and cost consistency should pass
    bs = next(r for r in results if r.check_name == Evaluator.CHECK_BASE_STOCK)
    assert bs.passed

    cc = next(r for r in results if r.check_name == Evaluator.CHECK_COST_CONSISTENCY)
    assert cc.passed
