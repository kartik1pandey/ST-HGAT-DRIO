"""
Tests for DROModule (src/optimization/dro.py).

Covers:
- Property 9: DRO Non-Negative Orders (Hypothesis, 100 examples)
- Unit test: solver fallback on infeasibility (μ + 2σ heuristic + warning)
- Unit test: RuntimeError when no solver is available
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.optimization.dro import DROModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(solver: str = "gurobi") -> dict:
    return {
        "optimization": {
            "solver": solver,
            "epsilon": 0.1,
            "gamma": 0.99,
            "holding_cost": 1.0,
            "stockout_penalty": 5.0,
        }
    }


def _mock_solver_ok(mu, sigma, epsilon, gamma, h, p):
    """Solver that always returns a valid (non-negative) solution."""
    return np.maximum(mu + sigma, 0.0)


def _mock_solver_infeasible(mu, sigma, epsilon, gamma, h, p):
    """Solver that signals infeasibility by returning None."""
    return None


# ---------------------------------------------------------------------------
# Property 9: DRO Non-Negative Orders
# Feature: st-hgat-drio, Property 9: DRO non-negative orders
# Validates: Requirements 6.4
# ---------------------------------------------------------------------------

@given(
    n_nodes=st.integers(min_value=1, max_value=20),
    horizon=st.integers(min_value=1, max_value=10),
    mu_vals=st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=200,
    ),
    sigma_vals=st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=200,
    ),
)
@settings(max_examples=100)
def test_dro_non_negative_orders(n_nodes, horizon, mu_vals, sigma_vals):
    # Feature: st-hgat-drio, Property 9: DRO non-negative orders
    # Validates: Requirements 6.4
    #
    # Build [N, H] arrays by tiling/truncating the generated lists.
    size = n_nodes * horizon
    mu_flat = np.array((mu_vals * (size // len(mu_vals) + 1))[:size], dtype=float)
    sigma_flat = np.array((sigma_vals * (size // len(sigma_vals) + 1))[:size], dtype=float)
    mu = mu_flat.reshape(n_nodes, horizon)
    sigma = sigma_flat.reshape(n_nodes, horizon)

    dro = DROModule(_make_cfg(), solver_fn=_mock_solver_ok)
    orders = dro.solve(mu, sigma)

    assert orders.shape == (n_nodes, horizon), (
        f"Expected shape ({n_nodes}, {horizon}), got {orders.shape}"
    )
    assert np.all(orders >= 0), (
        f"Found negative order quantities: {orders[orders < 0]}"
    )


# ---------------------------------------------------------------------------
# Unit test 9.2: solver fallback on infeasibility → μ + 2σ + warning logged
# Validates: Requirements 6.5
# ---------------------------------------------------------------------------

def test_solver_fallback_on_infeasibility(caplog):
    """When solver returns None (infeasible), DROModule falls back to mu + 2*effective_sigma."""
    mu = np.array([[10.0, 20.0], [5.0, 15.0]])
    sigma = np.array([[1.0, 2.0], [0.5, 1.5]])

    dro = DROModule(_make_cfg(), solver_fn=_mock_solver_infeasible)

    with caplog.at_level(logging.WARNING, logger="src.optimization.dro"):
        orders = dro.solve(mu, sigma)

    # Orders must be >= mu (safety buffer added) and non-negative
    assert np.all(orders >= mu), "Fallback orders must be >= forecast mean"
    assert np.all(orders >= 0), "Fallback orders must be non-negative"

    # A warning must have been logged
    assert any("heuristic" in record.message.lower() or "fallback" in record.message.lower()
               for record in caplog.records), (
        "Expected a warning about fallback to heuristic, but none was logged."
    )


def test_solver_fallback_orders_non_negative(caplog):
    """Fallback heuristic must also produce non-negative orders."""
    # Even with zero mu and sigma, result should be >= 0
    mu = np.zeros((3, 4))
    sigma = np.zeros((3, 4))

    dro = DROModule(_make_cfg(), solver_fn=_mock_solver_infeasible)
    with caplog.at_level(logging.WARNING):
        orders = dro.solve(mu, sigma)

    assert np.all(orders >= 0)


# ---------------------------------------------------------------------------
# Unit test 9.3: RuntimeError when no solver is available
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------

def test_runtime_error_when_gurobi_unavailable():
    """RuntimeError raised when gurobi is configured but not installed."""
    mu = np.array([[5.0, 10.0]])
    sigma = np.array([[1.0, 2.0]])

    dro = DROModule(_make_cfg(solver="gurobi"))  # no solver_fn injected

    with patch.dict(sys.modules, {"gurobipy": None}):
        with pytest.raises(RuntimeError, match="[Gg]urobi"):
            dro.solve(mu, sigma)


def test_runtime_error_when_cplex_unavailable():
    """RuntimeError raised when cplex is configured but not installed."""
    mu = np.array([[5.0, 10.0]])
    sigma = np.array([[1.0, 2.0]])

    dro = DROModule(_make_cfg(solver="cplex"))  # no solver_fn injected

    with patch.dict(sys.modules, {"cplex": None}):
        with pytest.raises(RuntimeError, match="[Cc][Pp][Ll][Ee][Xx]"):
            dro.solve(mu, sigma)


def test_runtime_error_both_solvers_unavailable():
    """RuntimeError raised when both gurobi and cplex are unavailable."""
    mu = np.array([[5.0]])
    sigma = np.array([[1.0]])

    # Test gurobi path
    dro_g = DROModule(_make_cfg(solver="gurobi"))
    with patch.dict(sys.modules, {"gurobipy": None}):
        with pytest.raises(RuntimeError):
            dro_g.solve(mu, sigma)

    # Test cplex path
    dro_c = DROModule(_make_cfg(solver="cplex"))
    with patch.dict(sys.modules, {"cplex": None}):
        with pytest.raises(RuntimeError):
            dro_c.solve(mu, sigma)
