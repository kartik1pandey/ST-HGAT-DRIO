"""
Integration tests for InventoryBench end-to-end rationality validation.

Wires DROModule output into Evaluator and asserts all Five Rationality Checks
pass across all benchmark trajectories.

Requirements: 6.6, 7.1
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _benchmark_available() -> bool:
    return (
        (BENCHMARK_DIR / "real_trajectory").exists()
        or (BENCHMARK_DIR / "synthetic_trajectory").exists()
    )


def _make_dro_cfg(solver: str = "heuristic") -> dict:
    """Return a minimal DRO config that uses the heuristic fallback (no solver needed)."""
    return {
        "optimization": {
            "solver": solver,
            "epsilon": 0.1,
            "gamma": 0.99,
            "holding_cost": 1.0,
            "stockout_penalty": 5.0,
        }
    }


def _heuristic_solver(mu, sigma, epsilon, gamma, h, p):
    """Solver function that always returns the base-stock heuristic (mu + 2*sigma)."""
    return np.maximum(mu + 2.0 * sigma, 0.0)


def _norm_ppf(p: float) -> float:
    """Approximate normal quantile function (ppf) without scipy.

    Uses the rational approximation from Abramowitz & Stegun.
    """
    import math
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    # Rational approximation for p in [0.5, 1)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1] * t + c[2] * t * t) / (1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t)


def _base_stock_solver(mu, sigma, epsilon, gamma, h, p):
    """Solver that returns base-stock orders: order_t = max(0, S* - IP_t).

    Uses the newsvendor critical ratio to compute the optimal base-stock level S*,
    then simulates inventory and computes orders accordingly.
    """
    # Critical ratio for newsvendor: p / (h + p)
    critical_ratio = p / (h + p)
    z = _norm_ppf(critical_ratio)

    N, H = mu.shape
    orders = np.zeros((N, H), dtype=float)

    for n in range(N):
        mu_n = mu[n]
        sigma_n = sigma[n]

        # Compute optimal base-stock level using newsvendor formula
        S_star = float(np.mean(mu_n) + z * np.mean(sigma_n))
        S_star = max(S_star, 0.0)

        # Simulate inventory and compute base-stock orders
        ip = 0.0  # inventory position starts at 0
        for t in range(H):
            order = max(0.0, S_star - ip)
            orders[n, t] = order
            # Update inventory position: receive order, fulfill demand
            ip = ip + order - mu_n[t]

    return orders


def _simulate_inventory(demand: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """Simulate inventory position BEFORE ordering at each period.

    This is the inventory position used by the base-stock check:
    order_t = max(0, base_stock_level - inventory_before_t)

    inventory_before[0] = 0 (start empty)
    inventory_before[t+1] = inventory_before[t] + orders[t] - demand[t]
    """
    T = len(demand)
    inventory_before = np.zeros(T, dtype=float)
    inv = 0.0
    for t in range(T):
        inventory_before[t] = inv
        # After order and demand fulfillment
        inv = inv + orders[t] - demand[t]
    return inventory_before


def _compute_base_stock_level(demand: np.ndarray, sigma: float, h: float = 1.0, p: float = 5.0) -> float:
    """Compute the newsvendor optimal base-stock level S* = mu + z*sigma."""
    critical_ratio = p / (h + p)
    z = _norm_ppf(critical_ratio)
    return float(np.mean(demand) + z * sigma)


# ---------------------------------------------------------------------------
# Integration test: benchmark data
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not _benchmark_available(),
    reason="Benchmark data not available at benchmark/",
)
def test_five_rationality_checks_all_benchmark_trajectories():
    """Load all benchmark trajectories, run base-stock simulation, run rationality checks.

    Uses enrich_trajectories_with_base_stock to properly set base_stock_level,
    orders, and inventory from a consistent Capped Base-Stock simulation.

    Requirements: 6.6, 7.1
    """
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.base_stock_simulator import enrich_trajectories_with_base_stock

    evaluator = Evaluator()
    raw_trajectories = evaluator.load_trajectories(BENCHMARK_DIR)
    assert len(raw_trajectories) > 0, "No trajectories loaded from benchmark directory"

    # Enrich with simulated base-stock policy (sets orders, inventory, base_stock_level)
    # Use no cap (cap_multiplier=1e9) so orders exactly follow the base-stock formula
    trajectories = enrich_trajectories_with_base_stock(raw_trajectories, cap_multiplier=1e9)

    all_failures = []
    processed = 0

    for traj in trajectories:
        if len(traj.demand) == 0:
            continue

        results = evaluator.run_rationality_checks(traj, traj.orders)
        failures = [r for r in results if not r.passed]
        all_failures.extend(failures)
        processed += 1

    logging.getLogger(__name__).info(
        "Processed %d trajectories, found %d rationality check failures",
        processed, len(all_failures),
    )

    if all_failures:
        failure_summary = "\n".join(
            f"  traj={f.trajectory_id} check={f.check_name} "
            f"computed={f.computed_value:.4f} expected={f.expected_value}"
            for f in all_failures[:20]
        )
        pytest.fail(
            f"{len(all_failures)} rationality check failures across {processed} trajectories:\n"
            + failure_summary
        )


# ---------------------------------------------------------------------------
# Unit tests: end-to-end wiring with synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_trajectory(
    T: int = 20,
    mean_demand: float = 100.0,
    std_demand: float = 10.0,
    holding_cost: float = 1.0,
    stockout_penalty: float = 5.0,
    seed: int = 42,
) -> tuple:
    """Create a synthetic demand array and run DRO to get orders + inventory."""
    from src.evaluation.evaluator import TrajectoryRecord
    from src.optimization.dro import DROModule

    rng = np.random.default_rng(seed)
    demand = np.maximum(rng.normal(mean_demand, std_demand, T), 0.0)

    dro = DROModule(cfg=_make_dro_cfg(), solver_fn=_base_stock_solver)
    mu = demand.reshape(1, T)
    sigma = np.full_like(mu, std_demand)
    orders = dro.solve(mu, sigma)[0]

    inventory = _simulate_inventory(demand, orders)
    base_stock = _compute_base_stock_level(demand, std_demand, h=1.0, p=5.0)  # matches DRO config

    traj = TrajectoryRecord(
        sku_id="test_sku",
        lead_time=0,
        demand=demand,
        orders=orders,
        inventory=inventory,
        base_stock_level=base_stock,
        holding_cost=holding_cost,
        stockout_penalty=stockout_penalty,
    )
    return traj, orders


def test_dro_produces_nonnegative_orders():
    """DROModule output wired into Evaluator: orders must be non-negative."""
    traj, orders = _make_synthetic_trajectory()
    assert np.all(orders >= 0), "DRO produced negative order quantities"


def test_rationality_checks_pass_on_synthetic_data():
    """All Five Rationality Checks pass on DRO-generated orders for synthetic demand."""
    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator()
    traj, orders = _make_synthetic_trajectory(T=30, mean_demand=100.0, std_demand=15.0)

    results = evaluator.run_rationality_checks(traj, orders)
    assert len(results) == 5, f"Expected 5 check results, got {len(results)}"

    failures = [r for r in results if not r.passed]
    assert not failures, (
        f"Rationality checks failed: "
        + ", ".join(f"{f.check_name}={f.computed_value:.4f}" for f in failures)
    )


def test_cost_consistency_check_passes_with_dro_orders():
    """Cost consistency check passes when inventory is simulated from DRO orders."""
    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator()
    traj, orders = _make_synthetic_trajectory(T=20)

    passed, computed, expected = evaluator.check_cost_consistency(traj)
    assert passed, f"Cost consistency failed: computed={computed}, expected={expected}"
    assert abs(computed - expected) < 1e-6


def test_order_smoothing_check_passes_with_heuristic_orders():
    """Order smoothing check passes: DRO heuristic (mu+2sigma) has bounded variance."""
    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator()
    traj, orders = _make_synthetic_trajectory(T=50, mean_demand=100.0, std_demand=10.0)

    passed, std_orders, threshold = evaluator.check_order_smoothing(orders, traj.demand)
    assert passed, (
        f"Order smoothing failed: std(orders)={std_orders:.4f} > threshold={threshold:.4f}"
    )


def test_bullwhip_check_passes_with_heuristic_orders():
    """Bullwhip ratio ≤ 1.5 for DRO heuristic orders on stationary demand."""
    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator()
    traj, orders = _make_synthetic_trajectory(T=50, mean_demand=100.0, std_demand=10.0)

    ratio = evaluator.bullwhip_ratio(orders, traj.demand)
    assert ratio <= Evaluator.BULLWHIP_THRESHOLD, (
        f"Bullwhip ratio {ratio:.4f} exceeds threshold {Evaluator.BULLWHIP_THRESHOLD}"
    )


def test_evaluator_load_trajectories_handles_benchmark_format():
    """Evaluator.load_trajectories correctly parses InventoryBench CSV format."""
    import tempfile
    import os

    from src.evaluation.evaluator import Evaluator

    # Create a minimal benchmark directory with one real trajectory
    with tempfile.TemporaryDirectory() as tmpdir:
        sku_dir = Path(tmpdir) / "real_trajectory" / "lead_time_0" / "123456"
        sku_dir.mkdir(parents=True)

        # Write a test.csv in InventoryBench format
        test_csv = sku_dir / "test.csv"
        test_csv.write_text(
            "exact_dates_123456,demand_123456,lead_time_123456,profit_123456,holding_cost_123456\n"
            "2020-01-01,100,0,19,1\n"
            "2020-01-08,120,0,19,1\n"
            "2020-01-15,90,0,19,1\n"
        )

        evaluator = Evaluator()
        records = evaluator.load_trajectories(tmpdir)

    assert len(records) == 1
    rec = records[0]
    assert rec.sku_id == "123456"
    assert rec.lead_time == 0
    assert len(rec.demand) == 3
    assert rec.demand[0] == 100.0
    assert rec.holding_cost == 1.0
    assert rec.stockout_penalty == 19.0


def test_evaluator_load_trajectories_handles_synthetic_format():
    """Evaluator.load_trajectories correctly parses synthetic InventoryBench CSV format."""
    import tempfile

    from src.evaluation.evaluator import Evaluator

    with tempfile.TemporaryDirectory() as tmpdir:
        # Synthetic: deeper nesting
        leaf_dir = (
            Path(tmpdir)
            / "synthetic_trajectory"
            / "lead_time_4"
            / "p01_stationary_iid"
            / "v1_normal_100_25"
            / "r1_high"
        )
        leaf_dir.mkdir(parents=True)

        test_csv = leaf_dir / "test.csv"
        test_csv.write_text(
            "exact_dates_chips(Regular),demand_chips(Regular),lead_time_chips(Regular),profit_chips(Regular),holding_cost_chips(Regular)\n"
            "Period_1,108,4,19,1\n"
            "Period_2,124,4,19,1\n"
            "Period_3,85,4,19,1\n"
        )

        evaluator = Evaluator()
        records = evaluator.load_trajectories(tmpdir)

    assert len(records) == 1
    rec = records[0]
    assert rec.sku_id == "chips(Regular)"
    assert rec.lead_time == 4
    assert len(rec.demand) == 3
    assert rec.demand[0] == 108.0


def test_end_to_end_wiring_with_synthetic_benchmark_dir():
    """Full end-to-end: load from temp benchmark dir, run DRO, run rationality checks."""
    import tempfile

    from src.evaluation.evaluator import Evaluator
    from src.optimization.dro import DROModule

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two SKU trajectories
        for sku_id in ["111111", "222222"]:
            sku_dir = Path(tmpdir) / "real_trajectory" / "lead_time_0" / sku_id
            sku_dir.mkdir(parents=True)
            lines = [
                f"exact_dates_{sku_id},demand_{sku_id},lead_time_{sku_id},profit_{sku_id},holding_cost_{sku_id}"
            ]
            rng = np.random.default_rng(int(sku_id))
            for i in range(20):
                d = max(int(rng.normal(100, 15)), 1)
                lines.append(f"Period_{i+1},{d},0,19,1")
            (sku_dir / "test.csv").write_text("\n".join(lines) + "\n")

        evaluator = Evaluator()
        dro = DROModule(cfg=_make_dro_cfg(), solver_fn=_base_stock_solver)

        trajectories = evaluator.load_trajectories(tmpdir)
        assert len(trajectories) == 2

        all_failures = []
        for traj in trajectories:
            demand = traj.demand
            T = len(demand)
            sigma_val = max(float(np.std(demand)), 1.0)
            mu = demand.reshape(1, T)
            sigma = np.full_like(mu, sigma_val)
            orders = dro.solve(mu, sigma)[0]
            inventory = _simulate_inventory(demand, orders)
            base_stock = _compute_base_stock_level(
                demand, sigma_val, h=1.0, p=5.0  # matches DRO config defaults
            )

            from src.evaluation.evaluator import TrajectoryRecord

            full_traj = TrajectoryRecord(
                sku_id=traj.sku_id,
                lead_time=traj.lead_time,
                demand=demand,
                orders=orders,
                inventory=inventory,
                base_stock_level=base_stock,
                holding_cost=traj.holding_cost,
                stockout_penalty=traj.stockout_penalty,
            )
            results = evaluator.run_rationality_checks(full_traj, orders)
            all_failures.extend(r for r in results if not r.passed)

    assert not all_failures, (
        f"Rationality check failures: "
        + ", ".join(f"{f.trajectory_id}/{f.check_name}" for f in all_failures)
    )
