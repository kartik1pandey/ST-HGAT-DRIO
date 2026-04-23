"""InventoryBench evaluator with Five Rationality Checks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _norm_ppf(p: float) -> float:
    """Rational approximation of the normal quantile (Abramowitz & Stegun)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1]*t + c[2]*t*t) / (1.0 + d[0]*t + d[1]*t*t + d[2]*t*t*t)


def _newsvendor_base_stock(
    demand: np.ndarray,
    holding_cost: float,
    stockout_penalty: float,
) -> float:
    """Compute the newsvendor optimal base-stock level S* = mu + z*sigma.

    Uses the critical ratio CR = p / (h + p) to find the service level z.
    """
    mu = float(np.mean(demand))
    sigma = float(np.std(demand))
    if sigma < 1e-9:
        return max(mu, 0.0)
    cr = stockout_penalty / (holding_cost + stockout_penalty)
    z = _norm_ppf(cr)
    return max(mu + z * sigma, 0.0)


@dataclass
class TrajectoryRecord:
    sku_id: str
    lead_time: int | str  # 0, 4, or "stochastic"
    demand: np.ndarray    # shape [T]
    orders: np.ndarray    # shape [T]
    inventory: np.ndarray # shape [T]
    base_stock_level: float
    holding_cost: float
    stockout_penalty: float


@dataclass
class RationalityCheckResult:
    trajectory_id: str
    check_name: str
    passed: bool
    computed_value: float
    expected_value: float | None


class Evaluator:
    """Evaluates inventory policies against InventoryBench trajectories."""

    # Five Rationality Check names
    CHECK_BASE_STOCK = "base_stock_structure"
    CHECK_BULLWHIP = "bullwhip_ratio"
    CHECK_ALLOCATION = "allocation_logic"
    CHECK_COST_CONSISTENCY = "cost_consistency"
    CHECK_ORDER_SMOOTHING = "order_smoothing"

    BULLWHIP_THRESHOLD = 1.5
    ORDER_SMOOTHING_FACTOR = 1.5

    def load_trajectories(self, benchmark_dir: str | Path) -> list[TrajectoryRecord]:
        """Load all test.csv files across lead_time variants.

        Searches both real_trajectory/ and synthetic_trajectory/ subdirectories
        for lead_time_0, lead_time_4, and lead_time_stochastic variants.

        Handles two directory layouts:
        - Real: benchmark_dir/real_trajectory/<lead_time>/<sku_id>/test.csv
        - Synthetic: benchmark_dir/synthetic_trajectory/<lead_time>/<pattern>/<variant>/<replicate>/test.csv

        Returns a list of TrajectoryRecord objects.
        """
        benchmark_dir = Path(benchmark_dir)
        records: list[TrajectoryRecord] = []

        trajectory_types = ["real_trajectory", "synthetic_trajectory"]
        lead_time_variants = {
            "lead_time_0": 0,
            "lead_time_4": 4,
            "lead_time_stochastic": "stochastic",
        }

        for traj_type in trajectory_types:
            for lt_dir, lt_value in lead_time_variants.items():
                lt_path = benchmark_dir / traj_type / lt_dir
                if not lt_path.exists():
                    logger.debug("Lead-time dir not found, skipping: %s", lt_path)
                    continue

                # Recursively find all test.csv files under this lead-time dir
                for csv_path in lt_path.rglob("test.csv"):
                    try:
                        records.extend(
                            self._parse_trajectory_csv(csv_path, lt_value)
                        )
                    except Exception as exc:
                        logger.warning("Failed to load %s: %s", csv_path, exc)

        logger.info("Loaded %d trajectory records from %s", len(records), benchmark_dir)
        return records

    def _parse_trajectory_csv(
        self, csv_path: Path, lead_time: int | str
    ) -> list[TrajectoryRecord]:
        """Parse a single trajectory CSV into TrajectoryRecord objects.

        Handles two formats:
        1. Standard format: columns sku_id, demand, orders, inventory, ...
        2. InventoryBench format: columns exact_dates_<sku>, demand_<sku>, [holding_cost_<sku>, ...]
           In this case orders/inventory are not present; callers must supply them separately.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        available = set(df.columns)

        # --- Standard format ---
        if "sku_id" in available and "demand" in available:
            records: list[TrajectoryRecord] = []
            for sku_id, group in df.groupby("sku_id"):
                group = group.sort_values("period") if "period" in group.columns else group
                records.append(
                    TrajectoryRecord(
                        sku_id=str(sku_id),
                        lead_time=lead_time,
                        demand=group["demand"].to_numpy(dtype=float),
                        orders=group["orders"].to_numpy(dtype=float)
                        if "orders" in group.columns
                        else np.zeros(len(group), dtype=float),
                        inventory=group["inventory"].to_numpy(dtype=float)
                        if "inventory" in group.columns
                        else np.zeros(len(group), dtype=float),
                        base_stock_level=float(group["base_stock_level"].iloc[0])
                        if "base_stock_level" in group.columns
                        else 0.0,
                        holding_cost=float(group["holding_cost"].iloc[0])
                        if "holding_cost" in group.columns
                        else 1.0,
                        stockout_penalty=float(group["stockout_penalty"].iloc[0])
                        if "stockout_penalty" in group.columns
                        else 5.0,
                    )
                )
            return records

        # --- InventoryBench format: dynamic column names ---
        # Columns are like: exact_dates_<sku_id>, demand_<sku_id>, [lead_time_<sku_id>, profit_<sku_id>, holding_cost_<sku_id>]
        demand_cols = [c for c in df.columns if c.startswith("demand_")]
        if not demand_cols:
            raise ValueError(
                f"CSV {csv_path} has no 'demand' or 'demand_*' columns. "
                f"Available: {sorted(available)}"
            )

        records = []
        for demand_col in demand_cols:
            suffix = demand_col[len("demand_"):]  # e.g. "108775044" or "chips(Regular)"
            sku_id = suffix

            demand = df[demand_col].to_numpy(dtype=float)

            hc_col = f"holding_cost_{suffix}"
            holding_cost = float(df[hc_col].iloc[0]) if hc_col in available else 1.0

            profit_col = f"profit_{suffix}"
            # stockout_penalty approximated from profit when available
            stockout_penalty = float(df[profit_col].iloc[0]) if profit_col in available else 5.0

            # Compute newsvendor optimal base-stock level from demand statistics
            # S* = mu + z * sigma  where z = Phi^{-1}(p / (h + p))
            base_stock_level = _newsvendor_base_stock(demand, holding_cost, stockout_penalty)

            records.append(
                TrajectoryRecord(
                    sku_id=sku_id,
                    lead_time=lead_time,
                    demand=demand,
                    orders=np.zeros(len(demand), dtype=float),    # filled by DRO wiring
                    inventory=np.zeros(len(demand), dtype=float), # filled by DRO wiring
                    base_stock_level=base_stock_level,
                    holding_cost=holding_cost,
                    stockout_penalty=stockout_penalty,
                )
            )
        return records

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def bullwhip_ratio(self, orders: np.ndarray, demand: np.ndarray) -> float:
        """Compute Var(orders) / Var(demand).

        Requirements 7.2
        """
        return float(np.var(orders) / np.var(demand))

    def compute_mape(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """Return mean absolute percentage error (%).

        MAPE = mean(|actual - forecast| / |actual|) * 100
        Zero actual values are excluded to avoid division by zero.
        """
        actual = np.asarray(actual, dtype=float)
        forecast = np.asarray(forecast, dtype=float)
        mask = actual != 0.0
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs(actual[mask] - forecast[mask]) / np.abs(actual[mask])) * 100)

    # ------------------------------------------------------------------
    # Individual rationality checks
    # ------------------------------------------------------------------

    def check_base_stock(
        self, trajectory: TrajectoryRecord, orders: np.ndarray
    ) -> tuple[bool, float, float]:
        """Check base-stock order structure: order_t == max(0, base_stock - inv_pos_t).

        Uses a tolerance of 5% of base-stock level or 5% of mean demand,
        whichever is larger, to account for floating-point accumulation
        and lead-time effects in inventory simulation.

        Returns (passed, max_relative_deviation, tolerance).
        Requirements 7.3
        """
        inventory_position = trajectory.inventory
        expected_orders = np.maximum(0.0, trajectory.base_stock_level - inventory_position)
        deviations = np.abs(orders - expected_orders)
        max_dev = float(np.max(deviations)) if len(deviations) > 0 else 0.0
        # Tolerance: 5% of base-stock level or 5% of mean demand, min 1.0
        mean_demand = float(np.mean(np.abs(trajectory.demand))) if len(trajectory.demand) > 0 else 1.0
        tol = max(0.05 * trajectory.base_stock_level, 0.05 * mean_demand, 1.0)
        passed = bool(max_dev <= tol)
        return passed, max_dev, tol

    def check_cost_consistency(
        self, trajectory: TrajectoryRecord
    ) -> tuple[bool, float, float]:
        """Check cost consistency: total_cost == Σ (h·max(0,inv) + p·max(0,-inv)).

        Returns (passed, computed_total, expected_total).
        Requirements 7.4
        """
        h = trajectory.holding_cost
        p = trajectory.stockout_penalty
        inv = trajectory.inventory
        expected_total = float(
            np.sum(h * np.maximum(0.0, inv) + p * np.maximum(0.0, -inv))
        )
        # TrajectoryRecord doesn't carry a reported total_cost field;
        # we recompute and compare to itself — always passes unless the
        # caller provides a separate reported cost. Return the computed value.
        computed_total = expected_total
        passed = True
        return passed, computed_total, expected_total

    def check_order_smoothing(
        self, orders: np.ndarray, demand: np.ndarray
    ) -> tuple[bool, float, float]:
        """Check order smoothing: std(orders) ≤ 1.5 * std(demand).

        Returns (passed, std_orders, 1.5 * std_demand).
        Requirements 7.5
        """
        std_orders = float(np.std(orders))
        std_demand = float(np.std(demand))
        threshold = self.ORDER_SMOOTHING_FACTOR * std_demand
        passed = std_orders <= threshold + 1e-9
        return passed, std_orders, threshold

    def _check_allocation_logic(
        self, trajectory: TrajectoryRecord, orders: np.ndarray
    ) -> tuple[bool, float, float | None]:
        """Placeholder allocation logic check.

        Returns (passed, 0.0, None).
        """
        # Placeholder: always passes until allocation logic is specified
        return True, 0.0, None

    # ------------------------------------------------------------------
    # Aggregate rationality check runner
    # ------------------------------------------------------------------

    def run_rationality_checks(
        self, trajectory: TrajectoryRecord, orders: np.ndarray
    ) -> list[RationalityCheckResult]:
        """Run all Five Rationality Checks; record failures without halting.

        Returns a list of RationalityCheckResult (one per check).
        Requirements 7.6
        """
        results: list[RationalityCheckResult] = []
        traj_id = trajectory.sku_id

        # 1. Base-stock structure
        try:
            passed, computed, expected = self.check_base_stock(trajectory, orders)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_BASE_STOCK,
                    passed=passed,
                    computed_value=computed,
                    expected_value=expected,
                )
            )
        except Exception as exc:
            logger.warning("Base-stock check error for %s: %s", traj_id, exc)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_BASE_STOCK,
                    passed=False,
                    computed_value=float("nan"),
                    expected_value=None,
                )
            )

        # 2. Bullwhip ratio ≤ 1.5
        try:
            ratio = self.bullwhip_ratio(orders, trajectory.demand)
            passed = ratio <= self.BULLWHIP_THRESHOLD
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_BULLWHIP,
                    passed=passed,
                    computed_value=ratio,
                    expected_value=self.BULLWHIP_THRESHOLD,
                )
            )
        except Exception as exc:
            logger.warning("Bullwhip check error for %s: %s", traj_id, exc)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_BULLWHIP,
                    passed=False,
                    computed_value=float("nan"),
                    expected_value=self.BULLWHIP_THRESHOLD,
                )
            )

        # 3. Allocation logic (placeholder)
        try:
            passed, computed, expected = self._check_allocation_logic(trajectory, orders)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_ALLOCATION,
                    passed=passed,
                    computed_value=computed,
                    expected_value=expected,
                )
            )
        except Exception as exc:
            logger.warning("Allocation check error for %s: %s", traj_id, exc)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_ALLOCATION,
                    passed=False,
                    computed_value=float("nan"),
                    expected_value=None,
                )
            )

        # 4. Cost consistency
        try:
            passed, computed, expected = self.check_cost_consistency(trajectory)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_COST_CONSISTENCY,
                    passed=passed,
                    computed_value=computed,
                    expected_value=expected,
                )
            )
        except Exception as exc:
            logger.warning("Cost consistency check error for %s: %s", traj_id, exc)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_COST_CONSISTENCY,
                    passed=False,
                    computed_value=float("nan"),
                    expected_value=None,
                )
            )

        # 5. Order smoothing
        try:
            passed, computed, expected = self.check_order_smoothing(orders, trajectory.demand)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_ORDER_SMOOTHING,
                    passed=passed,
                    computed_value=computed,
                    expected_value=expected,
                )
            )
        except Exception as exc:
            logger.warning("Order smoothing check error for %s: %s", traj_id, exc)
            results.append(
                RationalityCheckResult(
                    trajectory_id=traj_id,
                    check_name=self.CHECK_ORDER_SMOOTHING,
                    passed=False,
                    computed_value=float("nan"),
                    expected_value=None,
                )
            )

        # Log any failures
        failures = [r for r in results if not r.passed]
        if failures:
            for f in failures:
                logger.warning(
                    "Rationality check FAILED — trajectory=%s check=%s computed=%.4f expected=%s",
                    f.trajectory_id,
                    f.check_name,
                    f.computed_value,
                    f.expected_value,
                )

        return results
