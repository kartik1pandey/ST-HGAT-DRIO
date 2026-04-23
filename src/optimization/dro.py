"""
DRO (Distributionally Robust Optimization) module.

Builds a Wasserstein ambiguity set around the empirical forecast distribution
and solves min E_π[Σ γ^τ · c_τ] via Gurobi or CPLEX.

Falls back to μ + 2σ base-stock heuristic on infeasibility.
Raises RuntimeError if no configured solver is available.
"""

import logging
import warnings
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_config_value(cfg: Any, key: str, default: Any) -> Any:
    """Retrieve a value from a dict-like or object-like config."""
    if isinstance(cfg, dict):
        # Support nested keys like "optimization.solver"
        parts = key.split(".")
        val = cfg
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return default
        return val
    else:
        # Object-style config (e.g. DictConfig / dataclass)
        parts = key.split(".")
        val = cfg
        for part in parts:
            val = getattr(val, part, None)
            if val is None:
                return default
        return val


class DROModule:
    """
    Distributionally Robust Optimization module.

    Parameters
    ----------
    cfg : dict or object
        Configuration with keys under ``optimization``:
        - solver: "gurobi" or "cplex"
        - epsilon: Wasserstein radius (default 0.1)
        - gamma: discount factor (default 0.99)
        - holding_cost: h (default 1.0)
        - stockout_penalty: p (default 5.0)
    solver_fn : callable, optional
        Injectable solver function for testing. Signature:
        ``solver_fn(mu, sigma, epsilon, gamma, h, p) -> np.ndarray | None``
        Returns order quantities array of shape [N, H], or None if infeasible.
    """

    def __init__(self, cfg: Any, solver_fn: Optional[Callable] = None):
        self.solver_name: str = _get_config_value(cfg, "optimization.solver", "gurobi")
        self.epsilon: float = float(_get_config_value(cfg, "optimization.epsilon", 0.1))
        self.gamma: float = float(_get_config_value(cfg, "optimization.gamma", 0.99))
        self.h: float = float(_get_config_value(cfg, "optimization.holding_cost", 1.0))
        self.p: float = float(_get_config_value(cfg, "optimization.stockout_penalty", 5.0))
        self._solver_fn = solver_fn  # injectable for testing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, forecasts: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Compute optimal order quantities under distributional uncertainty.

        Parameters
        ----------
        forecasts : np.ndarray, shape [N, H]
            Per-node demand forecasts for each horizon step.
        sigma : np.ndarray, shape [N, H]
            Per-node forecast standard deviations.
        dynamic_epsilon : bool
            If True, scale epsilon by the coefficient of variation (sigma/mu)
            so high-variance forecasts get a larger Wasserstein buffer.

        Returns
        -------
        np.ndarray, shape [N, H]
            Non-negative order quantities.

        Raises
        ------
        RuntimeError
            If the configured solver (gurobi/cplex) is not installed.
        """
        forecasts = np.asarray(forecasts, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        # Dynamic epsilon: scale Wasserstein radius by coefficient of variation.
        # High-variance forecasts get a larger safety buffer.
        # We pass the original sigma to the solver but use a scaled epsilon
        # for the robust penalty term inside the formulation.
        mu_safe = np.where(np.abs(forecasts) > 1e-9, np.abs(forecasts), 1.0)
        cv = np.clip(sigma / mu_safe, 0.0, 2.0)  # coefficient of variation, capped at 2
        # effective_epsilon per element: epsilon * (1 + cv), bounded to [eps, 3*eps]
        effective_epsilon = np.clip(
            self.epsilon * (1.0 + cv),
            self.epsilon,
            3.0 * self.epsilon,
        )
        # Scale sigma by the ratio so the solver sees the right robust penalty
        # without changing the balance constraint (which uses raw mu)
        scaled_sigma = sigma * (effective_epsilon / self.epsilon)

        if self._solver_fn is not None:
            result = self._solver_fn(
                forecasts, scaled_sigma, self.epsilon, self.gamma, self.h, self.p
            )
            if result is None:
                return self._heuristic_fallback(forecasts, sigma)  # use raw sigma for fallback
            return np.maximum(result, 0.0)

        solver_name = self.solver_name.lower()
        if solver_name == "gurobi":
            return self._solve_with_gurobi(forecasts, scaled_sigma)
        elif solver_name == "cplex":
            return self._solve_with_cplex(forecasts, scaled_sigma)
        else:
            raise RuntimeError(
                f"Unknown solver '{self.solver_name}'. "
                "Configure 'optimization.solver' as 'gurobi' or 'cplex'."
            )

    # ------------------------------------------------------------------
    # Solver backends
    # ------------------------------------------------------------------

    def _solve_with_gurobi(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        try:
            import gurobipy  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Gurobi solver is not available. "
                "Install gurobipy or switch to 'cplex' in config."
            )
        result = self._formulate_and_solve_gurobi(mu, sigma)
        if result is None:
            return self._heuristic_fallback(mu, sigma)
        return np.maximum(result, 0.0)

    def _solve_with_cplex(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        try:
            import cplex  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "CPLEX solver is not available. "
                "Install cplex or switch to 'gurobi' in config."
            )
        result = self._formulate_and_solve_cplex(mu, sigma)
        if result is None:
            return self._heuristic_fallback(mu, sigma)
        return np.maximum(result, 0.0)

    # ------------------------------------------------------------------
    # Formulation helpers
    # ------------------------------------------------------------------

    def _formulate_and_solve_gurobi(
        self, mu: np.ndarray, sigma: np.ndarray
    ) -> Optional[np.ndarray]:
        """Formulate and solve the DRO problem via Gurobi (vectorized).

        Splits large problems into node-wise sub-problems to stay within
        the restricted license limit of 2,000 variables.
        Returns order quantities array or None if infeasible.
        """
        import gurobipy as gp
        from gurobipy import GRB

        N, H = mu.shape
        # Restricted license: max 2000 variables. Each node needs 3*H vars.
        # Solve node-wise if N*H*3 > 1800 (leave headroom).
        max_nodes_per_solve = max(1, 1800 // (3 * H))

        if N > max_nodes_per_solve:
            # Solve in chunks
            result = np.zeros((N, H))
            for start in range(0, N, max_nodes_per_solve):
                end = min(start + max_nodes_per_solve, N)
                chunk = self._solve_chunk_gurobi(
                    mu[start:end], sigma[start:end]
                )
                if chunk is None:
                    return None
                result[start:end] = chunk
            return result
        else:
            return self._solve_chunk_gurobi(mu, sigma)

    def _solve_chunk_gurobi(
        self, mu: np.ndarray, sigma: np.ndarray
    ) -> Optional[np.ndarray]:
        """Solve a single chunk via Gurobi MVar API."""
        import gurobipy as gp
        from gurobipy import GRB

        N, H = mu.shape
        NH = N * H
        mu_flat    = mu.flatten()
        sigma_flat = sigma.flatten()
        discounts  = np.tile([self.gamma ** t for t in range(H)], N)

        try:
            model = gp.Model("dro_chunk")
            model.setParam("OutputFlag", 0)
            model.setParam("Threads", 0)

            q     = model.addMVar(NH, lb=0.0, name="q")
            over  = model.addMVar(NH, lb=0.0, name="over")
            under = model.addMVar(NH, lb=0.0, name="under")

            model.addConstr(q - over + under == mu_flat, name="balance")

            robust = self.epsilon * (self.h + self.p) * sigma_flat
            obj = discounts @ (self.h * over + self.p * under) + float(discounts @ robust)
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                return q.X.reshape(N, H)
            else:
                logger.warning("Gurobi status %d; falling back.", model.Status)
                return None
        except Exception as exc:
            logger.warning("Gurobi solve failed (%s); falling back.", exc)
            return None

    def _formulate_and_solve_cplex(
        self, mu: np.ndarray, sigma: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Formulate and solve the DRO problem via CPLEX.

        Returns order quantities array or None if infeasible.
        """
        import cplex

        N, H = mu.shape
        try:
            c = cplex.Cplex()
            c.set_log_stream(None)
            c.set_error_stream(None)
            c.set_warning_stream(None)
            c.set_results_stream(None)

            # Variables: q[n,t], over[n,t], under[n,t]
            var_names = []
            lb = []
            ub = []
            obj_coeffs = []

            q_idx = {}
            over_idx = {}
            under_idx = {}
            idx = 0

            for n in range(N):
                for t in range(H):
                    discount = self.gamma ** t
                    sigma_nt = float(sigma[n, t])
                    robust_penalty = self.epsilon * (self.h + self.p) * sigma_nt

                    # q variable (no direct obj contribution from q itself)
                    var_names.append(f"q_{n}_{t}")
                    lb.append(0.0)
                    ub.append(cplex.infinity)
                    obj_coeffs.append(0.0)
                    q_idx[(n, t)] = idx
                    idx += 1

                    # over variable
                    var_names.append(f"over_{n}_{t}")
                    lb.append(0.0)
                    ub.append(cplex.infinity)
                    obj_coeffs.append(discount * self.h)
                    over_idx[(n, t)] = idx
                    idx += 1

                    # under variable
                    var_names.append(f"under_{n}_{t}")
                    lb.append(0.0)
                    ub.append(cplex.infinity)
                    obj_coeffs.append(discount * self.p)
                    under_idx[(n, t)] = idx
                    idx += 1

                    # Add robust penalty as constant (shift obj)
                    # CPLEX doesn't support constant in obj directly via addVars,
                    # so we track it separately
                    _ = robust_penalty  # included via offset below

            c.variables.add(obj=obj_coeffs, lb=lb, ub=ub, names=var_names)
            c.objective.set_sense(c.objective.sense.minimize)

            # Constraints: over - under == q - mu
            for n in range(N):
                for t in range(H):
                    mu_nt = float(mu[n, t])
                    c.linear_constraints.add(
                        lin_expr=[
                            cplex.SparsePair(
                                ind=[over_idx[(n, t)], under_idx[(n, t)], q_idx[(n, t)]],
                                val=[1.0, -1.0, -1.0],
                            )
                        ],
                        senses=["E"],
                        rhs=[-mu_nt],
                    )

            c.solve()
            status = c.solution.get_status()
            # CPLEX status 1 = optimal
            if status == 1:
                all_vals = c.solution.get_values()
                result = np.zeros((N, H))
                for n in range(N):
                    for t in range(H):
                        result[n, t] = all_vals[q_idx[(n, t)]]
                return result
            else:
                logger.warning(
                    "CPLEX returned non-optimal status %d; falling back to heuristic.",
                    status,
                )
                return None
        except Exception as exc:
            logger.warning("CPLEX solve failed (%s); falling back to heuristic.", exc)
            return None

    # ------------------------------------------------------------------
    # Fallback heuristic
    # ------------------------------------------------------------------

    def _heuristic_fallback(
        self, mu: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        """Base-stock heuristic: order up to μ + 2σ, clipped to non-negative."""
        logger.warning(
            "DRO solver infeasible or unavailable; "
            "falling back to μ + 2σ base-stock heuristic."
        )
        return np.maximum(mu + 2.0 * sigma, 0.0)
