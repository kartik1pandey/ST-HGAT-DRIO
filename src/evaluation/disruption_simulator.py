"""Disruption simulator for run_submission.py.

Provides M5-scale disruption scenarios and three-way comparison:
  - Naive base-stock (reactive)
  - DRO adaptive (epsilon scales with disruption signal)
  - SC-RIHN adaptive (proactive pre-surge safety stock)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np


@dataclass
class DisruptionScenario:
    sku_id: str
    demand: np.ndarray
    base_stock: float
    shock_start: int
    shock_duration: int
    shock_scale: float
    holding_cost: float
    stockout_penalty: float


@dataclass
class SimResult:
    recovery_time: float
    total_cost: float
    stockout_periods: int
    inventory: np.ndarray
    orders: np.ndarray


def _norm_ppf(p: float) -> float:
    if p <= 0: return float("-inf")
    if p >= 1: return float("inf")
    if p < 0.5: return -_norm_ppf(1 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0]+c[1]*t+c[2]*t*t) / (1+d[0]*t+d[1]*t*t+d[2]*t*t*t)


def build_m5_disruption_scenarios(
    m5_dir: str | Path,
    n_series: int = 200,
    shock_fraction: float = 0.70,
    disruption_duration: int = 56,
    holding_cost: float = 1.0,
    stockout_penalty: float = 5.0,
) -> List[DisruptionScenario]:
    import pandas as pd
    m5_dir = Path(m5_dir)
    p = m5_dir / "sales_train_evaluation.csv"
    if not p.exists():
        p = m5_dir / "sales_train_validation.csv"
    df = pd.read_csv(p)
    day_cols = sorted([c for c in df.columns if c.startswith("d_")],
                      key=lambda c: int(c.split("_")[1]))
    vals = df[day_cols].to_numpy(dtype=float)[:n_series]
    ids  = df["id"].tolist()[:n_series] if "id" in df.columns else [str(i) for i in range(n_series)]

    scenarios = []
    for i, (row, sid) in enumerate(zip(vals, ids)):
        demand = row.astype(float)
        T = len(demand)
        mu = float(np.mean(demand))
        sigma = max(float(np.std(demand)), 1.0)
        cr = stockout_penalty / (holding_cost + stockout_penalty)
        S = max(mu + _norm_ppf(cr) * sigma, 0.0)
        shock_start = max(1, int(T * shock_fraction))
        scenarios.append(DisruptionScenario(
            sku_id=sid, demand=demand, base_stock=S,
            shock_start=shock_start, shock_duration=disruption_duration,
            shock_scale=3.0, holding_cost=holding_cost, stockout_penalty=stockout_penalty,
        ))
    return scenarios


def _run_policy(demand: np.ndarray, orders: np.ndarray,
                h: float, p: float,
                shock_start: int = 0) -> SimResult:
    T = len(demand)
    inventory = np.zeros(T)
    ip = 0.0
    for t in range(T):
        inventory[t] = ip
        ip = ip + orders[t] - demand[t]
    cost = float(np.sum(h * np.maximum(inventory, 0) + p * np.maximum(-inventory, 0)))
    stockouts = int(np.sum(inventory < 0))

    # Recovery time: periods from shock_start until inventory >= 0 again
    # (or until it returns to within 10% of pre-shock mean)
    pre_shock_mean = float(np.mean(demand[:max(shock_start, 1)]))
    threshold = max(abs(pre_shock_mean) * 0.10, 1.0)
    rt = T - shock_start  # default: never recovered
    for t in range(shock_start, T):
        if inventory[t] >= -threshold:
            rt = t - shock_start
            break
    return SimResult(recovery_time=float(rt), total_cost=cost,
                     stockout_periods=stockouts, inventory=inventory, orders=orders)


def simulate_disruption(sc: DisruptionScenario, policy: str = "naive",
                        dro_epsilon_disruption: float = 2.0) -> SimResult:
    demand = sc.demand.copy()
    demand[sc.shock_start:sc.shock_start+sc.shock_duration] *= sc.shock_scale
    T = len(demand)
    orders = np.zeros(T)
    ip = 0.0
    for t in range(T):
        if policy == "dro_adaptive" and sc.shock_start <= t < sc.shock_start + sc.shock_duration:
            sigma = max(float(np.std(sc.demand)), 1.0)
            S_dro = sc.base_stock + dro_epsilon_disruption * sigma
        else:
            S_dro = sc.base_stock
        orders[t] = max(0.0, S_dro - ip)
        ip = ip + orders[t] - demand[t]
    return _run_policy(demand, orders, sc.holding_cost, sc.stockout_penalty,
                       shock_start=sc.shock_start)


def compute_presurge_signal(demand: np.ndarray, window: int = 28) -> np.ndarray:
    """Rolling z-score of demand — negative values signal pre-surge dip."""
    T = len(demand)
    signal = np.zeros(T)
    for t in range(window, T):
        w = demand[t-window:t]
        mu = float(np.mean(w)); sigma = max(float(np.std(w)), 1e-6)
        signal[t] = (demand[t] - mu) / sigma
    return signal


def simulate_scrihn_adaptive(
    sc: DisruptionScenario,
    presurge_signal: np.ndarray,
    presurge_window: int = 28,
    presurge_threshold: float = -0.08,
    epsilon_presurge: float = 4.0,
    epsilon_disruption: float = 2.0,
    epsilon_base: float = 0.1,
) -> SimResult:
    """SC-RIHN adaptive policy: pre-builds safety stock when presurge signal fires."""
    demand = sc.demand.copy()
    demand[sc.shock_start:sc.shock_start+sc.shock_duration] *= sc.shock_scale
    T = len(demand)
    sigma = max(float(np.std(sc.demand)), 1.0)
    orders = np.zeros(T)
    ip = 0.0
    for t in range(T):
        sig = presurge_signal[t] if t < len(presurge_signal) else 0.0
        in_disruption = sc.shock_start <= t < sc.shock_start + sc.shock_duration
        if in_disruption:
            eps = epsilon_disruption
        elif sig < presurge_threshold:
            eps = epsilon_presurge
        else:
            eps = epsilon_base
        S_adaptive = sc.base_stock + eps * sigma
        orders[t] = max(0.0, S_adaptive - ip)
        ip = ip + orders[t] - demand[t]
    return _run_policy(demand, orders, sc.holding_cost, sc.stockout_penalty,
                       shock_start=sc.shock_start)
