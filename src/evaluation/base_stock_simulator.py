"""Base-stock simulator — enriches TrajectoryRecord with computed orders/inventory."""
from __future__ import annotations
import math
import numpy as np
from typing import List
from src.evaluation.evaluator import TrajectoryRecord


def _norm_ppf(p: float) -> float:
    if p <= 0: return float("-inf")
    if p >= 1: return float("inf")
    if p < 0.5: return -_norm_ppf(1 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0]+c[1]*t+c[2]*t*t) / (1+d[0]*t+d[1]*t*t+d[2]*t*t*t)


def enrich_trajectories_with_base_stock(
    trajectories: List[TrajectoryRecord],
    cap_multiplier: float = 1e9,
) -> List[TrajectoryRecord]:
    """Compute newsvendor base-stock orders and simulated inventory for each trajectory."""
    enriched = []
    for traj in trajectories:
        demand = traj.demand.astype(float)
        T = len(demand)
        if T < 2:
            enriched.append(traj)
            continue
        h = traj.holding_cost
        p = traj.stockout_penalty
        mu = float(np.mean(demand))
        sigma = max(float(np.std(demand)), 1.0)
        cr = p / (h + p)
        z = _norm_ppf(cr)
        S = max(mu + z * sigma, 0.0)

        orders = np.zeros(T)
        inventory = np.zeros(T)
        ip = 0.0
        for t in range(T):
            inventory[t] = ip
            order = max(0.0, S - ip)
            orders[t] = order
            ip = ip + order - demand[t]

        enriched.append(TrajectoryRecord(
            sku_id=traj.sku_id,
            lead_time=traj.lead_time,
            demand=demand,
            orders=orders,
            inventory=inventory,
            base_stock_level=S,
            holding_cost=h,
            stockout_penalty=p,
        ))
    return enriched
