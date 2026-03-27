"""Reusable abstract optimization model for economic dispatch.

This module consolidates the repeated Gurobi optimization cells from the notebook
into reusable functions. It supports:
- Baseline dispatch (no battery)
- Dispatch with battery storage
- Batch runs across seasons/scenarios

Typical usage from notebook:

    from Abstract_model import DispatchInput, BatteryConfig, compare_baseline_vs_battery

    data = DispatchInput(
        demand=p50_demand,
        variable_costs=cost,
        tech_capacities=tech_caps,
        renewable_profiles={"Wind": p50_wind, "Solar": p50_solar, "Hydro": p50_hydro},
    )

    battery = BatteryConfig(
        power_mw=500,
        capacity_mwh=1500,
        charging_efficiency=0.92,
        discharging_efficiency=0.92,
        initial_soc_fraction=0.5,
        cyclic_mode="daily",
    )

    results = compare_baseline_vs_battery(data, battery)
    baseline = results["baseline"]
    with_battery = results["battery"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:
    raise ImportError(
        "gurobipy is required for Abstract_model.py. Install with: pip install gurobipy"
    ) from exc


DEFAULT_RENEWABLE_TECHS = ("Wind", "Solar", "Hydro")


@dataclass(frozen=True)
class BatteryConfig:
    """Battery optimization parameters."""

    power_mw: float
    capacity_mwh: float
    charging_efficiency: float = 1.0
    discharging_efficiency: float = 1.0
    initial_soc_fraction: float = 0.5
    cyclic_mode: str = "daily"  # one of: none, end, daily


@dataclass(frozen=True)
class DispatchInput:
    """Input container for one optimization run."""

    demand: np.ndarray
    variable_costs: Mapping[str, float]
    tech_capacities: Mapping[str, float]
    renewable_profiles: Mapping[str, np.ndarray]
    techs: Optional[Iterable[str]] = None


@dataclass
class DispatchResult:
    """Output container from one optimization run."""

    status: int
    objective_value: Optional[float]
    solution: pd.DataFrame
    hourly_prices: np.ndarray
    model: gp.Model
    techs: List[str]


def _as_1d_array(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least 1 value.")
    return arr


def _build_tech_list(data: DispatchInput) -> List[str]:
    if data.techs is not None:
        techs = list(data.techs)
    else:
        techs = sorted(
            set(data.variable_costs.keys())
            | set(data.tech_capacities.keys())
            | set(data.renewable_profiles.keys())
        )
    if not techs:
        raise ValueError("No technologies were provided in DispatchInput.")
    return techs


def _renewable_upper_bound(
    tech: str,
    t: int,
    renewable_profiles: Mapping[str, np.ndarray],
    tech_capacities: Mapping[str, float],
) -> float:
    if tech in renewable_profiles:
        return float(renewable_profiles[tech][t])
    return float(tech_capacities.get(tech, 0.0))


def solve_dispatch(
    data: DispatchInput,
    battery: Optional[BatteryConfig] = None,
    *,
    model_name: str = "economic_dispatch",
    solver_output: bool = False,
    clip_negative_prices: bool = True,
) -> DispatchResult:
    """Solve one dispatch optimization run.

    Objective is generation cost minimization:
    sum_t sum_tech variable_cost[tech] * generation[tech,t]

    Baseline mode (battery=None):
        sum_tech generation[tech,t] == demand[t]

    Battery mode:
        sum_tech generation[tech,t] + P_discharge[t] == demand[t] + P_charge[t]
        SOC dynamics and cyclic constraints as configured.
    """

    demand = _as_1d_array("demand", data.demand)
    renewable_profiles = {
        tech: _as_1d_array(f"renewable_profiles[{tech}]", profile)
        for tech, profile in data.renewable_profiles.items()
    }

    horizon = demand.size
    for tech, profile in renewable_profiles.items():
        if profile.size != horizon:
            raise ValueError(
                f"Profile length mismatch for {tech}: expected {horizon}, got {profile.size}."
            )

    techs = _build_tech_list(data)

    model = gp.Model(model_name)
    model.setParam("OutputFlag", 1 if solver_output else 0)

    generation = {}
    for t in range(horizon):
        for tech in techs:
            ub = _renewable_upper_bound(tech, t, renewable_profiles, data.tech_capacities)
            generation[(tech, t)] = model.addVar(lb=0.0, ub=max(0.0, ub), name=f"g_{tech}_{t}")

    p_charge = {}
    p_discharge = {}
    soc = {}

    if battery is not None:
        if not (0.0 <= battery.initial_soc_fraction <= 1.0):
            raise ValueError("battery.initial_soc_fraction must be between 0 and 1.")
        if battery.charging_efficiency <= 0 or battery.discharging_efficiency <= 0:
            raise ValueError("Battery efficiencies must be > 0.")

        for t in range(horizon):
            p_charge[t] = model.addVar(lb=0.0, ub=battery.power_mw, name=f"P_ch_{t}")
            p_discharge[t] = model.addVar(lb=0.0, ub=battery.power_mw, name=f"P_dis_{t}")
            soc[t] = model.addVar(lb=0.0, ub=battery.capacity_mwh, name=f"SOC_{t}")

        initial_soc = battery.initial_soc_fraction * battery.capacity_mwh
        model.addConstr(soc[0] == initial_soc, name="initial_soc")

        for t in range(horizon - 1):
            model.addConstr(
                soc[t + 1]
                == soc[t]
                + (p_charge[t] * battery.charging_efficiency)
                - (p_discharge[t] / battery.discharging_efficiency),
                name=f"soc_dynamics_{t}",
            )

        cyclic_mode = battery.cyclic_mode.lower()
        if cyclic_mode == "daily":
            n_days = horizon // 24
            for day in range(n_days):
                end_hour = (day + 1) * 24 - 1
                model.addConstr(
                    initial_soc
                    == soc[end_hour]
                    + (p_charge[end_hour] * battery.charging_efficiency)
                    - (p_discharge[end_hour] / battery.discharging_efficiency),
                    name=f"cyclic_soc_day_{day}",
                )
        elif cyclic_mode == "end":
            end_hour = horizon - 1
            model.addConstr(
                initial_soc
                == soc[end_hour]
                + (p_charge[end_hour] * battery.charging_efficiency)
                - (p_discharge[end_hour] / battery.discharging_efficiency),
                name="cyclic_soc_end",
            )
        elif cyclic_mode != "none":
            raise ValueError("battery.cyclic_mode must be one of: none, end, daily")

    for t in range(horizon):
        if battery is None:
            model.addConstr(
                gp.quicksum(generation[(tech, t)] for tech in techs) == demand[t],
                name=f"demand_hour_{t}",
            )
        else:
            model.addConstr(
                gp.quicksum(generation[(tech, t)] for tech in techs)
                + p_discharge[t]
                == demand[t]
                + p_charge[t],
                name=f"power_balance_{t}",
            )

    generation_cost = gp.quicksum(
        float(data.variable_costs.get(tech, 0.0)) * generation[(tech, t)]
        for tech in techs
        for t in range(horizon)
    )
    model.setObjective(generation_cost, GRB.MINIMIZE)

    model.optimize()

    if model.status != GRB.OPTIMAL:
        return DispatchResult(
            status=model.status,
            objective_value=None,
            solution=pd.DataFrame(),
            hourly_prices=np.array([]),
            model=model,
            techs=techs,
        )

    sol = pd.DataFrame(index=np.arange(horizon))
    for tech in techs:
        sol[tech] = [generation[(tech, t)].X for t in range(horizon)]
    sol["demand"] = demand

    if battery is not None:
        sol["P_charge"] = [p_charge[t].X for t in range(horizon)]
        sol["P_discharge"] = [p_discharge[t].X for t in range(horizon)]
        sol["SOC"] = [soc[t].X for t in range(horizon)]

    prices = []
    constr_prefix = "demand_hour" if battery is None else "power_balance"
    for t in range(horizon):
        constr = model.getConstrByName(f"{constr_prefix}_{t}")
        prices.append(float(constr.Pi) if constr is not None else np.nan)

    hourly_prices = np.array(prices, dtype=float)
    if clip_negative_prices:
        hourly_prices = np.maximum(hourly_prices, 0.0)

    return DispatchResult(
        status=model.status,
        objective_value=float(model.objVal),
        solution=sol,
        hourly_prices=hourly_prices,
        model=model,
        techs=techs,
    )


def compare_baseline_vs_battery(
    data: DispatchInput,
    battery: BatteryConfig,
    *,
    base_model_name: str = "dispatch_baseline",
    battery_model_name: str = "dispatch_battery",
    solver_output: bool = False,
    clip_negative_prices: bool = True,
) -> Dict[str, Any]:
    """Run baseline + battery optimization and summarize key deltas."""

    baseline = solve_dispatch(
        data,
        battery=None,
        model_name=base_model_name,
        solver_output=solver_output,
        clip_negative_prices=clip_negative_prices,
    )
    with_battery = solve_dispatch(
        data,
        battery=battery,
        model_name=battery_model_name,
        solver_output=solver_output,
        clip_negative_prices=clip_negative_prices,
    )

    if baseline.objective_value is None or with_battery.objective_value is None:
        return {
            "baseline": baseline,
            "battery": with_battery,
            "cost_reduction": None,
            "cost_reduction_pct": None,
            "total_charged": None,
            "total_discharged": None,
            "battery_profit": None,
        }

    total_charged = float(with_battery.solution["P_charge"].sum())
    total_discharged = float(with_battery.solution["P_discharge"].sum())

    battery_revenue = float(
        (with_battery.solution["P_discharge"].values * with_battery.hourly_prices).sum()
    )
    battery_charging_cost = float(
        (with_battery.solution["P_charge"].values * with_battery.hourly_prices).sum()
    )
    battery_profit = battery_revenue - battery_charging_cost

    cost_reduction = baseline.objective_value - with_battery.objective_value
    cost_reduction_pct = (
        100.0 * cost_reduction / baseline.objective_value
        if baseline.objective_value != 0
        else np.nan
    )

    return {
        "baseline": baseline,
        "battery": with_battery,
        "cost_reduction": cost_reduction,
        "cost_reduction_pct": cost_reduction_pct,
        "total_charged": total_charged,
        "total_discharged": total_discharged,
        "battery_profit": battery_profit,
    }


def run_for_seasons(
    seasonal_inputs: Mapping[str, DispatchInput],
    battery: Optional[BatteryConfig] = None,
    *,
    solver_output: bool = False,
    clip_negative_prices: bool = True,
) -> Dict[str, DispatchResult]:
    """Run one optimization per season using a shared model structure."""

    results: Dict[str, DispatchResult] = {}
    for season, season_input in seasonal_inputs.items():
        results[season] = solve_dispatch(
            season_input,
            battery=battery,
            model_name=f"dispatch_{season}",
            solver_output=solver_output,
            clip_negative_prices=clip_negative_prices,
        )
    return results
