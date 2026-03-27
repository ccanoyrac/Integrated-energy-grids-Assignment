"""Multi-country economic dispatch optimization with interconnections and batteries.

This module supports optimization across multiple countries (Spain, France, Italy, Portugal)
with:
- Inter-country power flows with capacity limits
- Battery storage in Spain only
- Renewable profiles constrained to hourly limits
- Non-renewable generation constrained to installed capacity

Typical usage from notebook:

    from Multi_country_dispatch import MultiCountryDispatchInput, solve_multi_country_dispatch

    data = MultiCountryDispatchInput(
        demands={
            'ES': p50_demand_es,
            'FR': p50_demand_fr,
            'IT': p50_demand_it,
            'PT': p50_demand_pt,
        },
        variable_costs={
            'ES': costs_es,
            'FR': costs_fr,
            'IT': costs_it,
            'PT': costs_pt,
        },
        tech_capacities={
            'ES': caps_es,
            'FR': caps_fr,
            'IT': caps_it,
            'PT': caps_pt,
        },
        renewable_profiles={
            'ES': {'Wind': wind_es, 'Solar': solar_es, 'Hydro': hydro_es},
            'FR': {'Wind': wind_fr, 'Solar': solar_fr, 'Hydro': hydro_fr},
            'IT': {'Wind': wind_it, 'Solar': solar_it, 'Hydro': hydro_it},
            'PT': {'Wind': wind_pt, 'Solar': solar_pt, 'Hydro': hydro_pt},
        },
        interconnections=[
            ("ES", "FR", 3500),  # (from, to, capacity_MW)
            ("FR", "IT", 4500),
            ("ES", "IT", 2000),
            ("ES", "PT", 4500),
        ],
        battery_country='ES',
        battery_config={
            'power_mw': 500,
            'capacity_mwh': 1500,
            'charging_efficiency': 0.92,
            'discharging_efficiency': 0.92,
            'initial_soc_fraction': 0.5,
        },
    )

    results = solve_multi_country_dispatch(data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:
    raise ImportError(
        "gurobipy is required. Install with: pip install gurobipy"
    ) from exc


@dataclass(frozen=True)
class MultiCountryDispatchInput:
    """Input container for multi-country optimization."""

    demands: Mapping[str, np.ndarray]  # {country_code: hourly_demand_array}
    variable_costs: Mapping[str, Mapping[str, float]]  # {country_code: {tech: cost}}
    tech_capacities: Mapping[str, Mapping[str, float]]  # {country_code: {tech: capacity}}
    renewable_profiles: Mapping[str, Mapping[str, np.ndarray]]  # {country_code: {tech: profile}}
    interconnections: List[Tuple[str, str, float]]  # [(from, to, capacity_mw), ...]
    battery_country: str = "ES"
    battery_config: Optional[Dict[str, float]] = None


@dataclass
class MultiCountryDispatchResult:
    """Output container from multi-country optimization."""

    status: int
    objective_value: Optional[float]
    generation_by_country: Dict[str, pd.DataFrame]  # {country: generation_df}
    hourly_prices_by_country: Dict[str, np.ndarray]  # {country: price_array}
    power_flows: pd.DataFrame  # interconnection flows (time, from, to, flow_mw)
    model: gp.Model
    countries: List[str]


def _as_1d_array(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least 1 value.")
    return arr


def _get_renewable_ub(
    country: str,
    tech: str,
    t: int,
    renewable_profiles: Mapping[str, Mapping[str, np.ndarray]],
    tech_capacities: Mapping[str, Mapping[str, float]],
) -> float:
    """Get upper bound for renewable generation (either profile or capacity)."""
    if country in renewable_profiles and tech in renewable_profiles[country]:
        return float(renewable_profiles[country][tech][t])
    if country in tech_capacities and tech in tech_capacities[country]:
        return float(tech_capacities[country][tech])
    return 0.0


def solve_multi_country_dispatch(
    data: MultiCountryDispatchInput,
    *,
    model_name: str = "multi_country_dispatch",
    solver_output: bool = False,
    clip_negative_prices: bool = True,
) -> MultiCountryDispatchResult:
    """Solve multi-country economic dispatch with interconnections.

    Objective: Minimize total generation cost across all countries
    subject to:
        - Demand balance in each country (including battery in Spain)
        - Inter-country power flow limits
        - Generation capacity limits
        - Renewable generation limits (P50 profiles)
        - Battery dynamics (if configured)
    """

    countries = sorted(data.demands.keys())
    horizon = _as_1d_array("demands", data.demands[countries[0]]).size

    # Validate
    for country in countries:
        demand_arr = _as_1d_array(f"demand[{country}]", data.demands[country])
        if demand_arr.size != horizon:
            raise ValueError(f"Demand mismatch for {country}: expected {horizon}, got {demand_arr.size}")

    # Prepare arrays
    demands = {c: _as_1d_array(f"demand[{c}]", data.demands[c]) for c in countries}
    renewable_profiles = {
        c: {
            tech: _as_1d_array(f"renewable_profiles[{c}][{tech}]", profile)
            for tech, profile in data.renewable_profiles.get(c, {}).items()
        }
        for c in countries
    }

    # Create model
    model = gp.Model(model_name)
    model.setParam("OutputFlag", 1 if solver_output else 0)

    # Generation variables: generation[(country, tech, t)]
    generation = {}
    for c in countries:
        for tech in data.tech_capacities.get(c, {}):
            for t in range(horizon):
                ub = _get_renewable_ub(c, tech, t, data.renewable_profiles, data.tech_capacities)
                generation[(c, tech, t)] = model.addVar(lb=0.0, ub=max(0.0, ub), name=f"g_{c}_{tech}_{t}")

    # Power flow variables: power_flow[(from_c, to_c, t)]
    power_flow = {}
    flow_from_to = {(f, t): 0 for f, t in [(ic[0], ic[1]) for ic in data.interconnections]}
    for from_c, to_c, capacity in data.interconnections:
        for t in range(horizon):
            power_flow[(from_c, to_c, t)] = model.addVar(lb=-capacity, ub=capacity, name=f"flow_{from_c}_{to_c}_{t}")

    # Battery variables (only in battery_country)
    p_charge = {}
    p_discharge = {}
    soc = {}

    if data.battery_config is not None:
        batt_country = data.battery_country
        power_mw = data.battery_config.get("power_mw", 500.0)
        capacity_mwh = data.battery_config.get("capacity_mwh", 1500.0)
        charge_eff = data.battery_config.get("charging_efficiency", 0.92)
        discharge_eff = data.battery_config.get("discharging_efficiency", 0.92)
        initial_soc_frac = data.battery_config.get("initial_soc_fraction", 0.5)

        for t in range(horizon):
            p_charge[t] = model.addVar(lb=0.0, ub=power_mw, name=f"P_ch_{t}")
            p_discharge[t] = model.addVar(lb=0.0, ub=power_mw, name=f"P_dis_{t}")
            soc[t] = model.addVar(lb=0.0, ub=capacity_mwh, name=f"SOC_{t}")

        initial_soc = initial_soc_frac * capacity_mwh
        model.addConstr(soc[0] == initial_soc, name="battery_initial_soc")

        for t in range(horizon - 1):
            model.addConstr(
                soc[t + 1]
                == soc[t] + (p_charge[t] * charge_eff) - (p_discharge[t] / discharge_eff),
                name=f"battery_soc_dynamics_{t}",
            )

        # Daily cyclic constraint: SOC returns to initial at end of each day (24h)
        n_days = horizon // 24
        for day in range(n_days):
            end_hour = (day + 1) * 24 - 1
            if end_hour < horizon:
                model.addConstr(
                    soc[end_hour] == initial_soc,
                    name=f"battery_cyclic_day_{day}",
                )

    # Demand balance constraints (net flows: outgoing - incoming)
    for c in countries:
        for t in range(horizon):
            # Generation in country
            gen_term = gp.quicksum(
                generation[(c, tech, t)]
                for tech in data.tech_capacities.get(c, {})
            )

            # Outgoing flows
            outgoing = gp.quicksum(
                power_flow[(c, to_c, t)]
                for from_c, to_c, _ in data.interconnections
                if from_c == c
            )

            # Incoming flows
            incoming = gp.quicksum(
                power_flow[(from_c, c, t)]
                for from_c, to_c, _ in data.interconnections
                if to_c == c
            )

            # Battery
            if c == data.battery_country and data.battery_config is not None:
                model.addConstr(
                    gen_term + incoming + p_discharge[t] == demands[c][t] + outgoing + p_charge[t],
                    name=f"balance_{c}_{t}",
                )
            else:
                model.addConstr(
                    gen_term + incoming == demands[c][t] + outgoing,
                    name=f"balance_{c}_{t}",
                )

    # Objective: minimize total generation cost
    cost = gp.quicksum(
        float(data.variable_costs.get(c, {}).get(tech, 0.0)) * generation[(c, tech, t)]
        for c in countries
        for tech in data.tech_capacities.get(c, {})
        for t in range(horizon)
    )
    model.setObjective(cost, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return MultiCountryDispatchResult(
            status=model.status,
            objective_value=None,
            generation_by_country={},
            hourly_prices_by_country={},
            power_flows=pd.DataFrame(),
            model=model,
            countries=countries,
        )

    # Extract results
    generation_by_country = {}
    for c in countries:
        gen_df = pd.DataFrame(index=np.arange(horizon))
        for tech in data.tech_capacities.get(c, {}):
            gen_df[tech] = [generation[(c, tech, t)].X for t in range(horizon)]
        gen_df["demand"] = demands[c]
        if c == data.battery_country and data.battery_config is not None:
            gen_df["P_charge"] = [p_charge[t].X for t in range(horizon)]
            gen_df["P_discharge"] = [p_discharge[t].X for t in range(horizon)]
            gen_df["SOC"] = [soc[t].X for t in range(horizon)]
        generation_by_country[c] = gen_df

    # Extract hourly prices
    hourly_prices_by_country = {}
    for c in countries:
        prices = []
        for t in range(horizon):
            constr = model.getConstrByName(f"balance_{c}_{t}")
            prices.append(float(constr.Pi) if constr is not None else np.nan)
        hourly_prices = np.array(prices, dtype=float)
        if clip_negative_prices:
            hourly_prices = np.maximum(hourly_prices, 0.0)
        hourly_prices_by_country[c] = hourly_prices

    # Extract power flows
    flows_data = []
    for from_c, to_c, _ in data.interconnections:
        for t in range(horizon):
            flow_val = power_flow[(from_c, to_c, t)].X
            flows_data.append({
                "time": t,
                "from": from_c,
                "to": to_c,
                "flow_mw": flow_val,
            })
    flows_df = pd.DataFrame(flows_data)

    return MultiCountryDispatchResult(
        status=model.status,
        objective_value=float(model.objVal),
        generation_by_country=generation_by_country,
        hourly_prices_by_country=hourly_prices_by_country,
        power_flows=flows_df,
        model=model,
        countries=countries,
    )
