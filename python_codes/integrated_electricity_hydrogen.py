"""Multi-country integrated electricity and hydrogen network optimization using PyPSA.

This module implements the mathematical model described in IEG_Electricity_H2_network copy.ipynb.

Network architecture (per country):
- Electricity bus  ({country}_elec): generators, battery, electricity demand
- Hydrogen bus     ({country}_h2):   H2 storage, H2 demand, electrolyzer output, fuel cell input

Coupling (H2 technology):
- Electrolyzer: Link bus0=elec, bus1=H2  (electricity → H2, efficiency η_elyz)
- Fuel cell:    Link bus0=H2,  bus1=elec (H2 → electricity, efficiency η_fc)
- H2 storage:   StorageUnit on H2 bus
  - Charges ONLY from electricity (via electrolyzer — the only H2 source)
  - Discharges to H2 bus → (a) H2 demand directly, or (b) fuel cell → electricity bus

Two power balance equations enforced by PyPSA (one per bus per hour):

  Electricity: sum_t(g_cth) + bat_dis + η_fc * fc_input + Σflows_in
               = D_elec + bat_ch + elyz_input + Σflows_out

  Hydrogen:    η_elyz * elyz_input + h2s_dis + Σh2pipe_in
               = D_h2 + h2s_ch + fc_input + Σh2pipe_out

H2 pipelines are bidirectional: two unidirectional Links added per country pair.

Typical usage from notebook:

    from python_codes.integrated_electricity_hydrogen import (
        optimize_multi_country_integrated_electricity_h2,
        IntegratedElectricityH2Result,
    )

    results = optimize_multi_country_integrated_electricity_h2(
        data=data_dict,
        h2_demand_fraction=0.10,
        h2_pipeline_capacities={('ES', 'FR'): 1500, ('FR', 'IT'): 2000, ...},
    )
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    conda_env_path = Path(sys.prefix)
    proj_lib = conda_env_path / "Library" / "share" / "proj"
    if proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pypsa
except ImportError as exc:
    raise ImportError("pypsa is required. Install with: pip install pypsa") from exc

warnings.filterwarnings("ignore")


@dataclass
class IntegratedElectricityH2Result:
    """Output container from integrated electricity + H2 optimization."""

    status: str
    objective_value: float
    generation_by_country: Dict[str, pd.DataFrame]   # electricity generation per country
    h2_production_by_country: Dict[str, pd.DataFrame]  # H2 production/dispatch per country
    power_flows: pd.DataFrame   # electricity interconnection flows
    h2_flows: pd.DataFrame      # H2 pipeline flows (both directions combined)
    network: pypsa.Network
    countries: List[str]
    electricity_prices: Dict[str, np.ndarray]
    h2_prices: Dict[str, np.ndarray]
    storage_capacities: Dict[str, Dict[str, float]]


def _get_opt_capacity(component_df: pd.DataFrame, name: str, col: str = "p_nom") -> float:
    """Return optimized capacity, falling back to p_nom if p_nom_opt not available."""
    if name not in component_df.index:
        return 0.0
    opt_col = col.replace("p_nom", "p_nom_opt").replace("e_nom", "e_nom_opt")
    if opt_col in component_df.columns:
        return float(component_df.loc[name, opt_col])
    return float(component_df.loc[name, col])


def optimize_multi_country_integrated_electricity_h2(
    data: Dict,
    h2_demand_fraction: float = 0.10,
    h2_pipeline_capacities: Optional[Dict[Tuple[str, str], float]] = None,
    battery_max_hours: float = 4.0,
    battery_charging_efficiency: float = 0.95,
    battery_discharging_efficiency: float = 0.9,
    battery_standing_loss: float = 0.0,
    battery_fixed_cost: float = 100.0,
    battery_variable_cost: float = 0.0,
    battery_max_capacity_limit: float = 100_000.0,
    hydrogen_max_hours: float = 168.0,
    hydrogen_charging_efficiency: float = 0.65,
    hydrogen_discharging_efficiency: float = 0.65,
    hydrogen_standing_loss: float = 0.0,
    hydrogen_fixed_cost: float = 100.0,
    hydrogen_variable_cost: float = 0.0,
    hydrogen_max_capacity_limit: float = 100_000.0,
    electrolyzer_efficiency: float = 0.65,
    fuel_cell_efficiency: float = 0.65,
    electrolyzer_fixed_cost: float = 500.0,
    fuel_cell_fixed_cost: float = 500.0,
    electrolyzer_max_capacity: float = 100_000.0,
    fuel_cell_max_capacity: float = 100_000.0,
    solver_name: str = "gurobi",
) -> IntegratedElectricityH2Result:
    """Optimize multi-country capacity expansion with integrated electricity and H2 networks.

    See module docstring and notebook mathematical model for full formulation details.

    Parameters
    ----------
    data : dict
        'demands'            : {country: hourly_demand_array}
        'tech_params'        : {tech: {'variable_cost', 'fixed_cost', 'min_cap', 'max_cap_by_country'}}
        'renewable_profiles' : {country: {tech: hourly_profile_array}}
        'interconnections'   : [(from_country, to_country, ntc_mw), ...]

    h2_demand_fraction : float
        H2 demand as fraction of electricity demand (default 0.10 = 10%).

    h2_pipeline_capacities : dict, optional
        {(from_c, to_c): capacity_mwh_h2_per_h}.
        Bidirectional: reverse direction uses the same value unless explicitly set.
        If None, defaults to 50% of the corresponding electricity NTC.

    battery_max_hours, battery_*_efficiency, battery_fixed_cost, battery_max_capacity_limit
        Battery StorageUnit parameters.

    hydrogen_max_hours, hydrogen_*_efficiency, hydrogen_fixed_cost, hydrogen_max_capacity_limit
        H2 StorageUnit parameters.

    electrolyzer_efficiency : float
        Electrolyzer conversion efficiency η_elyz (electricity → H2).

    fuel_cell_efficiency : float
        Fuel cell conversion efficiency η_fc (H2 → electricity).

    electrolyzer_fixed_cost, fuel_cell_fixed_cost : float
        Annualized capital costs for conversion equipment (€/MW/year).

    electrolyzer_max_capacity, fuel_cell_max_capacity : float
        Upper bounds on extendable converter capacities (MW).

    solver_name : str
        Solver to use: 'gurobi', 'highs', 'glpk', etc.

    Returns
    -------
    IntegratedElectricityH2Result
    """
    print("\n" + "=" * 80)
    print("MULTI-COUNTRY INTEGRATED ELECTRICITY + HYDROGEN OPTIMIZATION")
    print("=" * 80)

    # =========================================================================
    # 1. EXTRACT AND VALIDATE INPUT DATA
    # =========================================================================
    demands = data["demands"]
    tech_params = data["tech_params"]
    renewable_profiles = data["renewable_profiles"]
    interconnections = data["interconnections"]

    countries = sorted(demands.keys())
    technologies = sorted(tech_params.keys())

    print(f"\nCountries    : {countries}")
    print(f"Technologies : {technologies}")
    print(f"H2 demand    : {100 * h2_demand_fraction:.1f}% of electricity demand per country per hour")

    # Synchronize all demand/profile arrays to the shortest length
    demand_lengths = {c: np.asarray(demands[c], dtype=float).reshape(-1).size for c in countries}
    min_length = min(demand_lengths.values())
    max_length = max(demand_lengths.values())

    if min_length != max_length:
        print(f"\nWARNING: Demand length mismatch — truncating all to {min_length} hours")
        demands = {c: np.asarray(demands[c], dtype=float).reshape(-1)[:min_length] for c in countries}
        renewable_profiles = {
            c: {t: np.asarray(p, dtype=float).reshape(-1)[:min_length]
                for t, p in renewable_profiles[c].items()}
            for c in countries
        }

    n_hours = min_length
    print(f"Time horizon : {n_hours} hours")

    # Default H2 pipeline capacities (50% of electricity NTC if not provided)
    if h2_pipeline_capacities is None:
        h2_pipeline_capacities = {(f, t): 0.5 * ntc for f, t, ntc in interconnections}
        print(f"\nH2 pipeline capacities (auto — 50% of electricity NTC):")
    else:
        print(f"\nH2 pipeline capacities (user-defined):")

    for (fc, tc), cap in sorted(h2_pipeline_capacities.items()):
        print(f"  {fc} → {tc}: {cap:,.0f} MWh_H2/h")

    # =========================================================================
    # 2. CREATE PYPSA NETWORK WITH DUAL-BUS ARCHITECTURE
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("Building dual-bus network (electricity + hydrogen per country)")
    print(f"{'─' * 80}")

    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2024-01-01", periods=n_hours, freq="h"))

    # One electricity bus + one H2 bus per country
    for country in countries:
        network.add("Bus", f"{country}_elec", carrier="AC")
        network.add("Bus", f"{country}_h2", carrier="H2")
        print(f"  {country}: buses '{country}_elec'  '{country}_h2'")

    # =========================================================================
    # 3. ELECTRICITY NETWORK: DEMAND + GENERATORS + BATTERY + INTERCONNECTIONS
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("ELECTRICITY NETWORK")
    print(f"{'─' * 80}")

    # Electricity demand (inelastic)
    for country in countries:
        demand_arr = np.asarray(demands[country], dtype=float).reshape(-1)
        network.add("Load", f"demand_elec_{country}", bus=f"{country}_elec", p_set=demand_arr)
        print(f"  Elec demand {country}: {demand_arr.sum() / 1e3:,.1f} GWh/year")

    # Extensible generation for each technology in each country
    for tech in technologies:
        td = tech_params[tech]
        variable_cost = td.get("variable_cost", 0.0)
        fixed_cost = td.get("fixed_cost", 0.0)
        min_cap = td.get("min_cap", 0.0)
        max_cap_by_country = td.get("max_cap_by_country", {})

        for country in countries:
            max_cap = (
                max_cap_by_country.get(country, 50_000.0)
                if isinstance(max_cap_by_country, dict)
                else 50_000.0
            )

            if tech in renewable_profiles.get(country, {}):
                p_max_pu = np.asarray(renewable_profiles[country][tech], dtype=float).reshape(-1)
                if p_max_pu.size != n_hours:
                    raise ValueError(
                        f"Profile length mismatch for {country}/{tech}: "
                        f"expected {n_hours}, got {p_max_pu.size}"
                    )
            else:
                p_max_pu = 1.0

            network.add(
                "Generator",
                f"gen_{country}_{tech}",
                bus=f"{country}_elec",
                carrier="AC",
                p_nom_extendable=True,
                p_nom_min=min_cap,
                p_nom_max=max_cap,
                p_max_pu=p_max_pu,
                marginal_cost=variable_cost,
                capital_cost=fixed_cost,
            )

    # Battery storage (on electricity bus)
    for country in countries:
        network.add(
            "StorageUnit",
            f"battery_{country}",
            bus=f"{country}_elec",
            carrier="AC",
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_nom_max=battery_max_capacity_limit,
            max_hours=battery_max_hours,
            efficiency_store=battery_charging_efficiency,
            efficiency_dispatch=battery_discharging_efficiency,
            standing_loss=battery_standing_loss,
            marginal_cost=battery_variable_cost,
            capital_cost=battery_fixed_cost,
            cyclic_state_of_charge=False,
        )

    # Electricity interconnection lines (bidirectional by PyPSA Line convention)
    line_reactances = {
        ("ES", "FR"): 0.12,
        ("FR", "IT"): 0.095,
        ("ES", "IT"): 0.19,
        ("ES", "PT"): 0.08,
    }
    for from_c, to_c, ntc_mw in interconnections:
        x = line_reactances.get((from_c, to_c), 0.12)
        network.add(
            "Line",
            f"line_{from_c}_{to_c}",
            bus0=f"{from_c}_elec",
            bus1=f"{to_c}_elec",
            x=x,
            r=0.0,
            s_nom=ntc_mw,
        )

    print(f"  Generators + battery added for all countries")
    print(f"  Electricity interconnections added ({len(interconnections)} lines, bidirectional)")

    # =========================================================================
    # 4. HYDROGEN NETWORK: DEMAND + ELECTROLYZERS + FUEL CELLS + STORAGE + PIPELINES
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("HYDROGEN NETWORK")
    print(f"{'─' * 80}")

    # H2 demand (inelastic, proportional to electricity demand)
    for country in countries:
        elec_dem = np.asarray(demands[country], dtype=float).reshape(-1)
        h2_dem = h2_demand_fraction * elec_dem
        network.add("Load", f"demand_h2_{country}", bus=f"{country}_h2", p_set=h2_dem)
        print(f"  H2 demand {country}: {h2_dem.sum() / 1e3:,.1f} GWh_H2/year "
              f"({100 * h2_demand_fraction:.0f}% of electricity)")

    # Electrolyzers: electricity bus → H2 bus  (electricity → H2)
    # Appear on DEMAND side of electricity balance and SUPPLY side of H2 balance
    for country in countries:
        network.add(
            "Link",
            f"electrolyzer_{country}",
            bus0=f"{country}_elec",   # consumes electricity
            bus1=f"{country}_h2",     # produces H2
            efficiency=electrolyzer_efficiency,
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_nom_max=electrolyzer_max_capacity,
            capital_cost=electrolyzer_fixed_cost,
            marginal_cost=0.0,
        )

    # Fuel cells: H2 bus → electricity bus  (H2 → electricity)
    # Appear on DEMAND side of H2 balance and SUPPLY side of electricity balance
    for country in countries:
        network.add(
            "Link",
            f"fuel_cell_{country}",
            bus0=f"{country}_h2",      # consumes H2
            bus1=f"{country}_elec",    # produces electricity
            efficiency=fuel_cell_efficiency,
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_nom_max=fuel_cell_max_capacity,
            capital_cost=fuel_cell_fixed_cost,
            marginal_cost=0.0,
        )

    # H2 storage: StorageUnit on H2 bus
    # - Charging: from H2 bus (H2 produced by electrolyzers — the only H2 source)
    # - Discharging: to H2 bus → supplies H2 demand OR fuel cell → electricity bus
    for country in countries:
        network.add(
            "StorageUnit",
            f"h2_storage_{country}",
            bus=f"{country}_h2",
            carrier="H2",
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_nom_max=hydrogen_max_capacity_limit,
            max_hours=hydrogen_max_hours,
            efficiency_store=hydrogen_charging_efficiency,
            efficiency_dispatch=hydrogen_discharging_efficiency,
            standing_loss=hydrogen_standing_loss,
            marginal_cost=hydrogen_variable_cost,
            capital_cost=hydrogen_fixed_cost,
            cyclic_state_of_charge=False,
        )

    # H2 pipelines: bidirectional — add one Link per direction
    # The H2 pipeline capacities dict may have only the "forward" direction;
    # we use the same capacity for the reverse direction unless explicitly set.
    added_h2_pipes = set()
    for from_c, to_c, ntc_mw in interconnections:
        # Forward direction
        fwd_key = (from_c, to_c)
        if fwd_key not in added_h2_pipes:
            h2_cap_fwd = h2_pipeline_capacities.get(fwd_key, 0.5 * ntc_mw)
            network.add(
                "Link",
                f"h2_pipe_{from_c}_{to_c}",
                bus0=f"{from_c}_h2",
                bus1=f"{to_c}_h2",
                p_nom=h2_cap_fwd,
                p_nom_extendable=False,
                efficiency=1.0,
                capital_cost=0.0,
                marginal_cost=0.0,
            )
            added_h2_pipes.add(fwd_key)

        # Reverse direction
        rev_key = (to_c, from_c)
        if rev_key not in added_h2_pipes:
            h2_cap_rev = h2_pipeline_capacities.get(rev_key, h2_pipeline_capacities.get(fwd_key, 0.5 * ntc_mw))
            network.add(
                "Link",
                f"h2_pipe_{to_c}_{from_c}",
                bus0=f"{to_c}_h2",
                bus1=f"{from_c}_h2",
                p_nom=h2_cap_rev,
                p_nom_extendable=False,
                efficiency=1.0,
                capital_cost=0.0,
                marginal_cost=0.0,
            )
            added_h2_pipes.add(rev_key)

    print(f"\n  Electrolyzers (elec → H2) added for all countries")
    print(f"  Fuel cells    (H2 → elec) added for all countries")
    print(f"  H2 storage    (on H2 bus) added for all countries")
    print(f"  H2 pipelines  added ({len(added_h2_pipes)} directions, bidirectional)")

    # Validate H2 demand setup
    print(f"\n  H2 network validation:")
    for country in countries:
        load_name = f"demand_h2_{country}"
        if load_name in network.loads.index:
            total_h2 = network.loads_t.p_set[load_name].sum()
            peak_h2 = network.loads_t.p_set[load_name].max()
            print(f"    {country}: annual={total_h2 / 1e3:.1f} GWh_H2, peak={peak_h2:.1f} MWh_H2/h  [OK]")
        else:
            print(f"    {country}: H2 demand load NOT FOUND — check setup!")

    # =========================================================================
    # 5. OPTIMIZE
    # =========================================================================
    print(f"\n{'─' * 80}")
    print(f"RUNNING OPTIMIZATION  (solver: {solver_name})")
    print(f"{'─' * 80}\n")

    optimize_successful = False
    try:
        network.optimize(
            solver_name=solver_name,
            log_to_console=(solver_name == "gurobi"),
        )
        print(f"\nOptimization complete.")
        optimize_successful = True
    except AttributeError as e:
        msg = str(e)
        if "shadow-prices" in msg or "was not assigned" in msg:
            print(f"Optimization complete (shadow-price assignment warning — PyPSA compatibility).")
            print(f"  Objective = {network.objective:.4e}")
            optimize_successful = True
        else:
            raise
    except Exception as e:
        print(f"WARNING: {type(e).__name__}: {e!s:.200} — retrying...")
        try:
            network.optimize(solver_name=solver_name)
            print("Optimization complete (retry succeeded).")
            optimize_successful = True
        except Exception as e2:
            raise RuntimeError(f"Optimization failed: {e2}") from e2

    if not optimize_successful:
        raise RuntimeError("Optimization did not complete successfully.")

    # =========================================================================
    # 6. REPORT KEY CAPACITIES
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("OPTIMIZED H2 INFRASTRUCTURE CAPACITIES")
    print(f"{'─' * 80}")
    for country in countries:
        elyz_cap = _get_opt_capacity(network.links, f"electrolyzer_{country}")
        fc_cap = _get_opt_capacity(network.links, f"fuel_cell_{country}")
        h2s_cap = _get_opt_capacity(network.storage_units, f"h2_storage_{country}")
        bat_cap = _get_opt_capacity(network.storage_units, f"battery_{country}")
        h2_annual = demands[country].sum() * h2_demand_fraction
        print(f"\n  {country}:")
        print(f"    H2 demand (annual)   : {h2_annual / 1e3:,.1f} GWh_H2")
        print(f"    Electrolyzer capacity: {elyz_cap:,.1f} MW_elec")
        print(f"    Fuel cell capacity   : {fc_cap:,.1f} MW_H2")
        print(f"    H2 storage power     : {h2s_cap:,.1f} MW_H2")
        print(f"    Battery power        : {bat_cap:,.1f} MW")
        if elyz_cap < 1 and h2_annual > 0:
            print(f"    WARNING: H2 demand exists but electrolyzer is near-zero!")

    # =========================================================================
    # 7. EXTRACT RESULTS
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("EXTRACTING RESULTS")
    print(f"{'─' * 80}")

    # Electricity generation by country
    generation_by_country: Dict[str, pd.DataFrame] = {}
    for country in countries:
        df = pd.DataFrame(index=network.snapshots)
        for tech in technologies:
            gen_name = f"gen_{country}_{tech}"
            df[tech] = (
                network.generators_t.p[gen_name]
                if gen_name in network.generators_t.p.columns
                else 0.0
            )
        batt_name = f"battery_{country}"
        if batt_name in network.storage_units_t.p.columns:
            df["Battery_discharge"] = network.storage_units_t.p[batt_name].clip(lower=0)
            df["Battery_charge"] = network.storage_units_t.p[batt_name].clip(upper=0).abs()
        df["demand_elec"] = demands[country]
        generation_by_country[country] = df

    # H2 production / dispatch by country
    h2_production_by_country: Dict[str, pd.DataFrame] = {}
    for country in countries:
        df = pd.DataFrame(index=network.snapshots)

        # Electrolyzer H2 output = p1 (negative convention in Links, take absolute)
        elyz_name = f"electrolyzer_{country}"
        if elyz_name in network.links_t.p1.columns:
            df["Electrolyzer_H2_output"] = network.links_t.p1[elyz_name].abs()
        else:
            df["Electrolyzer_H2_output"] = 0.0

        # Electrolyzer electricity input = p0
        if elyz_name in network.links_t.p0.columns:
            df["Electrolyzer_elec_input"] = network.links_t.p0[elyz_name]
        else:
            df["Electrolyzer_elec_input"] = 0.0

        # Fuel cell H2 input = p0
        fc_name = f"fuel_cell_{country}"
        if fc_name in network.links_t.p0.columns:
            df["FuelCell_H2_input"] = network.links_t.p0[fc_name]
        else:
            df["FuelCell_H2_input"] = 0.0

        # Fuel cell electricity output = p1 (negative convention, take absolute)
        if fc_name in network.links_t.p1.columns:
            df["FuelCell_elec_output"] = network.links_t.p1[fc_name].abs()
        else:
            df["FuelCell_elec_output"] = 0.0

        # H2 storage dispatch (positive = discharge, negative = charge)
        h2s_name = f"h2_storage_{country}"
        if h2s_name in network.storage_units_t.p.columns:
            df["H2_storage_discharge"] = network.storage_units_t.p[h2s_name].clip(lower=0)
            df["H2_storage_charge"] = network.storage_units_t.p[h2s_name].clip(upper=0).abs()
        else:
            df["H2_storage_discharge"] = 0.0
            df["H2_storage_charge"] = 0.0

        df["demand_h2"] = h2_demand_fraction * demands[country]
        h2_production_by_country[country] = df

    # Electricity flows
    flows_elec: List[Dict] = []
    for from_c, to_c, _ in interconnections:
        line_name = f"line_{from_c}_{to_c}"
        if line_name in network.lines_t.p0.columns:
            for h, flow in enumerate(network.lines_t.p0[line_name].values):
                flows_elec.append({"hour": h, "from": from_c, "to": to_c, "flow_mw": flow})
    flows_df = pd.DataFrame(flows_elec)

    # H2 flows (both directions combined; net flow = fwd - rev)
    flows_h2: List[Dict] = []
    # Build unique set of forward interconnections
    for from_c, to_c, _ in interconnections:
        fwd_pipe = f"h2_pipe_{from_c}_{to_c}"
        rev_pipe = f"h2_pipe_{to_c}_{from_c}"

        fwd_flows = (
            network.links_t.p0[fwd_pipe].values
            if fwd_pipe in network.links_t.p0.columns
            else np.zeros(n_hours)
        )
        rev_flows = (
            network.links_t.p0[rev_pipe].values
            if rev_pipe in network.links_t.p0.columns
            else np.zeros(n_hours)
        )

        for h, (f_fwd, f_rev) in enumerate(zip(fwd_flows, rev_flows)):
            # Report net flow in the forward direction; also store individual directions
            flows_h2.append({
                "hour": h,
                "from": from_c,
                "to": to_c,
                "flow_mwh_h2": f_fwd,          # forward direction flow
                "flow_mwh_h2_reverse": f_rev,   # reverse direction flow
                "flow_mwh_h2_net": f_fwd - f_rev,  # net forward flow
            })
    flows_h2_df = pd.DataFrame(flows_h2)

    # Marginal prices
    elec_prices: Dict[str, np.ndarray] = {}
    h2_prices: Dict[str, np.ndarray] = {}
    for country in countries:
        elec_col = f"{country}_elec"
        h2_col = f"{country}_h2"
        if hasattr(network.buses_t, "marginal_price"):
            mp = network.buses_t.marginal_price
            elec_prices[country] = mp[elec_col].values if elec_col in mp.columns else np.full(n_hours, np.nan)
            h2_prices[country] = mp[h2_col].values if h2_col in mp.columns else np.full(n_hours, np.nan)
        else:
            elec_prices[country] = np.full(n_hours, np.nan)
            h2_prices[country] = np.full(n_hours, np.nan)

    # Storage and conversion capacities (use p_nom_opt for extendable components)
    storage_caps: Dict[str, Dict[str, float]] = {}
    for country in countries:
        bat_name = f"battery_{country}"
        h2s_name = f"h2_storage_{country}"
        elyz_name = f"electrolyzer_{country}"
        fc_name = f"fuel_cell_{country}"

        bat_power = _get_opt_capacity(network.storage_units, bat_name)
        bat_energy = bat_power * battery_max_hours  # energy = power × max_hours

        h2s_power = _get_opt_capacity(network.storage_units, h2s_name)
        h2s_energy = h2s_power * hydrogen_max_hours

        elyz_cap = _get_opt_capacity(network.links, elyz_name)
        fc_cap = _get_opt_capacity(network.links, fc_name)

        storage_caps[country] = {
            "Battery_power": bat_power,
            "Battery_energy": bat_energy,
            "H2_storage_power": h2s_power,
            "H2_storage_energy": h2s_energy,
            "Electrolyzer": elyz_cap,
            "Fuel_cell": fc_cap,
        }

    print("  Results extracted successfully.\n")

    return IntegratedElectricityH2Result(
        status="optimal" if optimize_successful else "unknown",
        objective_value=float(network.objective) if hasattr(network, "objective") else np.nan,
        generation_by_country=generation_by_country,
        h2_production_by_country=h2_production_by_country,
        power_flows=flows_df,
        h2_flows=flows_h2_df,
        network=network,
        countries=countries,
        electricity_prices=elec_prices,
        h2_prices=h2_prices,
        storage_capacities=storage_caps,
    )
