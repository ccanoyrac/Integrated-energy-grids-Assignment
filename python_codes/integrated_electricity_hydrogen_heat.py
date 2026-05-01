"""Multi-country integrated electricity, hydrogen, and heat network optimization using PyPSA.

Network architecture (per country):
- Electricity bus  ({country}_elec): generators, battery, electricity demand
- Hydrogen bus     ({country}_h2):   H2 storage, H2 demand, electrolyzer output, fuel cell input
- Heat bus         ({country}_heat): heat demand, heat pump output, CHP heat output
- Gas bus          ({country}_gas):  unlimited gas supply at fuel cost (feeds CHP)

Coupling:
- Electrolyzer: Link bus0=elec,  bus1=H2   (electricity → H2, efficiency η_elyz)
- Fuel cell:    Link bus0=H2,    bus1=elec  (H2 → electricity, efficiency η_fc)
- Heat pump:    Link bus0=elec,  bus1=heat  (electricity → heat, efficiency = COP ≥ 1)
- CHP:          Link bus0=gas,   bus1=elec, bus2=heat (gas → electricity + heat)

Three power balance equations enforced by PyPSA (one per bus per hour):

  Electricity: sum_t(g_cth) + bat_dis + η_fc·fc_input + η_chp_e·chp_gas + Σflows_in
               = D_elec + bat_ch + elyz_input + hp_elec_input + Σflows_out

  Hydrogen:    η_elyz·elyz_input + h2s_dis + Σh2pipe_in
               = D_h2 + h2s_ch + fc_input + Σh2pipe_out

  Heat:        Σ_hp(COP_hp · hp_elec_input) + η_chp_h · chp_gas_input
               = D_heat

  Gas:         gas_source = chp_gas_input
               (implicit: unlimited gas generator at chp_fuel_cost)

H2 pipelines are bidirectional: two unidirectional Links added per country pair.

Typical usage from notebook:

    from python_codes.integrated_electricity_hydrogen_heat_copy import (
        optimize_multi_country_integrated_electricity_h2_heat,
        IntegratedElectricityH2HeatResult,
    )

    results = optimize_multi_country_integrated_electricity_h2_heat(
        data=data_dict,
        h2_demand_fraction=0.10,
        heat_demand_fraction=0.15,
        h2_pipeline_capacities={('ES', 'FR'): 1500, ('FR', 'IT'): 2000, ...},
        chp_fuel_cost=87.50,
        chp_fixed_cost=511.0,
        heat_pump_params={
            'HP_air_central': {'cop': 3.0, 'fixed_cost': 8469.0, 'variable_cost': 10.0, 'max_cap': 100_000.0},
            'HP_ground':      {'cop': 3.5, 'fixed_cost': 19763.0, 'variable_cost': 8.57, 'max_cap': 100_000.0},
        },
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

# Default heat pump technologies (costs computed as: annualized * FOM% * 1000 * COP, per MW_elec)
_DEFAULT_HEAT_PUMP_PARAMS: Dict[str, Dict] = {
    "HP_air_central":   {"cop": 3.0, "fixed_cost":  8_469.0, "variable_cost": 10.00, "max_cap": 100_000.0},
    "HP_air_decentral": {"cop": 3.0, "fixed_cost": 12_705.0, "variable_cost": 10.00, "max_cap": 100_000.0},
    "HP_ground":        {"cop": 3.5, "fixed_cost": 19_763.0, "variable_cost":  8.57, "max_cap": 100_000.0},
}


@dataclass
class IntegratedElectricityH2HeatResult:
    """Output container from integrated electricity + H2 + heat optimization."""

    status: str
    objective_value: float
    generation_by_country: Dict[str, pd.DataFrame]   # electricity generation per country
    h2_production_by_country: Dict[str, pd.DataFrame]  # H2 production/dispatch per country
    heat_production_by_country: Dict[str, pd.DataFrame]  # heat supply/dispatch per country
    power_flows: pd.DataFrame    # electricity interconnection flows
    h2_flows: pd.DataFrame       # H2 pipeline flows (both directions combined)
    network: pypsa.Network
    countries: List[str]
    electricity_prices: Dict[str, np.ndarray]
    h2_prices: Dict[str, np.ndarray]
    heat_prices: Dict[str, np.ndarray]
    storage_capacities: Dict[str, Dict[str, float]]


def _get_opt_capacity(component_df: pd.DataFrame, name: str, col: str = "p_nom") -> float:
    """Return optimized capacity, falling back to p_nom if p_nom_opt not available."""
    if name not in component_df.index:
        return 0.0
    opt_col = col.replace("p_nom", "p_nom_opt").replace("e_nom", "e_nom_opt")
    if opt_col in component_df.columns:
        return float(component_df.loc[name, opt_col])
    return float(component_df.loc[name, col])


def optimize_multi_country_integrated_electricity_h2_heat(
    data: Dict,
    h2_demand_fraction: float = 0.10,
    heat_demand_fraction: float = 0.15,
    h2_pipeline_capacities: Optional[Dict[Tuple[str, str], float]] = None,
    # Battery storage
    battery_max_hours: float = 4.0,
    battery_charging_efficiency: float = 0.95,
    battery_discharging_efficiency: float = 0.9,
    battery_standing_loss: float = 0.0,
    battery_fixed_cost: float = 100.0,
    battery_variable_cost: float = 0.0,
    battery_max_capacity_limit: float = 100_000.0,
    # H2 storage
    hydrogen_max_hours: float = 168.0,
    hydrogen_charging_efficiency: float = 0.65,
    hydrogen_discharging_efficiency: float = 0.65,
    hydrogen_standing_loss: float = 0.0,
    hydrogen_fixed_cost: float = 100.0,
    hydrogen_variable_cost: float = 0.0,
    hydrogen_max_capacity_limit: float = 100_000.0,
    # Electrolyzer and fuel cell
    electrolyzer_efficiency: float = 0.65,
    fuel_cell_efficiency: float = 0.65,
    electrolyzer_fixed_cost: float = 500.0,
    fuel_cell_fixed_cost: float = 500.0,
    electrolyzer_max_capacity: float = 100_000.0,
    fuel_cell_max_capacity: float = 100_000.0,
    # CHP (gas → electricity + heat)
    chp_eta_elec: float = 0.40,
    chp_eta_heat: float = 0.40,
    chp_fuel_cost: float = 87.50,
    chp_fixed_cost: float = 511.0,
    chp_max_capacity: float = 100_000.0,
    # Heat pump technologies  {name: {cop, fixed_cost, variable_cost, max_cap}}
    heat_pump_params: Optional[Dict[str, Dict]] = None,
    solver_name: str = "gurobi",
) -> IntegratedElectricityH2HeatResult:
    """Optimize multi-country capacity expansion with electricity, H2, and heat networks.

    Parameters
    ----------
    data : dict
        'demands'            : {country: hourly_demand_array}
        'tech_params'        : {tech: {'variable_cost', 'fixed_cost', 'min_cap', 'max_cap_by_country'}}
        'renewable_profiles' : {country: {tech: hourly_profile_array}}
        'interconnections'   : [(from_country, to_country, ntc_mw), ...]

    h2_demand_fraction : float
        H2 demand as fraction of electricity demand (default 0.10 = 10%).

    heat_demand_fraction : float
        Heat demand as fraction of electricity demand (default 0.15 = 15%).

    h2_pipeline_capacities : dict, optional
        {(from_c, to_c): capacity_mwh_h2_per_h}.
        If None, defaults to 50% of the corresponding electricity NTC.

    chp_eta_elec : float
        CHP electrical efficiency (electricity output / gas input). Default 0.40.

    chp_eta_heat : float
        CHP thermal efficiency (heat output / gas input). Default 0.40.

    chp_fuel_cost : float
        Gas fuel cost for CHP in €/MWh_gas. Default 87.50.

    chp_fixed_cost : float
        Annualized capital cost of CHP in €/MW_gas/year. Default 511.

    heat_pump_params : dict, optional
        {name: {cop, fixed_cost (€/MW_elec/year), variable_cost (€/MWh_elec), max_cap (MW_elec)}}.
        If None, defaults to three HP types (air central, air decentral, ground source).

    solver_name : str
        Solver to use: 'gurobi', 'highs', 'glpk', etc.

    Returns
    -------
    IntegratedElectricityH2HeatResult
    """
    if heat_pump_params is None:
        heat_pump_params = _DEFAULT_HEAT_PUMP_PARAMS.copy()

    print("\n" + "=" * 80)
    print("MULTI-COUNTRY INTEGRATED ELECTRICITY + HYDROGEN + HEAT OPTIMIZATION")
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
    print(f"H2 demand    : {100 * h2_demand_fraction:.1f}% of electricity demand")
    print(f"Heat demand  : {100 * heat_demand_fraction:.1f}% of electricity demand")
    print(f"Heat pumps   : {list(heat_pump_params.keys())}")

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

    # Default H2 pipeline capacities
    if h2_pipeline_capacities is None:
        h2_pipeline_capacities = {(f, t): 0.5 * ntc for f, t, ntc in interconnections}
        print(f"\nH2 pipeline capacities (auto — 50% of electricity NTC):")
    else:
        print(f"\nH2 pipeline capacities (user-defined):")
    for (fc, tc), cap in sorted(h2_pipeline_capacities.items()):
        print(f"  {fc} → {tc}: {cap:,.0f} MWh_H2/h")

    # =========================================================================
    # 2. CREATE PYPSA NETWORK WITH TRI-BUS ARCHITECTURE
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("Building tri-bus network (electricity + hydrogen + heat per country)")
    print(f"{'─' * 80}")

    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2024-01-01", periods=n_hours, freq="h"))

    for country in countries:
        network.add("Bus", f"{country}_elec", carrier="AC")
        network.add("Bus", f"{country}_h2",   carrier="H2")
        network.add("Bus", f"{country}_heat", carrier="heat")
        network.add("Bus", f"{country}_gas",  carrier="gas")
        print(f"  {country}: buses '{country}_elec'  '{country}_h2'  '{country}_heat'  '{country}_gas'")

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

    # Extensible generation for each electricity technology
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
                "Generator", f"gen_{country}_{tech}",
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
            "StorageUnit", f"battery_{country}",
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

    # Electricity interconnection lines
    line_reactances = {
        ("ES", "FR"): 0.12, ("FR", "IT"): 0.095,
        ("ES", "IT"): 0.19, ("ES", "PT"): 0.08,
    }
    for from_c, to_c, ntc_mw in interconnections:
        x = line_reactances.get((from_c, to_c), 0.12)
        network.add(
            "Line", f"line_{from_c}_{to_c}",
            bus0=f"{from_c}_elec", bus1=f"{to_c}_elec",
            x=x, r=0.0, s_nom=ntc_mw,
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

    # Electrolyzers: electricity bus → H2 bus
    for country in countries:
        network.add(
            "Link", f"electrolyzer_{country}",
            bus0=f"{country}_elec", bus1=f"{country}_h2",
            efficiency=electrolyzer_efficiency,
            p_nom_extendable=True, p_nom_min=0.0, p_nom_max=electrolyzer_max_capacity,
            capital_cost=electrolyzer_fixed_cost, marginal_cost=0.0,
        )

    # Fuel cells: H2 bus → electricity bus
    for country in countries:
        network.add(
            "Link", f"fuel_cell_{country}",
            bus0=f"{country}_h2", bus1=f"{country}_elec",
            efficiency=fuel_cell_efficiency,
            p_nom_extendable=True, p_nom_min=0.0, p_nom_max=fuel_cell_max_capacity,
            capital_cost=fuel_cell_fixed_cost, marginal_cost=0.0,
        )

    # H2 storage: StorageUnit on H2 bus
    for country in countries:
        network.add(
            "StorageUnit", f"h2_storage_{country}",
            bus=f"{country}_h2", carrier="H2",
            p_nom_extendable=True, p_nom_min=0.0, p_nom_max=hydrogen_max_capacity_limit,
            max_hours=hydrogen_max_hours,
            efficiency_store=hydrogen_charging_efficiency,
            efficiency_dispatch=hydrogen_discharging_efficiency,
            standing_loss=hydrogen_standing_loss,
            marginal_cost=hydrogen_variable_cost, capital_cost=hydrogen_fixed_cost,
            cyclic_state_of_charge=False,
        )

    # H2 pipelines: bidirectional (two unidirectional Links per pair)
    added_h2_pipes: set = set()
    for from_c, to_c, ntc_mw in interconnections:
        fwd_key = (from_c, to_c)
        if fwd_key not in added_h2_pipes:
            h2_cap_fwd = h2_pipeline_capacities.get(fwd_key, 0.5 * ntc_mw)
            network.add(
                "Link", f"h2_pipe_{from_c}_{to_c}",
                bus0=f"{from_c}_h2", bus1=f"{to_c}_h2",
                p_nom=h2_cap_fwd, p_nom_extendable=False,
                efficiency=1.0, capital_cost=0.0, marginal_cost=0.0,
            )
            added_h2_pipes.add(fwd_key)
        rev_key = (to_c, from_c)
        if rev_key not in added_h2_pipes:
            h2_cap_rev = h2_pipeline_capacities.get(
                rev_key, h2_pipeline_capacities.get(fwd_key, 0.5 * ntc_mw)
            )
            network.add(
                "Link", f"h2_pipe_{to_c}_{from_c}",
                bus0=f"{to_c}_h2", bus1=f"{from_c}_h2",
                p_nom=h2_cap_rev, p_nom_extendable=False,
                efficiency=1.0, capital_cost=0.0, marginal_cost=0.0,
            )
            added_h2_pipes.add(rev_key)

    print(f"  Electrolyzers, fuel cells, H2 storage, H2 pipelines added")

    # =========================================================================
    # 5. HEAT NETWORK: DEMAND + CHP + HEAT PUMPS
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("HEAT NETWORK")
    print(f"{'─' * 80}")

    # Heat demand (inelastic, proportional to electricity demand)
    for country in countries:
        elec_dem = np.asarray(demands[country], dtype=float).reshape(-1)
        heat_dem = heat_demand_fraction * elec_dem
        network.add("Load", f"demand_heat_{country}", bus=f"{country}_heat", p_set=heat_dem)
        print(f"  Heat demand {country}: {heat_dem.sum() / 1e3:,.1f} GWh_th/year "
              f"({100 * heat_demand_fraction:.0f}% of electricity)")

    # Gas source: unlimited supply at fuel cost (feeds CHP)
    # Each country has its own gas bus; the gas Generator acts as infinite supply
    for country in countries:
        network.add(
            "Generator", f"gas_source_{country}",
            bus=f"{country}_gas",
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_nom_max=chp_max_capacity * 2,      # generous upper bound
            marginal_cost=chp_fuel_cost,          # €/MWh_gas
            capital_cost=0.0,                     # investment cost is on the CHP Link
        )

    # CHP: gas bus → electricity bus (bus1) + heat bus (bus2)
    # - p_nom is in MW_gas (gas input capacity)
    # - efficiency  = η_elec (electricity output per unit gas)
    # - efficiency2 = η_heat (heat output per unit gas)
    for country in countries:
        network.add(
            "Link", f"chp_{country}",
            bus0=f"{country}_gas",     # gas input
            bus1=f"{country}_elec",    # electricity output
            bus2=f"{country}_heat",    # heat output
            efficiency=chp_eta_elec,   # η_elec: MW_elec / MW_gas
            efficiency2=chp_eta_heat,  # η_heat: MW_heat / MW_gas
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_nom_max=chp_max_capacity,
            capital_cost=chp_fixed_cost,  # €/MW_gas/year
            marginal_cost=0.0,            # fuel cost on gas_source Generator
        )
        print(f"  CHP {country}: η_e={chp_eta_elec}, η_h={chp_eta_heat}, "
              f"fuel={chp_fuel_cost} €/MWh_gas, capex={chp_fixed_cost:.0f} €/MW_gas/yr")

    # Heat pumps: electricity bus → heat bus (efficiency = COP ≥ 1)
    for hp_name, hp in heat_pump_params.items():
        cop = float(hp["cop"])
        fc = float(hp["fixed_cost"])
        vc = float(hp["variable_cost"])
        mc = float(hp.get("max_cap", 100_000.0))
        for country in countries:
            network.add(
                "Link", f"hp_{hp_name}_{country}",
                bus0=f"{country}_elec",   # electricity input
                bus1=f"{country}_heat",   # heat output
                efficiency=cop,           # COP: MW_heat / MW_elec
                p_nom_extendable=True,
                p_nom_min=0.0,
                p_nom_max=mc,
                capital_cost=fc,          # €/MW_elec/year
                marginal_cost=vc,         # €/MWh_elec additional variable cost
            )
        print(f"  {hp_name}: COP={cop}, capex={fc:.0f} €/MW_elec/yr, "
              f"var={vc} €/MWh_elec")

    # Validate heat demand setup
    print(f"\n  Heat network validation:")
    for country in countries:
        load_name = f"demand_heat_{country}"
        if load_name in network.loads.index:
            total_h = network.loads_t.p_set[load_name].sum()
            peak_h = network.loads_t.p_set[load_name].max()
            print(f"    {country}: annual={total_h / 1e3:.1f} GWh_th, peak={peak_h:.1f} MW_th  [OK]")

    # =========================================================================
    # 6. OPTIMIZE
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
            print(f"Optimization complete (shadow-price warning — PyPSA compatibility).")
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
    # 7. REPORT KEY CAPACITIES
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("OPTIMIZED HEAT INFRASTRUCTURE CAPACITIES")
    print(f"{'─' * 80}")
    for country in countries:
        chp_cap = _get_opt_capacity(network.links, f"chp_{country}")
        bat_cap = _get_opt_capacity(network.storage_units, f"battery_{country}")
        elyz_cap = _get_opt_capacity(network.links, f"electrolyzer_{country}")
        h2s_cap = _get_opt_capacity(network.storage_units, f"h2_storage_{country}")
        heat_annual = np.asarray(demands[country], dtype=float).reshape(-1).sum() * heat_demand_fraction
        print(f"\n  {country}:")
        print(f"    Heat demand (annual)   : {heat_annual / 1e3:,.1f} GWh_th")
        print(f"    CHP capacity (MW_gas)  : {chp_cap:,.1f}")
        for hp_name in heat_pump_params:
            hp_cap = _get_opt_capacity(network.links, f"hp_{hp_name}_{country}")
            print(f"    {hp_name} (MW_elec): {hp_cap:,.1f}")
        print(f"    Battery (MW)           : {bat_cap:,.1f}")
        print(f"    Electrolyzer (MW_elec) : {elyz_cap:,.1f}")
        print(f"    H2 storage (MW_H2)     : {h2s_cap:,.1f}")

    # =========================================================================
    # 8. EXTRACT RESULTS
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("EXTRACTING RESULTS")
    print(f"{'─' * 80}")

    # --- Electricity generation by country ---
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
        # CHP electricity output (p1 of CHP Link, negative convention → abs)
        chp_name = f"chp_{country}"
        if chp_name in network.links_t.p1.columns:
            df["CHP_elec_output"] = network.links_t.p1[chp_name].abs()
        else:
            df["CHP_elec_output"] = 0.0

        batt_name = f"battery_{country}"
        if batt_name in network.storage_units_t.p.columns:
            df["Battery_discharge"] = network.storage_units_t.p[batt_name].clip(lower=0)
            df["Battery_charge"] = network.storage_units_t.p[batt_name].clip(upper=0).abs()
        df["demand_elec"] = demands[country]
        generation_by_country[country] = df

    # --- H2 production / dispatch by country ---
    h2_production_by_country: Dict[str, pd.DataFrame] = {}
    for country in countries:
        df = pd.DataFrame(index=network.snapshots)
        elyz_name = f"electrolyzer_{country}"
        df["Electrolyzer_H2_output"] = (
            network.links_t.p1[elyz_name].abs()
            if elyz_name in network.links_t.p1.columns else 0.0
        )
        df["Electrolyzer_elec_input"] = (
            network.links_t.p0[elyz_name]
            if elyz_name in network.links_t.p0.columns else 0.0
        )
        fc_name = f"fuel_cell_{country}"
        df["FuelCell_H2_input"] = (
            network.links_t.p0[fc_name]
            if fc_name in network.links_t.p0.columns else 0.0
        )
        df["FuelCell_elec_output"] = (
            network.links_t.p1[fc_name].abs()
            if fc_name in network.links_t.p1.columns else 0.0
        )
        h2s_name = f"h2_storage_{country}"
        if h2s_name in network.storage_units_t.p.columns:
            df["H2_storage_discharge"] = network.storage_units_t.p[h2s_name].clip(lower=0)
            df["H2_storage_charge"] = network.storage_units_t.p[h2s_name].clip(upper=0).abs()
        else:
            df["H2_storage_discharge"] = 0.0
            df["H2_storage_charge"] = 0.0
        df["demand_h2"] = h2_demand_fraction * demands[country]
        h2_production_by_country[country] = df

    # --- Heat production / dispatch by country ---
    heat_production_by_country: Dict[str, pd.DataFrame] = {}
    for country in countries:
        df = pd.DataFrame(index=network.snapshots)
        # CHP heat output (p2 of CHP Link, negative convention → abs)
        chp_name = f"chp_{country}"
        if chp_name in network.links_t.p2.columns:
            df["CHP_gas_input"] = network.links_t.p0[chp_name] if chp_name in network.links_t.p0.columns else 0.0
            df["CHP_heat_output"] = network.links_t.p2[chp_name].abs()
        else:
            df["CHP_gas_input"] = 0.0
            df["CHP_heat_output"] = 0.0
        # Heat pump outputs
        for hp_name in heat_pump_params:
            link_name = f"hp_{hp_name}_{country}"
            df[f"{hp_name}_elec_input"] = (
                network.links_t.p0[link_name]
                if link_name in network.links_t.p0.columns else 0.0
            )
            df[f"{hp_name}_heat_output"] = (
                network.links_t.p1[link_name].abs()
                if link_name in network.links_t.p1.columns else 0.0
            )
        df["demand_heat"] = heat_demand_fraction * demands[country]
        heat_production_by_country[country] = df

    # --- Electricity flows ---
    flows_elec: List[Dict] = []
    for from_c, to_c, _ in interconnections:
        line_name = f"line_{from_c}_{to_c}"
        if line_name in network.lines_t.p0.columns:
            for h, flow in enumerate(network.lines_t.p0[line_name].values):
                flows_elec.append({"hour": h, "from": from_c, "to": to_c, "flow_mw": flow})
    flows_df = pd.DataFrame(flows_elec)

    # --- H2 flows ---
    flows_h2: List[Dict] = []
    for from_c, to_c, _ in interconnections:
        fwd_pipe = f"h2_pipe_{from_c}_{to_c}"
        rev_pipe = f"h2_pipe_{to_c}_{from_c}"
        fwd_flows = (
            network.links_t.p0[fwd_pipe].values
            if fwd_pipe in network.links_t.p0.columns else np.zeros(n_hours)
        )
        rev_flows = (
            network.links_t.p0[rev_pipe].values
            if rev_pipe in network.links_t.p0.columns else np.zeros(n_hours)
        )
        for h, (f_fwd, f_rev) in enumerate(zip(fwd_flows, rev_flows)):
            flows_h2.append({
                "hour": h, "from": from_c, "to": to_c,
                "flow_mwh_h2": f_fwd,
                "flow_mwh_h2_reverse": f_rev,
                "flow_mwh_h2_net": f_fwd - f_rev,
            })
    flows_h2_df = pd.DataFrame(flows_h2)

    # --- Marginal prices ---
    elec_prices: Dict[str, np.ndarray] = {}
    h2_prices:   Dict[str, np.ndarray] = {}
    heat_prices: Dict[str, np.ndarray] = {}
    for country in countries:
        elec_col = f"{country}_elec"
        h2_col   = f"{country}_h2"
        heat_col = f"{country}_heat"
        if hasattr(network.buses_t, "marginal_price"):
            mp = network.buses_t.marginal_price
            elec_prices[country]  = mp[elec_col].values  if elec_col  in mp.columns else np.full(n_hours, np.nan)
            h2_prices[country]    = mp[h2_col].values    if h2_col    in mp.columns else np.full(n_hours, np.nan)
            heat_prices[country]  = mp[heat_col].values  if heat_col  in mp.columns else np.full(n_hours, np.nan)
        else:
            elec_prices[country]  = np.full(n_hours, np.nan)
            h2_prices[country]    = np.full(n_hours, np.nan)
            heat_prices[country]  = np.full(n_hours, np.nan)

    # --- Storage and conversion capacities ---
    storage_caps: Dict[str, Dict[str, float]] = {}
    for country in countries:
        bat_power  = _get_opt_capacity(network.storage_units, f"battery_{country}")
        h2s_power  = _get_opt_capacity(network.storage_units, f"h2_storage_{country}")
        elyz_cap   = _get_opt_capacity(network.links,         f"electrolyzer_{country}")
        fc_cap     = _get_opt_capacity(network.links,         f"fuel_cell_{country}")
        chp_cap    = _get_opt_capacity(network.links,         f"chp_{country}")

        caps: Dict[str, float] = {
            "Battery_power":    bat_power,
            "Battery_energy":   bat_power * battery_max_hours,
            "H2_storage_power": h2s_power,
            "H2_storage_energy": h2s_power * hydrogen_max_hours,
            "Electrolyzer":     elyz_cap,
            "Fuel_cell":        fc_cap,
            "CHP_gas_cap":      chp_cap,
            "CHP_elec_cap":     chp_cap * chp_eta_elec,
            "CHP_heat_cap":     chp_cap * chp_eta_heat,
        }
        for hp_name in heat_pump_params:
            hp_cap = _get_opt_capacity(network.links, f"hp_{hp_name}_{country}")
            caps[f"{hp_name}_elec_cap"] = hp_cap
            caps[f"{hp_name}_heat_cap"] = hp_cap * heat_pump_params[hp_name]["cop"]
        storage_caps[country] = caps

    print("  Results extracted successfully.\n")

    return IntegratedElectricityH2HeatResult(
        status="optimal" if optimize_successful else "unknown",
        objective_value=float(network.objective) if hasattr(network, "objective") else np.nan,
        generation_by_country=generation_by_country,
        h2_production_by_country=h2_production_by_country,
        heat_production_by_country=heat_production_by_country,
        power_flows=flows_df,
        h2_flows=flows_h2_df,
        network=network,
        countries=countries,
        electricity_prices=elec_prices,
        h2_prices=h2_prices,
        heat_prices=heat_prices,
        storage_capacities=storage_caps,
    )
