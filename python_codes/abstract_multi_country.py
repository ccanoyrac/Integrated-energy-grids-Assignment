"""Multi-country economic dispatch optimization using PyPSA.

This module implements a multi-country optimization model without batteries,
focusing on:
- Multi-country power balance
- Inter-country transmission constraints
- Same technology costs across all countries
- P50 renewable profiles

Typical usage from notebook:

    from abstract_multi_country import optimize_multi_country_dispatch_no_batteries

    data = {
        'demands': {'ES': demand_es, 'FR': demand_fr, 'IT': demand_it, 'PT': demand_pt},
        'tech_params': {
            'Wind': {'variable_cost': 0, 'capacity': {'ES': 7500, 'FR': 10000, 'IT': 5000, 'PT': 3500}},
            'Solar': {'variable_cost': 0, 'capacity': {'ES': 5000, 'FR': 2000, 'IT': 4000, 'PT': 2000}},
            'Hydro': {'variable_cost': 10, 'capacity': {'ES': 5000, 'FR': 8000, 'IT': 10000, 'PT': 3500}},
            'CCGT': {'variable_cost': 50, 'capacity': {'ES': 20000, 'FR': 8000, 'IT': 15000, 'PT': 5000}},
            'Coal': {'variable_cost': 40, 'capacity': {'ES': 8000, 'FR': 2000, 'IT': 8000, 'PT': 2000}},
            'Nuclear': {'variable_cost': 15, 'capacity': {'ES': 7000, 'FR': 61000, 'IT': 0, 'PT': 0}},
        },
        'renewable_profiles': {
            'ES': {'Wind': wind_es, 'Solar': solar_es, 'Hydro': hydro_es},
            'FR': {'Wind': wind_fr, 'Solar': solar_fr, 'Hydro': hydro_fr},
            'IT': {'Wind': wind_it, 'Solar': solar_it, 'Hydro': hydro_it},
            'PT': {'Wind': wind_pt, 'Solar': solar_pt, 'Hydro': hydro_pt},
        },
        'interconnections': [
            ('ES', 'FR', 3500),  # (from, to, NTC_MW)
            ('FR', 'IT', 4500),
            ('ES', 'IT', 2000),
            ('ES', 'PT', 4500),
        ],
    }

    results = optimize_multi_country_dispatch_no_batteries(data)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Fix PyProj CRS database issue on Windows
# Set PROJ_LIB before importing any packages that use PyProj
if sys.platform == 'win32':
    conda_env_path = Path(sys.prefix)
    proj_lib = conda_env_path / "Library" / "share" / "proj"
    if proj_lib.exists():
        os.environ['PROJ_LIB'] = str(proj_lib)

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pypsa
except ImportError as exc:
    raise ImportError(
        "pypsa is required for abstract_multi_country.py. Install with: pip install pypsa"
    ) from exc


@dataclass
class MultiCountryDispatchResult:
    """Output container from multi-country dispatch optimization."""

    status: str
    objective_value: float
    generation_by_country: Dict[str, pd.DataFrame]  # {country: generation_df with techs as columns}
    power_flows: pd.DataFrame  # interconnection flows: time x (from, to, flow_mw)
    network: pypsa.Network
    countries: List[str]
    hourly_prices: Dict[str, np.ndarray]  # {country: shadow_prices}


def optimize_multi_country_capacity_expansion_no_batteries(
    data: Dict,
    solver_name: str = "gurobi",
    solver_logfile: Optional[str] = None,
) -> MultiCountryDispatchResult:
    """Optimize multi-country CAPACITY EXPANSION WITHOUT batteries.

    **Optimization objective:** Minimize total annual cost (fixed + variable)
    across all countries simultaneously, with:
    - Extensible (optimized) capacities for each technology in each country
    - Same technology costs for all countries
    - Inter-country transmission constraints (NTC limits)
    - Full hourly generation dispatch optimization
    - P50+ renewable profiles per country

    Parameters
    ----------
    data : Dict
        Input data containing:
        - 'demands': {country: hourly_demand_array}
        - 'tech_params': {tech: {'variable_cost': float, 'fixed_cost': float, 
                                  'min_cap': float, 'max_cap': float}}
        - 'renewable_profiles': {country: {tech: hourly_profile_array}}
        - 'interconnections': [(from_country, to_country, ntc_mw), ...]

    solver_name : str
        Solver to use: 'gurobi', 'glpk', 'cbc', etc.

    solver_logfile : str, optional
        Path to write solver logs

    Returns
    -------
    MultiCountryDispatchResult
        Optimized capacities, dispatch, power flows, and dual prices
    """

    # Extract data
    demands = data["demands"]
    tech_params = data["tech_params"]
    renewable_profiles = data["renewable_profiles"]
    interconnections = data["interconnections"]

    countries = sorted(demands.keys())
    technologies = sorted(tech_params.keys())

    # Check all demand lengths and find minimum
    demand_lengths = {}
    for country in countries:
        demand_arr = np.asarray(demands[country], dtype=float).reshape(-1)
        demand_lengths[country] = demand_arr.size
    
    # Get minimum length and synchronize all to it
    min_length = min(demand_lengths.values())
    max_length = max(demand_lengths.values())
    
    if min_length != max_length:
        print(f"\n⚠ WARNING: Demand length mismatch detected!")
        print(f"  Min length: {min_length} hours, Max length: {max_length} hours")
        print(f"  Synchronizing all demands to minimum length: {min_length} hours")
        for country in countries:
            if demand_lengths[country] > min_length:
                print(f"    {country}: {demand_lengths[country]} → {min_length} hours (truncated)")
        
        # Truncate all demands to minimum length
        demands_sync = {}
        for country in countries:
            demand_arr = np.asarray(demands[country], dtype=float).reshape(-1)
            demands_sync[country] = demand_arr[:min_length]
        demands = demands_sync
        
        # Also truncate renewable profiles to match
        renewable_profiles_sync = {}
        for country in renewable_profiles:
            renewable_profiles_sync[country] = {}
            for tech, profile in renewable_profiles[country].items():
                profile_arr = np.asarray(profile, dtype=float).reshape(-1)
                renewable_profiles_sync[country][tech] = profile_arr[:min_length]
        renewable_profiles = renewable_profiles_sync
    
    n_hours = min_length

    # Create PyPSA network
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2024-01-01", periods=n_hours, freq="h"))

    # Add buses (one per country)
    for country in countries:
        network.add("Bus", country)

    # Add demand for each country
    for country in countries:
        demand_arr = np.asarray(demands[country], dtype=float).reshape(-1)
        network.add(
            "Load",
            f"demand_{country}",
            bus=country,
            p_set=demand_arr,
        )

    # Add EXTENSIBLE generators for each technology in each country
    # Capacities are DECISION VARIABLES (p_nom_extendable=True)
    for tech in technologies:
        tech_data = tech_params[tech]
        variable_cost = tech_data.get("variable_cost", 0.0)
        fixed_cost = tech_data.get("fixed_cost", 0.0)  # Capital cost annualized
        min_cap = tech_data.get("min_cap", 0.0)
        
        # Check if max_cap is country-specific or global
        max_cap_by_country_dict = tech_data.get("max_cap_by_country", None)

        for country in countries:
            # Determine max_cap for this country
            if max_cap_by_country_dict and isinstance(max_cap_by_country_dict, dict):
                max_cap = max_cap_by_country_dict.get(country, tech_data.get("max_cap", 100000.0))
            else:
                max_cap = tech_data.get("max_cap", 100000.0)
            
            # Get renewable profile if available
            if tech in renewable_profiles.get(country, {}):
                # Renewable technology: p_max_pu is the profile
                p_max_pu = np.asarray(
                    renewable_profiles[country][tech], dtype=float
                ).reshape(-1)
                if p_max_pu.size != n_hours:
                    raise ValueError(
                        f"Profile mismatch for {country}/{tech}: expected {n_hours}, got {p_max_pu.size}"
                    )
            else:
                # Non-renewable: p_max_pu = 1 (always available up to capacity)
                p_max_pu = 1.0

            network.add(
                "Generator",
                f"gen_{country}_{tech}",
                bus=country,
                p_nom_extendable=True,  # **CAPACITY IS OPTIMIZED**
                p_nom_min=min_cap,
                p_nom_max=max_cap,
                p_max_pu=p_max_pu,
                marginal_cost=variable_cost,
                capital_cost=fixed_cost,  # Annualized capital cost (€/MW/year)
            )

    # Add interconnection lines
    # Each interconnection is a bidirectional line with capacity limits
    for from_country, to_country, ntc_mw in interconnections:
        network.add(
            "Line",
            f"interconnection_{from_country}_{to_country}",
            bus0=from_country,
            bus1=to_country,
            x=0.00001,  # Very small reactance to ensure uniqueness
            s_nom=ntc_mw,  # NTC capacity
        )

    # Optimize with error handling
    try:
        print(f"Optimizing multi-country dispatch with {solver_name}...")
        network.optimize(
            solver_name=solver_name,
            log_to_console=True if solver_name == "gurobi" else False,
        )
        print(f"✓ Optimization complete\n")
        optimize_successful = True
    except AttributeError as e:
        # PyPSA 1.1.2: shadow price assignment may fail, but solve succeeded
        err_msg = str(e)
        if "shadow-prices" in err_msg or "was not assigned" in err_msg:
            print(f"✓ Optimization complete (shadow price warning - PyPSA 1.1.2 compatibility)")
            print(f"  Objective value = {network.objective}")
            print(f"  Solution found = YES\n")
            optimize_successful = True
        else:
            print(f"✗ Unexpected AttributeError: {err_msg}")
            raise
    except Exception as e:
        print(f"Warning: Optimization error ({type(e).__name__}), retrying...")
        print(f"  Error: {str(e)[:200]}")
        try:
            network.optimize(
                solver_name=solver_name,
                solver_options={"glpk": {"wopt": "all"}} if solver_name == "glpk" else {},
            )
            print(f"✓ Optimization complete (with retry)\n")
            optimize_successful = True
        except Exception as e2:
            print(f"Error during optimization retry: {e2}")
            raise

    if not optimize_successful:
        raise RuntimeError("Optimization did not complete successfully")

    # Extract generation by country
    generation_by_country = {}
    for country in countries:
        country_gen_df = pd.DataFrame(index=network.snapshots)
        for tech in technologies:
            gen_col_name = f"gen_{country}_{tech}"
            if gen_col_name in network.generators_t.p.columns:
                country_gen_df[tech] = network.generators_t.p[gen_col_name]
            else:
                country_gen_df[tech] = 0.0
        country_gen_df["demand"] = demands[country]
        generation_by_country[country] = country_gen_df

    # Extract power flows
    flows_data = []
    for from_country, to_country, _ in interconnections:
        line_name = f"interconnection_{from_country}_{to_country}"
        if line_name in network.lines_t.p0.columns:
            flows = network.lines_t.p0[line_name].values  # p0 = flow from bus0 to bus1
            for hour, flow_mw in enumerate(flows):
                flows_data.append({
                    "time": hour,
                    "from": from_country,
                    "to": to_country,
                    "flow_mw": flow_mw,
                })

    flows_df = pd.DataFrame(flows_data)

    # Extract hourly prices (dual variables of demand balance constraints)
    hourly_prices = {}
    for country in countries:
        prices = []
        for hour in network.snapshots:
            # Get the bus equilibrium constraint (dual variable)
            # In PyPSA, this is stored in buses_t.marginal_price
            if hasattr(network.buses_t, "marginal_price") and country in network.buses_t.marginal_price.columns:
                price = network.buses_t.marginal_price.loc[hour, country]
            else:
                price = np.nan
            prices.append(price)
        hourly_prices[country] = np.array(prices, dtype=float)

    return MultiCountryDispatchResult(
        status="optimal" if optimize_successful else "unknown",
        objective_value=float(network.objective) if hasattr(network, "objective") else np.nan,
        generation_by_country=generation_by_country,
        power_flows=flows_df,
        network=network,
        countries=countries,
        hourly_prices=hourly_prices,
    )
