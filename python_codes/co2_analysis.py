"""CO2-constrained capacity expansion model for the integrated electricity + H2 network.

Builds on the same PyPSA network architecture as integrated_electricity_hydrogen.py
but adds:
  - CO2 emission factors on fossil-fuel generators (Coal, CCGT)
  - An optional system-wide CO2 cap (GlobalConstraint)
  - Extraction of the CO2 shadow price (dual variable → €/tCO2)

Emission factors (EEA EMEP/EEA Guidebook 2023, tCO2/MWh_e):
  Coal : 94.6 kgCO2/GJ_fuel × 3.6 GJ/MWh / 0.38 η ≈ 0.896 tCO2/MWh_e
  CCGT : 56.1 kgCO2/GJ_fuel × 3.6 GJ/MWh / 0.55 η ≈ 0.367 tCO2/MWh_e

Typical usage from notebook:

    from python_codes.co2_analysis import run_co2_constrained

    # Step 1 – baseline (unconstrained, already run as results_integrated_h2)
    baseline_co2 = compute_baseline_co2(results_integrated_h2)

    # Step 2 – choose a target and solve
    result_co2 = run_co2_constrained(
        data=data_integrated_h2,
        co2_limit=baseline_co2 * 0.50,   # 50 % reduction
        <same kwargs as original optimization call>,
    )

    print(f"Shadow price: €{result_co2.co2_shadow_price:.2f}/tCO2")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pypsa
except ImportError as exc:
    raise ImportError("pypsa is required. Install with: pip install pypsa") from exc

from python_codes.integrated_electricity_hydrogen import (
    IntegratedElectricityH2Result,
    optimize_multi_country_integrated_electricity_h2,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Emission factors (tCO2 / MWh_e, direct combustion at plant level)
# Source: EEA EMEP/EEA Air Pollutant Emission Inventory Guidebook 2023
#   Hard coal: 94.6 kgCO2/GJ_fuel → /0.38 η → 0.896 tCO2/MWh_e
#   Natural gas (CCGT): 56.1 kgCO2/GJ_fuel → /0.55 η → 0.367 tCO2/MWh_e
# ---------------------------------------------------------------------------
EMISSION_FACTORS: Dict[str, float] = {
    "Coal": 0.896,
    "CCGT": 0.367,
}


@dataclass
class CO2Result:
    """Extended result container that adds CO2 metrics to the base result."""

    base: IntegratedElectricityH2Result   # full PyPSA result (network, flows, …)
    co2_limit_tonnes: float               # the cap that was imposed (tCO2/year)
    co2_emissions_tonnes: float           # actual emissions in the solution
    co2_shadow_price: Optional[float]     # €/tCO2 (None if not retrievable)
    reduction_pct: float                  # achieved % reduction vs baseline


def compute_baseline_co2(results: IntegratedElectricityH2Result) -> float:
    """Compute total system CO2 from an unconstrained result.

    Parameters
    ----------
    results : IntegratedElectricityH2Result
        Output of optimize_multi_country_integrated_electricity_h2 (no CO2 cap).

    Returns
    -------
    float
        Total CO2 in tCO2/year.
    """
    network = results.network
    total = 0.0
    for tech, ef in EMISSION_FACTORS.items():
        for country in results.countries:
            col = f"gen_{country}_{tech}"
            if col in network.generators_t.p.columns:
                total += float(network.generators_t.p[col].sum() * ef)
    return total


def run_co2_constrained(
    data: Dict,
    co2_limit: float,
    baseline_co2: Optional[float] = None,
    # --- pass-through kwargs for the underlying optimization ---
    h2_demand_fraction: float = 0.10,
    h2_pipeline_capacities=None,
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
) -> CO2Result:
    """Run the integrated electricity + H2 model with a system-wide CO2 cap.

    The function:
      1. Builds a fresh PyPSA network (same architecture as the base model).
      2. Attaches CO2 emission factors to Coal and CCGT generators.
      3. Adds a GlobalConstraint capping total annual emissions to `co2_limit`.
      4. Solves the LP.
      5. Reads the dual variable (shadow price) of the CO2 constraint.

    Parameters
    ----------
    data : dict
        Same input dict used by optimize_multi_country_integrated_electricity_h2.
    co2_limit : float
        Maximum allowed system-wide CO2 emissions in tCO2/year.
    baseline_co2 : float, optional
        Unconstrained baseline emissions (tCO2/year) used to compute the
        achieved reduction percentage. If None, reduction_pct is set to NaN.

    Returns
    -------
    CO2Result
    """
    print("\n" + "=" * 80)
    print("CO2-CONSTRAINED INTEGRATED ELECTRICITY + H2 OPTIMIZATION")
    print("=" * 80)
    print(f"CO2 cap  : {co2_limit / 1e6:.3f} Mt CO2/year")
    if baseline_co2 is not None:
        reduction = 100.0 * (1.0 - co2_limit / baseline_co2)
        print(f"Reduction: {reduction:.1f}% vs baseline ({baseline_co2 / 1e6:.3f} Mt)")
    else:
        reduction = float("nan")

    # ------------------------------------------------------------------
    # 1. Run the base optimizer to get a fully built PyPSA network
    # ------------------------------------------------------------------
    base_result = optimize_multi_country_integrated_electricity_h2(
        data=data,
        h2_demand_fraction=h2_demand_fraction,
        h2_pipeline_capacities=h2_pipeline_capacities,
        battery_max_hours=battery_max_hours,
        battery_charging_efficiency=battery_charging_efficiency,
        battery_discharging_efficiency=battery_discharging_efficiency,
        battery_standing_loss=battery_standing_loss,
        battery_fixed_cost=battery_fixed_cost,
        battery_variable_cost=battery_variable_cost,
        battery_max_capacity_limit=battery_max_capacity_limit,
        hydrogen_max_hours=hydrogen_max_hours,
        hydrogen_charging_efficiency=hydrogen_charging_efficiency,
        hydrogen_discharging_efficiency=hydrogen_discharging_efficiency,
        hydrogen_standing_loss=hydrogen_standing_loss,
        hydrogen_fixed_cost=hydrogen_fixed_cost,
        hydrogen_variable_cost=hydrogen_variable_cost,
        hydrogen_max_capacity_limit=hydrogen_max_capacity_limit,
        electrolyzer_efficiency=electrolyzer_efficiency,
        fuel_cell_efficiency=fuel_cell_efficiency,
        electrolyzer_fixed_cost=electrolyzer_fixed_cost,
        fuel_cell_fixed_cost=fuel_cell_fixed_cost,
        electrolyzer_max_capacity=electrolyzer_max_capacity,
        fuel_cell_max_capacity=fuel_cell_max_capacity,
        solver_name=solver_name,
    )

    # ------------------------------------------------------------------
    # 2. Copy the solved network and inject CO2 carriers + GlobalConstraint
    #    then re-optimize so all capacities are free to adjust
    # ------------------------------------------------------------------
    network = base_result.network.copy()
    countries = base_result.countries

    # Add fossil carriers with emission factors
    for tech, ef in EMISSION_FACTORS.items():
        carrier_name = f"carrier_{tech.lower()}"
        if carrier_name not in network.carriers.index:
            network.add("Carrier", carrier_name, co2_emissions=ef)
        for country in countries:
            gen_name = f"gen_{country}_{tech}"
            if gen_name in network.generators.index:
                network.generators.loc[gen_name, "carrier"] = carrier_name

    # Add the CO2 cap as a GlobalConstraint
    network.add(
        "GlobalConstraint",
        "co2_cap",
        sense="<=",
        constant=float(co2_limit),
        type="primary_energy",
        carrier_attribute="co2_emissions",
    )

    # Re-optimize with the CO2 cap active
    print(f"\n{'─' * 80}")
    print(f"RE-OPTIMIZING WITH CO2 CAP  (solver: {solver_name})")
    print(f"{'─' * 80}\n")

    try:
        network.optimize(
            solver_name=solver_name,
            log_to_console=(solver_name == "gurobi"),
        )
    except AttributeError as e:
        if "shadow-prices" in str(e) or "was not assigned" in str(e):
            pass  # PyPSA compatibility warning — optimization still succeeded
        else:
            raise
    except Exception as e:
        raise RuntimeError(f"CO2-constrained optimization failed: {e}") from e

    # ------------------------------------------------------------------
    # 3. Extract CO2 shadow price (dual variable of the GlobalConstraint)
    # ------------------------------------------------------------------
    co2_shadow_price: Optional[float] = None
    if "co2_cap" in network.global_constraints.index:
        try:
            mu_val = float(network.global_constraints.loc["co2_cap", "mu"])
            co2_shadow_price = abs(mu_val) if not np.isnan(mu_val) else None
        except Exception as e:
            print(f"WARNING: could not extract CO2 shadow price: {e}")

    # ------------------------------------------------------------------
    # 4. Compute actual emissions in the CO2-constrained solution
    # ------------------------------------------------------------------
    actual_co2 = 0.0
    for tech, ef in EMISSION_FACTORS.items():
        for country in countries:
            col = f"gen_{country}_{tech}"
            if col in network.generators_t.p.columns:
                actual_co2 += float(network.generators_t.p[col].sum() * ef)

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("CO2-CONSTRAINED RESULTS")
    print(f"{'─' * 80}")
    print(f"  CO2 cap imposed       : {co2_limit / 1e6:.3f} Mt CO2/year")
    print(f"  Actual CO2 emissions  : {actual_co2 / 1e6:.3f} Mt CO2/year")
    if co2_shadow_price is not None:
        print(f"  CO2 shadow price      : €{co2_shadow_price:.2f}/tCO2")
        print(f"  EU ETS 2024 average   : €64.74/tCO2  (EC Carbon Market Report 2024)")
        ratio = co2_shadow_price / 64.74
        print(f"  Ratio vs EU ETS       : {ratio:.2f}x")
    else:
        print("  CO2 shadow price      : not available")

    return CO2Result(
        base=base_result,
        co2_limit_tonnes=co2_limit,
        co2_emissions_tonnes=actual_co2,
        co2_shadow_price=co2_shadow_price,
        reduction_pct=reduction,
    )


def sweep_co2_targets(
    data: Dict,
    baseline_co2: float,
    reductions: List[float] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run run_co2_constrained for several reduction targets and return a summary table.

    Parameters
    ----------
    data : dict
        Input data dict.
    baseline_co2 : float
        Unconstrained baseline CO2 (tCO2/year) from compute_baseline_co2().
    reductions : list of float
        Fractional reductions to test, e.g. [0.20, 0.40, 0.50, 0.60].
        Defaults to [0.20, 0.30, 0.40, 0.50, 0.60].
    **kwargs
        Passed through to run_co2_constrained.

    Returns
    -------
    pd.DataFrame with columns: reduction_pct, co2_limit_MtCO2, co2_actual_MtCO2,
                                shadow_price_EUR_tCO2
    """
    if reductions is None:
        reductions = [0.20, 0.30, 0.40, 0.50, 0.60]

    rows = []
    for r in reductions:
        limit = baseline_co2 * (1.0 - r)
        print(f"\n{'=' * 60}")
        print(f"TARGET: {r * 100:.0f}% reduction → cap = {limit / 1e6:.2f} Mt CO2")
        print(f"{'=' * 60}")
        res = run_co2_constrained(
            data=data,
            co2_limit=limit,
            baseline_co2=baseline_co2,
            **kwargs,
        )
        rows.append({
            "reduction_pct": r * 100,
            "co2_limit_MtCO2": limit / 1e6,
            "co2_actual_MtCO2": res.co2_emissions_tonnes / 1e6,
            "shadow_price_EUR_tCO2": res.co2_shadow_price,
        })

    return pd.DataFrame(rows)
