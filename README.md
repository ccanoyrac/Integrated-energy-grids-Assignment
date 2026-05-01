# Integrated Energy Grids — Course Assignment (46770)

**DTU — Technical University of Denmark**
Master's-level course project covering capacity expansion planning and multi-sector energy system optimization for Southern European countries.

---

## Overview

This project models and optimizes integrated electricity, hydrogen, and heat grids using linear programming. It progresses from a single-country electricity dispatch to a fully coupled multi-country, multi-sector energy system — reflecting real-world energy transition challenges such as renewable integration, storage sizing, transmission planning, and sector coupling.

The geographic focus is **Spain, France, Italy, and Portugal**, using real hourly demand and renewable generation profiles.

---

## Project Structure

```
Integrated-energy-grids-Assignment/
├── python_codes/                          # Core optimization modules
│   ├── Abstract_model.py                  # Single-country capacity expansion
│   ├── abstract_multi_country.py          # Multi-country dispatch
│   ├── integrated_electricity_hydrogen.py # Electricity + H2 network
│   ├── integrated_electricity_hydrogen_heat.py  # Electricity + H2 + heat
│   └── co2_analysis.py                    # CO2-constrained optimization
│
├── IEG_1_Single_country.ipynb             # Assignment 1a–c: single country
├── IEG_2_Interannual_Variability.ipynb    # Assignment 1b: climate variability
├── IEG_3_Multi_country.ipynb              # Assignment 1d: multi-country network
├── IEG_4_Elec_&_H2.ipynb                 # Assignment 2g: electricity + hydrogen
├── IEG_5_Elec_&_H2_&_Heat.ipynb          # Assignment 2i: full sector coupling
├── IEG_6_Elec_&_H2_&_Heat_(Nuclear_Constrained).ipynb  # Regional policy scenario
│
├── Data/                                  # Input data (CSVs, Excel)
├── Plots/                                 # Generated figures
├── IEG_Course_Project_Part1.pdf           # Assignment specification (Part 1)
└── IEG_Course_Project_Part2.pdf           # Assignment specification (Part 2)
```

---

## Assignment Tasks

### Part 1 — Electricity Grid Fundamentals

| Task | Description | Notebook |
|------|-------------|----------|
| **a** | Single-country optimal capacity expansion (wind, solar, hydro, CCGT, coal, nuclear) | `IEG_1_Single_country.ipynb` |
| **b** | Interannual weather variability across 2020–2024 | `IEG_2_Interannual_Variability.ipynb` |
| **c** | Battery and hydrogen storage integration | `IEG_1_Single_country.ipynb` |
| **d** | Multi-country network with HVAC interconnections | `IEG_3_Multi_country.ipynb` |
| **e** | Network analysis: incidence matrix and PTDF calculations *(pen & paper)* | — |

### Part 2 — Multi-Sector Energy Integration

| Task | Description | Notebook |
|------|-------------|----------|
| **f** | CO2 sensitivity analysis on capacity mix | `co2_analysis.py` |
| **g** | Hydrogen and methane pipeline integration | `IEG_4_Elec_&_H2.ipynb` |
| **h** | CO2 pricing and carbon tax comparison (shadow prices) | `IEG_4_Elec_&_H2.ipynb` |
| **i** | Electricity + hydrogen + heat sector coupling | `IEG_5_Elec_&_H2_&_Heat.ipynb` |
| **j** | Regional policy scenario (nuclear constraint, France) | `IEG_6_...Nuclear_Constrained.ipynb` |

---

## Technology Database

Technologies are parameterized from IEA and ENTSO-E cost data. Three progressive cost files are used across assignments.

| Category | Technology | Annualized Cost (€/kW/a) | Efficiency | Fuel Cost (€/MWh) |
|----------|-----------|--------------------------|------------|-------------------|
| Wind | Onshore wind | 83.78 | 1.0 | 0 |
| Wind | Offshore wind | 177.78 | 1.0 | 0 |
| Solar | Utility-scale PV | 30.15 | 1.0 | 0 |
| Solar | Rooftop PV | 51.44 | 1.0 | 0 |
| Hydro | Pumped hydro | 102.04 | 0.87 | 0 |
| Hydro | Reservoir | 102.04 | 0.90 | 0 |
| Hydro | Run-of-river | 153.05 | 0.90 | 0 |
| Fossil | OCGT | 26.02 | 0.39 | 30.00 |
| Fossil | Coal | 174.61 | 0.40 | 61.52 |
| Nuclear | Nuclear | 317.04 | 0.33 | 8.50 |
| CHP | CHP central | 42.57 | 0.80 | 87.50 |
| Storage | Battery | 43.35 | 0.90 | 0 |
| Storage | Hydrogen | 198.65 | 0.35 | 0 |
| Heat pump | Air-source (decentral) | 121.00 | COP 3.0 | 0 |
| Heat pump | Air-source (central) | 80.67 | COP 3.0 | 0 |
| Heat pump | Ground-source (decentral) | 161.33 | COP 3.5 | 0 |

---

## Core Optimization Modules

### `Abstract_model.py`
Single-country linear program for capacity expansion. Minimizes annualized capital cost plus hourly variable cost. Supports:
- Basic generation mix (no storage)
- Battery + hydrogen dual-storage co-optimization
- CO2-capped scenarios via global constraint

### `abstract_multi_country.py`
Extends the model to a four-country network (ES–FR–IT–PT). One bus per country; bidirectional HVAC interconnections with NTC limits. Returns hourly power flows and shadow prices (marginal costs by country and hour).

### `integrated_electricity_hydrogen.py`
Adds a hydrogen bus per country. Coupling elements: electrolyzer (electricity → H2), fuel cell (H2 → electricity), H2 storage (≥168 h duration), and cross-border H2 pipelines. H2 demand is configurable as a fraction of electricity demand.

### `integrated_electricity_hydrogen_heat.py`
Extends to three sectors per country: electricity, hydrogen, and heat. Adds heat pumps (COP 3.0–3.5) and CHP units (≈40% electrical + 40% thermal efficiency). Enables full sector-coupling cost analysis.

### `co2_analysis.py`
Parametric CO2 sweep over a range of emission caps. Extracts the CO2 shadow price (€/tCO₂) from the LP dual variable. Emission factors: Coal 0.896 tCO₂/MWh, CCGT 0.367 tCO₂/MWh.

---

## Data Sources

| File | Description |
|------|-------------|
| `Techs_cost_FOM_&_Fuel_cost.csv` | Technology costs for Part 1 (12 technologies) |
| `Techs_&_storage_cost_FOM_&_Fuel_cost.csv` | Adds battery + H2 storage |
| `Techs_&_storage_&_Heat_pump_cost_FOM_&_Fuel_cost.csv` | Adds heat pumps (Part 2) |
| `generation_by_tech_2024.csv` | Hourly generation per technology (8 760 values) |
| `Spain_future_generation_&_demand.csv` | 2024–2030 demand and generation projections |
| `International_exchanges_January_2024.xlsx` | ENTSO-E cross-border flow data |
| `Gas prices Spain 2025.csv` | Monthly gas price profiles |

Renewable generation profiles are P50 (median) hourly capacity factors. Temporal resolution is **hourly** over a full year (8 760 time steps).

---

## Modeling Approach

- **Optimization framework:** [PyPSA](https://pypsa.org) (Python for Power System Analysis)
- **Problem type:** Linear programming (LP) for dispatch; mixed-integer (MIP) for investment
- **Solver:** Gurobi (primary); GLPK / CBC (fallback)
- **Time horizon:** Full year, hourly resolution
- **Network topology:** DC power flow approximation; one bus per country

**Key constraints:**
- Nodal power balance (per bus, per hour)
- Renewable generation upper bound (p.u. capacity factor profiles)
- Transmission capacity limits (NTC between countries)
- Storage energy balance with cyclic end conditions (battery: 4 h; H2: 168 h)
- Optional global CO2 emission cap

**Key outputs:**
- Optimal installed capacity by technology (MW)
- Hourly dispatch (MWh/h) and storage state-of-charge
- Interconnection power and H2 pipeline flows
- Marginal prices by bus and hour (€/MWh)
- CO2 shadow price (€/tCO₂) for carbon-constrained scenarios
- Technology cost breakdown (fixed vs. variable)

---

## Requirements

```bash
pip install pypsa pandas numpy matplotlib seaborn xarray linopy
```

A Gurobi license is recommended for larger models. The code falls back to open-source solvers automatically.

---

## Usage

Run notebooks in order (`IEG_1` → `IEG_6`) to follow the pedagogical progression. Each notebook is self-contained: it loads data, runs the relevant optimization module, and generates plots to `Plots/`.

To run a standalone optimization:

```python
from python_codes.Abstract_model import optimize_capacity_expansion, CapacityExpansionInput

result = optimize_capacity_expansion(inputs)
print(result.optimal_capacities)
```

---

## Results

All generated figures are saved to `Plots/`, including:
- Optimal capacity mixes (stacked bar charts by country)
- Seasonal dispatch time series (winter and summer weeks)
- Interconnection flow maps
- Storage state-of-charge profiles
- CO2 cost curves (abatement cost vs. emission cap)
- Sector-coupling heat and hydrogen dispatch analyses
