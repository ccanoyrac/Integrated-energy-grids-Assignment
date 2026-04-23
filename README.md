# Integrated Energy Grids - Capacity Expansion Planning

A comprehensive energy systems optimization project for analyzing and optimizing electricity generation capacity expansion for Spain and neighboring European countries (France, Italy, Portugal). This project uses mathematical optimization to determine cost-optimal technology portfolios under different scenarios.

## Overview

This repository contains Jupyter notebooks and Python modules for solving **capacity expansion planning problems** in integrated energy grids. The analysis determines optimal installed capacities (MW) for different generation technologies to minimize total system costs while satisfying hourly electricity demand over a full year (8,760 hours).

### Key Features

- **Two Analysis Scenarios:**
  - **Single-country:** Spain's electricity system in isolation
  - **Multi-country:** Coupled optimization across Spain, France, Italy, and Portugal with HVDC interconnections

- **Technology Portfolio:** Wind, Solar, Hydro, CCGT (gas), Coal, Nuclear, Battery Storage, Hydrogen Storage

- **Optimization Engine:** PyPSA (Python for Power System Analysis) with multiple solver backends (Gurobi, HiGHS, GLPK)

- **Comprehensive Outputs:** Capacity expansion results, hourly dispatch schedules, cost analysis, visualization of electricity mix and interconnection flows

---

## Notebooks

### 1. **IEG_Assignment_Single_country.ipynb**
Single-country capacity expansion optimization for Spain.

**Workflow:**
1. **Data Collection & Preprocessing**
   - Load installed capacity data from Spanish operator (REE)
   - Extract hourly operating program (demand profile)
   - Import daily gas price data (MIBGAS-ES index)
   - Compile capacity by technology and compute capacity factors for renewables

2. **Input Analysis**
   - Visualize full-year demand patterns (hourly, seasonal)
   - Analyze renewable resource availability (wind, solar, hydro capacity factors)
   - Validate data consistency and coverage

3. **Capacity Expansion Optimization (without storage)**
   - Optimize technology capacities to minimize cost
   - Determine hourly dispatch for all 8,760 hours
   - Outputs: Optimal capacities, generation mix, total annual cost

4. **Capacity Expansion with Battery Storage**
   - Extend optimization to include battery storage (BESS)
   - Optimize battery power rating (MW) and energy capacity (MWh)
   - Compare costs with/without storage
   - Analyze battery charge/discharge patterns

5. **Multi-scenario Analysis (2030 Projections)**
   - Run optimization with different future demand scenarios
   - Include hydrogen storage option
   - Compare technology portfolios and costs across scenarios

6. **Results & Visualization**
   - Hourly dispatch schedule plots
   - Annual electricity generation mix (pie charts)
   - Cost breakdown by technology
   - Battery state-of-charge profiles
   - Interactive HTML plots of supply/demand matching

**Key Outputs:**
- Optimal installed capacities for each technology
- Hourly generation schedule
- Annual system cost (fixed + variable costs)
- Seasonal and hourly demand patterns
- Renewable generation capacity factors

---

### 2. **IEG_Assignment_Multi_country.ipynb**
Multi-country capacity expansion with international interconnections.

**Workflow:**
1. **Multi-country Data Setup**
   - Load capacity and generation profiles for Spain, France, Italy, Portugal
   - Extract hourly demand for each country
   - Calculate renewable profiles (wind, solar, hydro capacity factors)
   - Aggregate by country

2. **Country-level Analysis**
   - Analyze electricity demand patterns per country
   - Compare renewable resource availability across regions
   - Visualize full-year hourly profiles for each country

3. **Multi-country Interconnection Mapping**
   - Define HVDC transmission links between countries
   - Specify transmission capacity limits
   - Model transmission losses

4. **Coupled Multi-country Optimization**
   - Optimize capacities simultaneously across all countries
   - Model power flows through interconnections
   - Minimize combined system cost (all countries)
   - Account for transmission constraints and losses

5. **Multi-scenario Analysis**
   - Compare scenarios with/without storage
   - Analyze trade-offs between local generation and imports
   - Evaluate portfolio diversity across regions

6. **Visualization & Analysis**
   - Country-specific generation dispatch
   - Interconnection power flows (hourly, seasonal)
   - Cost comparison across countries and scenarios
   - Heatmaps of renewable generation by country
   - Interactive maps of electricity mix by region

**Key Outputs:**
- Optimal capacities for each technology in each country
- Hourly power flows on each interconnection
- Regional cost breakdowns
- Renewable integration levels per country
- System reliability metrics

---

## Project Structure

```
Integrated-energy-grids-Assignment/
├── README.md                                    # This file
├── IEG_Assignment_Single_country.ipynb          # Single-country optimization
├── IEG_Assignment_Multi_country.ipynb           # Multi-country optimization
├── python_codes/
│   ├── Abstract_model.py                        # Core optimization model
│   └── abstract_multi_country.py                # Multi-country model extension
├── Data/
│   ├── export_InstalledCapacityGenerationTotal_2026-03-03_17_12.csv
│   ├── export_OperatingHourlyProgramGeneration+Storage+BalearicHVDCLinkP48_2026-03-03_17_22.csv
│   ├── Gas prices Spain 2025.csv
│   ├── Spain_future_generation_&_demand.csv
│   ├── Techs_cost_FOM_&_Fuel_cost.csv
│   └── Countries interconnected Data/
│       ├── France_AGGREGATED_GENERATION_PER_TYPE_GENERATION.csv
│       ├── france_capacity_non_renewables.csv
│       ├── Italy_AGGREGATED_GENERATION_PER_TYPE_GENERATION.csv
│       ├── italy_capacity_non_renewables.csv
│       ├── Portugal_AGGREGATED_GENERATION_PER_TYPE_GENERATION.csv
│       └── portugal_capacity_non_renewables.csv
├── Plots/                                       # Generated visualizations
│   ├── 0_Input Data/
│   ├── 1_Spanish_economic_dispatch/
│   ├── 2_Battery_economic_dispatch_(Spain)/
│   ├── 3_Battery_economic_dispatch_multiple_scenarios_(Spain)/
│   ├── 4_Battery_economic_dispatch_2030/
│   ├── 5_Multi_country_economic_dispatch/
│   └── 7_Annual_electricity_mix_*.html         # Interactive HTML maps
└── back-up projects/                            # Previous assignment versions

```

---

## Mathematical Model

### Objective Function

Minimize total annual system cost:

$$Z = \sum_{t \in T} FC_t \cdot C_t + \sum_{t \in T} \sum_{h \in H} VC_t \cdot G_{t,h}$$

Where:
- $FC_t$ = Fixed cost for technology $t$ (€/MW/year)
- $VC_t$ = Variable cost for technology $t$ (€/MWh)
- $C_t$ = Installed capacity (MW)
- $G_{t,h}$ = Generation at hour $h$ (MWh)
- $T$ = Set of technologies
- $H$ = Set of hourly time steps (8,760 hours/year)

### Key Constraints

1. **Power Balance (Demand satisfaction):**
$$\sum_{t \in T} G_{t,h} = D_h \quad \forall h \in H$$

2. **Renewable Generation Limits:**
$$G_{t,h} \leq CF_{t,h} \cdot C_t \quad \forall t \in T_{renewable}, h \in H$$

3. **Dispatchable Generation Limits:**
$$G_{t,h} \leq C_t \quad \forall t \in T_{dispatchable}, h \in H$$

4. **Capacity Bounds:**
$$0 \leq C_t \leq C_t^{max} \quad \forall t \in T$$

---

## Technologies Included

| Technology | Type | Cost Structure | Constraint |
|-----------|------|---------------|-----------| 
| **Wind** | Renewable | Fixed + Variable | Limited by capacity factor profile |
| **Solar** | Renewable | Fixed + Variable | Limited by capacity factor profile |
| **Hydro** | Renewable | Fixed + Variable | Capped at ~4,698 MW (Spain) |
| **CCGT** | Dispatchable | Fixed + Variable | Dispatchable up to capacity |
| **Coal** | Dispatchable | Fixed + Variable | Dispatchable up to capacity |
| **Nuclear** | Dispatchable | Fixed + Variable | Dispatchable up to capacity |
| **Battery Storage** | Storage | Fixed + Variable | Energy-limited (MWh) |
| **Hydrogen Storage** | Storage | Fixed + Variable | Energy-limited (MWh) |

---

## Requirements

### Python Packages
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pypsa` ≥ 0.26.0 - Power system optimization
- `linopy` ≥ 0.7.0 - Linear optimization interface
- `gurobi` or `highspy` - Optimization solvers
- `matplotlib` - Plotting
- `plotly` - Interactive visualizations
- `folium` - Interactive maps
- `xarray` < 2026 - Data structures (version constraint for PyPSA compatibility)

### Solvers
- **Gurobi** (academic license, recommended) - Robust, fast
- **HiGHS** (open-source) - Good alternative
- **GLPK** - Fallback option (avoid on Windows due to file locking issues)

### System Requirements
- **Disk Space:** ~2-5 GB for full-year optimization (linopy creates large temporary LP files)
- **RAM:** 8+ GB recommended
- **Windows Users:** See [PyProj CRS Database Error](https://github.com/DTU/integrated-energy-grids/wiki#pyproj-crs-database-error) in troubleshooting

---

## Usage

### Step 1: Setup Environment

```bash
# Create conda environment
conda create -n ieg_env python=3.10

# Activate environment
conda activate ieg_env

# Install dependencies
pip install pandas numpy pypsa>=0.26.0 linopy>=0.7.0 matplotlib plotly folium gurobi
```

### Step 2: Run Notebooks

#### Single-country Analysis:
```bash
jupyter notebook IEG_Assignment_Single_country.ipynb
```
Run cells sequentially from top to bottom. Each section outputs progress and generates plots in the `Plots/` directory.

#### Multi-country Analysis:
```bash
jupyter notebook IEG_Assignment_Multi_country.ipynb
```
Similar workflow to single-country, but extends to Spain + neighbors.

### Step 3: Interpret Results

**Look for these outputs in notebooks:**
- **Optimal capacities table:** Shows recommended MW for each technology
- **Generation dispatch plots:** Hourly generation by technology
- **Cost summary:** Breakdown of fixed vs. variable costs
- **Electricity mix:** Pie charts showing % generation by technology
- **Interactive maps:** Country-level visualization in `Plots/7_Annual_electricity_mix_*.html`

---

## Common Issues & Troubleshooting

### Issue 1: PermissionError with GLPK on Windows
**Error:** `PermissionError: [WinError 32] El proceso no tiene acceso al archivo`
**Solution:** Switch to Gurobi or HiGHS solver. Avoid retrying solve on same network object.

### Issue 2: DISK SPACE - OSError [Errno 28] No space left on device
**Root Cause:** linopy creates ~2GB+ temporary files during optimization
**Solutions:**
1. Use smaller time period (e.g., representative week = 168 hours)
2. Clean Windows temp folder: `C:\Users\<user>\Temp`
3. Use MPS format instead of LP (potentially smaller)

### Issue 3: PyProj CRS Database Error
**Error:** `pyproj.exceptions.CRSError: Invalid projection: EPSG:4326`
**Solution:**
```bash
pip install --upgrade pyproj
# OR (for Windows)
conda install -y --force-reinstall pyproj proj
```

### Issue 4: Solver Not Found
**Error:** `SolverError: Solver 'gurobi' not found`
**Solutions:**
- Install Gurobi: `pip install gurobi`
- Or use HiGHS: `pip install highspy`
- Check installation: `python -c "import gurobipy; print(gurobipy.gurobi.version())"`

---

## Output Files

### Automatically Generated
- **Plots/** - Directory containing all visualizations
  - Hourly dispatch charts (PNG)
  - Capacity and generation tables (HTML)
  - Interactive electricity mix maps (HTML with Folium)
  - Seasonal analysis plots
  - Cost breakdown charts

### Key Data Structures (in-notebook)
- `result` - CapacityExpansionResult object with optimal solution
- `solution_df` - DataFrame with hourly dispatch schedule
- `capacity_table` - DataFrame with optimized capacities by technology
- `cost_summary_df` - DataFrame with cost breakdown

---

## References

### Theory & Methods
- PyPSA Documentation: https://pypsa.readthedocs.io/
- Linear optimization with linopy: https://linopy.readthedocs.io/
- Capacity expansion planning in power systems (academic literature)

### Data Sources
- Spanish electricity operator (REE) - Capacity and generation data
- MIBGAS - Gas price data
- European TSOs - Multi-country generation and capacity data

---

## Contributing

To extend this project:
1. Modify `python_codes/Abstract_model.py` for model changes
2. Update data sources in the Data/ folder
3. Test with representative periods first (e.g., 1 week = 168 hours)
4. Create new notebooks for additional scenarios

---

## License & Attribution

Assignment project for DTU (Technical University of Denmark) - Integrated Energy Grids course.

Last updated: April 2026

---

## Contact & Support

For issues, questions, or contributions:
- Check the troubleshooting section above
- Review notebook markdown cells for detailed explanations
- Refer to PyPSA documentation for model-specific questions
