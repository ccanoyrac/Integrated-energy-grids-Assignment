# Integrated Energy Grids Assignment

## Course Project – Integrated Energy Systems Analysis

---

## 1. Introduction

This repository contains the implementation and results of a course project developed within the context of **Integrated Energy Grids**. The objective of the project is to analyze electricity system operation under different configurations, with a focus on **economic dispatch**, **battery storage integration**, and **cross-border electricity exchanges**.

The study primarily considers the Spanish power system and extends the analysis to a **multi-country interconnected network** including Spain, France, Italy, and Portugal.

---

## 2. Objectives

The main objectives of this project are:

* To model and solve the **economic dispatch problem** for a national electricity system
* To evaluate the impact of **battery energy storage systems (BESS)** on system operation and costs
* To compare multiple **storage scenarios** and assess their economic implications
* To analyze future system behavior under **2030 demand and generation projections**
* To extend the model to a **multi-country framework**, incorporating cross-border electricity exchanges

---

## 3. Methodology

The project is based on a **cost-minimization economic dispatch model**, formulated as an optimization problem. The model determines the optimal generation mix required to meet electricity demand at minimum cost while satisfying system constraints.

### 3.1 Model Characteristics

The optimization framework includes:

* **Demand balance constraint** (supply must meet hourly demand)
* **Generation capacity limits** per technology
* **Renewable generation profiles** (wind, solar, hydro)
* **Marginal cost-based dispatch**
* **Battery storage modeling**, including:

  * Charging and discharging decisions
  * State-of-charge constraints
  * Efficiency losses
* **Interconnection constraints** (multi-country model):

  * Transmission capacity limits
  * Cross-border power flows

### 3.2 Solver

The optimization problems are solved using **Gurobi** via the `gurobipy` interface.

---

## 4. Repository Structure

```
Integrated-energy-grids-Assignment/
│
├── Data/                         # Input datasets
│   ├── generation, demand, prices
│   ├── international exchanges
│   └── technology cost parameters
│
├── Plots/                        # Generated figures and results
│   ├── input data visualization
│   ├── Spanish dispatch results
│   ├── battery scenarios
│   ├── 2030 projections
│   └── multi-country analysis
│
├── python codes/                 # Core modeling modules
│   ├── Abstract_model.py
│   ├── Multi_country_dispatch.py
│   └── dispatch_plotting.py
│
├── Integrated_energy_grids_assignment_1.ipynb   # Main notebook
├── requirements.txt
└── IEG_Course_Project_Part1.pdf                 # Project description
```

---

## 5. Project Components

### 5.1 Main Notebook

The file `Integrated_energy_grids_assignment_1.ipynb` contains the full workflow, including:

* Data loading and preprocessing
* Model implementation
* Scenario analysis
* Visualization of results

### 5.2 Python Modules

* **Abstract_model.py**
  Implements the core economic dispatch model for:

  * baseline scenarios
  * battery-integrated systems

* **Multi_country_dispatch.py**
  Extends the model to a multi-country system with interconnections and power exchanges

* **dispatch_plotting.py**
  Provides visualization tools for:

  * generation dispatch
  * battery operation
  * price evolution
  * scenario comparisons

---

## 6. Data Description

The project relies on multiple datasets, including:

* Hourly electricity **demand profiles**
* Technology-specific **generation data**
* **Fuel prices** (e.g., gas prices in Spain)
* **International exchange data**
* **Technology cost and efficiency parameters**

These datasets are stored in the `Data/` directory and are used as inputs to the optimization models.

---

## 7. Analyses Performed

The project includes the following analyses:

### 7.1 Input Data Exploration

Visualization and validation of demand, renewable generation, and system parameters.

### 7.2 Spanish Economic Dispatch

* Optimal generation mix
* Cost-based dispatch results

### 7.3 Battery Integration (Spain)

* Charging/discharging schedules
* State-of-charge evolution
* Impact on system costs and electricity prices

### 7.4 Scenario Analysis

Comparison of multiple battery configurations to evaluate system performance under different storage assumptions.

### 7.5 2030 System Projection

Simulation of future scenarios using projected demand and renewable generation profiles.

### 7.6 Multi-Country Dispatch

* Joint optimization of Spain, France, Italy, and Portugal
* Analysis of cross-border electricity flows
* Impact of interconnection constraints on system operation

---

## 8. Installation and Setup

### 8.1 Clone Repository

```bash
git clone https://github.com/ccanoyrac/Integrated-energy-grids-Assignment.git
cd Integrated-energy-grids-Assignment
```

### 8.2 Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 8.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 9. Requirements

Main Python dependencies include:

* `pandas`
* `numpy`
* `matplotlib`
* `pulp`
* `gurobipy`
* `folium`

> Note: A valid **Gurobi license** is required to run the optimization models.

---

## 10. Usage

To reproduce the results:

```bash
jupyter notebook Integrated_energy_grids_assignment_1.ipynb
```

Run all cells sequentially to execute the full analysis.

---

## 11. Discussion

This project demonstrates how optimization techniques can be applied to analyze modern electricity systems. In particular, it highlights:

* The role of **storage technologies** in improving system flexibility
* The importance of **interconnections** in balancing supply and demand across regions
* The impact of **renewable integration** on dispatch and pricing

---

## 12. Limitations

* Simplified representation of network constraints
* Dependence on input data assumptions
* Requirement of a commercial solver (Gurobi)

---

## 13. Conclusion

The results illustrate the potential benefits of battery storage and regional integration in electricity markets. The modeling framework provides a foundation for further studies on energy system optimization and planning.
