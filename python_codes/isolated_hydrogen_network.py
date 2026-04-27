"""Isolated country-level hydrogen network construction using PyPSA.

This module deliberately builds a separate H2 carrier network and does not
connect it to the electricity model yet. All hydrogen flows are represented in
MWh_H2/h, which is PyPSA's usual power-flow unit interpreted here as hydrogen
energy flow rather than electricity.

Typical usage from a notebook:

    from isolated_hydrogen_network import build_isolated_hydrogen_network

    result = build_isolated_hydrogen_network()
    h2_network = result.network
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Keep the same Windows PyProj setup style used by the electricity modules.
if sys.platform == "win32":
    conda_env_path = Path(sys.prefix)
    proj_lib = conda_env_path / "Library" / "share" / "proj"
    if proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)

try:
    import pypsa
except ImportError as exc:
    raise ImportError(
        "pypsa is required for isolated_hydrogen_network.py. Install with: pip install pypsa"
    ) from exc


DEFAULT_COUNTRIES = ("ES", "FR", "IT", "PT")
DEFAULT_H2_DATA_DIR = Path(__file__).resolve().parents[1] / "Data" / "Hydrogen"


@dataclass
class IsolatedHydrogenNetworkResult:
    """Output container for the isolated H2 network builder."""

    network: pypsa.Network
    countries: List[str]
    snapshots: pd.DatetimeIndex
    data_dir: Path


def build_isolated_hydrogen_network(
    data_dir: Optional[Path | str] = None,
    countries: Optional[Iterable[str]] = None,
    n_hours: int = 8760,
    snapshots: Optional[pd.DatetimeIndex] = None,
) -> IsolatedHydrogenNetworkResult:
    """Build a standalone hydrogen network from simple CSV input files.

    The network is intentionally isolated from the electricity network:
    electrolysers are represented as exogenous H2 generators instead of electric
    loads, and fuel-cell operation is represented as fixed H2 demand instead of
    coupling to electric generation.

    Parameters
    ----------
    data_dir : path-like, optional
        Directory containing the hydrogen CSV files. Defaults to Data/Hydrogen.
    countries : iterable of str, optional
        Country codes already used by the electricity model. Defaults to
        ES, FR, IT, PT, matching the multi-country assignment.
    n_hours : int
        Number of hourly snapshots to create when snapshots is not supplied.
    snapshots : pandas.DatetimeIndex, optional
        Explicit snapshots for the H2 network.
    """

    h2_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_H2_DATA_DIR
    country_list = sorted(countries) if countries is not None else list(DEFAULT_COUNTRIES)
    if snapshots is None:
        snapshots = pd.date_range("2024-01-01", periods=n_hours, freq="h")

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Carrier", "H2")

    # Assumption: one aggregated H2 bus per electricity-model country. This is a
    # country-level copper plate for hydrogen, so no intra-country pipeline limits
    # are represented at this stage.
    for country in country_list:
        network.add("Bus", _h2_bus(country), carrier="H2")

    _add_h2_generators(network, h2_data_dir / "h2_electrolysers.csv", country_list)
    _add_h2_loads(
        network,
        h2_data_dir / "h2_fuel_cell_consumption.csv",
        country_list,
        name_prefix="fuel_cell_h2_consumption",
        value_column="h2_consumption_mwh_per_h",
    )
    _add_h2_loads(
        network,
        h2_data_dir / "h2_other_demand.csv",
        country_list,
        name_prefix="other_h2_demand",
        value_column="h2_demand_mwh_per_h",
    )
    _add_h2_stores(network, h2_data_dir / "h2_storage.csv", country_list)
    _add_h2_pipelines(network, h2_data_dir / "h2_pipelines.csv", country_list)

    return IsolatedHydrogenNetworkResult(
        network=network,
        countries=country_list,
        snapshots=snapshots,
        data_dir=h2_data_dir,
    )


def _add_h2_generators(network: pypsa.Network, csv_path: Path, countries: List[str]) -> None:
    table = _read_csv(csv_path)
    for _, row in table.iterrows():
        country = row["country"]
        _require_known_country(country, countries, csv_path)

        # Assumption: electrolyser H2 output is exogenous for now. It enters the
        # H2 network as a Generator with fixed installed H2 production capacity,
        # rather than being linked to electricity consumption and conversion
        # efficiency. p_nom is therefore in MWh_H2/h.
        network.add(
            "Generator",
            f"electrolyser_h2_{country}",
            bus=_h2_bus(country),
            carrier="H2",
            p_nom=float(row["p_nom_mwh_per_h"]),
            p_max_pu=float(row.get("availability_pu", 1.0)),
            marginal_cost=float(row.get("marginal_cost_eur_per_mwh", 0.0)),
        )


def _add_h2_loads(
    network: pypsa.Network,
    csv_path: Path,
    countries: List[str],
    name_prefix: str,
    value_column: str,
) -> None:
    table = _read_csv(csv_path)
    for _, row in table.iterrows():
        country = row["country"]
        _require_known_country(country, countries, csv_path)

        # Assumption: H2 consumption is fixed and inelastic. Fuel-cell H2 use is
        # not converted back into electricity here; it is only a hydrogen sink in
        # MWh_H2/h. Other H2 demand represents industry/transport/etc. as a flat
        # placeholder profile until real time-series demand is available.
        network.add(
            "Load",
            f"{name_prefix}_{country}",
            bus=_h2_bus(country),
            carrier="H2",
            p_set=float(row[value_column]),
        )


def _add_h2_stores(network: pypsa.Network, csv_path: Path, countries: List[str]) -> None:
    table = _read_csv(csv_path)
    for _, row in table.iterrows():
        country = row["country"]
        _require_known_country(country, countries, csv_path)

        # Assumption: storage is an energy reservoir only. PyPSA Stores model the
        # H2 inventory in MWh_H2 and exchange power with the H2 bus in MWh_H2/h.
        # Charging/discharging conversion losses are omitted until a coupled
        # electrolysis/fuel-cell representation is introduced.
        network.add(
            "Store",
            f"h2_store_{country}",
            bus=_h2_bus(country),
            carrier="H2",
            e_nom=float(row["e_nom_mwh"]),
            e_initial=float(row.get("e_initial_mwh", 0.0)),
            standing_loss=float(row.get("standing_loss_per_hour", 0.0)),
            e_cyclic=_as_bool(row.get("e_cyclic", False)),
            marginal_cost=float(row.get("marginal_cost_eur_per_mwh", 0.0)),
        )


def _add_h2_pipelines(network: pypsa.Network, csv_path: Path, countries: List[str]) -> None:
    table = _read_csv(csv_path)
    for _, row in table.iterrows():
        country0 = row["country0"]
        country1 = row["country1"]
        _require_known_country(country0, countries, csv_path)
        _require_known_country(country1, countries, csv_path)

        capacity = float(row["p_nom_mwh_per_h"])
        efficiency = float(row.get("efficiency", 1.0))
        marginal_cost = float(row.get("marginal_cost_eur_per_mwh", 0.0))

        # Assumption: pipelines are linear transport Links with fixed capacities.
        # We add one Link in each direction to approximate bidirectional trade.
        # This intentionally avoids nonlinear pressure-flow or Weymouth equations.
        for from_country, to_country in ((country0, country1), (country1, country0)):
            network.add(
                "Link",
                f"h2_pipeline_{from_country}_{to_country}",
                bus0=_h2_bus(from_country),
                bus1=_h2_bus(to_country),
                carrier="H2",
                p_nom=capacity,
                efficiency=efficiency,
                marginal_cost=marginal_cost,
            )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing H2 input file: {path}. Placeholder CSVs are provided under Data/Hydrogen."
        )
    return pd.read_csv(path, comment="#")


def _h2_bus(country: str) -> str:
    return f"{country}_H2"


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _require_known_country(country: str, countries: List[str], source: Path) -> None:
    if country not in countries:
        raise ValueError(
            f"{source} contains country '{country}', but expected one of {countries}."
        )
