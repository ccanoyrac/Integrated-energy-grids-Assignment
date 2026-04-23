"""Reusable plotting helpers for battery-vs-baseline dispatch analysis."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_COLOR_MAP = {
    "Wind": "green",
    "Solar": "gold",
    "Hydro": "royalblue",
    "Coal": "darkred",
    "Nuclear": "purple",
    "Diesel": "red",
    "CCGT": "orange",
    "CHP": "saddlebrown",
    "HVDC": "black",
    "OtherRenew": "lightgreen",
    "Waste": "darkgray",
    "Battery": "cyan",
}

DEFAULT_TECH_COST_ORDER = [
    "Wind",
    "Solar",
    "Hydro",
    "OtherRenew",
    "Nuclear",
    "Waste",
    "HVDC",
    "CHP",
    "Coal",
    "CCGT",
    "Diesel",
]


def plot_battery_dispatch_dashboard(
    *,
    results_by_season: Mapping[str, Mapping[str, object]],
    seasons: Iterable[str],
    suptitle: str,
    battery_capacity_mwh: float,
    filename: Optional[str] = None,
    price_with_battery_key: str = "battery_prices",
    color_map: Optional[Mapping[str, str]] = None,
    tech_cost_order: Optional[Iterable[str]] = None,
) -> None:
    """Plot 4x2 dashboard: generation, technology delta, battery operation, prices."""

    cmap = dict(DEFAULT_COLOR_MAP if color_map is None else color_map)
    order = list(DEFAULT_TECH_COST_ORDER if tech_cost_order is None else tech_cost_order)
    seasons = list(seasons)

    n_cols = max(1, len(seasons))
    hours = np.arange(168)
    fig, axes = plt.subplots(4, n_cols, figsize=(8 * n_cols, 18))

    # Normalize axes shape for the 1-season case.
    if n_cols == 1:
        axes = np.array(axes).reshape(4, 1)

    for col_idx, season in enumerate(seasons):
        if season not in results_by_season:
            for row_idx in range(4):
                axes[row_idx, col_idx].text(
                    0.5, 0.5, "No Data", ha="center", va="center", fontsize=14
                )
                axes[row_idx, col_idx].set_title(
                    f"{season}", fontsize=12, fontweight="bold"
                )
            continue

        res = results_by_season[season]
        sol = pd.DataFrame(res["sol"]).reset_index(drop=True)

        # Support both legacy and current naming for baseline solution payload.
        baseline_sol = res.get("sol_base")
        if baseline_sol is None:
            baseline_sol = res.get("baseline_sol")
        if baseline_sol is None:
            baseline_sol = res.get("sol")
        sol_base = pd.DataFrame(baseline_sol).reset_index(drop=True)
        techs = list(res["techs"])

        # ===== ROW 1: Generation mix =====
        ax_gen = axes[0, col_idx]
        plot_cols = [tech for tech in order if tech in techs and sol[tech].sum() > 0]
        plot_cols += [tech for tech in techs if tech not in plot_cols and sol[tech].sum() > 0]
        colors_list = [cmap.get(col, "gray") for col in plot_cols]

        stack_data = [sol[col].values for col in plot_cols]
        stack_data.append(sol["P_discharge"].values)
        colors_list.append(cmap["Battery"])

        ax_gen.stackplot(hours, *stack_data, colors=colors_list, alpha=0.8)
        ax_gen.plot(
            hours,
            sol["demand"].values,
            color="black",
            linewidth=2,
            linestyle="--",
            label="Demand",
        )
        ax_gen.plot(
            hours,
            sol["demand"].values + sol["P_charge"].values,
            color="magenta",
            linewidth=1.5,
            linestyle=":",
            label="Demand + Charging",
        )

        ax_gen.set_xlim(0, 167)
        ax_gen.set_title(f"{season} - Generation Mix", fontsize=12, fontweight="bold")
        ax_gen.set_ylabel("Generation (MW)", fontsize=10)
        ax_gen.grid(True, alpha=0.3)
        if col_idx == n_cols - 1:
            ax_gen.legend(loc="upper right", fontsize=8)

        # ===== ROW 2: Generation change by technology =====
        ax_delta = axes[1, col_idx]
        gen_change_data = []
        for tech in techs:
            if tech in sol.columns and tech in sol_base.columns:
                base_total = sol_base[tech].sum()
                bat_total = sol[tech].sum()
                if abs(base_total) > 1:
                    gen_change_data.append(
                        {
                            "Technology": tech,
                            "Change (MWh)": bat_total - base_total,
                        }
                    )

        gen_change_df = pd.DataFrame(gen_change_data)
        if not gen_change_df.empty:
            gen_change_df = gen_change_df.sort_values("Change (MWh)")
            bar_colors = [cmap.get(t, "gray") for t in gen_change_df["Technology"]]
            y_pos = np.arange(len(gen_change_df))
            ax_delta.barh(
                y_pos,
                gen_change_df["Change (MWh)"].values,
                color=bar_colors,
                alpha=0.8,
            )
            ax_delta.set_yticks(y_pos)
            ax_delta.set_yticklabels(gen_change_df["Technology"].values)
        else:
            ax_delta.text(0.5, 0.5, "No tech deltas", ha="center", va="center")

        ax_delta.axvline(x=0, color="black", linewidth=1)
        ax_delta.set_xlabel("Generation Change (MWh)", fontsize=10)
        ax_delta.set_title(
            f"{season} - Generation Change by Technology\n(With Battery vs Without)",
            fontsize=11,
            fontweight="bold",
        )
        ax_delta.grid(True, alpha=0.3, axis="x")

        # ===== ROW 3: Battery operation =====
        ax_bat = axes[2, col_idx]
        ax_bat2 = ax_bat.twinx()

        soc_vals = sol["SOC"].values
        discharge_vals = sol["P_discharge"].values
        charge_vals = sol["P_charge"].values

        ax_bat.fill_between(hours, soc_vals, alpha=0.3, color="purple", label="SOC")
        ax_bat.plot(hours, soc_vals, color="purple", linewidth=2)
        ax_bat.set_ylabel("State of Charge (MWh)", color="purple", fontsize=10)
        ax_bat.tick_params(axis="y", labelcolor="purple")
        ax_bat.set_ylim(0, battery_capacity_mwh * 1.1)
        ax_bat.set_xlim(0, 167)

        ax_bat2.bar(
            hours, discharge_vals, alpha=0.7, color="green", label="Discharge", width=1
        )
        ax_bat2.bar(hours, -charge_vals, alpha=0.7, color="red", label="Charge", width=1)
        ax_bat2.set_ylabel("Power (MW)", fontsize=10)
        ax_bat2.axhline(y=0, color="black", linewidth=0.5)
        ax_bat2.set_xlim(0, 167)

        ax_bat.set_title(
            f"{season} - Battery Operation\n"
            f"Charged: {res['total_charged']:,.0f} MWh | "
            f"Discharged: {res['total_discharged']:,.0f} MWh",
            fontsize=11,
            fontweight="bold",
        )
        ax_bat.grid(True, alpha=0.3)
        lines1, labels1 = ax_bat.get_legend_handles_labels()
        lines2, labels2 = ax_bat2.get_legend_handles_labels()
        ax_bat.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

        # ===== ROW 4: Price comparison =====
        ax_price = axes[3, col_idx]
        baseline_p = np.asarray(res["baseline_prices"])
        battery_p = np.asarray(res[price_with_battery_key])

        ax_price.plot(
            hours,
            baseline_p,
            color="darkred",
            linewidth=1.5,
            label="Price (No Battery)",
            alpha=0.7,
        )
        ax_price.plot(
            hours,
            battery_p,
            color="darkgreen",
            linewidth=2,
            label="Price (With Battery)",
        )
        ax_price.fill_between(
            hours,
            baseline_p,
            battery_p,
            where=(baseline_p > battery_p),
            alpha=0.3,
            color="green",
            label="Price Reduction",
        )
        ax_price.fill_between(
            hours,
            baseline_p,
            battery_p,
            where=(baseline_p < battery_p),
            alpha=0.3,
            color="red",
            label="Price Increase",
        )
        ax_price.set_xlim(0, 167)
        ax_price.set_title(
            f"{season} - Electricity Prices\n"
            f"Avg: EUR {baseline_p.mean():.2f} -> EUR {battery_p.mean():.2f}/MWh",
            fontsize=11,
            fontweight="bold",
        )
        ax_price.set_xlabel("Hour", fontsize=10)
        ax_price.set_ylabel("Price (EUR/MWh)", fontsize=10)
        ax_price.legend(loc="upper right", fontsize=8)
        ax_price.grid(True, alpha=0.3)

    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
