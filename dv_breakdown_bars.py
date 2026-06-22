#!/usr/bin/env python3
"""
Create stacked delta-v breakdown bars vs surface pressure.
"""
import math
import os

import numpy as np
import matplotlib.pyplot as plt

import terrestrial_planet as tp
import rocket_optimizer as ro
from constants import M_EARTH, R_EARTH, g_EARTH
from spacefaring import (
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_DRAG_MODEL,
    DEFAULT_MAX_STAGES,
    DEFAULT_MIN_PRESSURE,
    DEFAULT_MAX_PRESSURE,
    MAX_ENGINES,
    MAX_MASS_FLOW_PER_ENGINE,
)


SELECTED_MASSES_EARTH = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
PRESSURE_POINTS = 12
OUTPUT_DIR = "validation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "deltav_breakdown_stacked.png")


def compute_dv_components(mass_earth, pressure_bar):
    planetary_mass = mass_earth * M_EARTH
    planetary_radius = tp.calculate_planetary_radius(planetary_mass)
    escape_velocity = tp.calculate_escape_velocity(planetary_mass, planetary_radius)
    result = ro.optimize_multi_stage_rocket(
        g_EARTH,
        M_EARTH,
        R_EARTH,
        planetary_mass,
        planetary_radius,
        escape_velocity,
        surface_pressure=pressure_bar,
        payload_mass=DEFAULT_PAYLOAD_MASS,
        max_stages=DEFAULT_MAX_STAGES,
        drag_model=DEFAULT_DRAG_MODEL,
        max_engines=MAX_ENGINES,
        max_mass_flow_per_engine=MAX_MASS_FLOW_PER_ENGINE,
    )

    if not result["success"]:
        return math.nan, math.nan, math.nan

    dv_escape = escape_velocity / 1000.0
    dv_gravity = result["gravity_loss"] / 1000.0
    dv_drag = result["drag_loss"] / 1000.0
    return dv_escape, dv_gravity, dv_drag


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pressure_array = np.logspace(
        math.log10(DEFAULT_MIN_PRESSURE),
        math.log10(DEFAULT_MAX_PRESSURE),
        PRESSURE_POINTS,
    )

    n_masses = len(SELECTED_MASSES_EARTH)
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 9), sharey=True)
    axes = axes.flatten()

    x = np.arange(len(pressure_array))
    bar_width = 0.8

    for idx, mass_earth in enumerate(SELECTED_MASSES_EARTH):
        dv_escape_list = []
        dv_gravity_list = []
        dv_drag_list = []

        for pressure in pressure_array:
            dv_escape, dv_gravity, dv_drag = compute_dv_components(mass_earth, pressure)
            dv_escape_list.append(dv_escape)
            dv_gravity_list.append(dv_gravity)
            dv_drag_list.append(dv_drag)

        dv_escape_arr = np.array(dv_escape_list)
        dv_gravity_arr = np.array(dv_gravity_list)
        dv_drag_arr = np.array(dv_drag_list)

        ax = axes[idx]
        ax.bar(x, dv_escape_arr, bar_width, label="DV_escape")
        ax.bar(x, dv_gravity_arr, bar_width, bottom=dv_escape_arr, label="DV_gravity")
        ax.bar(
            x,
            dv_drag_arr,
            bar_width,
            bottom=dv_escape_arr + dv_gravity_arr,
            label="DV_drag",
        )

        ax.set_title(f"{mass_earth:.1f} M_Earth")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p:.2f}" for p in pressure_array], rotation=45, ha="right")
        ax.set_xlabel("Surface Pressure (bar)")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    for ax in axes[::n_cols]:
        ax.set_ylabel("Delta-V (km/s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Delta-V Component Breakdown vs Surface Pressure", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

