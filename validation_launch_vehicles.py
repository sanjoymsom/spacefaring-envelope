#!/usr/bin/env python3
"""
Validation: anchor the launch model against real launch vehicles.

The model has several sub-models that are best validated separately, on their
own terms, rather than forcing every vehicle through an "escape" mission it was
not designed to fly:

  (A) Rocket-equation + staging core  -> validated by MASS RATIO vs MISSION Δv,
      using each vehicle's ACTUAL mission Δv (LEO, TLI, Mars). This is
      target-agnostic, so LEO vehicles (Electron, N1) are legitimate low-Δv
      anchors here. (These same mass ratios are the quantities Hippke 2018
      quotes, e.g. Saturn V ~68, Falcon Heavy ~83.)

  (B) Absolute launch mass -> validated by ESCAPE-CLASS vehicles (Saturn V at
      TLI, Falcon Heavy at Mars transfer, SLS Block 1 at TLI, Atlas V 551 at
      C3=0), because the model computes mass to escape velocity. LEO-only
      vehicles (N1, Electron) are NOT used here: they were never designed to
      lift their payload to escape, so comparing them on an escape axis is not
      meaningful. The first-stage ENGINE-COUNT comparison is restricted further,
      to the all-liquid, engine-CLUSTERED vehicles for which the model's
      F-1-class engine counting is meaningful (Saturn V, Falcon Heavy); SLS and
      Atlas V are excluded from that axis because their liftoff thrust comes
      largely from SOLID boosters (plus a single liquid core engine, for
      Atlas V), which the liquid-turbopump engine model does not represent.

  (C) First-stage engine count -> validated for all-liquid, engine-CLUSTERED
      vehicles (Saturn V, Falcon Heavy). SLS and Atlas V are excluded because
      liftoff thrust comes largely from solid boosters.

PURPOSE
-------
Show the model reproduces real flight hardware so its super-Earth extrapolation
is anchored - a validation neither Hippke (2018) nor Gonzalez (2020) provides.

----------------------------------------------------------------------------
REAL-VEHICLE REFERENCE DATA  (citations in paper Table tab:valrefs)
----------------------------------------------------------------------------
URLs and DOIs: validation_data_sources.py (DATA_SOURCES dict).

Saturn V (Apollo 8--17, Orloff 2000)
  - Mean ground ignition weight and spacecraft mass (Apollo 8--17).
    Three stages. First stage (S-IC): 5x Rocketdyne F-1.
  - TLI C3: mean Apollo translunar cutoff C3 (8, 10--17; Apollo 9 omitted).
  - F-1 sizing: Hill & Petersen (1992).

Falcon Heavy (fully expendable)
  - Gross lift-off mass 1,420,788 kg; payload to Mars 16,800 kg;
    payload to LEO 63,800 kg. Two serial stages plus two strap-on boosters.
    First stage (3 cores combined): 27x Merlin 1D.

N1 (1964) (Soviet lunar rocket; draft project, never flown)
  - Gross lift-off mass 2.75e6 kg (2750 t); design LEO payload 95,000 kg. Five
    stages. First stage (Blok A): 30x NK-15.

Electron (Rocket Lab)
  - Gross lift-off mass 13,000 kg; payload to LEO up to 300 kg (User's Guide v7.0).
    Two stages + Kick Stage. First stage: 9x Rutherford (ELECTRIC pump-fed).

SLS Block 1 (NASA Space Launch System)
  - Gross lift-off mass ~2.608e6 kg (5.75 million lb); payload to trans-lunar
    injection (TLI) >27,000 kg. Two stages (core + ICPS) plus two five-segment
    solid rocket boosters. Core stage: 4x RS-25 (LH2/LOX). ESCAPE-CLASS
    absolute-mass anchor only: excluded from the engine-count axis (solid-
    booster-dominated liftoff, hydrolox core) and from the turbopump check (LH2).

Atlas V 551 (United Launch Alliance)
  - Gross lift-off mass ~5.87e5 kg; payload to Earth escape (C3 = 0 m^2/s^2)
    ~6,330 kg. First stage: 1x RD-180 + 5 solid boosters; Centaur upper stage.
    ESCAPE-CLASS absolute-mass anchor only: excluded from engine-count axis
    (single liquid core engine + solids) and from turbopump check (staged-
    combustion RD-180).

MISSION Δv BUDGETS (panel A; computed in mission_dv.py)
  - Saturn V TLI: mean Apollo 8--17 TLI C3 from Orloff (2000).
  - SLS Block 1 TLI: geocentric Earth–Moon Hohmann C3.
  - LEO / Mars / escape: as in mission_dv.py module docstring.

ENGINE DATA (F-1 turbopump anchor; Sec. enginesize in paper)
  - F-1: total propellant flow ~2600 kg/s; turbopump 60,000 hp (~45 MW).
    Ref: Stangeland (1992), Fig. 6; mdot from Hill & Petersen (1992).
"""
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import validation_data_sources as vds
import terrestrial_planet as tp
import rocket_optimizer as ro
import mission_dv as mdv
from constants import M_EARTH, R_EARTH, g_EARTH
from spacefaring import (
    DEFAULT_DRAG_MODEL,
    DEFAULT_MAX_STAGES,
    STAGE_RELIABILITY,
    MAX_LAUNCH_MASS_KG,
)

OUTPUT_DIR = "validation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "launch_vehicle_validation.png")
# LaTeX includes this figure from paper/ (same directory as spacefaring_iau.tex).
PAPER_OUTPUT_FILE = os.path.join("paper", "launch_vehicle_validation.png")

# Panel A: short offset labels; arrows only where the escape-class cluster overlaps.
ANN_A = {
    "SLS Block 1": {"xytext": (-2, 12), "ha": "right", "va": "bottom", "arrow": True},
    "Atlas V 551": {"xytext": (9, 4), "ha": "left", "va": "center", "arrow": True},
    "Falcon Heavy (exp.)": {"xytext": (9, -7), "ha": "left", "va": "top", "arrow": True},
    "Saturn V": {"xytext": (-9, -7), "ha": "right", "va": "top", "arrow": True},
    "Electron": {"xytext": (0, 9), "ha": "center"},
    "N1 (1964)": {"xytext": (8, -11), "ha": "left"},
}
ANN_B = {"SLS Block 1": (-12, 10), "Saturn V": (6, -6),
         "Falcon Heavy (exp.)": (6, -14), "Atlas V 551": (6, 7)}

LABEL_FONT = 9
LABEL_ARROW = dict(arrowstyle="-", color="0.45", lw=0.7, shrinkA=0, shrinkB=2)

# Model propulsion assumptions (match rocket_optimizer.optimize_multi_stage_rocket)
ISP_LOW = 350.0
ISP_HIGH = 450.0
EPSILON = 0.10
# Turbopump model assumptions (match engineering_validation.check_turbopump_power_limitation)
PUMP_EFFICIENCY = 0.70
PROP_DENSITY = 1000.0  # kg/m^3 (RP-1/LOX lumped)
PUMP_RISE_Pa = 100e5  # ~100 bar pump discharge rise (match Eq. turbo in paper)
F1_MDOT_KG_S = 2600.0
F1_PUMP_MW_PUB = 45.0  # Stangeland (1992), 60,000 hp

# ---------------------------------------------------------------------------
# Reference vehicles (see docstring for citations)
#   mass_ratio = glom_kg / payload_kg
#   mission = key for mission_dv.mission_dv_kms; Saturn masses from Orloff means
# ---------------------------------------------------------------------------
# Flags:
#   escape_class   -> appears in panel (B), absolute launch mass to escape
#   engine_compare -> appears in panel (C), first-stage engine count. Restricted
#                     to all-liquid, engine-CLUSTERED vehicles, the only ones for
#                     which the model's F-1-class engine counting is meaningful.
#                     Solid-booster vehicles (SLS, Atlas V) and LEO-only vehicles
#                     (N1, Electron) are excluded from this axis.
# All vehicles appear in panel (A), mass ratio vs mission Δv (target-agnostic).
_VEHICLES_RAW = [
    {"name": "Saturn V",
     "glom_kg": mdv.mean_saturn_v_glom_kg(),
     "payload_kg": mdv.mean_saturn_v_spacecraft_kg(),
     "target": "TLI",          "mission": "lunar_tli_saturn", "escape_class": True,
     "engine_compare": True,  "stages": 3, "first_stage_eng": 5,
     "source_keys": ("Orloff2000",)},
    {"name": "Falcon Heavy (exp.)", "glom_kg": 1.421e6, "payload_kg": 16800.0,
     "target": "Mars (C3>0)",  "mission": "mars_hohmann", "escape_class": True,
     "engine_compare": True,  "stages": 2, "first_stage_eng": 27,
     "source_keys": ("SpaceXFH",)},
    {"name": "SLS Block 1",         "glom_kg": 2.608e6, "payload_kg": 27000.0,
     "target": "TLI",          "mission": "lunar_tli", "escape_class": True,
     "engine_compare": False, "stages": 2, "first_stage_eng": 4,
     "source_keys": ("NASASLS2022",)},
    {"name": "Atlas V 551",         "glom_kg": 5.87e5,  "payload_kg": 6330.0,
     "target": "Escape (C3=0)", "mission": "earth_escape_c3_0", "escape_class": True,
     "engine_compare": False, "stages": 2, "first_stage_eng": 1,
     "source_keys": ("AstronautixAtlas551", "Schmidt2010")},
    {"name": "N1 (1964)",           "glom_kg": 2.75e6,  "payload_kg": 95000.0,
     "target": "LEO (design)", "mission": "leo", "escape_class": False,
     "engine_compare": False, "stages": 5, "first_stage_eng": 30,
     "source_keys": ("AstronautixN1",)},
    {"name": "Electron",            "glom_kg": 1.3e4,   "payload_kg": 300.0,
     "target": "LEO",          "mission": "leo", "escape_class": False,
     "engine_compare": False, "stages": 2, "first_stage_eng": 9,
     "source_keys": ("RocketLabElectron",)},
]


def _vehicles_with_dv():
    out = []
    for v in _VEHICLES_RAW:
        row = dict(v)
        row["mission_dv_kms"] = round(mdv.mission_dv_kms(v["mission"]), 3)
        keys = row.pop("source_keys", ())
        row["sources"] = {k: vds.DATA_SOURCES[k] for k in keys}
        out.append(row)
    return out


VEHICLES = _vehicles_with_dv()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def model_launch_mass_for_dv(total_dv, payload=1000.0):
    """Reliability-weighted optimal launch mass for a given total Δv (m/s).

    Mirrors the selection in rocket_optimizer (minimize launch_mass / R_s**n).
    Mass ratio is independent of payload for fixed Δv, so any payload works.
    """
    best_expected = float("inf")
    best_lm = None
    best_n = None
    no_improvement = 0
    for n in range(1, 200):
        cfg = ro.optimize_single_stage_count(
            n, total_dv, payload, ISP_LOW, ISP_HIGH, EPSILON, g_EARTH
        )
        if cfg is None:
            continue
        expected = cfg["launch_mass"] / (STAGE_RELIABILITY ** n)
        if expected < best_expected:
            best_expected = expected
            best_lm = cfg["launch_mass"]
            best_n = n
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 6:
                break
    return best_lm, best_n


def run_model_at_earth(payload_kg):
    """Run the full model for delivering payload_kg to Earth escape velocity."""
    planetary_mass = M_EARTH
    planetary_radius = tp.calculate_planetary_radius(planetary_mass)
    escape_velocity = tp.calculate_escape_velocity(planetary_mass, planetary_radius)
    return ro.optimize_multi_stage_rocket(
        g_EARTH, M_EARTH, R_EARTH,
        planetary_mass, planetary_radius, escape_velocity,
        surface_pressure=1.0,
        payload_mass=payload_kg,
        max_stages=DEFAULT_MAX_STAGES,
        drag_model=DEFAULT_DRAG_MODEL,
        stage_reliability=STAGE_RELIABILITY,
        max_launch_mass=MAX_LAUNCH_MASS_KG,
    )


def f1_turbopump_power_MW(mdot=F1_MDOT_KG_S):
    """Turbopump model power (MW): P = mdot * dP / (eta * rho), dP ~ 100 bar rise."""
    return mdot * PUMP_RISE_Pa / (PUMP_EFFICIENCY * PROP_DENSITY) / 1.0e6


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_tables():
    print("=" * 96)
    print("(A) ROCKET-EQUATION CORE: mass ratio vs mission dv  (all vehicles)")
    print("=" * 96)
    print(f"{'Vehicle':<20} {'Mission':>12} {'dv(km/s)':>9} "
          f"{'ratio act':>10} {'ratio mod':>10} {'mod/act':>8}")
    print("-" * 96)
    for v in VEHICLES:
        ratio_act = v["glom_kg"] / v["payload_kg"]
        lm, n = model_launch_mass_for_dv(v["mission_dv_kms"] * 1000.0)
        ratio_mod = lm / 1000.0  # payload used = 1000 kg -> ratio = lm/1000
        print(f"{v['name']:<20} {v['target']:>12} {v['mission_dv_kms']:>9.1f} "
              f"{ratio_act:>10.1f} {ratio_mod:>10.1f} {ratio_mod/ratio_act:>8.2f}")
    print("-" * 96)
    print("Model uses eps=0.10 and Isp 350/450 s (idealized) -> a best-case lower")
    print("bound; real vehicles sit above (heavier) due to real eps, lower Isp, margins.")
    print()

    print("=" * 96)
    print("(B) ABSOLUTE MASS + ENGINES: escape-class vehicles only")
    print("=" * 96)
    print(f"{'Vehicle':<20} {'Payload(kg)':>11} {'GLOM act(t)':>11} {'GLOM mod(t)':>11} "
          f"{'stg a/m':>8} {'eng a/m':>8}")
    print("-" * 96)
    for v in VEHICLES:
        if not v["escape_class"]:
            continue
        r = run_model_at_earth(v["payload_kg"])
        glom_mod = r["launch_mass"] / 1000.0
        # Engine count compared only for all-liquid, clustered vehicles.
        eng_act = f"{v['first_stage_eng']}" if v["engine_compare"] else "--"
        eng_col = f"{eng_act}/{r['engine_number']}"
        print(f"{v['name']:<20} {v['payload_kg']:>11,.0f} {v['glom_kg']/1000:>11,.0f} "
              f"{glom_mod:>11,.0f} {v['stages']}/{r['num_stages']:<6} "
              f"{eng_col:<8}")
    print("-" * 96)
    print("Engine count (F-1-class-equivalent units) compared only for all-liquid,")
    print("clustered vehicles; '--' = solid-augmented (SLS, Atlas V), not comparable.")
    print()

    p_f1 = f1_turbopump_power_MW()
    print("=" * 96)
    print("F-1 TURBOPUMP CHECK (Eq. turbo in paper; Stangeland 1992)")
    print("=" * 96)
    print(f"Model F-1 unit ({F1_MDOT_KG_S:.0f} kg/s @ 100-bar pump rise) = "
          f"{p_f1:.1f} MW vs published ~{F1_PUMP_MW_PUB:.0f} MW -> "
          f"{p_f1/F1_PUMP_MW_PUB:.2f}x")
    print("=" * 96)


# ---------------------------------------------------------------------------
# Plotting (1x3)
# ---------------------------------------------------------------------------
def _annotate_vehicle(ax, name, xy, spec):
    """Place a vehicle label; optional short leader line to the marker."""
    kw = dict(fontsize=LABEL_FONT)
    for key in ("ha", "va"):
        if key in spec:
            kw[key] = spec[key]
    textcoords = spec.get("textcoords", "offset points")
    xytext = spec.get("xytext", (6, 6))
    arrowprops = LABEL_ARROW if spec.get("arrow") else None
    ax.annotate(
        name,
        xy,
        xytext=xytext,
        textcoords=textcoords,
        arrowprops=arrowprops,
        **kw,
    )


def make_plot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (A) Mass ratio vs mission Δv  -- all vehicles
    axA = axes[0]
    dv_grid = np.linspace(7000.0, 14000.0, 50)
    ratio_curve = []
    for dv in dv_grid:
        lm, _ = model_launch_mass_for_dv(dv)
        ratio_curve.append(lm / 1000.0 if lm else np.nan)
    axA.plot(dv_grid / 1000.0, ratio_curve, color="tab:blue", lw=2,
             label="Model (reliability-optimal)")
    for v in VEHICLES:
        ratio = v["glom_kg"] / v["payload_kg"]
        marker = "o" if v["escape_class"] else "s"
        face = "tab:orange" if v["escape_class"] else "none"
        axA.scatter(v["mission_dv_kms"], ratio, marker=marker, facecolors=face,
                    edgecolors="black", s=110, zorder=5, linewidths=1.6)
        _annotate_vehicle(
            axA, v["name"], (v["mission_dv_kms"], ratio),
            ANN_A.get(v["name"], {"xytext": (6, 6), "textcoords": "offset points"}),
        )
    axA.set_yscale("log")
    axA.set_xlim(7.2, 14.2)
    ymax = max(v["glom_kg"] / v["payload_kg"] for v in VEHICLES)
    axA.set_ylim(8, ymax * 1.22)
    axA.set_xlabel("Mission Δv (km/s)")
    axA.set_ylabel("Mass ratio (GLOM / payload)")
    axA.set_title("(A) Rocket-equation core: mass ratio vs mission Δv\n(all vehicles, target-agnostic)")
    axA.grid(True, which="both", alpha=0.25, linestyle="--", linewidth=0.5, zorder=0)

    # (B) Launch mass vs payload  -- escape-class only
    axB = axes[1]
    payloads = np.logspace(2.0, 5.2, 50)
    lm_curve, stg_curve, eng_curve = [], [], []
    for p in payloads:
        r = run_model_at_earth(p)
        lm_curve.append(r["launch_mass"] / 1000.0 if r["success"] else np.nan)
        stg_curve.append(r["num_stages"] if r["success"] else np.nan)
        eng_curve.append(r["engine_number"] if r["success"] else np.nan)
    axB.plot(payloads, lm_curve, color="tab:blue", lw=2, label="Model (to escape)")
    for v in VEHICLES:
        if not v["escape_class"]:
            continue
        axB.scatter(v["payload_kg"], v["glom_kg"] / 1000.0, marker="o",
                    facecolors="tab:orange", edgecolors="black", s=90, zorder=5,
                    linewidths=1.5)
        axB.annotate(v["name"], (v["payload_kg"], v["glom_kg"] / 1000.0),
                     textcoords="offset points",
                     xytext=ANN_B.get(v["name"], (6, 6)), fontsize=8)
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("Payload to escape (kg)")
    axB.set_ylabel("Launch mass (t)")
    axB.set_title("(B) Absolute launch mass vs payload\n(escape-class anchors only)")
    axB.grid(True, which="both", alpha=0.3, linestyle="--", linewidth=0.5)

    # (C) First-stage engines vs payload -- escape-class only
    axC = axes[2]
    axC.plot(payloads, eng_curve, color="tab:blue", lw=2, label="Model (F-1-class equiv.)")
    for v in VEHICLES:
        if not v["engine_compare"]:
            continue
        axC.scatter(v["payload_kg"], v["first_stage_eng"], marker="o",
                    facecolors="tab:orange", edgecolors="black", s=90, zorder=5,
                    linewidths=1.5)
        axC.annotate(v["name"], (v["payload_kg"], v["first_stage_eng"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    axC.set_xscale("log")
    axC.set_xlabel("Payload to escape (kg)")
    axC.set_ylabel("First-stage engines")
    axC.set_title("(C) First-stage engine count vs payload\n(all-liquid clustered vehicles; F-1-class-equivalent units)")
    axC.grid(True, which="both", alpha=0.3, linestyle="--", linewidth=0.5)

    # Shared marker legend for A-C
    legend_elems = [
        Line2D([0], [0], color="tab:blue", lw=2, label="Model"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="tab:orange",
               linestyle="none", markersize=9, label="Escape-class vehicle"),
        Line2D([0], [0], marker="s", color="black", markerfacecolor="none",
               linestyle="none", markersize=9, label="LEO vehicle (panel A only)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Launch-model validation against real vehicles",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(OUTPUT_FILE, dpi=250, bbox_inches="tight")
    os.makedirs(os.path.dirname(PAPER_OUTPUT_FILE), exist_ok=True)
    fig.savefig(PAPER_OUTPUT_FILE, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_FILE}")
    print(f"Saved plot: {PAPER_OUTPUT_FILE}")


if __name__ == "__main__":
    print_tables()
    print()
    make_plot()
