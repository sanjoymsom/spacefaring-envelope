#!/usr/bin/env python3
"""
Reproduce published benchmarks from Hippke (2018).

Compares Hippke's quoted values against this model in two modes:
  (A) matching assumptions — single-stage Tsiolkovsky (Isp = 350 s, no structure)
      or ideal multistage (epsilon = 0.10, no engineering caps, R_s = 1);
  (B) full engineering constraints — reliability-weighted staging, 100-engine
      clustering limit, and Hippke's Cheops launch-mass budget (4e5 t).

Ref: Hippke, M. 2018, Int. J. Astrobiol., 18, 393 (arXiv:1803.11384)
"""
import math
import os

import numpy as np

import terrestrial_planet as tp
import rocket_optimizer as ro
import mission_dv as mdv
from constants import G, M_EARTH, R_EARTH, g_EARTH
from spacefaring import (
    DEFAULT_DRAG_MODEL,
    DEFAULT_MAX_STAGES,
    DEFAULT_PAYLOAD_MASS,
    MAX_ENGINES,
    MAX_LAUNCH_MASS_KG,
    MAX_MASS_FLOW_PER_ENGINE,
    STAGE_RELIABILITY,
)

OUTPUT_DIR = "validation"
LATEX_SNIPPET = os.path.join(OUTPUT_DIR, "prior_literature_table.tex")

# Hippke (2018), Int. J. Astrobiol., 18, 393 — arXiv:1803.11384
LIT_HIPPKE = {
    "single_stage_ratio_earth": 26.0,
    "single_stage_ratio_10m_kepler20b": 2700.0,
    "v_esc_10m_kepler20b_kms": 27.1,
    "saturn_v_ratio": 68.0,
    "falcon_heavy_ratio": 83.0,
    "apollo_45t_launch_mass_10m_t": 400_000.0,
    "max_mass_earth_chemical": 10.0,
}

ISP_HIPPKE = 350.0  # s, single-stage analytic benchmark in Hippke Eq. (3)
KEPLER20B_M_EARTH = 10.0
KEPLER20B_R_EARTH = 1.7


def hippke_single_stage_ratio(v_esc_ms, isp=ISP_HIPPKE):
    """Hippke Eq. (3): m0/mf = exp(v_esc / v_exhaust), no structure."""
    v_exhaust = isp * g_EARTH
    return math.exp(v_esc_ms / v_exhaust)


def v_esc_kepler20b():
    """Hippke's 10 M_E, 1.7 R_E reference planet."""
    m = KEPLER20B_M_EARTH * M_EARTH
    r = KEPLER20B_R_EARTH * R_EARTH
    return tp.calculate_escape_velocity(m, r)


def run_model(mass_earth, payload_kg=DEFAULT_PAYLOAD_MASS, pressure_bar=1.0,
              engineering=False, mission_dv_ms=None):
    """Run optimizer; optionally override total dv (for flown-vehicle ratios)."""
    planetary_mass = mass_earth * M_EARTH
    planetary_radius = tp.calculate_planetary_radius(planetary_mass)
    escape_velocity = tp.calculate_escape_velocity(planetary_mass, planetary_radius)

    if engineering:
        max_launch = MAX_LAUNCH_MASS_KG
        max_eng = MAX_ENGINES
        r_s = STAGE_RELIABILITY
    else:
        max_launch = 1.0e15
        max_eng = 10_000
        r_s = 1.0

    result = ro.optimize_multi_stage_rocket(
        g_EARTH, M_EARTH, R_EARTH,
        planetary_mass, planetary_radius, escape_velocity,
        surface_pressure=pressure_bar,
        payload_mass=payload_kg,
        max_stages=DEFAULT_MAX_STAGES,
        drag_model=DEFAULT_DRAG_MODEL,
        max_engines=max_eng,
        max_mass_flow_per_engine=MAX_MASS_FLOW_PER_ENGINE,
        stage_reliability=r_s,
        max_launch_mass=max_launch,
    )

    if mission_dv_ms is not None:
        # Mass ratio at a fixed mission dv (ideal multistage, Hippke-flown comparison)
        lm, _ = _launch_mass_for_dv(mission_dv_ms, payload_kg, engineering=engineering)
        ratio = lm / payload_kg if lm else float("nan")
        return {"mass_ratio": ratio, "launch_mass_kg": lm}

    ratio = result["launch_mass"] / payload_kg if payload_kg > 0 else float("nan")
    return {
        **result,
        "mass_ratio": ratio,
        "v_esc_kms": escape_velocity / 1000.0,
    }


def _launch_mass_for_dv(total_dv, payload, engineering=False):
    """Reliability-optimal launch mass for a fixed total dv (km/s input as m/s)."""
    if engineering:
        r_s = STAGE_RELIABILITY
    else:
        r_s = 1.0
    best_lm = None
    best_n = None
    best_expected = float("inf")
    no_improve = 0
    for n in range(1, 200):
        cfg = ro.optimize_single_stage_count(
            n, total_dv, payload, 350.0, 450.0, 0.10, g_EARTH
        )
        if cfg is None:
            continue
        expected = cfg["launch_mass"] / (r_s ** n)
        if expected < best_expected:
            best_expected = expected
            best_lm = cfg["launch_mass"]
            best_n = n
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 6:
                break
    return best_lm, best_n


def find_ideal_cheops_limit(masses=None):
    """Mass where ideal multistage launch mass exceeds Hippke's Cheops budget."""
    if masses is None:
        # Scan well past the 0.5--20 survey grid so the reported ceiling is the
        # true Cheops-budget crossing, not the grid edge.
        masses = np.linspace(0.5, 40.0, 396)
    last_ok = None
    for m in masses:
        r = run_model(m, engineering=False)
        if r["launch_mass"] <= MAX_LAUNCH_MASS_KG:
            last_ok = m
    return last_ok


def find_engineering_ceiling(masses=None):
    """Mass where launch budget or 100-engine limit is exceeded (1000 kg payload)."""
    if masses is None:
        masses = np.linspace(0.5, 15.0, 150)
    last_ok = None
    first_fail_reason = None
    for m in masses:
        r = run_model(m, engineering=True)
        ok_mass = r["launch_mass"] <= MAX_LAUNCH_MASS_KG
        ok_eng = r.get("engine_number", 0) <= MAX_ENGINES
        if ok_mass and ok_eng and r["launch_mass"] < float("inf"):
            last_ok = m
        elif first_fail_reason is None:
            if not ok_eng:
                first_fail_reason = f"engines>{MAX_ENGINES} at {m:.2f} M_E"
            elif not ok_mass:
                first_fail_reason = f"launch_mass>{MAX_LAUNCH_MASS_KG/1e3:.0f} kt at {m:.2f} M_E"
    return last_ok, first_fail_reason


def fmt_ratio(x, lit=None):
    if x is None or not math.isfinite(x):
        return "---"
    if lit is not None and lit > 0:
        return f"{x:.0f} ({x/lit:.2f}$\\times$)"
    if x >= 100:
        return f"{x:.0f}"
    return f"{x:.1f}"


def fmt_mass_t(tonnes, infeasible=False):
    """Format launch mass in tonnes for the LaTeX table."""
    if tonnes is None or not math.isfinite(tonnes) or tonnes <= 0:
        return "---"
    if tonnes >= 5000:
        thousands = int(round(tonnes / 1000.0))
        cell = f"{thousands}\\," + "000"
    else:
        cell = f"{tonnes:.0f}"
    if infeasible:
        cell += "$^*$"
    return cell


def lit_cell(value, mass=False):
    """Format a literature table cell; None -> em dash."""
    if value is None:
        return "---"
    if mass:
        return fmt_mass_t(value)
    if isinstance(value, float) and value < 100:
        return f"{value:.1f}"
    return f"{int(round(value))}"


def build_rows():
    v_esc_earth = tp.calculate_escape_velocity(M_EARTH, R_EARTH)
    v_esc_k20 = v_esc_kepler20b()

    # --- single-stage (Hippke analytic) ---
    ss_earth_ours = hippke_single_stage_ratio(v_esc_earth)
    ss_10_ours = hippke_single_stage_ratio(v_esc_k20)

    # --- Valencia v_esc at 10 M_E ---
    r10 = run_model(10.0, engineering=False)
    v_esc_valencia = r10["v_esc_kms"]

    # --- flown vehicle mass ratios at mission dv ---
    saturn_dv = mdv.mission_dv_ms("lunar_tli_saturn")
    fh_dv = mdv.mission_dv_ms("mars_hohmann")
    saturn_ideal = run_model(1.0, mission_dv_ms=saturn_dv, engineering=False)
    saturn_eng = run_model(1.0, mission_dv_ms=saturn_dv, engineering=True)
    fh_ideal = run_model(1.0, mission_dv_ms=fh_dv, engineering=False)
    fh_eng = run_model(1.0, mission_dv_ms=fh_dv, engineering=True)

    # --- 1000 kg to escape ---
    esc1_ideal = run_model(1.0, engineering=False)
    esc1_eng = run_model(1.0, engineering=True)
    esc10_ideal = run_model(10.0, engineering=False)
    esc10_eng = run_model(10.0, engineering=True)

    # --- Apollo-class 45 t at 10 M_E (Hippke Kepler-20 b scaling) ---
    apollo_ideal = run_model(10.0, payload_kg=45_000.0, engineering=False)
    apollo_eng = run_model(10.0, payload_kg=45_000.0, engineering=True)

    ceiling, fail_reason = find_engineering_ceiling()
    ideal_cheops = find_ideal_cheops_limit()

    rows = [
        {
            "benchmark": r"1-stage $m_0/m_f$, 1 $M_\oplus$",
            "hippke": lit_cell(LIT_HIPPKE["single_stage_ratio_earth"]),
            "matching": f"{ss_earth_ours:.0f}",
            "engineering": "---",
            "note": "Hippke Eq. (3); $I_{\rm sp}=350$ s",
        },
        {
            "benchmark": r"1-stage $m_0/m_f$, 10 $M_\oplus$",
            "hippke": lit_cell(LIT_HIPPKE["single_stage_ratio_10m_kepler20b"]),
            "matching": f"{ss_10_ours:.0f}",
            "engineering": "---",
            "note": "Hippke: $1.7 R_\oplus$, $v_{\rm esc}=27.1$ km/s",
        },
        {
            "benchmark": r"$v_{\rm esc}$, 10 $M_\oplus$",
            "hippke": lit_cell(LIT_HIPPKE["v_esc_10m_kepler20b_kms"]),
            "matching": f"{v_esc_valencia:.1f}",
            "engineering": f"{v_esc_valencia:.1f}",
            "note": "This work: Valencia $M$--$R$",
        },
        {
            "benchmark": "Saturn~V $m_0/m_f$ (TLI)",
            "hippke": lit_cell(LIT_HIPPKE["saturn_v_ratio"]),
            "matching": f"{saturn_ideal['mass_ratio']:.0f}",
            "engineering": f"{saturn_eng['mass_ratio']:.0f}",
            "note": "GLOM / lunar payload",
        },
        {
            "benchmark": "Falcon Heavy $m_0/m_f$",
            "hippke": lit_cell(LIT_HIPPKE["falcon_heavy_ratio"]),
            "matching": f"{fh_ideal['mass_ratio']:.0f}",
            "engineering": f"{fh_eng['mass_ratio']:.0f}",
            "note": "Fully expendable",
        },
        {
            "benchmark": r"$m_0$ (t), 1 t ESC @ 1 $M_\oplus$",
            "hippke": "---",
            "matching": fmt_mass_t(esc1_ideal["launch_mass"] / 1000),
            "engineering": fmt_mass_t(esc1_eng["launch_mass"] / 1000),
            "note": "Multistage + losses",
        },
        {
            "benchmark": r"$m_0$ (t), 1 t ESC @ 10 $M_\oplus$",
            "hippke": "---",
            "matching": fmt_mass_t(esc10_ideal["launch_mass"] / 1000),
            "engineering": fmt_mass_t(esc10_eng["launch_mass"] / 1000),
            "note": "Multistage + losses",
        },
        {
            "benchmark": r"$m_0$ (t), 45 t ESC @ 10 $M_\oplus$",
            "hippke": lit_cell(LIT_HIPPKE["apollo_45t_launch_mass_10m_t"], mass=True),
            "matching": fmt_mass_t(apollo_ideal["launch_mass"] / 1000, infeasible=True),
            "engineering": fmt_mass_t(apollo_eng["launch_mass"] / 1000, infeasible=True),
            "note": "Apollo on Kepler-20 b",
        },
        {
            "benchmark": r"Max.\ $M_\oplus$, 1 t to ESC",
            "hippke": f"$\\lesssim {LIT_HIPPKE['max_mass_earth_chemical']:.0f}$",
            "matching": f"$\\sim\\!{ideal_cheops:.1f}$",
            "engineering": f"$\\sim\\!{ceiling:.1f}$",
            "note": fail_reason or "",
        },
    ]
    return rows, {
        "ss_earth_ours": ss_earth_ours,
        "ceiling": ceiling,
        "fail_reason": fail_reason,
    }


def print_table(rows):
    print("=" * 100)
    print("HIPPKE (2018) REPRODUCTION TABLE")
    print("=" * 100)
    print(f"{'Benchmark':<48} {'Hippke':>8} {'Ideal':>8} {'Full':>8}")
    print("-" * 100)
    for row in rows:
        print(f"{row['benchmark']:<48} {row['hippke']:>8} "
              f"{row['matching']:>8} {row['engineering']:>8}  {row['note']}")
    print("=" * 100)


def write_latex(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = [
        "% Auto-generated by validation_prior_literature.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{See manuscript for full caption.}",
        "\\label{tab:benchmark}",
        "{\\tablefont\\small\\begin{tabular}{@{}lrrr@{}}",
        "\\midrule",
        "Benchmark & Hippke (2018) & idealized & full model \\\\",
        "\\midrule",
    ]
    for row in rows:
        b = row["benchmark"].replace("%", "\\%")
        lines.append(
            f"{b} & {row['hippke']} & {row['matching']} & {row['engineering']} \\\\"
        )
    lines += [
        "\\midrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]
    with open(LATEX_SNIPPET, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote LaTeX snippet: {LATEX_SNIPPET}")


if __name__ == "__main__":
    rows, meta = build_rows()
    print_table(rows)
    print()
    print(f"Single-stage Earth ratio: ours={meta['ss_earth_ours']:.2f} "
          f"(Hippke={LIT_HIPPKE['single_stage_ratio_earth']})")
    print(f"Engineering ceiling: {meta['ceiling']:.2f} M_E  ({meta['fail_reason']})")
    print()
    write_latex(rows)
