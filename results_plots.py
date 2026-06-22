#!/usr/bin/env python3
"""
Line plots and grid-survey output for spacefaring analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from csv_generator import export_grid_results_to_csv
from constants import KEPLER_20B_MASS_ME


LINE_PRESSURES_BAR = (0.1, 1.0, 10.0)
LINE_PLOT_FIGSIZE = (8, 5)
LINE_PLOT_DPI = 300
# Okabe--Ito, readable on white (avoid light yellow)
LINE_PLOT_COLORS = ('#0072B2', '#009E73', '#D55E00')
GRID_SAMPLE_TICKS = 10


def _nearest_pressure_index(pressure_array, target_bar):
    """Index of pressure_array value closest to target_bar."""
    return int(np.argmin(np.abs(pressure_array - target_bar)))


def _add_launchmass_reference_lines(ax, saturn_v_wet_mass_tons, cheops_launch_mass_tons=None):
    """Horizontal benchmarks: Saturn V GLOM and Hippke Cheops pyramid limit."""
    ax.axhline(
        saturn_v_wet_mass_tons,
        color='red',
        ls='--',
        lw=1.5,
        label=f'Saturn V GLOM ({saturn_v_wet_mass_tons:.0f} t)',
    )
    if cheops_launch_mass_tons is not None:
        ax.axhline(
            cheops_launch_mass_tons,
            color='#984EA3',
            ls='-.',
            lw=1.5,
            label=f'Cheops pyramid ($4\\times10^5$ t)',
        )


def _add_kepler20b_reference_line(ax, mass_me=KEPLER_20B_MASS_ME):
    """Vertical benchmark: Hippke (2018) Kepler-20 b case study at fixed mass."""
    ax.axvline(
        mass_me,
        color='tab:green',
        ls='-.',
        lw=1.5,
        label=f'Kepler-20 b (${mass_me:g}\\,M_\\oplus$)',
    )


def create_launchmass_lines_plot(
    launch_mass_matrix,
    mass_earth_array,
    pressure_array,
    output_filename,
    saturn_v_wet_mass_tons,
    cheops_launch_mass_tons=None,
    pressures_bar=LINE_PRESSURES_BAR,
    paper_output_filename=None,
):
    """Planetary mass vs required launch mass at fixed surface pressures."""
    launch_mass_tons = launch_mass_matrix / 1000.0

    fig, ax = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

    for p_bar, color in zip(pressures_bar, LINE_PLOT_COLORS):
        j = _nearest_pressure_index(pressure_array, p_bar)
        p_actual = pressure_array[j]
        y = launch_mass_tons[:, j].astype(float)
        valid = np.isfinite(y) & (y > 0)
        label = f'{p_bar:g} bar'
        if abs(p_actual - p_bar) / p_bar > 0.05:
            label += f' (${p_actual:.2g}$ bar)'
        ax.plot(
            mass_earth_array[valid],
            y[valid],
            color=color,
            lw=2,
            label=label,
        )

    _add_launchmass_reference_lines(
        ax, saturn_v_wet_mass_tons, cheops_launch_mass_tons=cheops_launch_mass_tons
    )
    _add_kepler20b_reference_line(ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Planetary mass ($M_\\oplus$)', fontsize=12)
    ax.set_ylabel('Required launch mass (t)', fontsize=12)
    ax.set_title(
        'Launch mass to escape vs planetary mass\n(1000 kg payload)',
        fontsize=13,
        fontweight='bold',
    )
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(mass_earth_array[0], mass_earth_array[-1])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=LINE_PLOT_DPI, bbox_inches='tight')
    print(f"Launch-mass line plot saved: {output_filename}")
    if paper_output_filename:
        import os
        os.makedirs(os.path.dirname(paper_output_filename) or '.', exist_ok=True)
        plt.savefig(paper_output_filename, dpi=LINE_PLOT_DPI, bbox_inches='tight')
        print(f"Launch-mass line plot saved: {paper_output_filename}")
    plt.close(fig)


def create_launchmass_lines_from_csv(
    csv_path,
    output_filename,
    saturn_v_wet_mass_tons,
    cheops_launch_mass_tons=None,
    pressures_bar=LINE_PRESSURES_BAR,
    paper_output_filename=None,
):
    """Regenerate the launch-mass line plot from an existing launchmass CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    mass_col = 'M_planet (M_Earth)'
    p_col = 'Pressure (bar)'
    lm_col = 'Launch Mass (tons)'
    masses = np.sort(df[mass_col].unique())

    fig, ax = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

    for p_bar, color in zip(pressures_bar, LINE_PLOT_COLORS):
        m_vals, lm_vals = [], []
        for m in masses:
            rows = df.loc[df[mass_col] == m]
            idx = (rows[p_col] - p_bar).abs().idxmin()
            lm = float(rows.loc[idx, lm_col])
            p_actual = float(rows.loc[idx, p_col])
            if np.isfinite(lm) and lm > 0:
                m_vals.append(m)
                lm_vals.append(lm)
        label = f'{p_bar:g} bar'
        if m_vals and abs(p_actual - p_bar) / p_bar > 0.05:
            label += f' (${p_actual:.2g}$ bar)'
        ax.plot(m_vals, lm_vals, color=color, lw=2, label=label)

    _add_launchmass_reference_lines(
        ax, saturn_v_wet_mass_tons, cheops_launch_mass_tons=cheops_launch_mass_tons
    )
    _add_kepler20b_reference_line(ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Planetary mass ($M_\\oplus$)', fontsize=12)
    ax.set_ylabel('Required launch mass (t)', fontsize=12)
    ax.set_title(
        'Launch mass to escape vs planetary mass\n(1000 kg payload)',
        fontsize=13,
        fontweight='bold',
    )
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(masses[0], masses[-1])
    plt.tight_layout()
    plt.savefig(output_filename, dpi=LINE_PLOT_DPI, bbox_inches='tight')
    print(f"Launch-mass line plot saved: {output_filename}")
    if paper_output_filename:
        import os
        os.makedirs(os.path.dirname(paper_output_filename) or '.', exist_ok=True)
        plt.savefig(paper_output_filename, dpi=LINE_PLOT_DPI, bbox_inches='tight')
        print(f"Launch-mass line plot saved: {paper_output_filename}")
    plt.close(fig)


def create_engines_lines_plot(
    num_engines_matrix,
    mass_earth_array,
    pressure_array,
    output_filename,
    max_engines=100,
    pressures_bar=LINE_PRESSURES_BAR,
    paper_output_filename=None,
):
    """First-stage engine count (F-1-class equiv.) vs planetary mass at fixed pressures."""
    fig, ax = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

    for p_bar, color in zip(pressures_bar, LINE_PLOT_COLORS):
        j = _nearest_pressure_index(pressure_array, p_bar)
        p_actual = pressure_array[j]
        y = num_engines_matrix[:, j].astype(float)
        valid = np.isfinite(y) & (y > 0)
        label = f'{p_bar:g} bar'
        if abs(p_actual - p_bar) / p_bar > 0.05:
            label += f' (${p_actual:.2g}$ bar)'
        ax.plot(
            mass_earth_array[valid],
            y[valid],
            color=color,
            lw=2,
            label=label,
        )

    ax.axhline(
        max_engines,
        color='red',
        ls='--',
        lw=1.5,
        label=f'$N_{{\\rm eng}} = {max_engines}$ limit',
    )
    _add_kepler20b_reference_line(ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Planetary mass ($M_\\oplus$)', fontsize=12)
    ax.set_ylabel('First-stage engines (F-1-class equiv.)', fontsize=12)
    ax.set_title(
        'Engine count vs planetary mass\n(1000 kg payload)',
        fontsize=13,
        fontweight='bold',
    )
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(mass_earth_array[0], mass_earth_array[-1])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=LINE_PLOT_DPI, bbox_inches='tight')
    print(f"Engine-count line plot saved: {output_filename}")
    if paper_output_filename:
        import os
        os.makedirs(os.path.dirname(paper_output_filename) or '.', exist_ok=True)
        plt.savefig(paper_output_filename, dpi=LINE_PLOT_DPI, bbox_inches='tight')
        print(f"Engine-count line plot saved: {paper_output_filename}")
    plt.close(fig)


def create_engines_lines_from_csv(
    csv_path,
    output_filename,
    max_engines=100,
    pressures_bar=LINE_PRESSURES_BAR,
    paper_output_filename=None,
):
    """Regenerate the engine-count line plot from an existing engines CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    mass_col = 'M_planet (M_Earth)'
    p_col = 'Pressure (bar)'
    eng_col = 'Num Engines'
    masses = np.sort(df[mass_col].unique())

    fig, ax = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

    for p_bar, color in zip(pressures_bar, LINE_PLOT_COLORS):
        m_vals, eng_vals = [], []
        p_actual = p_bar
        for m in masses:
            rows = df.loc[df[mass_col] == m]
            idx = (rows[p_col] - p_bar).abs().idxmin()
            eng = float(rows.loc[idx, eng_col])
            p_actual = float(rows.loc[idx, p_col])
            if np.isfinite(eng) and eng > 0:
                m_vals.append(m)
                eng_vals.append(eng)
        label = f'{p_bar:g} bar'
        if m_vals and abs(p_actual - p_bar) / p_bar > 0.05:
            label += f' (${p_actual:.2g}$ bar)'
        ax.plot(m_vals, eng_vals, color=color, lw=2, label=label)

    ax.axhline(
        max_engines,
        color='red',
        ls='--',
        lw=1.5,
        label=f'$N_{{\\rm eng}} = {max_engines}$ limit',
    )
    _add_kepler20b_reference_line(ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Planetary mass ($M_\\oplus$)', fontsize=12)
    ax.set_ylabel('First-stage engines (F-1-class equiv.)', fontsize=12)
    ax.set_title(
        'Engine count vs planetary mass\n(1000 kg payload)',
        fontsize=13,
        fontweight='bold',
    )
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(masses[0], masses[-1])
    plt.tight_layout()
    plt.savefig(output_filename, dpi=LINE_PLOT_DPI, bbox_inches='tight')
    print(f"Engine-count line plot saved: {output_filename}")
    if paper_output_filename:
        import os
        os.makedirs(os.path.dirname(paper_output_filename) or '.', exist_ok=True)
        plt.savefig(paper_output_filename, dpi=LINE_PLOT_DPI, bbox_inches='tight')
        print(f"Engine-count line plot saved: {paper_output_filename}")
    plt.close(fig)


def run_mass_pressure_survey(
    explore_func,
    M_EARTH,
    SATURN_V_WET_MASS_TONS,
    CHEOPS_LAUNCH_MASS_TONS=4.0e5,
    mass_range_earth=None,
    pressure_range=None,
    n_mass_points=None,
    n_pressure_points=None,
    payload_mass=None,
    max_stages=None,
    drag_model=None,
    output_feasibility_csv=None,
    output_deltav_csv=None,
    output_stages_csv=None,
    output_engines_csv=None,
    output_launchmass_csv=None,
    output_reliability_csv=None,
    output_launchmass_lines_plot=None,
    output_launchmass_paper_plot=None,
    output_engines_lines_plot=None,
    output_engines_paper_plot=None,
    default_mass_range=(0.5, 20.0),
    default_pressure_range=(0.1, 10.0),
    default_n_mass_points=50,
    default_n_pressure_points=30,
    default_payload_mass=1000.0,
    max_engines_first_stage=100,
):
    """
    Compute feasibility/performance over a mass--pressure grid; export CSVs and line plots.
    """
    if mass_range_earth is None:
        mass_range_earth = default_mass_range
    if pressure_range is None:
        pressure_range = default_pressure_range
    if n_mass_points is None:
        n_mass_points = default_n_mass_points
    if n_pressure_points is None:
        n_pressure_points = default_n_pressure_points
    if payload_mass is None:
        payload_mass = default_payload_mass

    print("=" * 80)
    print("MASS--PRESSURE SURVEY: Planetary Mass vs Surface Pressure")
    print("=" * 80)
    print()
    print(f"Mass range: {mass_range_earth[0]:.1f} - {mass_range_earth[1]:.1f} M_Earth")
    print(f"Pressure range: {pressure_range[0]:.1f} - {pressure_range[1]:.1f} bar")
    print(f"Grid size: {n_mass_points} x {n_pressure_points} = {n_mass_points * n_pressure_points} points")
    print(f"Payload: {payload_mass} kg")
    print()

    mass_earth_array = np.logspace(
        np.log10(mass_range_earth[0]),
        np.log10(mass_range_earth[1]),
        n_mass_points,
    )
    mass_array = mass_earth_array * M_EARTH

    pressure_array = np.unique(np.sort(np.concatenate([
        np.logspace(
            np.log10(pressure_range[0]),
            np.log10(pressure_range[1]),
            n_pressure_points,
        ),
        np.array(LINE_PRESSURES_BAR, dtype=float),
    ])))
    n_pressure_points = len(pressure_array)

    print("Computing feasibility for each mass-pressure combination...")
    results = explore_func(
        mass_array,
        surface_pressure_array=pressure_array,
        payload_mass=payload_mass,
        max_stages=max_stages,
        drag_model=drag_model,
    )

    feasibility_matrix = results['feasible'].astype(float)
    launch_mass_matrix = results['launch_mass_array']
    num_stages_matrix = results['num_stages_array']
    num_engines_matrix = results['num_engines_array']
    deltav_matrix = results['deltav_array']
    reliability_matrix = results.get('reliability_array')

    print()
    print("Exporting results to CSV...")
    export_grid_results_to_csv(
        mass_earth_array,
        mass_array,
        pressure_array,
        feasibility_matrix,
        launch_mass_matrix,
        num_stages_matrix,
        num_engines_matrix,
        deltav_matrix,
        output_feasibility_csv,
        output_deltav_csv,
        output_stages_csv,
        output_engines_csv,
        output_launchmass_csv,
        reliability_matrix=reliability_matrix,
        output_reliability_csv=output_reliability_csv,
    )
    print()

    if output_launchmass_lines_plot:
        create_launchmass_lines_plot(
            launch_mass_matrix,
            mass_earth_array,
            pressure_array,
            output_launchmass_lines_plot,
            saturn_v_wet_mass_tons=SATURN_V_WET_MASS_TONS,
            cheops_launch_mass_tons=CHEOPS_LAUNCH_MASS_TONS,
            paper_output_filename=output_launchmass_paper_plot,
        )

    if output_engines_lines_plot:
        create_engines_lines_plot(
            num_engines_matrix,
            mass_earth_array,
            pressure_array,
            output_engines_lines_plot,
            max_engines=max_engines_first_stage,
            paper_output_filename=output_engines_paper_plot,
        )

    n_feasible = np.sum(feasibility_matrix == 1.0)
    n_infeasible = np.sum(feasibility_matrix == 0.0)
    total = n_feasible + n_infeasible
    feasible_fraction = n_feasible / total if total > 0 else 0.0

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total combinations tested: {total}")
    print(f"Feasible: {n_feasible} ({100 * feasible_fraction:.1f}%)")
    print(f"Infeasible: {n_infeasible} ({100 * (1 - feasible_fraction):.1f}%)")
    print()

    print("Maximum feasible mass for each pressure:")
    sample_interval = max(1, n_pressure_points // GRID_SAMPLE_TICKS)
    sample_indices = list(range(0, n_pressure_points, sample_interval))
    if sample_indices[-1] != n_pressure_points - 1:
        sample_indices.append(n_pressure_points - 1)

    for j in sample_indices:
        pressure = pressure_array[j]
        feasible_masses = mass_earth_array[feasibility_matrix[:, j] == 1.0]
        if len(feasible_masses) > 0:
            max_mass = np.max(feasible_masses)
            print(f"  {pressure:.2f} bar: {max_mass:.2f} M_Earth")
        else:
            print(f"  {pressure:.2f} bar: No feasible solutions")
    print("=" * 80)

    return {
        'mass_array': mass_array,
        'mass_earth_array': mass_earth_array,
        'pressure_array': pressure_array,
        'feasibility_matrix': feasibility_matrix,
        'deltav_matrix': deltav_matrix,
        'stages_matrix': num_stages_matrix,
        'engines_matrix': num_engines_matrix,
        'reliability_matrix': reliability_matrix,
        'csv_paths': {
            'feasibility': output_feasibility_csv,
            'deltav': output_deltav_csv,
            'stages': output_stages_csv,
            'engines': output_engines_csv,
            'launchmass': output_launchmass_csv,
            'reliability': output_reliability_csv,
        },
        'plot_paths': {
            'launchmass': output_launchmass_lines_plot,
            'engines': output_engines_lines_plot,
        },
    }
