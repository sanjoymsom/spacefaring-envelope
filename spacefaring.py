#!/usr/bin/env python3
"""
Spacefaring feasibility analysis.

Explores escape feasibility over planetary mass and surface pressure grids.
"""
import sys
import os
import math
import numpy as np
import terrestrial_planet as tp
#from terrestrial_planet import calculate_planetary_radius, calculate_escape_velocity
import rocket_optimizer as ro
from engineering_validation import check_turbopump_power_limitation
from csv_generator import export_results_to_csv, export_planetary_array_results_to_csv
from results_plots import run_mass_pressure_survey
from constants import G, g_EARTH, M_EARTH, R_EARTH
import mission_dv as _mdv


# ============================================================================
# HARDCODED VALUES - Modify these as needed
# ============================================================================

# Default rocket parameters
DEFAULT_PAYLOAD_MASS = 1000.0  # kg (Voyager-class probe)
DEFAULT_SURFACE_PRESSURE = 1.0  # bar (Earth-like atmosphere)
DEFAULT_DRAG_MODEL = 'altitude_dependent'  # 'simple' or 'altitude_dependent'
DEFAULT_MAX_STAGES = None  # None = adaptive, or specify maximum number

# Engine count constraint (legacy turbopump checks remain in validation module)
MAX_ENGINES = 100  # Maximum practical number of engines on first stage
MAX_MASS_FLOW_PER_ENGINE = 2600.0  # kg/s (engineering limit for F-1 engine)

# Staging risk and resource budget (govern feasibility)
# Per-stage reliability: each stage adds a separation and an ignition event.
# Mission reliability decays as R_s**n. 0.97 is a mid estimate (0.95 conservative,
# 0.98 mature); orbital launch vehicles (2-3 stages) imply ~0.95-0.98 per stage.
STAGE_RELIABILITY = 0.97
# Launch-mass budget: feasibility ceiling on the rocket that must be built.
# Default ~4e5 t follows Hippke (2018) "Pyramid of Cheops" practical limit.
MAX_LAUNCH_MASS_TONS = 4.0e5
MAX_LAUNCH_MASS_KG = MAX_LAUNCH_MASS_TONS * 1000.0

# Default mass ranges for exploration
DEFAULT_MIN_MASS_EARTH = 0.5  # M_Earth
DEFAULT_MAX_MASS_EARTH = 20.0  # M_Earth
DEFAULT_N_MASS_POINTS = 50  # Number of mass points for logarithmic spacing

# Default pressure ranges for mass--pressure survey
DEFAULT_MIN_PRESSURE = 0.1  # bar
DEFAULT_MAX_PRESSURE = 10.0  # bar
DEFAULT_N_PRESSURE_POINTS = 30  # Number of pressure points for grid survey

# Default output directories and filenames
DEFAULT_CSV_DIR = 'csv'
DEFAULT_PLOT_DIR = 'validation'
DEFAULT_CSV_FILENAME = os.path.join(DEFAULT_CSV_DIR, 'spacefaring_results.csv')
DEFAULT_FEASIBILITY_CSV = os.path.join(DEFAULT_CSV_DIR, 'feasibility_grid.csv')
DEFAULT_DELTAV_CSV = os.path.join(DEFAULT_CSV_DIR, 'deltav_grid.csv')
DEFAULT_STAGES_CSV = os.path.join(DEFAULT_CSV_DIR, 'stages_grid.csv')
DEFAULT_ENGINES_CSV = os.path.join(DEFAULT_CSV_DIR, 'engines_grid.csv')
DEFAULT_LAUNCHMASS_CSV = os.path.join(DEFAULT_CSV_DIR, 'launchmass_grid.csv')
DEFAULT_RELIABILITY_CSV = os.path.join(DEFAULT_CSV_DIR, 'reliability_grid.csv')
DEFAULT_ENGINES_LINES_FILENAME = os.path.join(DEFAULT_PLOT_DIR, 'engines_lines.png')
DEFAULT_PAPER_ENGINES_LINES = os.path.join('paper', 'engines_lines.png')
DEFAULT_LAUNCHMASS_LINES_FILENAME = os.path.join(DEFAULT_PLOT_DIR, 'launchmass_lines.png')
DEFAULT_PAPER_LAUNCHMASS_LINES = os.path.join('paper', 'launchmass_lines.png')
# Reference wet mass for Saturn V (mean Apollo 8--17, Orloff 2000)
SATURN_V_WET_MASS_TONS = _mdv.mean_saturn_v_glom_kg() / 1000.0


def ensure_output_directories():
    """
    Create output directories if they do not exist.
    """
    os.makedirs(DEFAULT_CSV_DIR, exist_ok=True)
    os.makedirs(DEFAULT_PLOT_DIR, exist_ok=True)


def explore_maximum_planetary_mass_for_escape(
    mass_array,
    surface_pressure_array=None,
    payload_mass=None,
    max_stages=None,
    drag_model=None
):
    """
    Evaluate feasibility and performance across mass and pressure grids.
    """
    if surface_pressure_array is None:
        surface_pressure_array = [1.0]
    if payload_mass is None:
        payload_mass = DEFAULT_PAYLOAD_MASS
    if max_stages is None:
        max_stages = DEFAULT_MAX_STAGES
    if drag_model is None:
        drag_model = DEFAULT_DRAG_MODEL
    
    # Convert to numpy array for easier handling
    mass_array = np.array(mass_array) #kg
    surface_pressure_array = np.array(surface_pressure_array)
    n_masses = len(mass_array)
    n_pressures = len(surface_pressure_array)
    
    # Initialize result arrays
    # 1D arrays (per mass only)
    radius_array = np.zeros(n_masses)
    escape_velocity_array = np.zeros(n_masses)
    surface_gravity_array = np.zeros(n_masses)
    
    # 2D arrays (per mass and pressure)
    feasible = np.zeros((n_masses, n_pressures), dtype=bool)
    launch_mass_array = np.full((n_masses, n_pressures), np.nan)
    num_stages_array = np.full((n_masses, n_pressures), np.nan, dtype=float)
    num_engines_array = np.full((n_masses, n_pressures), np.nan, dtype=float)
    deltav_array = np.full((n_masses, n_pressures), np.nan, dtype=float)
    reliability_array = np.full((n_masses, n_pressures), np.nan, dtype=float)

    # Set flag for option, which drives verbosity of result
    detailed_output = len(surface_pressure_array) == 1
    
    if detailed_output:
        print("-"*128)
        print("  Mass (M_Earth) | Radius (km) |  v_esc (km/s) | Total DV (km/s) | Launch Mass (tons) | Stages | Engines | Reliability")
        print("-"*128)
    
    # Loop over planetary masses
    for i, planetary_mass in enumerate(mass_array):
        # Calculate radius using Valencia et al. 2006 relationship
        planetary_radius = tp.calculate_planetary_radius(planetary_mass)
        
        # Calculate escape velocity
        escape_velocity = tp.calculate_escape_velocity(planetary_mass, planetary_radius)
        
        # Calculate surface gravity self-consistently with the mass--radius law
        # (g = GM/R^2, same gravity that sets v_esc); see rocket_optimizer.
        surface_g = G * planetary_mass / (planetary_radius ** 2)
        
        # Store 1D arrays (these are constant across pressures for a given mass)
        radius_array[i] = planetary_radius
        escape_velocity_array[i] = escape_velocity
        surface_gravity_array[i] = surface_g

        for j, surface_pressure in enumerate(surface_pressure_array):
            # Try to optimize rocket for this planet
            result = ro.optimize_multi_stage_rocket(
                g_EARTH, M_EARTH, R_EARTH,
                planetary_mass,
                planetary_radius,
                escape_velocity,
                surface_pressure=surface_pressure,
                payload_mass=payload_mass,
                max_stages=max_stages,
                drag_model=drag_model,
                max_engines=MAX_ENGINES,
                max_mass_flow_per_engine=MAX_MASS_FLOW_PER_ENGINE,
                stage_reliability=STAGE_RELIABILITY,
                max_launch_mass=MAX_LAUNCH_MASS_KG
            )

            # Check if optimization succeeded (feasibility set by launch-mass budget)
            if result['success']:
                total_deltav = result['total_deltav']
                launch_mass_display = f"{result['launch_mass']/1000:.1f}"
                stages_display = f"{result['num_stages']:.0f}"
                engine_number = f"{result.get('engine_number', 0):.0f}"
                reliability_display = f"{result.get('mission_reliability', 0.0):.3f}"
                # Print progress
                mass_earth = planetary_mass / M_EARTH
                if detailed_output:
                    print(f"{mass_earth:8.2f} M_Earth | {planetary_radius/1000:8.1f} km | "
                    f"{escape_velocity/1000:6.2f} km/s   | {total_deltav/1000:6.2f} km/s     | "
                    f"{launch_mass_display:>13s} tons | {stages_display:>6s} | "
                    f"{engine_number:>5s} | {reliability_display:>9s}")

            # Store results in 2D arrays [mass_index, pressure_index]
            launch_mass_array[i, j] = result['launch_mass'] if result['success'] else np.nan  # kg
            num_stages_array[i, j] = result['num_stages'] if result['success'] else np.nan
            num_engines_array[i, j] = result.get('engine_number', np.nan) if result['success'] else np.nan
            deltav_array[i, j] = result['total_deltav'] / 1000 if result['success'] else np.nan  # km/s
            reliability_array[i, j] = result.get('mission_reliability', np.nan) if result['success'] else np.nan
            feasible[i, j] = result['success']

    print("-"*128)
    
    return {
            'mass_array': mass_array,
            'radius_array': radius_array,
            'escape_velocity_array': escape_velocity_array,
            'surface_gravity_array': surface_gravity_array,
            'pressure_array': surface_pressure_array,
            'feasible': feasible,
            'launch_mass_array': launch_mass_array,
            'num_stages_array': num_stages_array,
            'num_engines_array': num_engines_array,
            'deltav_array': deltav_array,
            'reliability_array': reliability_array,
        }


def create_planetary_array_and_export(
    M_EARTH=M_EARTH,
    min_mass_earth=0.5,
    max_mass_earth=20.0,
    n_points=50,
    surface_pressure_array=[1.0],
    payload_mass=1000.0,
    max_stages=None,
    drag_model='altitude_dependent',
    export_csv=True,
    csv_filename='rename_me.csv',
):
    """
    Convenience wrapper to run a mass grid and optionally export CSV.
    """
    # Create logarithmic array of masses
    mass_earth_array = np.logspace(
        np.log10(min_mass_earth),
        np.log10(max_mass_earth),
        n_points
    )
    
    # Convert to kg
    mass_array = mass_earth_array * M_EARTH
    
    # Run exploration
    results = explore_maximum_planetary_mass_for_escape(
        mass_array,
        surface_pressure_array=surface_pressure_array,
        payload_mass=payload_mass,
        max_stages=max_stages,
        drag_model=drag_model,
    )

    # Export to CSV if requested
    if export_csv:
        print(f"Exporting results to CSV: {csv_filename}")
        export_planetary_array_results_to_csv(results, mass_array, mass_earth_array, surface_pressure_array, csv_filename)
        print()

     
    # Extract maximum feasible values
    if len(results['feasible']) > 0 and np.any(results['feasible']):
        # Find first feasible result (when single pressure is used)
        if results['feasible'].ndim == 2:
            # 2D case: find max mass that is feasible for ANY pressure
            feasible_indices = np.where(np.any(results['feasible'], axis=1))[0]
            if len(feasible_indices) > 0:
                max_idx = feasible_indices[-1]
            else:
                return {
                    'max_mass': None,
                    'max_mass_earth': None,
                    'max_radius': None,
                    'max_radius_km': None,
                    'max_escape_velocity': None,
                    'max_launch_mass': None,
                    'max_num_stages': None,
                    'feasible': False,
                    'full_results': results
                }
        else:
            # 1D case (shouldn't happen with updated code)
            feasible_indices = np.where(results['feasible'])[0]
            if len(feasible_indices) > 0:
                max_idx = feasible_indices[-1]
            else:
                return {
                    'max_mass': None,
                    'max_mass_earth': None,
                    'max_radius': None,
                    'max_radius_km': None,
                    'max_escape_velocity': None,
                    'max_launch_mass': None,
                    'max_num_stages': None,
                    'feasible': False,
                    'full_results': results
                }
        
        return {
            'max_mass': results['mass_array'][max_idx],
            'max_mass_earth': mass_earth_array[max_idx],
            'max_radius': results['radius_array'][max_idx],
            'max_radius_km': results['radius_array'][max_idx] / 1000.0,
            'max_escape_velocity': results['escape_velocity_array'][max_idx],
            'max_launch_mass': results['launch_mass_array'][max_idx, 0] if results['launch_mass_array'].ndim == 2 else results['launch_mass_array'][max_idx],
            'max_num_stages': results['num_stages_array'][max_idx, 0] if results['num_stages_array'].ndim == 2 else results['num_stages_array'][max_idx],
            'feasible': True,
            'full_results': results
        }


def run_mass_pressure_grid(
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
):
    """
    Run mass--pressure grid survey; export CSVs and paper line plots.
    """
    return run_mass_pressure_survey(
        explore_func=explore_maximum_planetary_mass_for_escape,
        M_EARTH=M_EARTH,
        SATURN_V_WET_MASS_TONS=SATURN_V_WET_MASS_TONS,
        CHEOPS_LAUNCH_MASS_TONS=MAX_LAUNCH_MASS_TONS,
        mass_range_earth=mass_range_earth,
        pressure_range=pressure_range,
        n_mass_points=n_mass_points,
        n_pressure_points=n_pressure_points,
        payload_mass=payload_mass,
        max_stages=max_stages,
        drag_model=drag_model,
        output_feasibility_csv=output_feasibility_csv or DEFAULT_FEASIBILITY_CSV,
        output_deltav_csv=output_deltav_csv or DEFAULT_DELTAV_CSV,
        output_stages_csv=output_stages_csv or DEFAULT_STAGES_CSV,
        output_engines_csv=output_engines_csv or DEFAULT_ENGINES_CSV,
        output_launchmass_csv=output_launchmass_csv or DEFAULT_LAUNCHMASS_CSV,
        output_reliability_csv=output_reliability_csv or DEFAULT_RELIABILITY_CSV,
        output_launchmass_lines_plot=output_launchmass_lines_plot or DEFAULT_LAUNCHMASS_LINES_FILENAME,
        output_launchmass_paper_plot=output_launchmass_paper_plot or DEFAULT_PAPER_LAUNCHMASS_LINES,
        output_engines_lines_plot=output_engines_lines_plot or DEFAULT_ENGINES_LINES_FILENAME,
        output_engines_paper_plot=output_engines_paper_plot or DEFAULT_PAPER_ENGINES_LINES,
        default_mass_range=(DEFAULT_MIN_MASS_EARTH, DEFAULT_MAX_MASS_EARTH),
        default_pressure_range=(DEFAULT_MIN_PRESSURE, DEFAULT_MAX_PRESSURE),
        default_n_mass_points=DEFAULT_N_MASS_POINTS,
        default_n_pressure_points=DEFAULT_N_PRESSURE_POINTS,
        default_payload_mass=DEFAULT_PAYLOAD_MASS,
        max_engines_first_stage=MAX_ENGINES,
    )


#######################################################################
# MAIN: Test with range of planetary masses
#######################################################################

if __name__ == "__main__":
    # Create output directories
    ensure_output_directories()
    
    # Find maximum planetary mass for rocket escape
    # This uses the engine-count constraint: max 100 engines on first stage
    
    print("#"*72)
    print("FINDING MAXIMUM PLANETARY MASS FOR ROCKET ESCAPE")
    print("#"*72)
    print()
    print("Constraints:")
    print(f"  - Payload: {DEFAULT_PAYLOAD_MASS} kg")
    print(f"  - Engine limit: max {MAX_ENGINES:.0f} engines (first stage)")
    print("  - Valencia scaling: R ~ M^0.27")
    print()
    print("Choose analysis type:")
    print("1. Set air pressure and find maximum planetary mass for rocket escape")
    print("   (detailed screen output)")
    print("2. Loop over planetary mass and air pressure (grid survey + line plots)")
    print()
    answer = input('How would you like to proceed (1 or 2)? ')
    # If option 1 is selected, follow up to ask for surface pressure
    if answer.strip() == '1':
        pressure_input = input(f'Enter surface pressure in bar (default {DEFAULT_SURFACE_PRESSURE}): ')
        try:
            surface_pressure = float(pressure_input)
        except ValueError:
            surface_pressure = DEFAULT_SURFACE_PRESSURE
        pressure_array = np.array([surface_pressure])
        
        # Use the convenience function to find maximum
        max_result = create_planetary_array_and_export(
            M_EARTH=M_EARTH,
            min_mass_earth=DEFAULT_MIN_MASS_EARTH,
            max_mass_earth=DEFAULT_MAX_MASS_EARTH,
            n_points=DEFAULT_N_MASS_POINTS,
            surface_pressure_array=pressure_array,
            payload_mass=DEFAULT_PAYLOAD_MASS,
            max_stages=DEFAULT_MAX_STAGES,
            drag_model=DEFAULT_DRAG_MODEL,
            export_csv=True,
            csv_filename=DEFAULT_CSV_FILENAME
        )
        
        print()
        print("="*80)
        print("FINAL RESULT: MAXIMUM PLANETARY MASS FOR ROCKET ESCAPE")
        print("="*80)
        print()
        
        if max_result['feasible']:
            print(f"Maximum feasible planetary mass: {max_result['max_mass']:.3e} M_Earth")
            print()
            print(f"Planetary radius (Valencia scaling): {max_result['max_radius_km']:.1f} km")
            print()
            print(f"Escape velocity: {max_result['max_escape_velocity']/1000:.2f} km/s")
            print()
            print(f"Required launch mass: {max_result['max_launch_mass']/1000:.1f} tons")
            print()
            print(f"Optimal number of stages: {max_result['max_num_stages']:.0f}")
            print()
            print("Engineering constraints (all passed):")
            print("  - Rocket optimization: PASSED")
            print(f"  - Engine count (max {MAX_ENGINES:.0f}): PASSED")
        else:
            print("No feasible solution found in the tested mass range.")
            print("Rocket escape may not be possible for planets in this range.")
        
        print()
        print("="*80)
        print("Analysis complete!")
        print("="*80)
        
    elif answer.strip() == '2':
        run_mass_pressure_grid(
            mass_range_earth=(DEFAULT_MIN_MASS_EARTH, DEFAULT_MAX_MASS_EARTH),
            pressure_range=(DEFAULT_MIN_PRESSURE, DEFAULT_MAX_PRESSURE),
            n_mass_points=DEFAULT_N_MASS_POINTS,
            n_pressure_points=DEFAULT_N_PRESSURE_POINTS,
            payload_mass=DEFAULT_PAYLOAD_MASS,
            max_stages=DEFAULT_MAX_STAGES,
            drag_model=DEFAULT_DRAG_MODEL,
        )
        
        print()
        print("="*80)
        print("Analysis complete!")
        print("="*80)
    else:
        print("Invalid choice. Please enter 1 or 2.")

