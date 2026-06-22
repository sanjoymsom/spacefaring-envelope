#!/usr/bin/env python3
"""
CSV export helpers for spacefaring analysis results.
"""
import csv
import numpy as np


def export_results_to_csv(results, filename):
    """
    Export per-mass results to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'M_planet (kg)',
            'R_planet (m)',
            'gravity (m/s^2)',
            'V_esc (m/s)',
            'wet_mass_liftoff (kg)',
            'num_stages',
            'num_engines',
        ])
        
        # Write data rows
        n = len(results['mass_array'])
        for i in range(n):
            # Handle NaN values (write as empty string or 'N/A')
            wet_mass = results['launch_mass_array'][i] if not np.isnan(results['launch_mass_array'][i]) else ''
            num_stages = results['num_stages_array'][i] if not np.isnan(results['num_stages_array'][i]) else ''
            num_engines = results['num_engines_array'][i] if not np.isnan(results['num_engines_array'][i]) else ''
            
            writer.writerow([
                f"{results['mass_array'][i]:.6e}",
                f"{results['radius_array'][i]:.6e}",
                f"{results['surface_gravity_array'][i]:.6f}",
                f"{results['escape_velocity_array'][i]:.6f}",
                f"{wet_mass:.6e}" if wet_mass != '' else '',
                f"{num_stages:.0f}" if num_stages != '' else '',
                f"{num_engines:.0f}" if num_engines != '' else ''
            ])
    
    return filename


def export_planetary_array_results_to_csv(results, mass_array, mass_earth_array, surface_pressure_array, csv_filename):
    """
    Export mass/pressure grid results to a CSV file.
    """
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'M_planet (M_Earth)',
            'M_planet (kg)',
            'Pressure (bar)',
            'Feasible',
            'Launch Mass (kg)',
            'Num Stages'
        ])
        
        # Write data rows
        for i, mass_earth in enumerate(mass_earth_array):
            for j, pressure in enumerate(surface_pressure_array):
                feasible = results['feasible'][i, j]
                launch_mass = results['launch_mass_array'][i, j] if not np.isnan(results['launch_mass_array'][i, j]) else ''
                num_stages = results['num_stages_array'][i, j] if not np.isnan(results['num_stages_array'][i, j]) else ''
                
                writer.writerow([
                    f"{mass_earth:.6e}",
                    f"{mass_array[i]:.6e}",
                    f"{pressure:.6e}",
                    f"{feasible:.0f}",
                    f"{launch_mass:.6e}" if launch_mass != '' else '',
                    f"{num_stages:.0f}" if num_stages != '' else ''
                ])
    
    print(f"CSV exported: {csv_filename}")


def export_grid_results_to_csv(
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
    reliability_matrix=None,
    output_reliability_csv=None
):
    """
    Export mass--pressure grid results to CSV files.
    """
    # Feasibility CSV
    with open(output_feasibility_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'M_planet (M_Earth)',
            'M_planet (kg)',
            'Pressure (bar)',
            'Feasible',
            'Launch Mass (kg)',
            'Num Stages',
            'Mission Reliability'
        ])
        for i, mass_earth in enumerate(mass_earth_array):
            for j, pressure in enumerate(pressure_array):
                feasible = feasibility_matrix[i, j]
                launch_mass = launch_mass_matrix[i, j] if not np.isnan(launch_mass_matrix[i, j]) else ''
                num_stages = num_stages_matrix[i, j] if not np.isnan(num_stages_matrix[i, j]) else ''
                if reliability_matrix is not None and not np.isnan(reliability_matrix[i, j]):
                    reliability = reliability_matrix[i, j]
                else:
                    reliability = ''
                writer.writerow([
                    f"{mass_earth:.6e}",
                    f"{mass_array[i]:.6e}",
                    f"{pressure:.6e}",
                    f"{feasible:.0f}",
                    f"{launch_mass:.6e}" if launch_mass != '' else '',
                    f"{num_stages:.0f}" if num_stages != '' else '',
                    f"{reliability:.6f}" if reliability != '' else ''
                ])
    print(f"csv saved: {output_feasibility_csv}")

    # Delta-V CSV
    with open(output_deltav_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['M_planet (M_Earth)', 'M_planet (kg)', 'Pressure (bar)', 'Total Delta-V (km/s)'])
        for i, mass_earth in enumerate(mass_earth_array):
            for j, pressure in enumerate(pressure_array):
                dv = deltav_matrix[i, j]
                writer.writerow([f"{mass_earth:.6e}", f"{mass_array[i]:.6e}", f"{pressure:.6e}", f"{dv:.6e}" if not np.isnan(dv) else ''])
    print(f"csv saved: {output_deltav_csv}")

    # Stages CSV
    with open(output_stages_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['M_planet (M_Earth)', 'M_planet (kg)', 'Pressure (bar)', 'Num Stages'])
        for i, mass_earth in enumerate(mass_earth_array):
            for j, pressure in enumerate(pressure_array):
                ns = num_stages_matrix[i, j]
                writer.writerow([f"{mass_earth:.6e}", f"{mass_array[i]:.6e}", f"{pressure:.6e}", f"{ns:.0f}" if not np.isnan(ns) else ''])
    print(f"csv saved: {output_stages_csv}")

    # Engines CSV
    with open(output_engines_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['M_planet (M_Earth)', 'M_planet (kg)', 'Pressure (bar)', 'Num Engines'])
        for i, mass_earth in enumerate(mass_earth_array):
            for j, pressure in enumerate(pressure_array):
                ne = num_engines_matrix[i, j]
                writer.writerow([f"{mass_earth:.6e}", f"{mass_array[i]:.6e}", f"{pressure:.6e}", f"{ne:.0f}" if not np.isnan(ne) else ''])
    print(f"csv saved: {output_engines_csv}")

    # Launch mass CSV
    with open(output_launchmass_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['M_planet (M_Earth)', 'M_planet (kg)', 'Pressure (bar)', 'Launch Mass (tons)'])
        for i, mass_earth in enumerate(mass_earth_array):
            for j, pressure in enumerate(pressure_array):
                lm = launch_mass_matrix[i, j]
                lm_tons = lm / 1000.0 if not np.isnan(lm) else np.nan
                writer.writerow([f"{mass_earth:.6e}", f"{mass_array[i]:.6e}", f"{pressure:.6e}", f"{lm_tons:.6e}" if not np.isnan(lm_tons) else ''])
    print(f"csv saved: {output_launchmass_csv}")

    # Mission reliability CSV
    if reliability_matrix is not None and output_reliability_csv is not None:
        with open(output_reliability_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['M_planet (M_Earth)', 'M_planet (kg)', 'Pressure (bar)', 'Mission Reliability'])
            for i, mass_earth in enumerate(mass_earth_array):
                for j, pressure in enumerate(pressure_array):
                    rel = reliability_matrix[i, j]
                    writer.writerow([f"{mass_earth:.6e}", f"{mass_array[i]:.6e}", f"{pressure:.6e}", f"{rel:.6f}" if not np.isnan(rel) else ''])
        print(f"csv saved: {output_reliability_csv}")
