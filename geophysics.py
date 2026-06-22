#!/usr/bin/env python3
#######################################################################
#Plate tectonics likelihood assessment for terrestrial planets.#
#
#This module assesses the likelihood of plate tectonics on terrestrial planets
#based on geophysical criteria 
#
#The assessment uses:
#1. Rayleigh number (mantle convection strength)
#2. Magnetic Reynolds number (dynamo activity)
#3. Planetary mass range
#
#The methodology follows Valencia et al. (2007) scaling relationships:
#- Density: ρ = ρ_⊕ (M/M_⊕)^0.2
#- Gravity: g = g_⊕ (M/M_⊕)^0.5
#- Mantle thickness: d = d_⊕ (M/M_⊕)^0.28
#- Temperature difference: ΔT = ΔT_⊕ (M/M_⊕)^0.17
#3- Viscosity: constant (isoviscous case, η = 10^21 Pa·s)
#
#Good References
#- Valencia, D., O'Connell, R. J., & Sasselov, D. D. (2006). Internal structure
#  of massive terrestrial planets. Icarus, 181(2), 545-554.
#
#- Valencia, D., O'Connell, R. J., & Sasselov, D. D. (2007). Inevitability of
#  plate tectonics on super-Earths. The Astrophysical Journal Letters, 670(1), L45-L48.
#
#- Turcotte, D. L., & Schubert, G. (2002). Geodynamics (2nd ed.). 
#  Cambridge University Press, Cambridge.
#
#- Foley, B. J., & Driscoll, P. E. (2016). Whole planet coupling between climate,
#  mantle, and core: Implications for the evolution of rocky planets.
#  Geochemistry, Geophysics, Geosystems, 17(5), 1885-1914.
#
#- Christensen, U. R., & Aubert, J. (2006). Scaling properties of convection-driven
#  dynamos in rotating spherical shells and application to planetary magnetic fields.
#  Geophysical Journal International, 166(1), 97-114.
#
# Sanjoy Som, Fall 2025 (code formatted and commented with AI assistance:
# Claude, Anthropic, via the Cursor IDE; see the paper Acknowledgments).
#######################################################################
import sys
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from terrestrial_planet import calculate_planetary_radius
from constants import M_EARTH, R_EARTH, G, M_MARS, KEPLER_20B_MASS


#######################################################################
# PHYSICAL CONSTANTS AND PARAMETERS
#######################################################################

# Mantle convection parameters (for Rayleigh number calculation)
# Based on Valencia et al. 2007 and Turcotte & Schubert 2002
RHO_MANTLE_EARTH = 4000.0   # kg/m³ (Earth mantle density, average mantle value)
                            # Average mantle density (Turcotte & Schubert 2002, p. 271)
D_MANTLE_EARTH = 2900e3     # m (Earth mantle thickness)
                            # 2900 km = 2900×10³ m (Valencia et al. 2007)                            
ALPHA_THERMAL = 3e-5        # 1/K (mantle thermal expansion coefficient)
                            # Standard value: 3×10⁻⁵ K⁻¹ (Turcotte & Schubert 2002)
DELTA_T_EARTH = 2500.0      # K (temperature difference across Earth's mantle)
                            # Typical estimate for temperature drop across the mantle
                            # For other planets: ΔT = 2500 (M/M_⊕)^0.17
KAPPA = 1e-6                # m²/s (thermal diffusivity, isothermal diffusivity)
                            # Standard value: ~1×10⁻⁶ m²/s (Turcotte & Schubert 2002)


# Viscosity (isoviscous case)
# Based on Valencia et al. 2007b: constant viscosity assumption
# The temperature beneath the lithosphere is almost independent of mass,
# justifying the isoviscous approximation
ETA_CONSTANT = 1e21         # Pa·s (constant dynamic viscosity, isoviscous case)
                            # Reference viscosity value (Valencia et al. 2007b)

# Reference Rayleigh number at Earth (Eq. rayleigh in paper; direct evaluation)
RA_EARTH = (
    RHO_MANTLE_EARTH
    * (G * M_EARTH / R_EARTH ** 2)
    * ALPHA_THERMAL
    * DELTA_T_EARTH
    * D_MANTLE_EARTH ** 3
    / (KAPPA * ETA_CONSTANT)
)  # ~7.2×10^7

# Core parameters (for dynamo calculation)
# The paper's magnetic Reynolds number uses the Christensen (2010) core-velocity
# scaling calibrated to the Earth value U_core,Earth (Roberts & Glatzmaier 2000);
# see calculate_dynamo_proxy below. The three constants immediately following
# (Q_CMB_EARTH, DELTA_T_CORE_EARTH, ALPHA_C) are LEGACY/UNUSED in that proxy and
# are retained only for reference; they do not affect any published result.
Q_CMB_EARTH = 8e12          # W (core-mantle boundary heat flux at Earth) [unused]
                            # Literature range: 10-15 TW (Christensen & Aubert 2006)
DELTA_T_CORE_EARTH = 1000.0 # K (superadiabatic core temperature contrast) [unused]
ALPHA_C = 1.5e-5            # 1/K (core thermal expansion coefficient) [unused]
ETA_MAG = 2.0               # m²/s (magnetic diffusivity, typical value ~1-2 m²/s)

# Reference magnetic Reynolds number at Earth (Christensen & Aubert 2006 scaling)
R_CORE_EARTH = R_EARTH - D_MANTLE_EARTH
U_CORE_EARTH = 5e-4  # m/s (Roberts & Glatzmaier 2000; paper Sec. geophys)
RM_EARTH = U_CORE_EARTH * R_CORE_EARTH / ETA_MAG  # ~8.7×10^2

# Solar-system reference bodies for geophysics plots (mass in M_Earth)
REFERENCE_BODIES = (
    {'name': 'Earth', 'mass_kg': M_EARTH, 'color': 'tab:blue'},
    {'name': 'Mars', 'mass_kg': M_MARS, 'color': 'tab:orange'},
    {'name': 'Kepler-20 b', 'mass_kg': KEPLER_20B_MASS, 'color': 'tab:green'},
)

# Default annotation offsets (offset points) for geophysics plots
GEOPHYS_BODY_OFFSETS_RA = {
    'Earth': (8, -18),
    'Mars': (8, 10),
    'Kepler-20 b': (0, 14),
}
GEOPHYS_BODY_OFFSETS_RM = {
    'Earth': (8, -18),
    'Mars': (8, 12),
    'Kepler-20 b': (0, 14),
}
GEOPHYS_BODY_HA = {
    'Kepler-20 b': 'center',
}

# Critical thresholds for plate tectonics assessment
# Based on Valencia et al. 2007, Foley & Driscoll 2016, and literature
RA_CRITICAL = 1000.0        # Critical Rayleigh number for onset of convection
                            # Classical value: Ra_c ≈ 1,708 for free-slip boundaries
                            # Using 1000 as conservative threshold for convection onset
                            # Earth Ra ~ 7×10^7 (direct evaluation; paper Eq. rascale)
RA_ROBUST = 1e6             # Robust-convection threshold (paper Sec. geophys)
RM_CRITICAL = 50.0          # Critical magnetic Reynolds number for dynamo
                            # Literature: Rm > 10-100 for dynamo action
RM_ROBUST = 100.0           # Robust-dynamo threshold (paper Sec. geophys)
MASS_MIN_LIKELY = 0.5 * M_EARTH   # Minimum mass for likely plate tectonics
                                  # Based on Valencia et al. 2007: very small planets
                                  # (< 0.5 M_Earth) are less likely to exhibit plate tectonics
MASS_MAX_LIKELY = 10.0 * M_EARTH  # Maximum mass for likely plate tectonics
                                  # Based on Valencia et al. 2007: very large planets
                                  # (> 10 M_Earth) are less likely to exhibit plate tectonics
                                  # due to different convective regimes

#######################################################################
# GEOPHYSICAL PROXY CALCULATIONS
#######################################################################

def calculate_rayleigh_number(planetary_mass, planetary_radius=None):
    """
    Calculate Rayleigh number as a proxy for mantle convection strength.
    
    The Rayleigh number (Ra) is a dimensionless number that characterizes
    the vigor of thermal convection. Higher Ra indicates stronger convection,
    which favors plate tectonics.
    
    Formula: Ra = (ρ·g·α·ΔT·d³) / (κ·η)
    
    Based on Valencia et al. (2007) scaling relationships and Turcotte & Schubert (2002).
    Uses isoviscous case with constant viscosity.
    
    Parameters:
    -----------
    planetary_mass : float
        Planetary mass in kg
    planetary_radius : float, optional
        Planetary radius in m. If None, calculated from mass using Valencia et al. 2006.
        (default: None)
    
    Returns:
    --------
    float
        Rayleigh number (dimensionless)
    """
    if planetary_radius is None:
        planetary_radius = calculate_planetary_radius(planetary_mass)
    
    # Scaling relationships from Valencia et al. (2007)
    # Assumes planetary internal heat generation occurs at the same rate as Earth
    mass_ratio = planetary_mass / M_EARTH
    
    # Density scaling: ρ = ρ_⊕ (M/M_⊕)^0.2
    rho_mantle = RHO_MANTLE_EARTH * (mass_ratio ** 0.2)
    
    # Surface gravity scaling: g = g_⊕ (M/M_⊕)^0.5
    # Alternatively: g = G * M / R², but using scaling for consistency
    g_earth = G * M_EARTH / (R_EARTH ** 2)
    surface_g = g_earth * (mass_ratio ** 0.5)
    
    # Mantle thickness scaling: d = d_⊕ (M/M_⊕)^0.28
    d_mantle = D_MANTLE_EARTH * (mass_ratio ** 0.28)
    
    # Temperature difference scaling: ΔT = ΔT_⊕ (M / M_⊕)^0.17
    # Based on boundary-layer theory: for n=1/3 (Nu ∝ Ra^(1/3)), 
    # ΔT ∝ q_tot^(3/4) (ρg)^(-1/4) where q_tot is heat flux per unit area.
    # For planets: q_tot ∝ M / R² ∝ M^(1-2β) with R ∝ M^β, β≈0.27
    # Therefore: q_tot ∝ M^0.46
    # Plugging into boundary-layer theory: ΔT ∝ M^(0.46 * 3/4) (M^0.2 * M^0.5)^(-1/4)
    # = M^0.345 * M^(-0.175) = M^0.17
    # This scaling is independent of mantle thickness d (boundary-layer theory result)
    delta_T = DELTA_T_EARTH * (mass_ratio ** 0.17)
    
    # Viscosity: constant (isoviscous case)
    # Based on Valencia et al. 2007b: temperature beneath lithosphere is
    # almost independent of mass, justifying constant viscosity
    viscosity = ETA_CONSTANT
    
    # Rayleigh number: Ra = (ρ·g·α·ΔT·d³) / (κ·η)
    # Scaling: Ra ~ M^0.2 (ρ) × M^0.5 (g) × M^0.17 (ΔT) × M^0.84 (d³) ~ M^1.71
    # This matches the theoretical scaling from boundary-layer theory
    rayleigh = (rho_mantle * surface_g * ALPHA_THERMAL * delta_T * 
                d_mantle ** 3) / (KAPPA * viscosity)
  
    return rayleigh


def calculate_dynamo_proxy(planetary_mass, planetary_radius=None, f_mt=1.0):
    """
    Based on Christensen & Aubert (2006) scaling laws for planetary dynamos.
    Strong dynamo activity suggests active core convection, which correlates
    with plate tectonics likelihood.
    
    Parameters:
    -----------
    planetary_mass : float
        Planetary mass in kg
    planetary_radius : float, optional
        Planetary radius in m. If None, calculated from mass.
        (default: None)
    f_mt : float, optional
        Mantle temperature factor (default: 1.0 for Earth-like)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'Rm': Magnetic Reynolds number (dimensionless)
    """
    if planetary_radius is None:
        planetary_radius = calculate_planetary_radius(planetary_mass)
    
    mass_ratio = planetary_mass / M_EARTH
    
    # Core radius scaling: R_core = R_core_⊕ (M / M_⊕)^0.27 (paper Eq. rm, R_core = R - d)
    # Based on Valencia (2006) scaling, using Earth's core radius as reference
    R_core_earth = R_CORE_EARTH
    R_core = R_core_earth * (mass_ratio ** 0.27)
    
    # Core convective velocity scaling: U_core = U_core_⊕ (M / M_⊕)^0.347 (paper Eq. ucore)
    # Derived from boundary-layer theory: U_core ~ (g × R_core × q_tot / ρ_core)^(1/3)
    # Where: g ~ M^0.5, R_core ~ M^0.27, q_tot ~ M^0.46, ρ_core ~ M^0.19
    # Therefore: U_core ~ (M^0.5 × M^0.27 × M^0.46 × M^-0.19)^(1/3) = M^0.347
    # Calibrated to Earth's observed core velocity
    U_core_earth = U_CORE_EARTH  # m/s (Earth's typical core velocity, from observations)
    U_core = U_core_earth * (mass_ratio ** 0.347)
    
    # Magnetic Reynolds number: Rm = (U_core * R_core) / η_mag (paper Eq. rm)
    # This is directly obtained from the scaled and calibrated U_core and R_core.
    # The magnetic Reynolds number is calibrated through:
    #   - U_core: scaled as M^0.347, calibrated to Earth (U_core_earth = 5×10⁻⁴ m/s)
    #   - R_core: scaled as M^0.27, calibrated to Earth (R_core_earth)
    #   - η_mag: constant magnetic diffusivity (2.0 m²/s)
    # Scaling: Rm ~ M^0.347 × M^0.27 = M^0.617
    # Rm characterizes the ratio of magnetic field advection to diffusion.
    # For dynamo action, Rm > 50 (conservative threshold).
    # For Earth: Rm ~ 100-1000 (literature range)
    Rm = U_core * R_core / ETA_MAG
    
    return {'Rm': Rm}

#######################################################################
# CSV AND PLOT GENERATION
#######################################################################

def generate_rayleigh_reynolds_csv_and_plot(
    output_csv='geophysics_rayleigh_reynolds.csv',
    output_plot='geophysics_rayleigh_reynolds.png'
):
    """
    LEGACY (not used for the published figures). Combined dual-axis Ra/Rm plot.

    The paper's Fig. 1 panels are produced by generate_rayleigh_mass_plot and
    generate_dynamo_mass_plot (with regime bands and reference bodies). This
    single-figure dual-axis variant is retained only for quick exploration and
    is reachable via the interactive menu (option 2).

    Generate a CSV file and plot with Planetary Mass, Rayleigh Number, and Magnetic Reynolds number.
    
    Parameters:
    -----------
    output_csv : str
        Output CSV filename (default: 'geophysics_rayleigh_reynolds.csv')
    output_plot : str
        Output plot filename (default: 'geophysics_rayleigh_reynolds.png')
    """
    # Hardcoded parameters
    min_mass_earth = 0.1
    max_mass_earth = 10.0
    n_points = 100
    
    # Generate mass array (linear spacing)
    mass_earth_array = np.linspace(
        min_mass_earth,
        max_mass_earth,
        n_points
    )
    mass_array = mass_earth_array * M_EARTH
    
    # Calculate Rayleigh and Magnetic Reynolds numbers for each mass
    rayleigh_numbers = []
    magnetic_reynolds_numbers = []
    
    print(f"Calculating geophysical parameters for {n_points} planetary masses...")
    for i, mass in enumerate(mass_array):
        Ra = calculate_rayleigh_number(mass)
        dynamo = calculate_dynamo_proxy(mass)
        Rm = dynamo['Rm']
        rayleigh_numbers.append(Ra)
        magnetic_reynolds_numbers.append(Rm)
        
        # Progress indicator
        if (i + 1) % 20 == 0 or (i + 1) == n_points:
            print(f"  Progress: {i + 1}/{n_points} ({100 * (i + 1) / n_points:.1f}%)")
    
    # Create DataFrame
    df = pd.DataFrame({
        'Planetary Mass (M_Earth)': mass_earth_array,
        'Planetary Mass (kg)': mass_array,
        'Rayleigh Number': rayleigh_numbers,
        'Magnetic Reynolds Number': magnetic_reynolds_numbers
    })
    
    # Save to CSV (only the 3 requested columns)
    csv_df = df[['Planetary Mass (kg)', 'Rayleigh Number', 'Magnetic Reynolds Number']]
    csv_df.to_csv(output_csv, index=False)
    print(f"\nCSV file saved: {output_csv}")
    
    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Set y-axis limits for Rayleigh Number
    y_min = 10**2
    y_max = 10**12
    
    # Left y-axis: Rayleigh Number
    color1 = 'tab:blue'
    ax1.set_xlabel('Planetary Mass (M$_\\oplus$)', fontsize=12)
    ax1.set_ylabel('Rayleigh Number', color=color1, fontsize=12)
    line1 = ax1.plot(mass_earth_array, rayleigh_numbers, color=color1, linewidth=2, label='Rayleigh Number', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('linear')
    ax1.set_xlim(0, 10.0)
    ax1.set_yscale('log')
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, alpha=0.3, zorder=1)
    
    # Right y-axis: Magnetic Reynolds Number
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Magnetic Reynolds Number', color=color2, fontsize=12)
    line2 = ax2.plot(mass_earth_array, magnetic_reynolds_numbers, color=color2, linewidth=2, label='Magnetic Reynolds Number')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('linear')
    ax2.set_ylim(bottom=0)
    
    # Add critical threshold lines
    threshold_ra = ax1.axhline(y=RA_CRITICAL, color=color1, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Ra critical ({RA_CRITICAL:.0f})')
    threshold_rm = ax2.axhline(y=RM_CRITICAL, color=color2, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Rm critical ({RM_CRITICAL:.0f})')
    
    # Combine legends from both axes
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    all_handles = ax1_handles + ax2_handles
    all_labels = ax1_labels + ax2_labels
    ax1.legend(all_handles, all_labels, loc='upper left', fontsize=10)
    
    plt.title('Geophysical Parameters vs Planetary Mass\nRayleigh Number (left) and Magnetic Reynolds Number (right)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_plot}")
    
    return df


def _ensure_parent_dirs(*paths):
    import os
    for path in paths:
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


def _annotate_reference_bodies(ax, value_fn, offsets=None, ha=None):
    """Mark Earth, Mars, and Kepler-20 b on a mass-scaling plot."""
    offsets = offsets or GEOPHYS_BODY_OFFSETS_RA
    ha = ha or GEOPHYS_BODY_HA
    for body in REFERENCE_BODIES:
        mu = body['mass_kg'] / M_EARTH
        val = value_fn(body['mass_kg'])
        ax.scatter([mu], [val], color=body['color'], s=60, zorder=5,
                   edgecolors='black', linewidths=0.8)
        ax.annotate(
            body['name'], (mu, val),
            xytext=offsets.get(body['name'], (8, 8)),
            textcoords='offset points', fontsize=10,
            ha=ha.get(body['name'], 'left'),
        )


def generate_rayleigh_mass_plot(
    min_mass_earth=0.05,
    max_mass_earth=20.0,
    n_points=200,
    output_csv='geophysics_rayleigh.csv',
    output_plot='geophysics_rayleigh.png',
    paper_plot=None,
):
    """
    Rayleigh number vs planetary mass with convection-regime bands.

    Regimes (paper thresholds):
      Ra < 1e3        — below classical convection onset
      1e3 <= Ra < 1e6 — weak convection
      Ra >= 1e6       — robust convection
    """
    _ensure_parent_dirs(output_csv, output_plot, paper_plot)

    mass_earth_array = np.linspace(min_mass_earth, max_mass_earth, n_points)
    mass_array = mass_earth_array * M_EARTH
    rayleigh_numbers = [calculate_rayleigh_number(m) for m in mass_array]

    df = pd.DataFrame({
        'Planetary Mass (M_Earth)': mass_earth_array,
        'Planetary Mass (kg)': mass_array,
        'Rayleigh Number': rayleigh_numbers,
        'log10_Ra': np.log10(rayleigh_numbers),
        'above_Ra_crit': np.array(rayleigh_numbers) >= RA_CRITICAL,
        'above_Ra_robust': np.array(rayleigh_numbers) >= RA_ROBUST,
    })
    df.to_csv(output_csv, index=False)
    print(f"CSV saved: {output_csv}")

    # Mass at which scaling law crosses each threshold
    mu_onset = (RA_CRITICAL / RA_EARTH) ** (1.0 / 1.71)
    mu_robust = (RA_ROBUST / RA_EARTH) ** (1.0 / 1.71)
    print(f"Scaling-law crossings: Ra={RA_CRITICAL:.0e} at M={mu_onset:.4f} M_E; "
          f"Ra={RA_ROBUST:.0e} at M={mu_robust:.4f} M_E")
    print(f"Ra at Earth: {RA_EARTH:.3e}")

    mars_mu = M_MARS / M_EARTH
    mars_ra = calculate_rayleigh_number(M_MARS)
    mars_rm = calculate_dynamo_proxy(M_MARS)['Rm']
    k20_ra = calculate_rayleigh_number(KEPLER_20B_MASS)
    k20_rm = calculate_dynamo_proxy(KEPLER_20B_MASS)['Rm']
    print(f"Mars ({mars_mu:.3f} M_E): Ra={mars_ra:.3e}, Rm={mars_rm:.1f}")
    print(f"Kepler-20 b ({KEPLER_20B_MASS/M_EARTH:.1f} M_E): Ra={k20_ra:.3e}, Rm={k20_rm:.1f}")

    in_range = df[
        (df['Planetary Mass (M_Earth)'] >= min_mass_earth)
        & (df['Planetary Mass (M_Earth)'] <= max_mass_earth)
    ]
    n_crit = in_range['above_Ra_crit'].sum()
    n_robust = in_range['above_Ra_robust'].sum()
    n_tot = len(in_range)
    print(f"Mass range [{min_mass_earth}, {max_mass_earth}] M_E ({n_tot} points): "
          f"{n_crit}/{n_tot} above Ra_crit; {n_robust}/{n_tot} above Ra_robust")
    if n_robust < n_tot:
        weak = in_range[~in_range['above_Ra_robust']]
        print(f"  Weak regime ({RA_CRITICAL:.0e} <= Ra < {RA_ROBUST:.0e}): "
              f"M = [{weak['Planetary Mass (M_Earth)'].min():.3f}, "
              f"{weak['Planetary Mass (M_Earth)'].max():.3f}] M_E")
    if n_crit < n_tot:
        sub = in_range[~in_range['above_Ra_crit']]
        print(f"  Sub-critical (Ra < {RA_CRITICAL:.0e}): "
              f"M = [{sub['Planetary Mass (M_Earth)'].min():.3f}, "
              f"{sub['Planetary Mass (M_Earth)'].max():.3f}] M_E")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Regime shading (horizontal bands in log-Ra space)
    ax.axhspan(1e-2, RA_CRITICAL, color='#fee2e2', alpha=0.5, zorder=0,
               label=f'Ra $<$ {RA_CRITICAL:.0e} (below onset)')
    ax.axhspan(RA_CRITICAL, RA_ROBUST, color='#fef9c3', alpha=0.5, zorder=0,
               label=f'{RA_CRITICAL:.0e} $\\leq$ Ra $<$ {RA_ROBUST:.0e} (weak convection)')
    ax.axhspan(RA_ROBUST, 1e12, color='#dcfce7', alpha=0.45, zorder=0,
               label=f'Ra $\\geq$ {RA_ROBUST:.0e} (robust convection)')

    ax.plot(mass_earth_array, rayleigh_numbers, color='tab:blue', linewidth=2.5,
            label='This model', zorder=4)
    _annotate_reference_bodies(ax, calculate_rayleigh_number, offsets=GEOPHYS_BODY_OFFSETS_RA)

    ax.axhline(RA_CRITICAL, color='0.35', linestyle='--', linewidth=1.2, zorder=2)
    ax.axhline(RA_ROBUST, color='0.35', linestyle='-.', linewidth=1.2, zorder=2)
    ax.text(max_mass_earth * 0.98, RA_CRITICAL * 1.15, r'$Ra_{\rm crit}=10^3$',
            ha='right', va='bottom', fontsize=9, color='0.35')
    ax.text(max_mass_earth * 0.98, RA_ROBUST * 1.15, r'Robust ($10^6$)',
            ha='right', va='bottom', fontsize=9, color='0.35')

    if min_mass_earth >= mu_robust:
        note = (f'All $M \\geq {min_mass_earth:.2f}$ $M_\\oplus$ in this range lie in the robust regime '
                f'($Ra \\geq {RA_ROBUST:.0e}$).')
    elif max_mass_earth <= mu_onset:
        note = f'All masses in this range lie below convection onset ($Ra < {RA_CRITICAL:.0e}$).'
    else:
        note = (f'Robust ($Ra \\geq {RA_ROBUST:.0e}$) for $M \\gtrsim {mu_robust:.2f}$ $M_\\oplus$; '
                f'weak convection between ${mu_onset:.3f}$ and ${mu_robust:.2f}$ $M_\\oplus$.')
    ax.text(0.02, 0.06, note, transform=ax.transAxes, fontsize=8.5, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.set_xlabel('Planetary mass ($M/M_\\oplus$)', fontsize=12)
    ax.set_ylabel('Rayleigh number Ra', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min_mass_earth, max_mass_earth)
    ax.set_ylim(1e2, 1e11)
    ax.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.5, zorder=1)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.92)
    ax.set_title('(A) Mantle Rayleigh number vs planetary mass',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    if paper_plot:
        plt.savefig(paper_plot, dpi=200, bbox_inches='tight')
        print(f"Plot saved: {paper_plot}")
    plt.close()
    print(f"Plot saved: {output_plot}")

    return df, mu_onset, mu_robust


def generate_dynamo_mass_plot(
    min_mass_earth=0.05,
    max_mass_earth=20.0,
    n_points=200,
    output_csv='geophysics_dynamo.csv',
    output_plot='geophysics_dynamo.png',
    paper_plot=None,
):
    """
    Magnetic Reynolds number vs planetary mass with dynamo-regime bands.

    Regimes (paper thresholds):
      Rm < 50   — below dynamo onset
      50 <= Rm < 100 — weak dynamo
      Rm >= 100 — robust dynamo
    """
    _ensure_parent_dirs(output_csv, output_plot, paper_plot)

    mass_earth_array = np.linspace(min_mass_earth, max_mass_earth, n_points)
    mass_array = mass_earth_array * M_EARTH
    rm_values = [calculate_dynamo_proxy(m)['Rm'] for m in mass_array]

    df = pd.DataFrame({
        'Planetary Mass (M_Earth)': mass_earth_array,
        'Planetary Mass (kg)': mass_array,
        'Magnetic Reynolds Number': rm_values,
        'log10_Rm': np.log10(rm_values),
        'above_Rm_crit': np.array(rm_values) >= RM_CRITICAL,
        'above_Rm_robust': np.array(rm_values) >= RM_ROBUST,
    })
    df.to_csv(output_csv, index=False)
    print(f"CSV saved: {output_csv}")

    mu_crit = (RM_CRITICAL / RM_EARTH) ** (1.0 / 0.617)
    mu_robust = (RM_ROBUST / RM_EARTH) ** (1.0 / 0.617)
    print(f"Scaling-law crossings: Rm={RM_CRITICAL:.0f} at M={mu_crit:.4f} M_E; "
          f"Rm={RM_ROBUST:.0f} at M={mu_robust:.4f} M_E")
    print(f"Rm at Earth: {RM_EARTH:.1f}")

    mars_mu = M_MARS / M_EARTH
    mars_rm = calculate_dynamo_proxy(M_MARS)['Rm']
    mars_ra = calculate_rayleigh_number(M_MARS)
    k20_ra = calculate_rayleigh_number(KEPLER_20B_MASS)
    k20_rm = calculate_dynamo_proxy(KEPLER_20B_MASS)['Rm']
    print(f"Mars ({mars_mu:.3f} M_E): Ra={mars_ra:.3e}, Rm={mars_rm:.1f}")
    print(f"Kepler-20 b ({KEPLER_20B_MASS/M_EARTH:.1f} M_E): Ra={k20_ra:.3e}, Rm={k20_rm:.1f}")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.axhspan(1e-1, RM_CRITICAL, color='#fee2e2', alpha=0.5, zorder=0,
               label=f'Rm $<$ {RM_CRITICAL:.0f} (below onset)')
    ax.axhspan(RM_CRITICAL, RM_ROBUST, color='#fef9c3', alpha=0.5, zorder=0,
               label=f'{RM_CRITICAL:.0f} $\\leq$ Rm $<$ {RM_ROBUST:.0f} (weak dynamo)')
    ax.axhspan(RM_ROBUST, 1e5, color='#dcfce7', alpha=0.45, zorder=0,
               label=f'Rm $\\geq$ {RM_ROBUST:.0f} (robust dynamo)')

    ax.plot(mass_earth_array, rm_values, color='tab:blue', linewidth=2.5,
            label='This model', zorder=4)
    _annotate_reference_bodies(
        ax, lambda m: calculate_dynamo_proxy(m)['Rm'],
        offsets=GEOPHYS_BODY_OFFSETS_RM,
    )

    ax.axhline(RM_CRITICAL, color='0.35', linestyle='--', linewidth=1.2, zorder=2)
    ax.axhline(RM_ROBUST, color='0.35', linestyle='-.', linewidth=1.2, zorder=2)
    ax.text(max_mass_earth * 0.98, RM_CRITICAL * 1.08, r'$Rm_{\rm crit}=50$',
            ha='right', va='bottom', fontsize=9, color='0.35')
    ax.text(max_mass_earth * 0.98, RM_ROBUST * 1.08, r'Robust ($100$)',
            ha='right', va='bottom', fontsize=9, color='0.35')

    if min_mass_earth >= mu_robust:
        note = (f'All $M \\geq {min_mass_earth:.2f}$ $M_\\oplus$ in this range exceed the '
                f'robust-dynamo threshold ($Rm \\geq {RM_ROBUST:.0f}$).')
    elif max_mass_earth <= mu_crit:
        note = f'All masses in this range lie below dynamo onset ($Rm < {RM_CRITICAL:.0f}$).'
    else:
        note = (f'Robust dynamo ($Rm \\geq {RM_ROBUST:.0f}$) for $M \\gtrsim {mu_robust:.2f}$ $M_\\oplus$; '
                f'dynamo onset at $M \\gtrsim {mu_crit:.3f}$ $M_\\oplus$.')
    ax.text(0.02, 0.06, note, transform=ax.transAxes, fontsize=8.5, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.set_xlabel('Planetary mass ($M/M_\\oplus$)', fontsize=12)
    ax.set_ylabel('Magnetic Reynolds number Rm', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min_mass_earth, max_mass_earth)
    ax.set_ylim(10, 1e4)
    ax.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.5, zorder=1)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.92)
    ax.set_title('(B) Core magnetic Reynolds number vs planetary mass',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    if paper_plot:
        plt.savefig(paper_plot, dpi=200, bbox_inches='tight')
        print(f"Plot saved: {paper_plot}")
    plt.close()
    print(f"Plot saved: {output_plot}")

    return df, mu_crit, mu_robust


#######################################################################
# MAIN: Test with various planetary masses
#######################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Geophysical proxy calculator')
    parser.add_argument(
        '--plot-ra',
        action='store_true',
        help='Generate Rayleigh-number vs mass plot and CSV (non-interactive)',
    )
    parser.add_argument(
        '--plot-dynamo',
        action='store_true',
        help='Generate magnetic-Reynolds-number vs mass plot and CSV',
    )
    parser.add_argument('--min-mass', type=float, default=0.05, help='Min mass (M_Earth)')
    parser.add_argument('--max-mass', type=float, default=20.0, help='Max mass (M_Earth)')
    args, _unknown = parser.parse_known_args()

    if args.plot_ra:
        generate_rayleigh_mass_plot(
            min_mass_earth=args.min_mass,
            max_mass_earth=args.max_mass,
            output_csv='csv/geophysics_rayleigh.csv',
            output_plot='validation/geophysics_rayleigh.png',
            paper_plot='paper/geophysics_rayleigh.png',
        )
        raise SystemExit(0)

    if args.plot_dynamo:
        generate_dynamo_mass_plot(
            min_mass_earth=args.min_mass,
            max_mass_earth=args.max_mass,
            output_csv='csv/geophysics_dynamo.csv',
            output_plot='validation/geophysics_dynamo.png',
            paper_plot='paper/geophysics_dynamo.png',
        )
        raise SystemExit(0)

    print("Geophysical Proxy Calculator")
    print("="*80)
    print()
    print("Rayleigh number and Magnetic Reynolds number")
    print("="*80)
    print()
    
    # Menu system
    print("Options:")
    print("  1. Calculate for a single planetary mass")
    print("  2. Generate CSV and plot for a range of planetary masses")
    print()
    
    while True:
        try:
            choice = input("Select an option (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                print("  Error: Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            exit(0)
    
    print()
    
    # Option 1: Single mass calculation
    if choice == '1':
        # Request planetary mass from user
        while True:
            try:
                mass_input = input("Enter planetary mass (in Earth masses or kg): ").strip()
                
                if not mass_input:
                    print("  Error: Please enter a value.")
                    continue
                
                # Check if input contains "earth" or "M_Earth" or similar (case insensitive)
                mass_input_lower = mass_input.lower()
                if any(keyword in mass_input_lower for keyword in ['earth', 'm_earth', 'm⊕', 'm_⊕']):
                    # Extract numeric value (handle scientific notation)
                    # Remove Earth-related keywords and extract number
                    for keyword in ['earth', 'm_earth', 'm⊕', 'm_⊕', '×', '*']:
                        mass_input_lower = mass_input_lower.replace(keyword, '')
                    # Extract number (including scientific notation)
                    number_match = re.search(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', mass_input_lower)
                    if number_match:
                        mass_value = float(number_match.group())
                    else:
                        raise ValueError("Could not extract numeric value")
                    planetary_mass = mass_value * M_EARTH
                    print(f"  Mass: {mass_value:.3f} M_Earth ({planetary_mass:.3e} kg)")
                else:
                    # Try to parse as float (could be in kg or Earth masses)
                    # If the number is very large (> 1e23), assume it's in kg
                    # Otherwise, assume it's in Earth masses
                    mass_value = float(mass_input)
                    if mass_value > 1e23:
                        # Large number, assume kg
                        planetary_mass = mass_value
                        mass_earth = planetary_mass / M_EARTH
                        print(f"  Mass: {mass_earth:.3f} M_Earth ({planetary_mass:.3e} kg)")
                    else:
                        # Small number, assume Earth masses
                        planetary_mass = mass_value * M_EARTH
                        print(f"  Mass: {mass_value:.3f} M_Earth ({planetary_mass:.3e} kg)")
                
                if planetary_mass <= 0:
                    print("  Error: Mass must be positive. Please try again.")
                    continue
                
                break
            except ValueError as e:
                print(f"  Error: Invalid input. Please enter a number (e.g., '1.0', '1.0 Earth', or '5.972e24').")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                exit(0)
        
        print()
        
        # Calculate Rayleigh number
        rayleigh_number_earth  = calculate_rayleigh_number(1 * M_EARTH)
        rayleigh_number        = calculate_rayleigh_number(planetary_mass)
        rayleign_number_scaled = RA_EARTH * (planetary_mass / M_EARTH) ** 1.71

        # Test Rayleigh scaling law for sanity
        if abs(rayleigh_number - rayleign_number_scaled) / rayleigh_number > 0.01:
            print("  [FAIL] Rayleigh number scaling test failed!")      
            sys.exit()
        else:
            print("  [OK] Rayleigh number scaling test passed.")
        
        # Calculate dynamo proxies (including magnetic Reynolds number)
        dynamo = calculate_dynamo_proxy(planetary_mass)
        magnetic_reynolds = dynamo['Rm']
        
        # Display result
        print("="*80)
        print("Result:")
        print(f"  Rayleigh number (Ra): {rayleigh_number:.2e}")
        print()
        print(f"  Critical Ra for convection onset: {RA_CRITICAL:.0f}")
        meets_threshold = rayleigh_number >= RA_CRITICAL
        status_symbol = "[OK]" if meets_threshold else "[X]"
        status_text = "Above" if meets_threshold else "Below"
        print(f"  {status_symbol} {status_text} critical threshold (convection {'can occur' if meets_threshold else 'cannot occur'})")
        
        # Additional thresholds for robust convection
        if rayleigh_number >= 1e6:
            print(f"  [OK] Robust convection (Ra >= 1e6, favors plate tectonics)")
        elif rayleigh_number >= 1e5:
            print(f"  [OK] Moderate convection (Ra >= 1e5)")
        elif meets_threshold:
            print(f"  [OK] Weak convection (Ra >= {RA_CRITICAL:.0f}, above critical but weak)")
        
        print()
        print(f"  Magnetic Reynolds number (Rm): {magnetic_reynolds:.2e}")
        print()
        print(f"  Critical Rm for dynamo action: {RM_CRITICAL:.0f}")
        meets_rm_threshold = magnetic_reynolds >= RM_CRITICAL
        rm_status_symbol = "[OK]" if meets_rm_threshold else "[X]"
        rm_status_text = "Above" if meets_rm_threshold else "Below"
        print(f"  {rm_status_symbol} {rm_status_text} critical threshold (dynamo {'can occur' if meets_rm_threshold else 'cannot occur'})")
        
        # Additional thresholds for robust dynamo
        if magnetic_reynolds >= 100:
            print(f"  [OK] Robust dynamo (Rm >= 100, favors plate tectonics)")
        elif magnetic_reynolds >= RM_CRITICAL:
            print(f"  [OK] Moderate dynamo (Rm >= {RM_CRITICAL:.0f})")
        
        print("="*80)
    
    # Option 2: Generate CSV and plot
    elif choice == '2':
        print("Generate CSV and plot for a range of planetary masses")
        print("="*80)
        print()
        print("Mass range: 0.1 to 10.0 M_Earth (hardcoded)")
        print("Number of points: 100 (hardcoded)")
        print()
        
        # Generate CSV and plot
        generate_rayleigh_reynolds_csv_and_plot()
        
        print()
        print("="*80)
        print("Done!")
        print("="*80)

