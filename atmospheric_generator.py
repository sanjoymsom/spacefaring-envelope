#!/usr/bin/env python3
"""
Atmospheric utilities for drag modeling.

Provides isothermal density/scale-height helpers and an altitude-dependent
drag-loss integrator used by the rocket optimizer.
"""
import sys
import math


# ============================================================================
# ATMOSPHERIC CONSTANTS
# ============================================================================

R_GAS = 8.31446261815324  # J mol^-1 K^-1 (universal gas constant)
MU_AIR = 0.0288           # kg/mol (mean molar mass, N2-O2 mix)
T_REF = 288.0             # K (reference temperature at 1 bar, 15°C)
FRAC_N2 = 0.78            # Fraction of N2 in Earth-like atmosphere

# ASSUMPTION: All planets have Earth-like atmospheric composition
# Reality: Different planets may have different compositions (CO2, H2, etc.)
# Impact: Affects scale height and density calculations

# ASSUMPTION: Constant temperature with altitude (isothermal atmosphere)
# Reality: Temperature typically decreases with altitude (~6.5 K/km on Earth)
# Impact: Moderate - affects scale height calculation


# ============================================================================
# ATMOSPHERIC PROPERTY CALCULATIONS
# ============================================================================

def calculate_surface_temperature(surface_pressure, T_base=T_REF, 
                                  include_pressure_broadening=False):
    """
    Return surface temperature in K.

    Pressure broadening is currently a no-op; this returns T_base.
    """
    if include_pressure_broadening:
        # Add temperature increase from other effects (e.g. pressure broadening)
        # Note: not implemented
        delta_T = 0
        return T_base + delta_T
    else:
        # Constant temperature (default)
        return T_base


def calculate_scale_height(surface_g, surface_pressure=None, T=None, mu=MU_AIR,
                           include_pressure_broadening=False):
    """
    Compute isothermal scale height H = RT/(mu*g) in meters.
    """
    # Calculate temperature
    if surface_pressure is not None and include_pressure_broadening:
        T = calculate_surface_temperature(surface_pressure, include_pressure_broadening=True)
    elif T is None:
        T = T_REF
    
    return R_GAS * T / (mu * surface_g)


def calculate_surface_density(surface_pressure, surface_g=None, T=None, mu=MU_AIR,
                             include_pressure_broadening=False):
    """
    Compute surface density from pressure via the ideal gas law.
    """
    # Calculate temperature
    if surface_g is not None and include_pressure_broadening:
        T = calculate_surface_temperature(surface_pressure, include_pressure_broadening=True)
    elif T is None:
        T = T_REF
    
    p_pa = surface_pressure * 1.0e5  # Convert bar to Pa
    return p_pa * mu / (R_GAS * T)  # kg/m³


def calculate_density_at_altitude(surface_pressure, surface_g, altitude, 
                                   T=None, mu=MU_AIR, include_pressure_broadening=False):
    """
    Compute density at altitude using an exponential profile.
    """
    # Calculate surface density
    rho0 = calculate_surface_density(surface_pressure, surface_g, T, mu, 
                                     include_pressure_broadening)
    
    # Calculate scale height
    scale_height_H = calculate_scale_height(surface_g, surface_pressure, T, mu,
                                           include_pressure_broadening)
    
    return rho0 * math.exp(-altitude / scale_height_H)


# ============================================================================
# DRAG LOSS CALCULATION
# ============================================================================
# Note: a Max-Q (peak dynamic pressure) diagnostic was explored during
# development but is not part of the published model; the drag loss used by the
# paper is the integrated quantity computed in calculate_altitude_dependent_drag
# below. The include_maxq / include_throttling flags are retained for API
# compatibility and are not exercised by the published results.

def calculate_altitude_dependent_drag(g_EARTH, surface_pressure, surface_g, stage1_wet_mass, 
                                      total_launch_mass, escape_velocity,
                                      include_maxq=False, include_throttling=False):
    """
    Integrate drag loss over a vertical ascent with exponential density.

    Returns the integrated drag loss in m/s (paper Sec. ascent). The include_maxq
    and include_throttling flags are accepted for API compatibility but are not
    used by the published model.
    """
    # ------------------------------------------------------------------------
    # Step 1: Physical Constants and Parameters
    # ------------------------------------------------------------------------
    Cd = 0.2      # Drag coefficient (fixed; conservative baseline)
    # Note: Real drag coefficients vary with Mach number and rise toward transonic speeds
    
    # ------------------------------------------------------------------------
    # Step 2: Calculate Atmospheric Properties
    # ------------------------------------------------------------------------
    # Use constant temperature (288 K) by default
    # Note: Pressure broadening can be enabled by setting include_pressure_broadening=True
    scale_height_H = calculate_scale_height(surface_g, surface_pressure, 
                                           include_pressure_broadening=False)
    rho0 = calculate_surface_density(surface_pressure, surface_g, 
                                     include_pressure_broadening=False)
    # ------------------------------------------------------------------------
    # Step 3: Estimate Rocket Cross-Sectional Area from Stage 1 Mass
    # ------------------------------------------------------------------------
    # Estimate area by assuming cylindrical stage 1 and calculating diameter
    # from propellant volume
    
    # ASSUMPTION: Cylindrical stage 1 with propellant density ~1000 kg/m³
    propellant_density = 1000.0  # kg/m³ (kerosene/LOX mixture)
    EPSILON = 0.10  # Structural fraction (10%)
    
    # Calculate propellant volume from stage 1 mass
    stage1_propellant_mass = stage1_wet_mass * (1.0 - EPSILON)
    propellant_volume = stage1_propellant_mass / propellant_density
    
    # ASSUMPTION: Aspect ratio (height/diameter) = 3 (typical for first stages)
    aspect_ratio = 3.0
    # For cylinder: V = π·(d/2)²·h = π·(d/2)²·(aspect_ratio·d) = π·aspect_ratio·d³/4
    # Solving for d: d = (4V/(π·aspect_ratio))^(1/3)
    diameter = (4 * propellant_volume / (math.pi * aspect_ratio)) ** (1.0/3.0)
    cross_sectional_area = math.pi * (diameter / 2.0) ** 2  # m²
    
    # ------------------------------------------------------------------------
    # Step 4: Constant Acceleration Model 
    # ------------------------------------------------------------------------
    # ASSUMPTION: Thrust-to-weight ratio (TWR) of 1.5 (typical for rockets)
    # Net acceleration a = (TWR - 1) * g = 0.5 * g
    # This scales naturally with planet gravity
    TWR = 1.5
    base_net_acceleration = (TWR - 1.0) * surface_g  # m/s²
    
    # ------------------------------------------------------------------------
    # Step 5: Average Mass Approximation
    # ------------------------------------------------------------------------
    # ASSUMPTION: Use half of launch mass as average mass during ascent
    # Note: This is a simplification - real rockets lose mass continuously
    # Using m_avg = m_rocket/2 provides a reasonable approximation
    m_avg = total_launch_mass / 2.0  # kg
    
    # ------------------------------------------------------------------------
    # Step 6: Integration Setup
    # ------------------------------------------------------------------------
    # Integration limits using scale height
    # ASSUMPTION: Integrate until density drops to < 0.1% of surface density
    # ρ(h)/ρ₀ = exp(-h/H) < 0.001 → h > -H·ln(0.001) ≈ 6.9·H
    # Use 10 scale heights to be safe (density ~ 0.0045% of surface)
    # ASSUMPTION: Drag is negligible above this altitude
    max_altitude = 10.0 * scale_height_H  # m
    
    # Altitude step size (use 1% of scale height for good resolution)
    dh = scale_height_H * 0.01  # m
    
    # ------------------------------------------------------------------------
    # Step 7: Integrate Drag Over Trajectory (with Max Q tracking)
    # ------------------------------------------------------------------------
    # Numerically integrate drag force over altitude to get total drag loss
    # Method: dv_drag = ∫ (F_drag / m) dt = ∫ a_drag dt
    
    total_drag_loss = 0.0
    h = 0.0  # Current altitude
     
    while h < max_altitude:
        # Calculate density at this altitude using exponential profile
        density = rho0 * math.exp(-h / scale_height_H)
        
        # Early exit if density is negligible (< 0.1% of surface)
        if density < rho0 * 0.001:
            break
        
        # Nothing fancy wrt acceleration
        net_acceleration = base_net_acceleration
        
        # Calculate velocity at this altitude
        # For constant acceleration: v(h) = sqrt(2·a·h) for vertical ascent
        # Note: With throttling, acceleration changes, so we need to integrate
        # For simplicity, use current acceleration (approximation)
        if h > 0:
            # Simplified: use current net acceleration
            # More accurate would integrate acceleration history
            velocity = math.sqrt(2.0 * net_acceleration * h)
        else:
            velocity = 0.0  # At surface, velocity is zero
        
        # Calculate drag force: F_drag = 0.5 · ρ · v² · Cd · A
        # This is the standard drag equation for incompressible flow
        drag_force = 0.5 * density * velocity ** 2 * Cd * cross_sectional_area  # N
        
        # Calculate drag acceleration: a_drag = F_drag / m
        drag_acceleration = drag_force / m_avg  # m/s²
        
        # Calculate time step: dt = dh / v (if v > 0)
        # This converts altitude step to time step
        if velocity > 0:
            dt = dh / velocity
        else:
            dt = 0.0  # At surface, no time passes for zero velocity
        
        # Integrate: dv_drag += a_drag · dt
        # This accumulates the velocity loss due to drag
        total_drag_loss += drag_acceleration * dt
        
        # Move to next altitude step
        h += dh
    
    return total_drag_loss
