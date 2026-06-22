#!/usr/bin/env python3
#######################################################################
#Terrestrial planet properties based on Valencia et al. 2006 mass-radius relationship.
#
#This module provides functions to calculate planetary radius and related properties
#for terrestrial (rocky) planets based on their mass, using the empirical mass-radius
#scaling relationship from Valencia et al. 2006.
#
#REFERENCE:
#Valencia, D., O'Connell, R. J., & Sasselov, D. D. (2006).
#Internal structure of massive terrestrial planets.
#Icarus, 181(2), 545-554.
#
#The mass-radius relationship for terrestrial planets follows a power-law:
#R = R_Earth * (M / M_Earth)^α
#
#where α ≈ 0.27 for Earth-like compositions.
#######################################################################

import math
from constants import G, M_EARTH, R_EARTH

# Valencia et al. 2006 mass-radius relationship parameters
# Power-law exponent for terrestrial planets with Earth-like composition
VALENCIA_EXPONENT = 0.27


def calculate_planetary_radius(planetary_mass, M_ref=M_EARTH, R_ref=R_EARTH, 
    exponent=VALENCIA_EXPONENT):
    if planetary_mass <= 0:
        raise ValueError("Planetary mass must be positive")
    # Power-law relationship: R = R_ref * (M / M_ref)^α
    radius = R_ref * (planetary_mass / M_ref) ** exponent
    return radius


def calculate_surface_gravity(planetary_mass, planetary_radius=None):
    if planetary_radius is None:
        planetary_radius = calculate_planetary_radius(planetary_mass) 
    surface_g = G * planetary_mass / (planetary_radius ** 2)
    return surface_g


def calculate_escape_velocity(planetary_mass, planetary_radius=None):
    if planetary_radius is None:
        planetary_radius = calculate_planetary_radius(planetary_mass)
    escape_velocity = math.sqrt(2.0 * G * planetary_mass / planetary_radius)
    return escape_velocity


#######################################################################
# MAIN: Test with Earth and other examples
#######################################################################

if __name__ == "__main__":
    print("Testing terrestrial planet calculations with Valencia et al. 2006 relationship:")
    print("="*80)
    print()
    
    # Test with Earth
    M_earth = M_EARTH
    R_earth_calc = calculate_planetary_radius(M_earth)
    g_earth_calc = calculate_surface_gravity(M_earth, R_earth_calc)
    v_esc_earth_calc = calculate_escape_velocity(M_earth, R_earth_calc)
    
    print("Earth (reference):")
    print(f"  Mass: {M_earth:.3e} kg")
    print(f"  Calculated radius: {R_earth_calc:.3e} m ({R_earth_calc/1000:.1f} km)")
    print(f"  Actual radius: {R_EARTH:.3e} m ({R_EARTH/1000:.1f} km)")
    print(f"  Error: {abs(R_earth_calc - R_EARTH)/R_EARTH*100:.2f}%")
    print(f"  Surface gravity: {g_earth_calc:.2f} m/s²")
    print(f"  Escape velocity: {v_esc_earth_calc:.2f} m/s ({v_esc_earth_calc/1000:.2f} km/s)")
    print()
    
    # Test with other masses
    test_masses = [0.5, 2.0, 5.0, 10.0]  # Earth masses
    
    print("Other planetary masses:")
    print("-"*80)
    print("Mass (M_Earth) | Radius (km) | Surface g (m/s²) | Escape v (km/s)")
    print("-"*80)
    
    for M_ratio in test_masses:
        M = M_ratio * M_EARTH
        R = calculate_planetary_radius(M)
        g = calculate_surface_gravity(M, R)
        v_esc = calculate_escape_velocity(M, R)
        
        print(f"{M_ratio:6.1f} M_Earth | {R/1000:8.1f} km | {g:12.2f} m/s² | {v_esc/1000:10.2f} km/s")









