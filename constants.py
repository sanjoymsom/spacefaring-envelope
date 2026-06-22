#!/usr/bin/env python3
"""
Shared physical constants used across the codebase.
"""

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2 (gravitational constant)
g_EARTH = 9.80665  # m/s^2 (standard gravity)
M_EARTH = 5.972e24  # kg (Earth mass)
R_EARTH = 6.371e6   # m (Earth radius)
M_MARS = 6.4171e23  # kg (Mars mass; NASA fact sheet)
# Kepler-20 b — Hippke (2018) working values for rocket benchmarks: 10 M_E, 1.7 R_E
# (intro cites Buchhave et al. 2016: ~9.7 M_E, ~1.9 R_E for the planet itself)
KEPLER_20B_MASS_ME = 10.0
KEPLER_20B_RADIUS_ME = 1.7
KEPLER_20B_MASS = KEPLER_20B_MASS_ME * M_EARTH
KEPLER_20B_RADIUS = KEPLER_20B_RADIUS_ME * R_EARTH

