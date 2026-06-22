#!/usr/bin/env python3
"""
Engineering checks for stage 1 turbopump power limits.
"""

import math


def check_turbopump_power_limitation(g_EARTH, total_launch_mass, surface_g,
                                     isp_stage1=None, max_engines=100,
                                     max_mass_flow_per_engine=2600.0):
    # Engine parameters (typical for first stage, RP-1/LOX)
    # Use provided Isp or default to ISP_LOW from rocket_optimizer.py (350.0 s)
    # Note: 350 s is vacuum Isp; sea-level Isp is ~330 s, but we use 350 s for consistency
    # with rocket_optimizer.py which uses ISP_LOW = 350.0 s for lower stages
    if isp_stage1 is None:
        Isp_stage1 = 350.0  # s (matches ISP_LOW from rocket_optimizer.py)
    else:
        Isp_stage1 = isp_stage1
    # CAVEAT: Isp_stage1 = 350 s is a vacuum/upper-stage kerolox value, but the
    # liftoff thrust->mass-flow conversion below should use the FIRST-STAGE
    # SEA-LEVEL Isp. The F-1 (whose 2600 kg/s sets max_mass_flow_per_engine) has
    # a sea-level Isp ~263 s. Using 350 s here understates the required total
    # mass flow and hence the engine count: e.g. for the Saturn-V-scale launch
    # mass the model predicts 4 engines, whereas a consistent ~263 s yields 5,
    # matching the real 5x F-1. Kept at 350 s for consistency with the optimizer;
    # pass isp_stage1=263 for a first-stage-consistent estimate.
    TWR_target = 1.5  # Thrust-to-weight ratio for liftoff
    
    # Turbopump parameters
    # NOTE: 100 bar is the pump pressure RISE (discharge), not the chamber
    # pressure. For the F-1 the chamber pressure is ~70 bar, but the pump must
    # discharge higher (~110 bar) to overcome the injector drop, cooling jacket,
    # and feed losses. Using ~100 bar reproduces the F-1 turbopump power
    # (60,000 hp, ~45 MW; Stangeland 1992, Fig. 6) to within ~20%. (Pressure
    # rise cancels in the engine-COUNT ratio
    # below, so this value only affects the absolute pump-power figures.)
    pressure_rise = 100e5  # Pa (~100 bar pump pressure rise, RP-1/LOX class)
    pump_efficiency = 0.70  # Typical pump efficiency (60-80%)
    
    # Propellant properties (RP-1/LOX)
    fuel_density = 1000  # kg/m³ average density of RP-1/LOX mixture
    
    # Calculate required thrust
    required_thrust = total_launch_mass * surface_g * TWR_target
    
    # Calculate total mass flow rate
    # From rocket equation: F = m_dot * Isp * g_EARTH
    # Therefore: m_dot = F / (Isp * g_EARTH)
    total_mass_flow_rate = required_thrust / (Isp_stage1 * g_EARTH)
        
    # Calculate pump power for each propellant
    # Power = (mass_flow * pressure_rise) / (density * efficiency)
    total_pump_power = (total_mass_flow_rate * pressure_rise) / (fuel_density * pump_efficiency)
    single_engine_pump_power = (max_mass_flow_per_engine * pressure_rise) / (fuel_density * pump_efficiency)
    
    # Calculate number of pumps needed (external analysis approach)
    num_engines_estimate = math.ceil(total_pump_power / single_engine_pump_power)
     
    engine_result = {
        'total_pump_power': total_pump_power,
        'single_engine_pump_power': single_engine_pump_power,
        'number_of_engines_estimate': num_engines_estimate}
    
    return engine_result

if __name__ == "__main__":
    # Import here to avoid circular import
    from rocket_optimizer import optimize_multi_stage_rocket
    from constants import G, M_EARTH, R_EARTH, g_EARTH
    
    # Calculate Earth's escape velocity
    v_esc_earth = math.sqrt(2.0 * G * M_EARTH / R_EARTH)
    P_EARTH = 1.0  # bar
    
    print("Running rocket optimization...")
    print()
    
    # Get rocket optimization results
    result = optimize_multi_stage_rocket(
        g_EARTH,
        M_EARTH,
        R_EARTH,
        M_EARTH,
        R_EARTH,
        v_esc_earth,
        surface_pressure=P_EARTH,
    )
    
    if not result['success']:
        print("ERROR: Rocket optimization failed")
        exit(1)
    
    print(f"Optimization complete: {result['num_stages']} stages, "
          f"{result['launch_mass']/1000:.2f} tons total mass")
    print()

