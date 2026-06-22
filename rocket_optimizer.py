#!/usr/bin/env python3
"""
Multi-stage rocket optimizer for planetary escape.

Supports a simple drag model (118 m/s per bar) and an altitude-dependent
drag model with iterative refinement.
"""
import sys
import math
import engineering_validation as ev
from atmospheric_generator import calculate_altitude_dependent_drag
from constants import G

# ============================================================================
# HELPER FUNCTION: Optimize Single Stage Count Configuration
# ============================================================================

def optimize_single_stage_count(num_stages, total_dv_required, payload_mass, 
                                ISP_LOW, ISP_HIGH, EPSILON_UNIFORM, g_EARTH, equal_dv_only=False):
    """
    Optimize stage masses for a fixed stage count and total delta-v.

    Returns a dict of masses and delta-v splits, or None if invalid.
    """
    # ------------------------------------------------------------------------
    # Step 1: Assign Isp values to stages
    # ------------------------------------------------------------------------
    # ASSUMPTION: Top 2 stages use high Isp (hydrolox), lower stages use low Isp (kerolox)
    # Rationale: Upper stages benefit more from high Isp, lower stages prioritize thrust
    isp_values = []
    epsilon_values = []
    for i in range(num_stages):
        if i < 2:  # Top two stages (i=0, i=1)
            isp_values.append(ISP_HIGH)
        else:  # Lower stages (i=2, i=3, ...)
            isp_values.append(ISP_LOW)
        epsilon_values.append(EPSILON_UNIFORM)
    
    # ------------------------------------------------------------------------
    # Step 2: Try equal delta-v distribution (baseline)
    # ------------------------------------------------------------------------
    # ASSUMPTION: Equal delta-v distribution is often near-optimal
    dv_per_stage = total_dv_required / num_stages
    
    # Solve from top to bottom (lightest to heaviest)
    # For top stage: m_above = payload (known)
    # For each subsequent stage: m_above = payload + all stages above it
    m_above = payload_mass
    stage_masses = []
    stage_deltav = []
    structural_masses = []
    propellant_masses = []
    valid = True
    for i in range(num_stages):  # i=0 is top stage, i=num_stages-1 is bottom stage
        isp = isp_values[i]
        epsilon = epsilon_values[i]
        
        try:
            # Tsiolkovsky rocket equation: dv = Isp * g_EARTH * ln(m0 / mf)
            # where m0 = m_wet + m_above, mf = m_above + epsilon * m_wet
            # Solving for m_wet: m_wet = m_above * (exp(dv/(Isp*g_EARTH)) - 1) / (1 - epsilon*exp(dv/(Isp*g_EARTH)))
            exp_term = math.exp(dv_per_stage / (isp * g_EARTH))
            
            # Check for invalid configuration (would require negative or infinite mass)
            if epsilon * exp_term >= 1.0:
                valid = False
                break
            
            m_wet = m_above * (exp_term - 1.0) / (1.0 - epsilon * exp_term)
            
            # Validate result
            if m_wet <= 0 or not math.isfinite(m_wet):
                valid = False
                break
            
            # Calculate structural and propellant masses
            m_struct = epsilon * m_wet
            m_prop = m_wet - m_struct
            
            stage_masses.append(m_wet)
            stage_deltav.append(dv_per_stage)
            structural_masses.append(m_struct)
            propellant_masses.append(m_prop)
            
            # For next stage down, m_above includes this stage's mass
            m_above += m_wet
            
        except (ValueError, OverflowError):
            valid = False
            break
    
    if not valid:
        return None
    
    launch_mass = sum(stage_masses) + payload_mass
    
    # Store result
    # The model uses an EQUAL Δv split across stages (Δv_i = Δv/n), matching the
    # "Δv split: Equal per stage" assumption in the paper (Table 1). The
    # `equal_dv_only` argument is retained for API compatibility; non-equal Δv
    # distributions were explored during development but are not used for any
    # published result.
    best_launch_mass = launch_mass
    best_stage_masses = stage_masses
    best_stage_deltav = stage_deltav
    best_structural_masses = structural_masses
    best_propellant_masses = propellant_masses

    return {
        'launch_mass': best_launch_mass,
        'stage_masses': best_stage_masses,
        'stage_deltav': best_stage_deltav,
        'structural_masses': best_structural_masses,
        'propellant_masses': best_propellant_masses,
        'isp_values': isp_values
    }


# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

def optimize_multi_stage_rocket(g_EARTH, M_EARTH, R_EARTH, planetary_mass, planetary_radius, escape_velocity, 
                                 surface_pressure=1.0, payload_mass=1000.0, max_stages=None,
                                 drag_model='altitude_dependent',max_engines=100,
                                 max_mass_flow_per_engine=2600.0,
                                 stage_reliability=0.97, max_launch_mass=4.0e8):
    """
    Find the reliability-optimal rocket configuration that reaches escape velocity.

    Rather than minimizing launch mass and capping the number of stages at an
    arbitrary value, this selects the stage count a rational builder would pick:
    the one that minimizes the EXPECTED launch mass (cost) to obtain one
    successful flight. Each stage contributes a separation and an ignition event,
    so mission reliability decays as ``stage_reliability ** n``. The expected
    mass ``launch_mass(n) / stage_reliability**n`` has an interior minimum n*
    because launch mass shows diminishing returns while 1/R_s**n grows
    geometrically. Feasibility is then governed by a physical resource budget on
    launch mass (``max_launch_mass``, default ~4e5 t following Hippke 2018's
    "Pyramid of Cheops" practical ceiling), not by a numerical stage cap.

    Returns a dict with mass breakdown, delta-v, losses, stage count, mission
    reliability, expected mass, and a success flag.
    """
   
    # Calculate surface gravity self-consistently with the adopted mass--radius
    # relation: g = GM/R^2 with R = R_EARTH (M/M_EARTH)^0.27. This is the same g
    # that defines v_esc = sqrt(2GM/R), so escape speed, gravity loss, and liftoff
    # thrust all use one gravity. (Valencia 2007's interior models give g ~ mu^0.5;
    # the adopted R--M law alone gives g ~ mu^0.46. The difference is immaterial to
    # the conclusions.)
    surface_g = G * planetary_mass / (planetary_radius ** 2)
    
    # ========================================================================
    # SECTION 2: Propulsion Parameters
    # ========================================================================
    # ASSUMPTION: Lower stages use kerosene/LOX (Isp ~ 350 s)
    # ASSUMPTION: Upper stages use hydrolox (Isp ~ 450 s)
    # Rationale: Upper stages benefit more from high Isp, lower stages prioritize thrust
    ISP_LOW = 350.0   # s (kerosene/LOX)
    ISP_HIGH = 450.0  # s (hydrolox)
    
    # Structural mass fraction (epsilon) - same for all stages
    # epsilon = structural_mass / wet_mass, typically 0.08-0.12 for well-designed stages
    # Using uniform 10% for all stages (simpler and easier to justify)
    EPSILON_UNIFORM = 0.10
    
    # ========================================================================
    # SECTION 3: Loss Calculations
    # ========================================================================
    # Gravity loss: Energy lost fighting gravity during ascent
    # ASSUMPTION: Gravity loss ~1500 m/s at Earth, scales with sqrt(g/g_EARTH)
    # Rationale: Gravity loss depends on burn time and trajectory, sqrt(g) scaling is approximate
    gravity_loss = 1500.0 * math.sqrt(surface_g / g_EARTH)
    
     # Drag loss: Energy lost to atmospheric drag
    # For altitude_dependent model, we use a two-pass solution:
    # Pass 1: Use simple drag to get stage 1 mass
    # Pass 2: Use stage 1 mass to calculate area, then recalculate drag and re-optimize
    if drag_model == 'simple':
        # ASSUMPTION: Drag loss scales linearly with surface pressure
    # ASSUMPTION: Drag loss = 118 m/s per bar at Earth
        # Rationale: Simplified model; actual drag depends on trajectory, vehicle shape, etc.
        drag_loss = 118.0 * surface_pressure  # m/s
        two_pass_needed = False
    elif drag_model == 'altitude_dependent':
        # Initial estimate for first pass (will be refined in second pass)
        drag_loss = 118.0 * surface_pressure  # m/s
        two_pass_needed = True
    else:
        raise ValueError(f"Unknown drag_model: {drag_model}. Use 'simple' or 'altitude_dependent'")
    
    # Total delta-v required = escape velocity + losses
    total_dv_required = escape_velocity + gravity_loss + drag_loss
    
    # ========================================================================
    # SECTION 4: Initialize Result Structure
    # ========================================================================
    best_result = {
        'launch_mass': float('inf'),
        'num_stages': None,
        'stage_masses': None,
        'stage_deltav': None,
        'total_deltav': total_dv_required,
        'structural_masses': None,
        'propellant_masses': None,
        'isp_values': None,
        'gravity_loss': gravity_loss,
        'drag_loss': drag_loss,
        'mission_reliability': 0.0,
        'expected_mass': float('inf'),
        'stage_reliability': stage_reliability,
        'success': False
    }

    # ========================================================================
    # SECTION 5: Reliability-Weighted Stage-Count Selection
    # ========================================================================
    # The expected mass to obtain one successful launch is
    #     M_expected(n) = launch_mass(n) / stage_reliability**n
    # which is unimodal in n (interior minimum n*). We search upward and stop
    # once we are clearly past the minimum. `absolute_search_max` is only a
    # numerical search bound, NOT a feasibility cap.
    if max_stages is None:
        absolute_search_max = 200
    else:
        absolute_search_max = max_stages

    def select_optimum_stage_count(total_dv):
        """Return (n*, cfg) minimizing reliability-weighted expected mass."""
        best_n_local = None
        best_cfg_local = None
        best_expected = float('inf')
        no_improvement = 0
        for n in range(1, absolute_search_max + 1):
            cfg = optimize_single_stage_count(
                n, total_dv, payload_mass,
                ISP_LOW, ISP_HIGH, EPSILON_UNIFORM, g_EARTH
            )
            # Too few stages for this dv leaves the per-stage mass ratio invalid;
            # keep adding stages until a valid configuration appears.
            if cfg is None:
                continue
            expected_mass = cfg['launch_mass'] / (stage_reliability ** n)
            if expected_mass < best_expected:
                best_expected = expected_mass
                best_n_local = n
                best_cfg_local = cfg
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= 6:  # safely past the unimodal minimum
                    break
        return best_n_local, best_cfg_local

    def store_cfg(cfg, n, drag_value, total_dv_value):
        best_result['launch_mass'] = cfg['launch_mass']
        best_result['num_stages'] = n
        best_result['stage_masses'] = cfg['stage_masses']
        best_result['stage_deltav'] = cfg['stage_deltav']
        best_result['structural_masses'] = cfg['structural_masses']
        best_result['propellant_masses'] = cfg['propellant_masses']
        best_result['isp_values'] = cfg['isp_values']
        best_result['drag_loss'] = drag_value
        best_result['total_deltav'] = total_dv_value

    # First pass with the initial drag estimate
    best_n, best_cfg = select_optimum_stage_count(total_dv_required)

    if best_cfg is None:
        # No valid configuration anywhere in the search range
        best_result['engine_number'] = 0
        return best_result

    store_cfg(best_cfg, best_n, drag_loss, total_dv_required)

    # ========================================================================
    # SECTION 6: Second Pass - Iterative Drag Refinement (altitude model only)
    # ========================================================================
    # Refine the altitude-dependent drag using the chosen first-stage mass, then
    # re-select the optimum stage count (n* can shift slightly). Converges in a
    # few iterations.
    if two_pass_needed:
        stage1_mass = best_cfg['stage_masses'][-1]
        current_launch_mass = best_cfg['launch_mass']
        drag_loss_old = 118.0 * surface_pressure
        max_iterations = 10
        for _ in range(max_iterations):
            drag_loss_new = calculate_altitude_dependent_drag(
                g_EARTH, surface_pressure, surface_g, stage1_mass,
                current_launch_mass, escape_velocity
            )
            if drag_loss_old > 0 and abs(drag_loss_new - drag_loss_old) / drag_loss_old < 0.05:
                break
            total_dv_refined = escape_velocity + gravity_loss + drag_loss_new
            refined_n, refined_cfg = select_optimum_stage_count(total_dv_refined)
            if refined_cfg is None:
                break
            stage1_mass = refined_cfg['stage_masses'][-1]
            current_launch_mass = refined_cfg['launch_mass']
            drag_loss_old = drag_loss_new
            best_n = refined_n
            store_cfg(refined_cfg, refined_n, drag_loss_new, total_dv_refined)

    # ========================================================================
    # SECTION 7: Mission Reliability, Expected Mass, and Feasibility Gate
    # ========================================================================
    mission_reliability = stage_reliability ** best_result['num_stages']
    best_result['mission_reliability'] = mission_reliability
    best_result['expected_mass'] = (
        best_result['launch_mass'] / mission_reliability if mission_reliability > 0 else float('inf')
    )
    # Feasibility is set by a physical resource budget on launch mass, not by a
    # numerical stage cap. Default ~4e5 t follows Hippke (2018).
    best_result['success'] = best_result['launch_mass'] <= max_launch_mass

    # ========================================================================
    # SECTION 8: Engineering diagnostics (engine count - reported, not a gate)
    # ========================================================================
    engine_result = ev.check_turbopump_power_limitation(g_EARTH, best_result['launch_mass'], surface_g)
    best_result['engine_number'] = engine_result['number_of_engines_estimate']
    return best_result


# ============================================================================
# MAIN: Test with Earth Parameters
# ============================================================================

if __name__ == "__main__":
    # Earth parameters for reference
    from constants import G, M_EARTH, R_EARTH, g_EARTH
    
    print("#" * 72)
    print("Multi-Stage Rocket Optimizer - Stage Count Analysis")
    print("#" * 72)
    print()
    
    # Get planetary mass from user
    while True:
        try:
            mass_input = input("Enter planetary mass (in Earth masses): ")
            mass_earth_masses = float(mass_input)
            if mass_earth_masses <= 0:
                print("Error: Mass must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid number. Please try again.")
    
    # Get surface pressure from user
    while True:
        try:
            pressure_input = input("Enter surface air pressure (in bar, e.g., 1.0 for Earth): ")
            surface_pressure = float(pressure_input)
            if surface_pressure < 0:
                print("Error: Pressure cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid number. Please try again.")
    
    # Get Isp configuration from user
    print()
    print("Isp configuration options:")
    print("  1. Uniform Isp (same for all stages)")
    print("  2. Mixed Isp (450s for upper 2 stages, 350s for lower stages)")
    print("     Note: Upper stages (last to fire) get 450s, lower stages (first to fire) get 350s")
    print("     Examples: 1 stage = 450s, 2 stages = both 450s, 3 stages = Stages 2-3 at 450s, Stage 1 at 350s")
    while True:
        try:
            isp_choice = input("Choose Isp configuration (1 or 2): ").strip()
            if isp_choice == "1":
                use_uniform_isp = True
                # Get uniform Isp value
                while True:
                    try:
                        isp_input = input("Enter specific impulse (Isp) in seconds (e.g., 350 for kerolox, 450 for hydrolox): ")
                        isp_value = float(isp_input)
                        if isp_value <= 0:
                            print("Error: Isp must be positive. Please try again.")
                            continue
                        ISP_LOW = isp_value
                        ISP_HIGH = isp_value
                        isp_description = f"{isp_value:.1f} s (uniform for all stages)"
                        break
                    except ValueError:
                        print("Error: Please enter a valid number. Please try again.")
                break
            elif isp_choice == "2":
                use_uniform_isp = False
                ISP_LOW = 350.0  # s (kerolox for lower stages)
                ISP_HIGH = 450.0  # s (hydrolox for upper 2 stages)
                isp_description = f"450s (upper 2 stages, last to fire), 350s (lower stages, first to fire)"
                break
            else:
                print("Error: Please enter 1 or 2.")
                continue
        except (ValueError, KeyboardInterrupt):
            print("Error: Invalid input. Please try again.")
            continue
    
    # Calculate planetary parameters
    planetary_mass = mass_earth_masses * M_EARTH
    mass_ratio     = planetary_mass / M_EARTH
    
    # Assume Valencia scaling for radius: R ~ M^0.27
    planetary_radius = R_EARTH * (mass_ratio ** 0.27)
    
    # Calculate escape velocity: v_esc = sqrt(2GM/R)
    escape_velocity = math.sqrt(2.0 * G * planetary_mass / planetary_radius)
    
    print()
    print(f"Planetary parameters:")
    print('---------------------')
    print(f"  Mass: {planetary_mass:.3e} kg ({mass_earth_masses:.3f} Earth masses)")
    print(f"  Radius: {planetary_radius:.3e} m ({planetary_radius/1000:.2f} km)")
    print(f"  Escape velocity: {escape_velocity:.2f} m/s ({escape_velocity/1000:.2f} km/s)")
    print(f"  Surface pressure: {surface_pressure:.2f} bar")
    print(f"  Specific impulse (Isp): {isp_description}")
    print()
    print("Calculating rocket masses for 1-10 stages with equal delta-v distribution...")
    print()
    
    # Calculate losses (same as in optimize_multi_stage_rocket)
    surface_g = G * planetary_mass / (planetary_radius ** 2)
    gravity_loss = 1500.0 * math.sqrt(surface_g / g_EARTH)
    drag_loss = 118.0 * surface_pressure  # Simple drag model to start
    total_dv_required = escape_velocity + gravity_loss + drag_loss
    
    # Propulsion parameters
    EPSILON_UNIFORM = 0.10
    payload_mass = 1000.0
    
    # Test 1-100 stages with equal delta-v distribution
    results = []
    for num_stages in range(1, 101):
        result = optimize_single_stage_count(
            num_stages, total_dv_required, payload_mass,
            ISP_LOW, ISP_HIGH, EPSILON_UNIFORM, g_EARTH, equal_dv_only=True
        )
        if result is not None:
            results.append({
                'num_stages': num_stages,
                'launch_mass': result['launch_mass'],
                'success': True
            })
        else:
            results.append({
                'num_stages': num_stages,
                'launch_mass': None,
                'success': False
            })
    
    # Display results table
    print("=" * 70)
    print(f"Results for {mass_earth_masses:.3f} Earth masses, {surface_pressure:.2f} bar pressure, Isp={isp_description}")
    print(f"Total delta-v required: {total_dv_required:.2f} m/s")
    print(f"  - Escape velocity: {escape_velocity:.2f} m/s")
    print(f"  - Gravity loss: {gravity_loss:.2f} m/s")
    print(f"  - Drag loss: {drag_loss:.2f} m/s")
    print("=" * 70)
    print()
    print(f"{'Stages':<10} {'Launch Mass (kg)':<20} {'Launch Mass (tons)':<20} {'Status':<15}")
    print("-" * 70)
    
    # Find optimal number of stages (minimum launch mass)
    optimal_stages = None
    optimal_mass = float('inf')
    for r in results:
        if r['success'] and r['launch_mass'] < optimal_mass:
            optimal_mass = r['launch_mass']
            optimal_stages = r['num_stages']
    
    for r in results:
        if r['success']:
            mass_kg = r['launch_mass']
            mass_tons = mass_kg / 1000.0
            # Mark optimal with asterisk
            optimal_marker = " *" if r['num_stages'] == optimal_stages else ""
            print(f"{r['num_stages']:<10} {mass_kg:>18.2e} {mass_tons:>18.2f} {'Success':<15}{optimal_marker}")
        else:
            print(f"{r['num_stages']:<10} {'N/A':<20} {'N/A':<20} {'Failed':<15}")
    
    print()
    if optimal_stages is not None:
        optimal_mass_tons = optimal_mass / 1000.0
        print(f"Optimal: {optimal_stages} stages with launch mass of {optimal_mass_tons:.2f} tons")
        print()
    print("Note: Delta-v is split evenly across all stages.")
    print("Note: * indicates optimal number of stages (minimum launch mass).")
