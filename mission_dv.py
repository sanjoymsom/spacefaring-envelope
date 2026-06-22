#!/usr/bin/env python3
"""
Mission Δv budgets for launch-vehicle validation (panel A).

Computed from the same loss model as rocket_optimizer / Eq. (dvbudget) in the paper:
  pad-to-parking: v_circ(r_park) + Δv_grav + Δv_drag
  parking departure: sqrt(2μ/r + C3) − v_circ   (prograde burn from circular LEO)

Hohmann C3 derivations and numerical values: paper Appendix A (app:missionc3).

Saturn V: mean Apollo 8--17 lunar statistics from Orloff (2000).
SLS Block 1: geocentric Earth--Moon Hohmann C3 (no Apollo heritage).
Mars: heliocentric Earth--Mars Hohmann C3.

Bibliographic URLs: validation_data_sources.py (ORLOFF_SP_4029, DATA_SOURCES).
"""

import math
from typing import Iterable, Optional, Tuple

from constants import G, M_EARTH, R_EARTH

# Match rocket_optimizer defaults at 1 M⊕, 1 bar
PARKING_ALTITUDE_M = 200_000.0
GRAVITY_LOSS_EARTH_MS = 1500.0
DRAG_LOSS_EARTH_1BAR_MS = 118.0

FT2_TO_M2 = 0.3048 ** 2
LB_TO_KG = 0.45359237

# Saturn V flights in Orloff (2000), Apollo 8--17.
SATURN_V_APOLLO_MISSIONS: Tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17)

# Translunar-injection C3 at cutoff (ft^2 s^-2), Orloff translunar table.
# Apollo 9 did not perform TLI; omitted from C3 mean only.
APOLLO_TLI_MISSIONS: Tuple[int, ...] = (8, 10, 11, 12, 13, 14, 15, 16, 17)
APOLLO_TLI_C3_FT2 = {
    8: -15_918_930,
    10: -14_084_265,
    11: -14_979_133,
    12: -19_745_586,
    13: -14_814_090,
    14: -18_096_135,
    15: -15_643_934,
    16: -16_881_439,
    17: -18_152_226,
}

# Ground ignition weight, total vehicle (lbm), Orloff Table "Ground Ignition Weights".
APOLLO_GLOM_LBM = {
    8: 6_221_823,
    9: 6_486_577,
    10: 6_486_873,
    11: 6_477_875,
    12: 6_487_742,
    13: 6_501_733,
    14: 6_505_548,
    15: 6_494_415,
    16: 6_537_238,
    17: 6_529_784,
}

# Total spacecraft mass at ignition (lbm), same source.
APOLLO_SPACECRAFT_LBM = {
    8: 96_272,
    9: 104_031,
    10: 107_200,
    11: 109_646,
    12: 110_044,
    13: 110_226,
    14: 111_122,
    15: 116_235,
    16: 116_314,
    17: 116_269,
}

MU_SUN = 1.32712440018e20  # m^3 s^-2
AU_M = 1.495978707e11
MARS_ORBIT_AU = 1.524
MOON_ORBIT_RADIUS_M = 384_400e3


def _mu_earth() -> float:
    return G * M_EARTH


def parking_radius(altitude_m: float = PARKING_ALTITUDE_M) -> float:
    return R_EARTH + altitude_m


def circular_speed(mu: float, radius_m: float) -> float:
    return math.sqrt(mu / radius_m)


def pad_to_parking_dv(
    mu: Optional[float] = None,
    altitude_m: float = PARKING_ALTITUDE_M,
    gravity_loss_ms: float = GRAVITY_LOSS_EARTH_MS,
    drag_loss_ms: float = DRAG_LOSS_EARTH_1BAR_MS,
) -> float:
    """Launch-site to circular parking orbit (m/s)."""
    mu = mu or _mu_earth()
    r = parking_radius(altitude_m)
    return circular_speed(mu, r) + gravity_loss_ms + drag_loss_ms


def departure_dv_from_parking(
    c3_m2_s2: float,
    mu: Optional[float] = None,
    altitude_m: float = PARKING_ALTITUDE_M,
) -> float:
    """Circular parking orbit to departure C3 (m/s). C3 = v_inf^2 (m^2 s^-2)."""
    mu = mu or _mu_earth()
    r = parking_radius(altitude_m)
    v_circ = circular_speed(mu, r)
    return math.sqrt(2.0 * mu / r + c3_m2_s2) - v_circ


def pad_to_c3_total(
    c3_m2_s2: float,
    mu: Optional[float] = None,
    altitude_m: float = PARKING_ALTITUDE_M,
    gravity_loss_ms: float = GRAVITY_LOSS_EARTH_MS,
    drag_loss_ms: float = DRAG_LOSS_EARTH_1BAR_MS,
) -> float:
    """Launch site to injection with characteristic energy C3 (m/s)."""
    return pad_to_parking_dv(mu, altitude_m, gravity_loss_ms, drag_loss_ms) + (
        departure_dv_from_parking(c3_m2_s2, mu, altitude_m)
    )


def surface_escape_dv(
    mu: Optional[float] = None,
    radius_m: float = R_EARTH,
    gravity_loss_ms: float = GRAVITY_LOSS_EARTH_MS,
    drag_loss_ms: float = DRAG_LOSS_EARTH_1BAR_MS,
) -> float:
    """Launch site to Earth escape (Eq. dvbudget, m/s)."""
    mu = mu or _mu_earth()
    v_esc = math.sqrt(2.0 * mu / radius_m)
    return v_esc + gravity_loss_ms + drag_loss_ms


def hohmann_moon_c3(
    r_park: Optional[float] = None,
    r_moon: float = MOON_ORBIT_RADIUS_M,
    mu: Optional[float] = None,
) -> float:
    """C3 (m^2 s^-2) for minimum-energy Earth--Moon transfer from parking."""
    mu = mu or _mu_earth()
    r_park = parking_radius() if r_park is None else r_park
    a = 0.5 * (r_park + r_moon)
    v_inj = math.sqrt(mu * (2.0 / r_park - 1.0 / a))
    return v_inj * v_inj - 2.0 * mu / r_park


def hohmann_mars_c3(
    r_earth_au: float = 1.0,
    r_mars_au: float = MARS_ORBIT_AU,
    mu_sun: float = MU_SUN,
    au_m: float = AU_M,
) -> float:
    """Characteristic energy C3 (m^2 s^-2) for minimum-energy Earth--Mars Hohmann."""
    r_e = r_earth_au * au_m
    r_m = r_mars_au * au_m
    a = 0.5 * (r_e + r_m)
    v_planet = math.sqrt(mu_sun / r_e)
    v_dep = math.sqrt(mu_sun * (2.0 / r_e - 1.0 / a))
    v_inf = v_dep - v_planet
    return v_inf * v_inf


def c3_ft2_to_si(c3_ft2_s2: float) -> float:
    return c3_ft2_s2 * FT2_TO_M2


def mean_apollo_tli_c3_si(
    missions: Iterable[int] = APOLLO_TLI_MISSIONS,
) -> float:
    """Mean flown lunar TLI C3 (m^2 s^-2), Orloff Apollo 8--17 (excl. Apollo 9)."""
    values = [c3_ft2_to_si(APOLLO_TLI_C3_FT2[m]) for m in missions]
    return sum(values) / len(values)


def mean_saturn_v_glom_kg(
    missions: Iterable[int] = SATURN_V_APOLLO_MISSIONS,
) -> float:
    """Mean Saturn V ground ignition mass (kg), Orloff Apollo 8--17."""
    masses = [APOLLO_GLOM_LBM[m] * LB_TO_KG for m in missions]
    return sum(masses) / len(masses)


def mean_saturn_v_spacecraft_kg(
    missions: Iterable[int] = SATURN_V_APOLLO_MISSIONS,
) -> float:
    """Mean Apollo spacecraft mass at ignition (kg), Orloff Apollo 8--17."""
    masses = [APOLLO_SPACECRAFT_LBM[m] * LB_TO_KG for m in missions]
    return sum(masses) / len(masses)


def mission_dv_ms(mission: str, **kwargs) -> float:
    """
    Total mission Δv in m/s for validation missions.

    mission:
      leo, earth_escape_c3_0, mars_hohmann, lunar_tli, lunar_tli_saturn,
      surface_escape
    """
    if mission == "leo":
        return pad_to_parking_dv(**kwargs)
    if mission == "earth_escape_c3_0":
        return pad_to_c3_total(0.0, **kwargs)
    if mission == "mars_hohmann":
        return pad_to_c3_total(hohmann_mars_c3(), **kwargs)
    if mission == "lunar_tli":
        return pad_to_c3_total(hohmann_moon_c3(), **kwargs)
    if mission == "lunar_tli_saturn":
        return pad_to_c3_total(mean_apollo_tli_c3_si(), **kwargs)
    if mission == "surface_escape":
        return surface_escape_dv(**kwargs)
    raise ValueError(f"Unknown mission: {mission}")


def mission_dv_kms(mission: str, **kwargs) -> float:
    return mission_dv_ms(mission, **kwargs) / 1000.0
