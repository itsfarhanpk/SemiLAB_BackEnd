"""Abrupt, one-dimensional p-n junction calculations in SI units."""

import math

import numpy as np

from .constants import CM3_TO_M3, K_B, M_TO_UM, Q
from .diode import normalized_current
from .materials import Material, intrinsic_concentration_cm3


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def validate_inputs(material: Material, na_cm3: float, nd_cm3: float, temperature_k: float, voltage_v: float):
    for name, value in (("Na", na_cm3), ("Nd", nd_cm3), ("temperature", temperature_k), ("applied voltage", voltage_v)):
        _finite(name, float(value))
    if not 1e14 <= na_cm3 <= 1e18 or not 1e14 <= nd_cm3 <= 1e18:
        raise ValueError("Na and Nd must be between 1e14 and 1e18 cm^-3.")
    if not 250.0 <= temperature_k <= 400.0:
        raise ValueError("Temperature must be between 250 K and 400 K.")
    if not -1.0 <= voltage_v <= 0.6:
        raise ValueError("Applied voltage must be between -1.0 V and +0.6 V.")


def simulate_junction(material: Material, na_cm3: float, nd_cm3: float, temperature_k: float, voltage_v: float) -> dict:
    validate_inputs(material, na_cm3, nd_cm3, temperature_k, voltage_v)
    ni_cm3 = intrinsic_concentration_cm3(material, temperature_k)
    thermal_v = K_B * temperature_k / Q
    v_bi = thermal_v * math.log(na_cm3 * nd_cm3 / ni_cm3**2)
    effective_drop = v_bi - voltage_v
    warnings = []
    valid = effective_drop > 0.0
    if not valid:
        warnings.append(
            "The depletion approximation is invalid because applied forward bias reaches the built-in potential. "
            "Profiles are omitted rather than shown as misleading values."
        )
        return {
            "valid": False, "warnings": warnings, "material": material.public_dict(),
            "built_in_potential_v": v_bi, "intrinsic_concentration_cm3": ni_cm3,
            "depletion_width_um": None, "p_side_width_um": None, "n_side_width_um": None,
            "maximum_electric_field_v_per_m": None, "position_um": [], "electric_field_v_per_m": [],
            "charge_density_c_per_m3": [], "electrostatic_potential_v": [],
            "conduction_band_ev": [], "valence_band_ev": [], "junction_position_um": 0.0,
            "diode_iv": _diode_curve(temperature_k, voltage_v),
        }

    na_m3, nd_m3 = na_cm3 * CM3_TO_M3, nd_cm3 * CM3_TO_M3
    width_m = math.sqrt(2.0 * material.permittivity / Q * (1.0 / na_m3 + 1.0 / nd_m3) * effective_drop)
    xp_m = width_m * nd_m3 / (na_m3 + nd_m3)
    xn_m = width_m * na_m3 / (na_m3 + nd_m3)
    # x=0 is the metallurgical junction; p-side is negative x. Sampling each
    # side independently keeps narrow depletion extensions visible.
    p_positions = np.linspace(-xp_m, 0.0, 161)
    n_positions = np.linspace(0.0, xn_m, 161)[1:]
    positions_m = np.concatenate((p_positions, n_positions))
    charge = np.where(positions_m < 0.0, -Q * na_m3, Q * nd_m3)
    field = np.where(
        positions_m < 0.0, -Q * na_m3 / material.permittivity * (positions_m + xp_m),
        Q * nd_m3 / material.permittivity * (positions_m - xn_m),
    )
    potential = np.where(
        positions_m <= 0.0,
        Q * na_m3 * (positions_m + xp_m) ** 2 / (2.0 * material.permittivity),
        effective_drop - Q * nd_m3 * (xn_m - positions_m) ** 2 / (2.0 * material.permittivity),
    )
    band_shift = -potential
    conduction = material.band_gap_ev_300k + band_shift
    valence = band_shift
    peak_field = Q * na_m3 * xp_m / material.permittivity
    if voltage_v > 0.8 * v_bi:
        warnings.append("Forward bias is close to the built-in potential; depletion results are increasingly approximate.")
    if ni_cm3 * 10.0 > min(na_cm3, nd_cm3):
        warnings.append("At least one side is weakly doped relative to ni; the depletion approximation is less reliable.")
    return {
        "valid": True, "warnings": warnings, "material": material.public_dict(),
        "built_in_potential_v": v_bi, "intrinsic_concentration_cm3": ni_cm3,
        "depletion_width_um": width_m * M_TO_UM, "p_side_width_um": xp_m * M_TO_UM,
        "n_side_width_um": xn_m * M_TO_UM, "maximum_electric_field_v_per_m": peak_field,
        "position_um": (positions_m * M_TO_UM).tolist(), "electric_field_v_per_m": field.tolist(),
        "charge_density_c_per_m3": charge.tolist(), "electrostatic_potential_v": potential.tolist(),
        "conduction_band_ev": conduction.tolist(), "valence_band_ev": valence.tolist(),
        "junction_position_um": 0.0, "diode_iv": _diode_curve(temperature_k, voltage_v),
    }


def _diode_curve(temperature_k: float, selected_voltage: float) -> dict:
    voltages = np.linspace(-1.0, 0.6, 161)
    currents = normalized_current(voltages, temperature_k)
    return {
        "voltage_v": voltages.tolist(), "normalized_current": currents.tolist(),
        "selected_voltage_v": selected_voltage,
        "selected_normalized_current": float(normalized_current(selected_voltage, temperature_k)),
        "model": "Ideal diode, I/I0 = exp(qVa/kBT) - 1, ideality n=1",
    }
