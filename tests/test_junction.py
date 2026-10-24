import numpy as np

from physics.junction import simulate_junction
from physics.materials import get_material


def run(**kwargs):
    values = {"na_cm3": 1e16, "nd_cm3": 1e16, "temperature_k": 300.0, "voltage_v": 0.0}
    values.update(kwargs)
    return simulate_junction(get_material("si"), **values)


def test_outputs_are_finite_and_charge_balances():
    result = run()
    assert result["valid"]
    assert np.isfinite(result["depletion_width_um"])
    assert np.isclose(np.trapezoid(result["charge_density_c_per_m3"][:121], np.array(result["position_um"][:121]) * 1e-6),
                      -np.trapezoid(result["charge_density_c_per_m3"][120:], np.array(result["position_um"][120:]) * 1e-6),
                      rtol=0.03)


def test_reverse_bias_increases_depletion_width():
    assert run(voltage_v=-0.8)["depletion_width_um"] > run(voltage_v=0.0)["depletion_width_um"]


def test_forward_limit_returns_warning_without_profiles():
    result = simulate_junction(get_material("ge"), 1e16, 1e16, 300.0, 0.6)
    assert not result["valid"]
    assert result["position_um"] == []
    assert result["warnings"]


def test_higher_doping_reduces_width():
    assert run(na_cm3=1e18, nd_cm3=1e18)["depletion_width_um"] < run()["depletion_width_um"]
