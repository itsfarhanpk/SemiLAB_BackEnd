import numpy as np
import pytest

from physics.diode import normalized_current
from physics.junction import simulate_junction
from physics.materials import MATERIALS, get_material


def run(material="si", **kwargs):
    values = {"na_cm3": 1e16, "nd_cm3": 1e16, "temperature_k": 300.0, "voltage_v": 0.0}
    values.update(kwargs)
    return simulate_junction(get_material(material), **values)


def test_default_reference_values_and_finite_profiles():
    result = run()
    assert result["valid"]
    assert result["built_in_potential_v"] == pytest.approx(0.714317, rel=2e-4)
    assert result["depletion_width_um"] == pytest.approx(0.429821, rel=2e-4)
    assert result["maximum_electric_field_v_per_m"] == pytest.approx(3.32e6, rel=0.02)
    assert np.all(np.isfinite(result["electric_field_v_per_m"]))
    assert np.all(np.isfinite(result["electrostatic_potential_v"]))


def test_field_direction_continuity_and_zero_edges():
    result = run()
    positions = np.asarray(result["position_um"])
    field = np.asarray(result["electric_field_v_per_m"])
    junction = np.flatnonzero(np.isclose(positions, 0.0))[0]
    assert field[0] == pytest.approx(0.0, abs=1e-9)
    assert field[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(field <= 1e-9)
    assert field[junction] == pytest.approx(-result["maximum_electric_field_v_per_m"], rel=1e-12)


def test_charge_neutrality_and_potential_continuity():
    result = run(na_cm3=1e16, nd_cm3=1e18)
    positions = np.asarray(result["position_um"])
    potential = np.asarray(result["electrostatic_potential_v"])
    junction = np.flatnonzero(np.isclose(positions, 0.0))[0]
    assert result["p_side_width_um"] * 1e16 == pytest.approx(result["n_side_width_um"] * 1e18, rel=1e-12)
    na_m3 = 1e16 * 1e6
    expected_junction = 1.602176634e-19 * na_m3 * (result["p_side_width_um"] * 1e-6) ** 2 / (
        2 * get_material("si").permittivity
    )
    assert potential[junction] == pytest.approx(expected_junction, rel=1e-10)
    assert potential[0] == pytest.approx(0.0, abs=1e-12)
    assert potential[-1] == pytest.approx(result["built_in_potential_v"], rel=1e-10)


def test_potential_field_and_poisson_consistency():
    result = run()
    x = np.asarray(result["position_um"]) * 1e-6
    e = np.asarray(result["electric_field_v_per_m"])
    phi = np.asarray(result["electrostatic_potential_v"])
    rho = np.asarray(result["charge_density_c_per_m3"])
    segment_field = -np.diff(phi) / np.diff(x)
    assert np.allclose(segment_field, (e[:-1] + e[1:]) / 2.0, rtol=2e-10, atol=2e-6)
    material = get_material("si")
    derivative = np.gradient(e, x)
    assert np.allclose(derivative[2:158], rho[2:158] / material.permittivity, rtol=2e-3, atol=1e-2)


def test_energy_bands_are_monotonic_and_gap_is_constant():
    result = run()
    ec = np.asarray(result["conduction_band_ev"])
    ev = np.asarray(result["valence_band_ev"])
    assert np.all(np.diff(ec) <= 1e-12)
    assert np.all(np.diff(ev) <= 1e-12)
    assert np.allclose(ec - ev, get_material("si").band_gap_ev_300k)
    assert ec[-1] - ec[0] == pytest.approx(-result["built_in_potential_v"], rel=1e-10)


@pytest.mark.parametrize("na,nd", [(1e16, 1e16), (1e18, 1e16), (1e16, 1e18), (1e18, 1e14)])
def test_sampling_includes_junction_and_preserves_order(na, nd):
    result = run(na_cm3=na, nd_cm3=nd)
    positions = np.asarray(result["position_um"])
    assert np.count_nonzero(np.isclose(positions, 0.0)) == 1
    assert np.all(np.diff(positions) > 0)
    assert len(positions) == len(result["electric_field_v_per_m"])


@pytest.mark.parametrize("material", list(MATERIALS))
@pytest.mark.parametrize("temperature", [250.0, 300.0, 400.0])
def test_all_materials_and_temperature_limits(material, temperature):
    result = run(material=material, temperature_k=temperature)
    assert result["valid"]
    assert result["intrinsic_concentration_cm3"] > 0
    assert result["depletion_width_um"] > 0


def test_bias_scaling_and_invalid_model_state():
    assert run(voltage_v=-0.8)["depletion_width_um"] == pytest.approx(0.625822, rel=2e-3)
    assert run(voltage_v=0.45)["depletion_width_um"] == pytest.approx(0.261460, rel=2e-3)
    invalid = simulate_junction(get_material("ge"), 1e16, 1e16, 300.0, 0.6)
    assert not invalid["valid"]
    assert invalid["position_um"] == []


def test_normalized_diode_current():
    result = run(voltage_v=0.1)
    expected = normalized_current(0.1, 300.0)
    assert result["diode_iv"]["selected_normalized_current"] == pytest.approx(float(expected))
    assert result["diode_iv"]["normalized_current"][0] < 0


def test_charge_integrals_balance():
    result = run(na_cm3=1e16, nd_cm3=1e18)
    p_charge_per_area = -1.602176634e-19 * 1e16 * 1e6 * result["p_side_width_um"] * 1e-6
    n_charge_per_area = 1.602176634e-19 * 1e18 * 1e6 * result["n_side_width_um"] * 1e-6
    assert p_charge_per_area + n_charge_per_area == pytest.approx(0.0, abs=1e-18)
