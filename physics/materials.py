"""Material parameters and a documented intrinsic-density approximation.

The intrinsic carrier density is approximated from its 300 K value with
ni(T) = ni(300) * (T / 300)^1.5 * exp[-Eg / (2 kB) * (1/T - 1/300)].
This is suitable for an educational model, not calibrated device design.
"""

from dataclasses import asdict, dataclass
import math

from .constants import K_B, Q


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    symbol: str
    relative_permittivity: float
    band_gap_ev_300k: float
    intrinsic_concentration_cm3_300k: float
    assumptions: str

    @property
    def permittivity(self) -> float:
        from .constants import EPSILON_0

        return self.relative_permittivity * EPSILON_0

    def public_dict(self) -> dict:
        data = asdict(self)
        data["permittivity_f_per_m"] = self.permittivity
        return data


MATERIALS = {
    "si": Material(
        "si", "Silicon", "Si", 11.7, 1.12, 1.0e10,
        "Non-degenerate, abrupt, one-dimensional junction; 300 K ni is approximate.",
    ),
    "ge": Material(
        "ge", "Germanium", "Ge", 16.0, 0.66, 2.4e13,
        "Non-degenerate, abrupt, one-dimensional junction; 300 K ni is approximate.",
    ),
    "gaas": Material(
        "gaas", "Gallium arsenide", "GaAs", 12.9, 1.42, 2.1e6,
        "Non-degenerate, abrupt, one-dimensional junction; 300 K ni is approximate.",
    ),
}


def get_material(key: str) -> Material:
    try:
        return MATERIALS[key.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown material '{key}'. Choose Si, Ge, or GaAs.") from exc


def intrinsic_concentration_cm3(material: Material, temperature_k: float) -> float:
    eg_j = material.band_gap_ev_300k * Q
    return material.intrinsic_concentration_cm3_300k * (temperature_k / 300.0) ** 1.5 * math.exp(
        -eg_j / (2.0 * K_B) * (1.0 / temperature_k - 1.0 / 300.0)
    )
