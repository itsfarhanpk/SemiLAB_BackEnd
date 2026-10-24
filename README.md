# SemiLab Backend

Flask and NumPy scientific API for the SemiLab interactive semiconductor physics laboratory.

## Run locally

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:FRONTEND_ORIGINS="http://localhost:5173"
python app.py
```

The API runs at `http://localhost:5000`.

## Test

```powershell
pytest tests
```

The regression suite checks field direction and continuity, zero-field depletion
edges, charge neutrality, analytical potential, `E = -dφ/dx`, Poisson
consistency, constant band gap, asymmetric sampling, all materials and
temperature limits, invalid model states, API validation, and normalized diode
current. Tests use a small local trapezoid calculation rather than
`np.trapezoid`, so they remain compatible with the declared NumPy 1.26+
dependency range.

## API

- `GET /api/health` — backend status.
- `GET /api/materials` — supported materials and documented model assumptions.
- `POST /api/simulate` — accepts `material`, `Na`, `Nd`, `temperature`, and `appliedVoltage`.

## Model scope

The API is the source of truth for the scientific outputs. It implements an ideal abrupt one-dimensional p–n junction using SI units internally, material-specific parameters for Si, Ge, and GaAs, a depletion approximation, electric-field and charge-density profiles, analytical electrostatic potential, simplified band edges, and normalized ideal-diode current. The coordinate convention places the P side at negative x and the N side at positive x; the depletion field is negative and the potential rises from 0 on P to `Vbi - Va` on N.

The model includes validity warnings near the built-in potential. The diode curve is normalized to `I/I0`; it does not claim absolute amperes because device geometry and a saturation-current model are outside version 1.

Set `FRONTEND_ORIGINS` to the deployed frontend origin when hosting the API. The default allows both local development and `https://semi-lab-front-end.vercel.app`.
