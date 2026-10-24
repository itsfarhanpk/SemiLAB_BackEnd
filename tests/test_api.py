from app import create_app


def test_health_and_simulation_api():
    client = create_app().test_client()
    assert client.get("/api/health").status_code == 200
    response = client.post("/api/simulate", json={
        "material": "si", "Na": 1e16, "Nd": 1e16, "temperature": 300, "appliedVoltage": 0,
    })
    assert response.status_code == 200
    assert response.get_json()["valid"]


def test_api_rejects_invalid_payload():
    client = create_app().test_client()
    assert client.post("/api/simulate", json={"material": "si"}).status_code == 400
    assert client.post("/api/simulate", json={
        "material": "unknown", "Na": 1e16, "Nd": 1e16, "temperature": 300, "appliedVoltage": 0,
    }).status_code == 400
    assert client.post("/api/simulate", json={
        "material": "si", "Na": 1e12, "Nd": 1e16, "temperature": 300, "appliedVoltage": 0,
    }).status_code == 400


def test_api_returns_model_warning_for_invalid_forward_state():
    response = create_app().test_client().post("/api/simulate", json={
        "material": "ge", "Na": 1e16, "Nd": 1e16, "temperature": 300, "appliedVoltage": 0.6,
    })
    assert response.status_code == 200
    assert response.get_json()["valid"] is False
    assert response.get_json()["warnings"]
