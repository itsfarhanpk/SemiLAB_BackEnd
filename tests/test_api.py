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
    response = create_app().test_client().post("/api/simulate", json={"material": "si"})
    assert response.status_code == 400
