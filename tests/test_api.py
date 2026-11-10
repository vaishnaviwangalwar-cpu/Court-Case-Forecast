import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained, monkeypatch):
    root, _ = trained
    monkeypatch.setenv("CCF_ARTIFACT_DIR", str(root / "artifacts"))
    from src.api.main import app
    return TestClient(app)


def case(client):
    m = client.get("/meta").json()
    return {"court_id": m["courts"][0], "case_type": m["case_types"][0], "act": m["acts"][0], "filing_date": "2014-05-10"}


def test_predict(client):
    r = client.post("/predict", json=case(client))
    assert r.status_code == 200
    j = r.json()
    s = j["survival"]
    assert all(a >= b for a, b in zip(s, s[1:])) and s[0] > 0.95            # monotone survival curve
    assert 0 <= j["interval"]["lower"] <= j["p_delay_24m"] <= j["interval"]["upper"] <= 1
    assert j["model_version"] and "disclaimer" in j


def test_curve_and_interval(client):
    c = case(client)
    assert "survival" in client.post("/curve", json=c).json()
    assert "interval" in client.post("/interval", json=c).json()


def test_validation_error(client):
    assert client.post("/predict", json={"court_id": "x"}).status_code == 422


def test_unknown_court_flagged(client):
    c = case(client) | {"court_id": "00_00_00"}
    assert client.post("/predict", json=c).json()["known_court"] is False


def test_simulate_capacity_reduces_delay(client):
    r = client.post("/simulate", json={"arrivals_per_month": 60, "slots": 20, "extra_capacity_pct": 50}).json()
    assert r["scenario"]["share_over_24m"] <= r["baseline"]["share_over_24m"]
