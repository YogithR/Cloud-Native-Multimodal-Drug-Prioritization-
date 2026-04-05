from fastapi.testclient import TestClient

from src.api.main import app


def _candidate(candidate_id: str, smiles: str) -> dict[str, str]:
    return {"candidate_id": candidate_id, "smiles": smiles}


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "http" in body.lower() or "process" in body.lower() or "python" in body.lower()


def test_predict_endpoint_returns_probabilities() -> None:
    client = TestClient(app)
    response = client.post("/predict", json=_candidate("c1", "CCO"))
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_id"] == "c1"
    assert 0.0 <= data["predicted_prob_positive"] <= 1.0
    assert 0.0 <= data["prediction_confidence"] <= 1.0
    assert 0.0 <= data["risk_probability"] <= 1.0


def test_rank_endpoint_returns_priority() -> None:
    client = TestClient(app)
    response = client.post("/rank", json=_candidate("c1", "CCO"))
    assert response.status_code == 200
    data = response.json()
    assert data["rank"] == 1
    assert "priority_score" in data


def test_batch_rank_orders_candidates() -> None:
    client = TestClient(app)
    payload = {"candidates": [_candidate("c1", "CCO"), _candidate("c2", "CCC")]}
    response = client.post("/batch-rank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["rank"] == 1
    assert data[1]["rank"] == 2
