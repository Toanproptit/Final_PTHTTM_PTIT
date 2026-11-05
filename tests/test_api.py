from fastapi.testclient import TestClient
import pytest
from diabetes_demo.api import create_app
from diabetes_demo.schema import DEMO_PATIENT

@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "demo.sqlite3")) as session:
        yield session

def test_prediction_trace_and_sqlite_roundtrip(client):
    from sqlalchemy import func, select
    from diabetes_demo.database import PredictionRecord
    response = client.post("/api/predict", json=DEMO_PATIENT)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["orchestration"]["framework"] == "LangGraph"
    assert 0 <= result["prediction"]["positive_class_score"] <= 1
    assert [item["agent"] for item in result["trace"]] == [
        "ValidationAgent", "SafetyAgent", "PredictionAgent", "KnowledgeAgent", "ExplanationAgent"
    ]
    assert result["trace"][2]["input"] == DEMO_PATIENT
    assert result["safety"]["level"] == "follow_up"
    assert result["routing_decisions"]
    assert result["explanation"]["general_information"]
    stored = client.get("/api/history").json()
    assert stored[0]["payload"]["run_id"] == result["run_id"]
    with client.app.state.orchestrator.store.sessions() as session:
        assert session.scalar(select(func.count()).select_from(PredictionRecord)) == 1

def test_invalid_input_does_not_run_or_save_agents(client):
    response = client.post("/api/predict", json={"age": 45})
    assert response.status_code == 422
    assert client.get("/api/history").json() == []

def test_outside_age_skips_prediction(client):
    response = client.post("/api/predict", json={**DEMO_PATIENT, "age": 100.0})
    result = response.json()
    assert result["prediction"] is None
    prediction_step = next(item for item in result["trace"] if item["agent"] == "PredictionAgent")
    assert prediction_step["status"] == "skipped"

def test_dynamic_router_skips_knowledge_for_low_signal_negative_profile(client):
    profile = {**DEMO_PATIENT, "hba1c_level": 5.2, "blood_glucose_level": 90.0}
    result = client.post("/api/predict", json=profile).json()
    knowledge_step = next(item for item in result["trace"] if item["agent"] == "KnowledgeAgent")
    if result["prediction"]["predicted_dataset_label"] == "diabetes = 0":
        assert knowledge_step["status"] == "skipped"
        assert result["knowledge"]["status"] == "not_requested"

def test_safety_agent_flags_conditional_urgent_review(client):
    result = client.post("/api/predict", json={**DEMO_PATIENT, "blood_glucose_level": 300.0}).json()
    assert result["safety"]["level"] == "urgent_review"
    assert "S_GLUCOSE_GE_300" in result["safety"]["matched_rules"]
    assert any("duy trì" in message for message in result["safety"]["messages"])

def test_question_history_clear_and_no_model_artifact_exposure(client):
    response = client.post("/api/ask", json={"question": "HbA1c là gì?"})
    assert response.json()["evidence"]
    assert client.get("/api/history").json()
    assert client.delete("/api/history").status_code == 200
    assert not client.get("/api/history").json()
    assert client.get("/api/artifact/ml/mlp.pt").status_code == 404

def test_whitespace_question_is_rejected(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422

def test_missing_model_returns_actionable_503(tmp_path, monkeypatch):
    import diabetes_demo.agents as agents
    monkeypatch.setattr(agents, "ML", tmp_path / "missing")
    with TestClient(create_app(tmp_path / "separate.sqlite3")) as session:
        response = session.post("/api/predict", json=DEMO_PATIENT)
        assert response.status_code == 503
        assert "diabetes_demo.train" in response.json()["detail"]

def test_ui_and_experiments_are_served(client):
    import re
    html = client.get("/").text
    assert "Diabetes Agent Lab" in html
    bundle = re.search(r'src="(/assets/[^"]+\.js)"', html).group(1)
    assert client.get(bundle).status_code == 200
    health = client.get("/api/health").json()
    assert health["prediction_ready"]
    assert health["dataset_ref"] == "ziya07/diabetes-clinical-dataset100k-rows"
    assert health["input_fields"] == 8
    assert health["database"]["backend"] == "sqlite"
    assert health["orchestrator"]["mode"] == "dynamic-routing"
    assert health["orchestrator"]["framework"] == "LangGraph"
    assert "validation" in health["orchestrator"]["graph"]["nodes"]
    assert client.get("/api/experiments").json()["ml"]["metrics"]

def test_demo_values_satisfy_browser_numeric_constraints(client):
    schema = client.get("/api/schema").json()
    for field in schema["features"]:
        value = schema["demo_patient"][field["name"]]
        if field["type"] == "number":
            assert field["min"] <= value <= field["max"]
            if field["step"] != "any":
                steps = (value - field["min"]) / field["step"]
                assert steps == pytest.approx(round(steps))
