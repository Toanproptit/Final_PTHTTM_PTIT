"""Kiểm tra server HTTP thật và lưu chứng cứ; không thay thế browser visual QA."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diabetes_demo.schema import DEMO_PATIENT

BASE = "http://127.0.0.1:8000"

def fetch(route: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE + route, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.headers.get_content_type(), response.read()

checks = {}
for route in ["/", "/assets/app.js", "/assets/style.css", "/docs", "/legacy-demo"]:
    status, content_type, body = fetch(route)
    checks[route] = {"passed": status == 200 and len(body) > 100, "status": status, "content_type": content_type, "bytes": len(body)}
_, _, raw = fetch("/api/health")
health = json.loads(raw)
checks["health_kaggle"] = {"passed": health["prediction_ready"] and health["dataset_ref"] == "ziya07/diabetes-clinical-dataset100k-rows" and health["input_fields"] == 8}
_, _, raw = fetch("/api/schema")
schema = json.loads(raw)
valid_constraints = True
for field in schema["features"]:
    value = schema["demo_patient"][field["name"]]
    if field["type"] == "number":
        valid_constraints = valid_constraints and field["min"] <= value <= field["max"]
        if field["step"] != "any":
            steps = (value - field["min"]) / field["step"]
            valid_constraints = valid_constraints and abs(steps - round(steps)) < 1e-6
checks["live_demo_form_constraints"] = {"passed": valid_constraints}
_, _, raw = fetch("/api/predict", DEMO_PATIENT)
prediction = json.loads(raw)
checks["live_prediction"] = {"passed": prediction["status"] == "completed" and len(prediction["trace"]) == 4, "score": prediction["prediction"]["positive_class_score"]}
_, _, raw = fetch("/api/ask", {"question": "HbA1c phản ánh điều gì?"})
checks["live_qa"] = {"passed": bool(json.loads(raw)["evidence"])}
_, _, raw = fetch("/api/experiments")
experiments = json.loads(raw)
checks["live_experiments"] = {"passed": experiments["ml"]["metadata"]["dataset_ref"] == health["dataset_ref"] and bool(experiments["rl_manifest"]) and bool(experiments["retrieval"])}
for group, name in [("ml", "training_loss.png"), ("ml", "roc_curves.png"), ("ml", "confusion_matrices.png"), ("rl", "episode_rewards.png"), ("rl", "learning_rate_comparison.png"), ("rl", "exploration_comparison.png")]:
    status, mime, body = fetch(f"/api/artifact/{group}/{name}")
    checks[f"image_{group}_{name}"] = {"passed": status == 200 and mime == "image/png" and body.startswith(b"\x89PNG")}
result = {"generated_utc": datetime.now(timezone.utc).isoformat(), "base_url": BASE, "checks": checks,
          "all_passed": all(check["passed"] for check in checks.values()), "browser_rendering_verified": False}
(ROOT / "artifacts" / "http_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["all_passed"]:
    raise SystemExit(1)
