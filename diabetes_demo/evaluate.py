"""Đánh giá retrieval và workflow trên câu hỏi/hồ sơ giả định, lưu kết quả thực."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .knowledge import KnowledgeAgent
from .orchestrator import OrchestratorAgent
from .paths import ARTIFACTS, KB
from .schema import DEMO_PATIENT, PatientInput
from .storage import DemoStore

def evaluate_retrieval() -> dict:
    agent = KnowledgeAgent()
    questions = json.loads((KB / "evaluation_questions.json").read_text(encoding="utf-8"))
    rows = []
    for case in questions:
        output = agent.run(case["question"])
        ranked = list(dict.fromkeys(item["document_id"] for item in output["evidence"]))
        expected = case["expected_document"]
        rows.append({"question": case["question"], "expected_document": expected,
                     "status": output["status"], "retrieved_documents": "|".join(ranked),
                     "top1_hit": int(bool(ranked) and ranked[0] == expected) if expected else None,
                     "top3_hit": int(expected in ranked[:3]) if expected else None,
                     "correct_rejection": int(not ranked) if expected is None else None})
    supported = [row for row in rows if row["expected_document"]]
    unsupported = [row for row in rows if not row["expected_document"]]
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "questions": len(rows),
               "supported_questions": len(supported), "unsupported_questions": len(unsupported),
               "hit_at_1": sum(row["top1_hit"] for row in supported) / len(supported),
               "hit_at_3": sum(row["top3_hit"] for row in supported) / len(supported),
               "rejection_accuracy": sum(row["correct_rejection"] for row in unsupported) / len(unsupported),
               "minimum_score": agent.minimum_score, "index_sha256": agent.index_hash,
               "mode": "retrieval-based QA", "evaluation_unit": "document id, deduplicated top-3 chunks",
               "limitation": "Bộ câu hỏi nhỏ do tác giả soạn, cùng ngôn ngữ/nội dung KB; không phải benchmark độc lập."}
    destination = ARTIFACTS / "retrieval"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "question_results.csv", index=False)
    (destination / "evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Hit@1", "Hit@3", "Rejection"], [summary["hit_at_1"], summary["hit_at_3"], summary["rejection_accuracy"]])
    axis.set(ylim=(0, 1.05), ylabel="Proportion", title="Small author-written retrieval evaluation")
    fig.tight_layout(); fig.savefig(destination / "retrieval_evaluation.png", dpi=160); plt.close(fig)
    return summary

def evaluate_workflow() -> dict:
    with TemporaryDirectory() as directory:
        agent = OrchestratorAgent(DemoStore(Path(directory) / "test.sqlite3"))
        prediction = agent.predict(PatientInput(**DEMO_PATIENT))
        question = agent.ask("HbA1c phản ánh điều gì?")
        unsupported = agent.predict(PatientInput(**{**DEMO_PATIENT, "age": 100.0}))
        checks = {
            "prediction_completed": prediction["status"] == "completed",
            "real_prediction_score": 0 <= prediction["prediction"]["positive_class_score"] <= 1,
            "agent_order": [item["agent"] for item in prediction["trace"]] == ["RuleBasedAgent", "PredictionAgent", "KnowledgeAgent", "ExplanationAgent"],
            "source_backed_answer": bool(question["evidence"]),
            "out_of_domain_skips_prediction": unsupported["prediction"] is None,
            "sqlite_saved_three_runs": len(agent.store.recent()) == 3,
        }
        result = {"generated_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
                  "all_passed": all(checks.values()), "scenario": "synthetic input for software flow, not medical evaluation",
                  "prediction_example": prediction, "question_example": question}
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "workflow_evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{"check": name, "passed": passed} for name, passed in checks.items()]).to_csv(ARTIFACTS / "workflow_checks.csv", index=False)
    return {"checks": checks, "all_passed": result["all_passed"]}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"retrieval": evaluate_retrieval(), "workflow": evaluate_workflow()}, indent=2, ensure_ascii=False))
