"""FastAPI phục vụ UI và agent. Chạy bằng uvicorn trên 127.0.0.1."""
from contextlib import asynccontextmanager
import csv
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .orchestrator import OrchestratorAgent
from .llm_agent import LLMToolAgent
from .paths import ARTIFACTS, ML, RL, ROOT, WEB
from .schema import AgentChatInput, CATEGORY_VALUES, DEMO_PATIENT, FEATURES, LABELS, LIMITATION, NUMERIC_FEATURES, PatientInput, QuestionInput
from .storage import DemoStore

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))

def create_app(store_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.orchestrator = OrchestratorAgent(DemoStore(store_path))
        application.state.llm_agent = LLMToolAgent(application.state.orchestrator)
        yield

    app = FastAPI(title="Diabetes Multi-Agent Research Demo", version="2.0.0", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=WEB), name="assets")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB / "index.html")

    @app.get("/legacy-demo", include_in_schema=False)
    def legacy_demo():
        return FileResponse(ROOT / "index.html")

    @app.get("/api/health")
    def health(request: Request):
        agent = request.app.state.orchestrator
        graph_health = agent.knowledge.graph.health() if agent.knowledge.graph else {
            "configured": False, "ready": False, "reason": "missing_credentials"
        }
        return {"status": "ok", "prediction_ready": agent.prediction.ready,
                "dataset_ref": agent.prediction.metadata.get("dataset_ref") if agent.prediction.metadata else None,
                "input_fields": len(FEATURES),
                "knowledge_chunks": len(agent.knowledge.chunks),
                "qa_mode": "hybrid semantic + GraphRAG" if graph_health.get("ready") else "TF-IDF fallback",
                "database": {"backend": agent.store.backend, "ready": agent.store.ping(), "url": agent.store.url},
                "neo4j": graph_health,
                "orchestrator": {"mode": "dynamic-routing", "framework": "LangGraph",
                                 "max_steps": agent.max_steps, "graph": agent.workflow.description()},
                "rl_ready": (RL / "manifest.json").exists(), "limitation": LIMITATION}

    @app.get("/api/knowledge/graph")
    def knowledge_graph(request: Request, limit: int = 80):
        graph = request.app.state.orchestrator.knowledge.graph
        if graph is None:
            raise HTTPException(status_code=503, detail="Neo4j chưa được cấu hình.")
        try:
            return graph.neighborhood(limit)
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Neo4j chưa sẵn sàng: {type(error).__name__}") from error

    @app.get("/api/agent/status")
    def agent_status(request: Request):
        return request.app.state.llm_agent.status()

    @app.post("/api/agent/chat")
    def agent_chat(payload: AgentChatInput, request: Request):
        agent = request.app.state.llm_agent
        if not agent.configured:
            raise HTTPException(status_code=503, detail="Chưa cấu hình OPENROUTER_API_KEY.")
        try:
            return agent.chat(payload.message, [turn.model_dump() for turn in payload.history])
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/schema")
    def schema():
        ranges = {"age": (0, 120, 0.01), "bmi": (0.01, 150, 0.01), "hba1c_level": (0.01, 25, "any"), "blood_glucose_level": (0.01, 1000, "any")}
        fields = []
        for name in FEATURES:
            field = {"name": name, "label": LABELS[name], "type": "number" if name in NUMERIC_FEATURES else "category" if name in CATEGORY_VALUES else "boolean"}
            if name in CATEGORY_VALUES:
                field["options"] = CATEGORY_VALUES[name]
            if name in ranges:
                field.update(dict(zip(["min", "max", "step"], ranges[name])))
            fields.append(field)
        return {"features": fields, "demo_patient": DEMO_PATIENT,
                "limitation": LIMITATION}

    @app.post("/api/predict")
    def predict(patient: PatientInput, request: Request):
        try:
            return request.app.state.orchestrator.predict(patient)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/ask")
    def ask(question: QuestionInput, request: Request):
        if len(question.question.strip()) < 3:
            raise HTTPException(422, "Vui lòng nhập câu hỏi có ít nhất 3 ký tự.")
        return request.app.state.orchestrator.ask(question.question)

    @app.get("/api/history")
    def history(request: Request):
        return request.app.state.orchestrator.store.recent()

    @app.delete("/api/history")
    def clear_history(request: Request):
        request.app.state.orchestrator.store.clear()
        return {"status": "cleared"}

    @app.get("/api/experiments")
    def experiments():
        return {"ml": read_json(ML / "evaluation.json"),
                "rl_manifest": read_json(RL / "manifest.json"),
                "rl_baseline": read_csv(RL / "baseline_evaluation.csv"),
                "rl_learning_rate": read_csv(RL / "learning_rate_comparison.csv"),
                "rl_exploration": read_csv(RL / "exploration_comparison.csv"),
                "rl_multi_seed": read_csv(RL / "multi_seed_summary.csv"),
                "rl_q_table": (RL / "q_tables.txt").read_text(encoding="utf-8") if (RL / "q_tables.txt").is_file() else None,
                "retrieval": read_json(ARTIFACTS / "retrieval" / "evaluation.json"),
                "workflow": read_json(ARTIFACTS / "workflow_evaluation.json")}

    @app.get("/api/artifact/{category}/{filename}")
    def artifact(category: str, filename: str):
        if category not in {"ml", "rl", "retrieval"} or Path(filename).name != filename or Path(filename).suffix not in {".csv", ".png", ".json", ".txt"}:
            raise HTTPException(404, "Không có artifact này")
        path = ARTIFACTS / category / filename
        if not path.is_file():
            raise HTTPException(404, "Artifact chưa được tạo; xem README để chạy thực nghiệm.")
        return FileResponse(path)

    return app

app = create_app()
