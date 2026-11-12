"""Transactional run storage backed by SQLAlchemy (MySQL in Compose, SQLite in tests)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from .database import AgentExecution, AgentMessage, AgentStep, Base, PredictionRecord, RetrievalLog, ToolCall, create_database_engine, session_factory


class DemoStore:
    def __init__(self, path_or_url: Path | str | None = None, create_schema: bool | None = None):
        self.engine = create_database_engine(path_or_url)
        self.sessions = session_factory(self.engine)
        self.url = self.engine.url.render_as_string(hide_password=True)
        should_create = create_schema if create_schema is not None else self.engine.url.get_backend_name() == "sqlite"
        if should_create:
            Base.metadata.create_all(self.engine)

    @property
    def backend(self) -> str:
        return self.engine.url.get_backend_name()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def has_execution(self, run_id: str) -> bool:
        with self.sessions() as session:
            return session.get(AgentExecution, run_id) is not None

    def get_execution(self, run_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(AgentExecution, run_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "kind": row.kind,
                "final_payload": row.final_payload or {},
            }

    def save(self, run_id: str, kind: str, payload: dict, intent: str | None = None) -> None:
        now = datetime.now(timezone.utc)
        trace = payload.get("trace", [])
        with self.sessions.begin() as session:
            session.merge(AgentExecution(
                id=run_id, kind=kind, intent=intent or kind,
                status=payload.get("status", "completed"),
                input_payload=payload.get("input", {"question": payload.get("question")} if payload.get("question") else {}),
                final_payload=payload, started_at=now, finished_at=now,
            ))
            session.execute(delete(AgentStep).where(AgentStep.execution_id == run_id))
            for index, item in enumerate(trace, start=1):
                session.add(AgentStep(
                    execution_id=run_id, step_order=int(item.get("order", index)),
                    agent_name=item.get("agent", "UnknownAgent"), action=item.get("action", "run"),
                    status=item.get("status", "ok"), routing_reason=item.get("routing_reason"),
                    input_summary=item.get("input", {}), output_summary=item.get("output", {}),
                    error_json=item.get("error"), started_at=now, finished_at=now,
                    duration_ms=float(item.get("duration_ms", 0.0)),
                ))

    def save_tool_calls(self, run_id: str, calls: list[dict[str, Any]]) -> None:
        with self.sessions.begin() as session:
            for call in calls:
                session.add(ToolCall(
                    execution_id=run_id, tool_name=call.get("name", "unknown"),
                    status=call.get("status", "unknown"), arguments_json=call.get("arguments", {}),
                    result_summary=call.get("summary", {}), duration_ms=call.get("duration_ms"),
                ))

    def save_messages(self, run_id: str, messages: list[dict[str, str]]) -> None:
        with self.sessions.begin() as session:
            session.execute(delete(AgentMessage).where(AgentMessage.execution_id == run_id))
            for message in messages:
                role, content = message.get("role"), message.get("content")
                if role in {"user", "assistant", "system"} and isinstance(content, str) and content:
                    session.add(AgentMessage(execution_id=run_id, role=role, content=content[:8000]))

    def save_retrieval(self, run_id: str, query: str, result: dict, duration_ms: float = 0.0) -> None:
        citations = [
            {key: item.get(key) for key in ("chunk_id", "title", "publisher", "url", "accessed", "score")}
            for item in result.get("evidence", [])
        ]
        with self.sessions.begin() as session:
            session.add(RetrievalLog(
                execution_id=run_id,
                mode=result.get("mode") or "unknown",
                query_text=query,
                result_count=len(result.get("evidence", [])),
                citations_json=citations,
                duration_ms=float(duration_ms),
            ))

    def save_prediction(self, run_id: str, inputs: dict, prediction: dict) -> None:
        with self.sessions.begin() as session:
            session.merge(PredictionRecord(
                id=run_id,
                execution_id=run_id,
                model_name=prediction["model"],
                model_version=prediction["model_version"],
                input_payload=inputs,
                positive_class_score=float(prediction["positive_class_score"]),
                predicted_label=prediction["predicted_dataset_label"],
                threshold=float(prediction["threshold"]),
            ))

    def recent(self, limit: int = 20) -> list[dict]:
        with self.sessions() as session:
            rows = session.scalars(select(AgentExecution).order_by(AgentExecution.started_at.desc()).limit(min(limit, 100))).all()
            return [{"id": row.id, "created_at": row.started_at.isoformat(), "kind": row.kind, "payload": row.final_payload or {}} for row in rows]

    def clear(self) -> None:
        with self.sessions.begin() as session:
            session.execute(delete(AgentExecution))
