"""Relational persistence models shared by MySQL production and SQLite tests/migration."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import get_settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_ref: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    license: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="dataset")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    records: Mapped[list["DatasetRecord"]] = relationship(back_populates="version")


class DatasetRecord(Base):
    __tablename__ = "dataset_records"
    __table_args__ = (
        UniqueConstraint("dataset_version_id", "source_row", name="uq_dataset_record_source"),
        Index("ix_dataset_records_version_target", "dataset_version_id", "target"),
        Index("ix_dataset_records_feature_group", "feature_group"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    source_row: Mapped[int] = mapped_column(Integer, nullable=False)
    age: Mapped[float] = mapped_column(Float, nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    hypertension: Mapped[bool] = mapped_column(Boolean, nullable=False)
    heart_disease: Mapped[bool] = mapped_column(Boolean, nullable=False)
    smoking_history: Mapped[str] = mapped_column(String(32), nullable=False)
    bmi: Mapped[float] = mapped_column(Float, nullable=False)
    hba1c_level: Mapped[float] = mapped_column(Float, nullable=False)
    blood_glucose_level: Mapped[float] = mapped_column(Float, nullable=False)
    target: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_group: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[DatasetVersion] = relationship(back_populates="records")


class DataImportJob(Base):
    __tablename__ = "data_import_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id", ondelete="SET NULL"))
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    rows_read: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    training_run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    split_name: Mapped[str] = mapped_column(String(32), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(80), nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float)
    metric_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("model_version_id", "split_name", "metric_name", name="uq_model_metric"),)


class AgentExecution(Base):
    __tablename__ = "agent_executions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    intent: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    final_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    steps: Mapped[list["AgentStep"]] = relationship(back_populates="execution", cascade="all, delete-orphan")


class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    execution_id: Mapped[str] = mapped_column(
        ForeignKey("agent_executions.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    positive_class_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)


class AgentStep(Base):
    __tablename__ = "agent_steps"
    __table_args__ = (UniqueConstraint("execution_id", "step_order", name="uq_execution_step_order"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(ForeignKey("agent_executions.id", ondelete="CASCADE"), nullable=False, index=True)
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(120), nullable=False)
    action: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    routing_reason: Mapped[str | None] = mapped_column(Text)
    input_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    execution: Mapped[AgentExecution] = relationship(back_populates="steps")


class AgentMessage(Base):
    __tablename__ = "agent_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(ForeignKey("agent_executions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ToolCall(Base):
    __tablename__ = "tool_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(ForeignKey("agent_executions.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_step_id: Mapped[int | None] = mapped_column(ForeignKey("agent_steps.id", ondelete="SET NULL"))
    tool_name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    arguments_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    result_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str | None] = mapped_column(ForeignKey("agent_executions.id", ondelete="SET NULL"), index=True)
    mode: Mapped[str] = mapped_column(String(64), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


def normalize_database_url(value: str | Path | None) -> str:
    if isinstance(value, Path):
        return f"sqlite:///{value.as_posix()}"
    return str(value) if value else get_settings().database_url


def create_database_engine(value: str | Path | None = None) -> Engine:
    url = normalize_database_url(value)
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update({"pool_size": 5, "max_overflow": 10, "pool_recycle": 1800})
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def session_factory(engine: Engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
