"""LangChain StructuredTool registry backed by the project's local services."""
from __future__ import annotations

import json

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .paths import ARTIFACTS, ML
from .schema import NUMERIC_FEATURES, PatientInput


class KnowledgeToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=1000)


class EmptyToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_langchain_tools(orchestrator) -> list[BaseTool]:
    def search_health_knowledge(question: str) -> dict:
        return orchestrator.knowledge.run(question)

    def run_research_prediction(**profile) -> dict:
        normalized = dict(profile)
        for field in NUMERIC_FEATURES:
            if isinstance(normalized.get(field), (int, float)) and not isinstance(normalized[field], bool):
                normalized[field] = float(normalized[field])
        patient = PatientInput.model_validate(normalized)
        result = orchestrator.predict(patient)
        return {
            "status": result["status"],
            "prediction": result["prediction"],
            "safety": result["safety"],
            "rule_warnings": result["rules"]["warnings"],
            "limitation": result["explanation"]["limitation"],
            "orchestration": result.get("orchestration"),
        }

    def get_experiment_summary() -> dict:
        evaluation_path = ML / "evaluation.json"
        if not evaluation_path.is_file():
            return {"status": "unavailable", "message": "No saved ML evaluation artifact was found."}
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        result = {
            "status": "available",
            "metadata": evaluation.get("metadata", {}),
            "metrics": evaluation.get("metrics", evaluation.get("results", {})),
        }
        workflow_path = ARTIFACTS / "workflow_evaluation.json"
        if workflow_path.is_file():
            result["workflow_evaluation"] = json.loads(workflow_path.read_text(encoding="utf-8"))
        return result

    return [
        StructuredTool.from_function(
            func=search_health_knowledge,
            name="search_health_knowledge",
            description=(
                "Search the curated diabetes knowledge base and GraphRAG. Use for general questions about "
                "symptoms, tests, risk, prevention or sourced context. Returns evidence and source URLs."
            ),
            args_schema=KnowledgeToolInput,
        ),
        StructuredTool.from_function(
            func=run_research_prediction,
            name="run_research_prediction",
            description=(
                "Run the research-only dataset classifier on a complete hypothetical profile. "
                "Never invent missing fields and never present the output as a diagnosis."
            ),
            args_schema=PatientInput,
        ),
        StructuredTool.from_function(
            func=get_experiment_summary,
            name="get_experiment_summary",
            description="Read saved model and workflow evaluation metrics from the project artifacts.",
            args_schema=EmptyToolInput,
        ),
    ]
