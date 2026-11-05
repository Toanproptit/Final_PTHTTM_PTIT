"""Typed shared state passed between the specialised agents."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    request_id: str
    kind: Literal["prediction", "question"]
    intent: Literal["profile_classification", "knowledge_query"]
    user_input: dict[str, Any]
    validation_result: dict[str, Any] | None = None
    safety_status: dict[str, Any] | None = None
    prediction_result: dict[str, Any] | None = None
    knowledge_result: dict[str, Any] | None = None
    explanation_result: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    agent_outputs: dict[str, Any] = Field(default_factory=dict)
    routing_decisions: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 12

    def route(self, next_agent: str, reason: str, selected: bool = True) -> None:
        self.routing_decisions.append({
            "order": len(self.routing_decisions) + 1,
            "next_agent": next_agent,
            "reason": reason,
            "selected": selected,
        })
