"""LangGraph state machine for the structured patient-profile workflow."""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent_state import AgentState
from .schema import PatientInput


class PredictionWorkflowState(TypedDict, total=False):
    audit_state: AgentState
    patient: PatientInput
    inputs: dict
    validation: dict
    safety: dict
    prediction: dict | None
    knowledge: dict
    knowledge_query: str | None
    explanation: dict


class LangGraphPredictionWorkflow:
    """Finite graph with explicit conditional routes and auditable node calls."""

    KNOWLEDGE_QUERY = "Ý nghĩa chung của HbA1c, glucose máu và khi nào cần hỗ trợ y tế"

    def __init__(self, owner):
        self.owner = owner
        builder = StateGraph(PredictionWorkflowState)
        builder.add_node("validation", self._validation)
        builder.add_node("safety", self._safety)
        builder.add_node("prediction", self._prediction)
        builder.add_node("skip_prediction", self._skip_prediction)
        builder.add_node("knowledge", self._knowledge)
        builder.add_node("skip_knowledge", self._skip_knowledge)
        builder.add_node("explanation", self._explanation)

        builder.add_edge(START, "validation")
        builder.add_edge("validation", "safety")
        builder.add_conditional_edges(
            "safety",
            self._route_prediction,
            {"prediction": "prediction", "skip_prediction": "skip_prediction"},
        )
        builder.add_conditional_edges(
            "prediction",
            self._route_knowledge,
            {"knowledge": "knowledge", "skip_knowledge": "skip_knowledge"},
        )
        builder.add_conditional_edges(
            "skip_prediction",
            self._route_knowledge,
            {"knowledge": "knowledge", "skip_knowledge": "skip_knowledge"},
        )
        builder.add_edge("knowledge", "explanation")
        builder.add_edge("skip_knowledge", "explanation")
        builder.add_edge("explanation", END)
        self.graph = builder.compile(name="diabetes-profile-multi-agent")

    def invoke(self, patient: PatientInput, audit_state: AgentState, inputs: dict) -> PredictionWorkflowState:
        return self.graph.invoke(
            {"patient": patient, "audit_state": audit_state, "inputs": inputs},
            config={"recursion_limit": audit_state.max_steps + 8},
        )

    def description(self) -> dict:
        graph = self.graph.get_graph()
        return {
            "framework": "LangGraph",
            "name": "diabetes-profile-multi-agent",
            "nodes": [name for name in graph.nodes if name not in {"__start__", "__end__"}],
            "conditional_routes": ["safety->prediction", "prediction->knowledge"],
        }

    def _validation(self, graph_state: PredictionWorkflowState) -> dict:
        audit, patient, inputs = graph_state["audit_state"], graph_state["patient"], graph_state["inputs"]
        output = self.owner._call(
            audit, "ValidationAgent", "validate_profile", inputs,
            "LangGraph entered the validation node after API schema validation.",
            lambda: self.owner.validation.run(patient),
        )
        audit.validation_result = output
        return {"validation": output, "audit_state": audit}

    def _safety(self, graph_state: PredictionWorkflowState) -> dict:
        audit, patient, inputs = graph_state["audit_state"], graph_state["patient"], graph_state["inputs"]
        output = self.owner._call(
            audit, "SafetyAgent", "evaluate_safety", inputs,
            "LangGraph always runs the independent safety node before model inference.",
            lambda: self.owner.safety.run(patient),
        )
        audit.safety_status = output
        audit.citations.extend(output["citations"])
        return {"safety": output, "audit_state": audit}

    @staticmethod
    def _route_prediction(graph_state: PredictionWorkflowState) -> Literal["prediction", "skip_prediction"]:
        return "prediction" if graph_state["validation"]["within_training_domain"] else "skip_prediction"

    def _prediction(self, graph_state: PredictionWorkflowState) -> dict:
        audit, patient, inputs = graph_state["audit_state"], graph_state["patient"], graph_state["inputs"]
        output = self.owner._call(
            audit, "PredictionAgent", "classify_dataset_label", inputs,
            "LangGraph selected inference because all numeric fields are inside the training domain.",
            lambda: self.owner.prediction.run(patient),
        )
        audit.prediction_result = output
        return {"prediction": output, "audit_state": audit}

    def _skip_prediction(self, graph_state: PredictionWorkflowState) -> dict:
        audit, inputs, validation = graph_state["audit_state"], graph_state["inputs"], graph_state["validation"]
        self.owner._skip(
            audit, "PredictionAgent", "LangGraph routed around inference because the profile is outside the training domain.",
            {name: inputs[name] for name in validation["outside_features"]},
        )
        return {"prediction": None, "audit_state": audit}

    @staticmethod
    def _route_knowledge(graph_state: PredictionWorkflowState) -> Literal["knowledge", "skip_knowledge"]:
        prediction = graph_state.get("prediction")
        needs_evidence = graph_state["safety"]["requires_knowledge"] or bool(
            prediction and prediction["predicted_dataset_label"] == "diabetes = 1"
        )
        return "knowledge" if needs_evidence else "skip_knowledge"

    def _knowledge(self, graph_state: PredictionWorkflowState) -> dict:
        audit = graph_state["audit_state"]
        output = self.owner._call(
            audit, "KnowledgeAgent", "retrieve_evidence", {"question": self.KNOWLEDGE_QUERY},
            "LangGraph selected retrieval because safety or the positive dataset label requires sourced context.",
            lambda: self.owner.knowledge.run(self.KNOWLEDGE_QUERY),
        )
        audit.knowledge_result = output
        audit.citations.extend([
            {key: item.get(key) for key in ("title", "publisher", "url", "accessed")}
            for item in output.get("evidence", [])
        ])
        return {"knowledge": output, "knowledge_query": self.KNOWLEDGE_QUERY, "audit_state": audit}

    def _skip_knowledge(self, graph_state: PredictionWorkflowState) -> dict:
        audit = graph_state["audit_state"]
        output = {"status": "not_requested", "evidence": [], "graph_facts": []}
        audit.knowledge_result = output
        self.owner._skip(
            audit, "KnowledgeAgent",
            "LangGraph skipped retrieval because no safety threshold fired and the dataset label is negative.",
        )
        return {"knowledge": output, "knowledge_query": None, "audit_state": audit}

    def _explanation(self, graph_state: PredictionWorkflowState) -> dict:
        audit = graph_state["audit_state"]
        prediction = graph_state.get("prediction")
        knowledge = graph_state["knowledge"]
        output = self.owner._call(
            audit, "ExplanationAgent", "compose_report",
            {"prediction_available": prediction is not None, "knowledge_status": knowledge["status"]},
            "LangGraph joined the selected branches and entered the final explanation node.",
            lambda: self.owner.explanation.run(
                prediction, graph_state["validation"], graph_state["safety"], knowledge
            ),
        )
        audit.explanation_result = output
        return {"explanation": output, "audit_state": audit}
