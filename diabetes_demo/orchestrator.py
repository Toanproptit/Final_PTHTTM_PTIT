"""Stateful, auditable routing between specialised agents."""
from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Callable
from uuid import uuid4

from .agent_state import AgentState
from .agents import ExplanationAgent, PredictionAgent, SafetyAgent, ValidationAgent
from .config import get_settings
from .knowledge import KnowledgeAgent
from .langgraph_workflow import LangGraphPredictionWorkflow
from .schema import PatientInput
from .storage import DemoStore


class StepLimitError(RuntimeError):
    pass


class OrchestratorAgent:
    def __init__(self, store: DemoStore):
        self.store = store
        self.prediction = PredictionAgent()
        if self.prediction.ready:
            self.prediction.load()
            dataset = self.prediction.metadata["dataset"]
            ranges = self.prediction.metadata.get("inference_numeric_ranges", dataset["numeric_ranges"])
            self.validation = ValidationAgent(ranges)
        else:
            self.validation = ValidationAgent()
        self.rules = self.validation  # compatibility for existing callers
        self.safety = SafetyAgent()
        self.knowledge = KnowledgeAgent()
        self.explanation = ExplanationAgent()
        self.max_steps = get_settings().agent_max_steps
        self.workflow = LangGraphPredictionWorkflow(self)

    def _call(
        self,
        state: AgentState,
        agent: str,
        action_name: str,
        inputs: dict,
        routing_reason: str,
        action: Callable[[], dict],
    ) -> dict:
        if state.step_count >= state.max_steps:
            raise StepLimitError(f"Agent step limit ({state.max_steps}) reached")
        state.step_count += 1
        state.route(agent, routing_reason)
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        try:
            output = action()
            status = "ok"
            error = None
            state.agent_outputs[agent] = output
        except Exception as exc:
            output = {"error": type(exc).__name__}
            status = "error"
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            state.errors.append({"agent": agent, **error})
            caught_error = exc
        else:
            caught_error = None
        finished_at = datetime.now(timezone.utc)
        state.trace.append({
            "order": len(state.trace) + 1,
            "agent": agent,
            "action": action_name,
            "status": status,
            "routing_reason": routing_reason,
            "input": inputs,
            "output": output,
            "error": error,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": round((perf_counter() - started) * 1000, 3),
        })
        if caught_error is not None:
            raise caught_error
        return output

    @staticmethod
    def _skip(state: AgentState, agent: str, reason: str, inputs: dict | None = None) -> None:
        state.route(agent, reason, selected=False)
        state.trace.append({
            "order": len(state.trace) + 1,
            "agent": agent,
            "action": "skip",
            "status": "skipped",
            "routing_reason": reason,
            "input": inputs or {},
            "output": {"reason": reason},
            "error": None,
            "duration_ms": 0.0,
        })

    def _predict_without_langgraph(self, patient: PatientInput) -> dict:
        run_id = str(uuid4())
        inputs = patient.model_dump()
        state = AgentState(
            request_id=run_id,
            kind="prediction",
            intent="profile_classification",
            user_input=inputs,
            max_steps=self.max_steps,
        )
        try:
            validation = self._call(
                state, "ValidationAgent", "validate_profile", inputs,
                "Structured profile passed API schema validation.",
                lambda: self.validation.run(patient),
            )
            state.validation_result = validation

            safety = self._call(
                state, "SafetyAgent", "evaluate_safety", inputs,
                "Safety evaluation is independent from the ML prediction.",
                lambda: self.safety.run(patient),
            )
            state.safety_status = safety
            state.citations.extend(safety["citations"])

            prediction = None
            if validation["within_training_domain"]:
                prediction = self._call(
                    state, "PredictionAgent", "classify_dataset_label", inputs,
                    "All numeric features are inside the observed training ranges.",
                    lambda: self.prediction.run(patient),
                )
                state.prediction_result = prediction
            else:
                self._skip(
                    state, "PredictionAgent", "One or more fields are outside the training domain.",
                    {name: inputs[name] for name in validation["outside_features"]},
                )

            needs_knowledge = safety["requires_knowledge"] or bool(
                prediction and prediction["predicted_dataset_label"] == "diabetes = 1"
            )
            if needs_knowledge:
                knowledge_query = "Ý nghĩa chung của HbA1c, glucose máu và khi nào cần hỗ trợ y tế"
                knowledge = self._call(
                    state, "KnowledgeAgent", "retrieve_evidence", {"question": knowledge_query},
                    "Safety or prediction output requires sourced background information.",
                    lambda: self.knowledge.run(knowledge_query),
                )
                state.knowledge_result = knowledge
                state.citations.extend([
                    {key: item.get(key) for key in ("title", "publisher", "url", "accessed")}
                    for item in knowledge.get("evidence", [])
                ])
            else:
                knowledge = {"status": "not_requested", "evidence": [], "graph_facts": []}
                state.knowledge_result = knowledge
                self._skip(
                    state, "KnowledgeAgent",
                    "No safety threshold fired and the model did not produce the positive dataset label.",
                )

            explanation = self._call(
                state, "ExplanationAgent", "compose_report",
                {"prediction_available": prediction is not None, "knowledge_status": knowledge["status"]},
                "A final report must combine validation, safety, model and evidence outputs.",
                lambda: self.explanation.run(prediction, validation, safety, knowledge),
            )
            state.explanation_result = explanation
            result = {
                "run_id": run_id,
                "status": "completed" if prediction else "outside_domain",
                "intent": state.intent,
                "input": inputs,
                "prediction": prediction,
                "rules": validation,
                "safety": safety,
                "knowledge": knowledge,
                "explanation": explanation,
                "citations": state.citations,
                "routing_decisions": state.routing_decisions,
                "trace": state.trace,
            }
            self.store.save(run_id, "prediction", result, intent=state.intent)
            if prediction is not None:
                self.store.save_prediction(run_id, inputs, prediction)
            if knowledge.get("status") == "answered":
                retrieval_step = next(
                    (item for item in state.trace if item["agent"] == "KnowledgeAgent" and item["status"] == "ok"),
                    {},
                )
                self.store.save_retrieval(run_id, knowledge_query, knowledge, retrieval_step.get("duration_ms", 0.0))
            return result
        except Exception:
            failure = {
                "run_id": run_id, "status": "failed", "intent": state.intent,
                "input": inputs, "errors": state.errors, "routing_decisions": state.routing_decisions,
                "trace": state.trace,
            }
            self.store.save(run_id, "prediction", failure, intent=state.intent)
            raise

    def predict(self, patient: PatientInput) -> dict:
        """Execute the production profile workflow through a compiled LangGraph."""
        run_id = str(uuid4())
        inputs = patient.model_dump()
        state = AgentState(
            request_id=run_id,
            kind="prediction",
            intent="profile_classification",
            user_input=inputs,
            max_steps=self.max_steps,
        )
        try:
            graph_output = self.workflow.invoke(patient, state, inputs)
            state = graph_output["audit_state"]
            validation = graph_output["validation"]
            safety = graph_output["safety"]
            prediction = graph_output.get("prediction")
            knowledge = graph_output["knowledge"]
            explanation = graph_output["explanation"]
            knowledge_query = graph_output.get("knowledge_query")
            result = {
                "run_id": run_id,
                "status": "completed" if prediction else "outside_domain",
                "intent": state.intent,
                "orchestration": self.workflow.description(),
                "input": inputs,
                "prediction": prediction,
                "rules": validation,
                "safety": safety,
                "knowledge": knowledge,
                "explanation": explanation,
                "citations": state.citations,
                "routing_decisions": state.routing_decisions,
                "trace": state.trace,
            }
            self.store.save(run_id, "prediction", result, intent=state.intent)
            if prediction is not None:
                self.store.save_prediction(run_id, inputs, prediction)
            if knowledge.get("status") == "answered" and knowledge_query:
                retrieval_step = next(
                    (item for item in state.trace if item["agent"] == "KnowledgeAgent" and item["status"] == "ok"),
                    {},
                )
                self.store.save_retrieval(
                    run_id, knowledge_query, knowledge, retrieval_step.get("duration_ms", 0.0)
                )
            return result
        except Exception:
            failure = {
                "run_id": run_id,
                "status": "failed",
                "intent": state.intent,
                "orchestration": self.workflow.description(),
                "input": inputs,
                "errors": state.errors,
                "routing_decisions": state.routing_decisions,
                "trace": state.trace,
            }
            self.store.save(run_id, "prediction", failure, intent=state.intent)
            raise

    def ask(self, question: str) -> dict:
        run_id = str(uuid4())
        state = AgentState(
            request_id=run_id,
            kind="question",
            intent="knowledge_query",
            user_input={"question": question},
            max_steps=self.max_steps,
        )
        output = self._call(
            state, "KnowledgeAgent", "retrieve_evidence", {"question": question},
            "The request is a knowledge question and does not require a patient prediction.",
            lambda: self.knowledge.run(question),
        )
        result = {
            "run_id": run_id,
            "intent": state.intent,
            "question": question,
            **output,
            "routing_decisions": state.routing_decisions,
            "trace": state.trace,
        }
        self.store.save(run_id, "question", result, intent=state.intent)
        self.store.save_retrieval(run_id, question, output, state.trace[0].get("duration_ms", 0.0))
        return result
