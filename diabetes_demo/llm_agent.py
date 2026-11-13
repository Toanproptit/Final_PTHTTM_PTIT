"""LangChain tool-calling agent backed by OpenRouter."""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter
from pydantic import ValidationError

from .config import get_settings
from .langchain_tools import build_langchain_tools

INSTRUCTIONS = """You are the tool-using assistant in a Vietnamese student research demo.
Respond in the user's language. You may choose and call the provided local tools when useful; decide which tools to use and in what order. Never invent tool results, sources, or profile fields. Ask a concise follow-up when prediction fields are missing. For general health information, call search_health_knowledge and ground factual claims in its evidence and source URLs. If the knowledge tool returns no evidence or outside_scope, say so. Treat all predictions as classification scores for labels in an inadequately documented Kaggle dataset, not diagnosis, future disease risk, or treatment advice. Never recommend medication or individual treatment. The project uses hypothetical demo profiles only. Do not expose internal chain-of-thought; a short user-facing summary of actions taken is fine."""

ADVICE_INSTRUCTIONS = """You explain a saved research-model result in Vietnamese using only the supplied profile summary and retrieved evidence.
The model predicts a label in an inadequately documented Kaggle dataset. Never call the label or score a diagnosis, an individual's disease probability, or future risk. Explain that limitation plainly. Do not recommend, start, stop, or change medication; do not give a treatment plan. Do not infer a medical condition from one numeric field. If evidence is insufficient for a requested conclusion, say so. Ground health facts only in the supplied evidence and cite its Vinmec source URL. Give general educational context and suggest discussing personal concerns with a qualified health professional."""


class LLMToolAgent:
    """LangChain agent loop that may execute only registered StructuredTools."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        settings = get_settings()
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model
        self.max_tool_rounds = max(1, min(8, settings.agent_max_steps // 2))
        self.timeout_seconds = settings.agent_tool_timeout_seconds
        self.tools = build_langchain_tools(orchestrator)
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.chat_model = None
        self.bound_model = None
        if self.configured:
            self.chat_model = ChatOpenRouter(
                api_key=self.api_key,
                model=self.model,
                temperature=0,
                max_tokens=1200,
                timeout=round(self.timeout_seconds * 1000),
                max_retries=0,
                app_title="Diabetes Agent Lab",
            )
            self.bound_model = self.chat_model.bind_tools(
                self.tools,
                tool_choice="auto",
                parallel_tool_calls=False,
            )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    def status(self) -> dict:
        return {
            "configured": self.configured,
            "provider": "OpenRouter",
            "framework": "LangChain",
            "integration": "ChatOpenRouter",
            "model": self.model if self.configured else None,
            "tools": list(self.tool_map),
        }

    def _invoke_model(self, messages: list[BaseMessage]) -> AIMessage:
        if self.bound_model is None:
            raise RuntimeError(
                "LLM chưa được cấu hình. Hãy đặt OPENROUTER_API_KEY và khởi động lại server."
            )
        try:
            response = self.bound_model.invoke(messages)
        except Exception as error:
            raise RuntimeError(
                "Không thể gọi OpenRouter qua LangChain. Hãy kiểm tra API key, model và kết nối mạng "
                f"({type(error).__name__})."
            ) from None
        if not isinstance(response, AIMessage):
            raise RuntimeError("LangChain returned an unexpected response type.")
        return response

    @staticmethod
    def _text(message: AIMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
            return "\n".join(parts).strip()
        return ""

    def _call_local_tool(self, name: str, arguments: dict) -> tuple[Any, dict]:
        tool = self.tool_map.get(name)
        if tool is None:
            raise ValueError("Tool is not allowlisted.")
        result = tool.invoke(arguments)
        if name == "search_health_knowledge":
            citations = [
                {key: item.get(key) for key in ("title", "publisher", "url", "accessed")}
                for item in result.get("evidence", [])
            ]
            summary = {
                "status": result.get("status"),
                "mode": result.get("mode"),
                "evidence_count": len(result.get("evidence", [])),
                "citations": citations,
            }
        elif name == "run_research_prediction":
            summary = {
                "status": result.get("status"),
                "prediction": result.get("prediction"),
                "rule_warnings": result.get("rule_warnings", []),
                "limitation": result.get("limitation"),
                "orchestration": result.get("orchestration"),
            }
        else:
            summary = {"status": result.get("status")}
        return result, summary

    def chat(self, message: str, history: list[dict] | None = None) -> dict:
        if not self.configured:
            raise RuntimeError(
                "LLM chưa được cấu hình. Hãy đặt OPENROUTER_API_KEY và khởi động lại server."
            )

        run_id = str(uuid4())
        conversation: list[BaseMessage] = [SystemMessage(content=INSTRUCTIONS)]
        for turn in (history or [])[-8:]:
            role = turn.get("role")
            content = turn.get("content")
            if not isinstance(content, str):
                continue
            if role == "user":
                conversation.append(HumanMessage(content=content[:4000]))
            elif role == "assistant":
                conversation.append(AIMessage(content=content[:4000]))
        conversation.append(HumanMessage(content=message))

        tool_trace = []
        for round_number in range(1, self.max_tool_rounds + 1):
            assistant_message = self._invoke_model(conversation)
            calls = assistant_message.tool_calls or []
            if not calls:
                answer = self._text(assistant_message)
                if not answer:
                    raise RuntimeError("Model returned no text response.")
                result = {
                    "run_id": run_id,
                    "status": "completed",
                    "model": self.model,
                    "framework": "LangChain",
                    "answer": answer,
                    "tool_calls": tool_trace,
                    "rounds": round_number,
                    "input": {"message": message},
                }
                self._persist_chat(result, conversation, answer)
                return result

            if round_number == self.max_tool_rounds:
                for call in calls:
                    tool_trace.append({
                        "name": call.get("name", ""),
                        "status": "skipped_limit",
                        "arguments": call.get("args", {}),
                        "summary": {"reason": "tool round limit reached"},
                        "round": round_number,
                    })
                answer = (
                    "Đã đến giới hạn số lượt lập kế hoạch cho một yêu cầu. "
                    "Hãy chia câu hỏi thành các bước nhỏ hơn."
                )
                result = {
                    "run_id": run_id,
                    "status": "step_limit",
                    "model": self.model,
                    "framework": "LangChain",
                    "answer": answer,
                    "tool_calls": tool_trace,
                    "rounds": self.max_tool_rounds,
                    "input": {"message": message},
                }
                self._persist_chat(result, conversation, answer)
                return result

            conversation.append(assistant_message)
            for call in calls:
                name = call.get("name", "")
                arguments = call.get("args", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                try:
                    tool_result, summary = self._call_local_tool(name, arguments)
                    status = "ok"
                    output_text = json.dumps(tool_result, ensure_ascii=False)
                except (ValueError, KeyError, ValidationError) as error:
                    status = "error"
                    summary = {"error": str(error)}
                    output_text = json.dumps({"error": str(error)}, ensure_ascii=False)
                except Exception as error:
                    status = "error"
                    summary = {"error": type(error).__name__}
                    output_text = json.dumps(
                        {"error": "Tool execution failed. Do not infer a result."}
                    )
                tool_trace.append({
                    "name": name,
                    "status": status,
                    "arguments": arguments,
                    "summary": summary,
                    "round": round_number,
                })
                conversation.append(ToolMessage(
                    content=output_text,
                    tool_call_id=call.get("id") or f"tool-{round_number}",
                    name=name or None,
                ))

        answer = (
            "Đã đạt giới hạn số lượt gọi công cụ cho một yêu cầu. "
            "Hãy chia câu hỏi thành các bước nhỏ hơn."
        )
        result = {
            "run_id": run_id,
            "status": "step_limit",
            "model": self.model,
            "framework": "LangChain",
            "answer": answer,
            "tool_calls": tool_trace,
            "rounds": self.max_tool_rounds,
            "input": {"message": message},
        }
        self._persist_chat(result, conversation, answer)
        return result

    def advise_from_prediction(self, source_run_id: str) -> dict:
        """Retrieve sourced context for a saved prediction, then explain it with the LLM."""
        if not self.configured or self.chat_model is None:
            raise RuntimeError("ChÆ°a cáº¥u hÃ¬nh OPENROUTER_API_KEY.")

        execution = self.orchestrator.store.get_execution(source_run_id)
        if execution is None or execution["kind"] != "prediction":
            raise KeyError("KhÃ´ng tÃ¬m tháº¥y láº§n dá»± Ä‘oÃ¡n nÃ y.")
        prediction_result = execution["final_payload"]
        prediction = prediction_result.get("prediction")
        if not prediction:
            raise ValueError("Láº§n cháº¡y nÃ y khÃ´ng cÃ³ káº¿t quáº£ dá»± Ä‘oÃ¡n Ä‘á»ƒ giáº£i thÃ­ch.")

        # Retrieve general evidence for every advice request. The RAG layer
        # still enforces its scope and minimum-score threshold.
        knowledge_query = "HbA1c glucose xét nghiệm yếu tố nguy cơ tiểu đường type 2 phòng ngừa"
        knowledge = self.orchestrator.knowledge.run(knowledge_query)
        evidence = knowledge.get("evidence", [])
        sources = []
        seen_urls = set()
        for item in [*evidence, *knowledge.get("graph_facts", [])]:
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "url": url,
                    "accessed": item.get("accessed"),
                })

        if not evidence:
            return {
                "status": "no_evidence",
                "source_prediction_run_id": source_run_id,
                "answer": "Mình chưa tìm được đoạn tài liệu đủ liên quan để đưa ra phần giải thích có dẫn nguồn. Kết quả mô hình chỉ là phân loại trên dataset nghiên cứu, không phải chẩn đoán.",
                "sources": [],
                "evidence_count": 0,
                "retrieval_status": knowledge.get("retrieval_status", {}),
            }

        context = {
            "profile_fields": prediction_result.get("input", {}),
            "prediction": prediction,
            "validation_warnings": prediction_result.get("rules", {}).get("warnings", []),
            "safety": prediction_result.get("safety", {}),
            "retrieved_evidence": [
                {key: item.get(key) for key in ("title", "publisher", "url", "accessed", "text", "score")}
                for item in evidence
            ],
            "graph_facts": knowledge.get("graph_facts", []),
        }
        messages = [
            SystemMessage(content=ADVICE_INSTRUCTIONS + "\n\nPrediction context and RAG evidence (JSON):\n" + json.dumps(context, ensure_ascii=False)),
            HumanMessage(content="Giải thích kết quả dự đoán này theo ngữ cảnh hồ sơ, nêu rõ giới hạn và chỉ đưa thông tin sức khỏe có căn cứ trong evidence. Trả lời ngắn gọn, có trích dẫn nguồn Vinmec."),
        ]
        try:
            response = self.chat_model.invoke(messages)
        except Exception as error:
            raise RuntimeError(
                f"KhÃ´ng thá»ƒ gá»i OpenRouter Ä‘á»ƒ giáº£i thÃ­ch káº¿t quáº£ ({type(error).__name__})."
            ) from None
        if not isinstance(response, AIMessage):
            raise RuntimeError("LangChain returned an unexpected response type.")
        answer = self._text(response)
        if not answer:
            raise RuntimeError("Model returned no text response.")

        run_id = str(uuid4())
        result = {
            "run_id": run_id,
            "status": "completed",
            "source_prediction_run_id": source_run_id,
            "model": self.model,
            "framework": "LangChain",
            "answer": answer,
            "sources": sources,
            "evidence_count": len(evidence),
            "graph_fact_count": len(knowledge.get("graph_facts", [])),
            "retrieval_status": knowledge.get("retrieval_status", {}),
        }
        self.orchestrator.store.save(run_id, "llm_advice", result, intent="prediction_advice")
        self.orchestrator.store.save_retrieval(run_id, knowledge_query, knowledge)
        return result

    def _persist_chat(self, result: dict, conversation: list[BaseMessage], answer: str) -> None:
        """Persist the plan/tool audit only after an execution row exists."""
        run_id = result["run_id"]
        payload = {**result, "trace": []}
        self.orchestrator.store.save(run_id, "llm_chat", payload, intent="llm_tool_planning")
        self.orchestrator.store.save_tool_calls(run_id, result.get("tool_calls", []))
        for call in result.get("tool_calls", []):
            if call.get("name") == "search_health_knowledge" and call.get("status") == "ok":
                summary = call.get("summary", {})
                retrieval_result = {
                    "mode": summary.get("mode") or "llm-tool-retrieval",
                    "evidence": summary.get("citations", []),
                }
                self.orchestrator.store.save_retrieval(
                    run_id,
                    call.get("arguments", {}).get("question", ""),
                    retrieval_result,
                    call.get("duration_ms", 0.0),
                )
        messages = []
        for item in conversation:
            if isinstance(item, HumanMessage) and isinstance(item.content, str):
                messages.append({"role": "user", "content": item.content})
            elif isinstance(item, AIMessage) and isinstance(item.content, str) and item.content:
                messages.append({"role": "assistant", "content": item.content})
        if not messages or messages[-1] != {"role": "assistant", "content": answer}:
            messages.append({"role": "assistant", "content": answer})
        self.orchestrator.store.save_messages(run_id, messages)
