from types import SimpleNamespace

from langchain_core.messages import AIMessage
from sqlalchemy import func, select

from diabetes_demo.database import AgentMessage, RetrievalLog, ToolCall
from diabetes_demo.llm_agent import LLMToolAgent
from diabetes_demo.storage import DemoStore


class KnowledgeStub:
    def run(self, question):
        return {
            "status": "answered",
            "question": question,
            "mode": "test",
            "evidence": [{"url": "https://example.test"}],
        }


def test_llm_tool_plan_is_persisted_for_audit(tmp_path, monkeypatch):
    store = DemoStore(tmp_path / "audit.sqlite3")
    orchestrator = SimpleNamespace(store=store, knowledge=KnowledgeStub())
    agent = LLMToolAgent(orchestrator)
    agent.api_key = "test-only"
    responses = iter([
        AIMessage(content="", tool_calls=[{
            "id": "call-1",
            "name": "search_health_knowledge",
            "args": {"question": "HbA1c là gì?"},
            "type": "tool_call",
        }]),
        AIMessage(content="Đây là câu trả lời có nguồn."),
    ])
    monkeypatch.setattr(agent, "_invoke_model", lambda _messages: next(responses))

    result = agent.chat("Giải thích HbA1c")

    assert result["status"] == "completed"
    assert result["framework"] == "LangChain"
    assert result["tool_calls"][0]["name"] == "search_health_knowledge"
    assert store.recent()[0]["id"] == result["run_id"]
    with store.sessions() as session:
        assert session.scalar(select(func.count()).select_from(ToolCall)) == 1
        assert session.scalar(select(func.count()).select_from(AgentMessage)) >= 2
        assert session.scalar(select(func.count()).select_from(RetrievalLog)) == 1
