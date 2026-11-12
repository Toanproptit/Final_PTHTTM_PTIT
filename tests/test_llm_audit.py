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


def test_openrouter_timeout_is_converted_to_milliseconds_and_disables_long_retries(monkeypatch):
    captured = {}

    class ChatModelStub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools, **kwargs):
            captured["tool_count"] = len(tools)
            captured["bind_options"] = kwargs
            return object()

    monkeypatch.setattr(
        "diabetes_demo.llm_agent.get_settings",
        lambda: SimpleNamespace(
            openrouter_api_key="test-only",
            openrouter_model="openai/gpt-4.1-mini",
            agent_max_steps=12,
            agent_tool_timeout_seconds=30,
        ),
    )
    monkeypatch.setattr("diabetes_demo.llm_agent.ChatOpenRouter", ChatModelStub)

    agent = LLMToolAgent(SimpleNamespace())

    assert agent.configured
    assert captured["timeout"] == 30_000
    assert captured["max_retries"] == 0
    assert captured["tool_count"] == 3
