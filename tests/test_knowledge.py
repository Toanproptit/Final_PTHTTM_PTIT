import pytest
from diabetes_demo.knowledge import KnowledgeAgent, chunk_markdown

@pytest.fixture(scope="module")
def agent():
    return KnowledgeAgent()

def test_chunks_bound_long_paragraph_and_preserve_tail():
    words = [f"word{i}" for i in range(205)]
    chunks = chunk_markdown(" ".join(words), max_words=90, overlap=15)
    assert all(len(chunk.split()) <= 90 for chunk in chunks)
    assert chunks[-1].endswith("word204")
    assert "word75" in chunks[0] and "word75" in chunks[1]

def test_retrieval_has_traceable_source_and_exact_stored_text(agent):
    result = agent.run("HbA1c phản ánh đường huyết trong bao lâu?")
    assert result["status"] == "answered"
    assert result["evidence"][0]["document_id"] == "testing"
    assert result["mode"] == "retrieval-based QA"
    for evidence in result["evidence"]:
        assert evidence["text"] == next(chunk["text"] for chunk in agent.chunks if chunk["chunk_id"] == evidence["chunk_id"])
        assert evidence["url"].startswith(("https://www.cdc.gov/", "https://www.who.int/"))
        assert evidence["accessed"] == "2026-11-02"

@pytest.mark.parametrize("question", ["Cách sửa động cơ xe máy?", "Giá cổ phiếu hôm nay?", "Tôi nên uống metformin liều bao nhiêu?", "Thời tiết tuần này thế nào?", "Hướng dẫn viết phiếu đăng ký?"])
def test_unknown_or_personalized_questions_do_not_create_answer(agent, question):
    result = agent.run(question)
    assert not result["evidence"]
    assert result["status"] in {"no_evidence", "outside_scope"}
