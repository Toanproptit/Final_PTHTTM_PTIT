"""Retrieval-based QA: Markdown -> chunks -> TF-IDF -> cosine -> trích đoạn."""
from __future__ import annotations
import hashlib
import json
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .paths import KB

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower()).replace("đ", "d")
    return "".join(char for char in text if not unicodedata.combining(char))

STOP_WORDS = "la va cua co khong nhung mot cac trong de duoc voi ve cho tu o nhu nay do thi khi ra hay nao toi ban se da phai gi".split()

def chunk_markdown(text: str, max_words: int = 90, overlap: int = 15) -> list[str]:
    """Giữ ngữ nghĩa đoạn; đoạn quá dài dùng cửa sổ từ có overlap."""
    if max_words <= overlap or overlap < 0:
        raise ValueError("Cần 0 <= overlap < max_words")
    paragraphs = [re.sub(r"^#+\s*", "", p.strip(), flags=re.MULTILINE) for p in text.split("\n\n") if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= max_words:
            chunks.append(paragraph)
        else:
            for start in range(0, len(words), max_words - overlap):
                chunks.append(" ".join(words[start:start + max_words]))
                if start + max_words >= len(words):
                    break
    return chunks

class KnowledgeAgent:
    """Không sinh văn bản bằng LLM; evidence giữ nguyên nội dung chunk đã biên soạn."""
    def __init__(self, minimum_score: float = 0.08):
        self.minimum_score = minimum_score
        self.documents = json.loads((KB / "manifest.json").read_text(encoding="utf-8"))
        self.chunks = []
        for document in self.documents:
            text = (KB / document["file"]).read_text(encoding="utf-8")
            for number, chunk in enumerate(chunk_markdown(text)):
                self.chunks.append({
                    "chunk_id": f"{document['id']}:{number}", "document_id": document["id"],
                    "title": document["title"], "publisher": document["publisher"],
                    "url": document["url"], "accessed": document["accessed"], "text": chunk,
                })
        self.vectorizer = TfidfVectorizer(preprocessor=normalize, stop_words=STOP_WORDS,
                                         ngram_range=(1, 2), sublinear_tf=True,
                                         token_pattern=r"(?u)\b\w+\b")
        self.matrix = self.vectorizer.fit_transform([f"{c['title']} {c['text']}" for c in self.chunks])
        self.index_hash = hashlib.sha256(json.dumps(self.chunks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.graph = None
        try:
            from .graph_knowledge import KnowledgeGraphService
            candidate = KnowledgeGraphService()
            if candidate.configured:
                self.graph = candidate
        except Exception:
            self.graph = None

    def run(self, question: str, top_k: int = 3) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống")
        normalized = normalize(question)
        # Chặn yêu cầu kê đơn/chẩn đoán cá nhân trước khi ghép các đoạn tổng quát.
        restricted = [r"\blieu (?:luong|dung|thuoc|bao nhieu)\b", r"ke don", r"toi.*(?:co bi|mac benh)",
                      r"(?:uong|dung|ngung|tang|giam).*thuoc", r"metformin", r"chan doan.*(?:toi|em)"]
        if any(re.search(pattern, normalized) for pattern in restricted):
            return {"mode": "retrieval-based QA", "status": "outside_scope", "question": question,
                    "answer": "Kho tri thức chỉ cung cấp thông tin chung, không kê liều thuốc hoặc xác nhận bệnh cho cá nhân. Hãy trao đổi với nhân viên y tế về đánh giá và điều trị.", "evidence": []}
        # Cổng phạm vi ngăn một từ tình cờ (ví dụ 'phiếu') kéo câu ngoài y tế vào KB.
        # Danh sách là heuristic của MVP, không phải bộ hiểu ngôn ngữ tổng quát.
        topics = ["tieu duong", "insulin", "hba1c", "a1c", "glucose", "duong huyet", "xet nghiem",
                  "trieu chung", "khat nhieu", "tieu nhieu", "sut can", "vet thuong", "nhin mo",
                  "thua can", "beo phi", "tien su gia dinh", "thai ky", "type 1", "type 2"]
        if not any(topic in normalized for topic in topics):
            return {"mode": "retrieval-based QA", "status": "no_evidence", "question": question,
                    "answer": "Câu hỏi chưa khớp phạm vi kho tri thức tiểu đường. Hãy nêu rõ chủ đề triệu chứng, xét nghiệm, nguy cơ hoặc phòng ngừa.", "evidence": []}
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix).ravel()
        order = scores.argsort()[::-1]
        evidence = [{**self.chunks[int(i)], "score": round(float(scores[i]), 5)}
                    for i in order[:max(1, min(top_k, 5))] if scores[i] >= self.minimum_score]
        if not evidence:
            return {"mode": "retrieval-based QA", "status": "no_evidence", "question": question,
                    "answer": "Chưa tìm thấy đoạn đủ liên quan trong kho tri thức hiện tại. Hãy hỏi cụ thể về triệu chứng, xét nghiệm, yếu tố nguy cơ hoặc phòng ngừa tiểu đường.", "evidence": []}
        result = {"mode": "retrieval-based QA", "status": "answered", "question": question,
                  "answer": "Các đoạn thông tin chung dưới đây được truy xuất từ bản diễn giải tiếng Việt có dẫn nguồn.",
                  "evidence": evidence, "graph_facts": [], "retrieval_status": {"tfidf": "ok", "neo4j": "not_configured"}}
        if self.graph:
            try:
                graph_result = self.graph.retrieve(question, top_k=top_k)
                by_id = {item["chunk_id"]: item for item in evidence}
                for item in graph_result["semantic_evidence"]:
                    item["score"] = round(float(item["score"]), 5)
                    by_id.setdefault(item["chunk_id"], item)
                result.update({"mode": "hybrid semantic + GraphRAG", "evidence": list(by_id.values())[:5],
                               "graph_facts": graph_result["graph_facts"],
                               "retrieval_status": {"tfidf": "ok", "neo4j": "ok",
                                                    "duration_ms": graph_result["duration_ms"]}})
            except Exception as error:
                result["retrieval_status"]["neo4j"] = f"unavailable:{type(error).__name__}"
        return result
