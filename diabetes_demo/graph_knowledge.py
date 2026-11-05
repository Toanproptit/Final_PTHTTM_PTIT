"""Neo4j knowledge graph import and hybrid semantic/graph retrieval."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from time import perf_counter
from typing import Any

from .config import Settings, get_settings
from .knowledge import chunk_markdown, normalize
from .paths import KB


class GraphUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _encoder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise GraphUnavailable("Chưa cài sentence-transformers") from error
    return SentenceTransformer(model_name)


class KnowledgeGraphService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._driver = None

    @property
    def configured(self) -> bool:
        return self.settings.neo4j_configured

    def _get_driver(self):
        if not self.configured:
            raise GraphUnavailable("Neo4j chưa được cấu hình")
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as error:
                raise GraphUnavailable("Chưa cài neo4j driver") from error
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                connection_timeout=5,
            )
        return self._driver

    def health(self) -> dict:
        if not self.configured:
            return {"configured": False, "ready": False, "reason": "missing_credentials"}
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            with driver.session(database=self.settings.neo4j_database) as session:
                counts = session.run(
                    "MATCH (d:SourceDocument) WITH count(d) AS documents "
                    "MATCH (c:DocumentChunk) WITH documents, count(c) AS chunks "
                    "MATCH (f:Fact) RETURN documents, chunks, count(f) AS facts"
                ).single()
            return {"configured": True, "ready": True, **dict(counts)}
        except Exception as error:
            return {"configured": True, "ready": False, "reason": type(error).__name__}

    def import_knowledge(self) -> dict:
        driver = self._get_driver()
        documents = json.loads((KB / "manifest.json").read_text(encoding="utf-8"))
        facts = json.loads((KB / "graph_facts.json").read_text(encoding="utf-8"))
        encoder = _encoder(self.settings.embedding_model)
        chunks: list[dict[str, Any]] = []
        for document in documents:
            content = (KB / document["file"]).read_text(encoding="utf-8")
            for number, text in enumerate(chunk_markdown(content)):
                chunks.append({
                    "chunk_id": f"{document['id']}:{number}", "document_id": document["id"],
                    "title": document["title"], "publisher": document["publisher"],
                    "url": document["url"], "accessed": document["accessed"], "text": text,
                })
        vectors = encoder.encode([f"{item['title']} {item['text']}" for item in chunks], normalize_embeddings=True)
        dimension = int(vectors.shape[1])
        by_document: dict[str, list[dict[str, Any]]] = {}
        for item, vector in zip(chunks, vectors):
            item["embedding"] = [float(value) for value in vector]
            by_document.setdefault(item["document_id"], []).append(item)
        with driver.session(database=self.settings.neo4j_database) as session:
            for statement in [
                "CREATE CONSTRAINT source_document_id IF NOT EXISTS FOR (d:SourceDocument) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT document_chunk_id IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
                "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            ]:
                session.run(statement).consume()
            vector_index_query = (
                "CREATE VECTOR INDEX knowledge_chunk_embeddings IF NOT EXISTS "
                "FOR (c:DocumentChunk) ON c.embedding OPTIONS {indexConfig: {"
                "`vector.dimensions`: %d, `vector.similarity_function`: 'cosine'}}"
            ) % dimension
            session.run(vector_index_query).consume()
            for document in documents:
                session.run(
                    "MERGE (d:SourceDocument {id:$id}) SET d.title=$title, d.publisher=$publisher, "
                    "d.url=$url, d.accessed=$accessed, d.usage=$usage",
                    **document,
                ).consume()
            for item in chunks:
                session.run(
                    "MATCH (d:SourceDocument {id:$document_id}) MERGE (c:DocumentChunk {id:$chunk_id}) "
                    "SET c.text=$text, c.title=$title, c.publisher=$publisher, c.url=$url, "
                    "c.accessed=$accessed, c.embedding=$embedding MERGE (d)-[:HAS_CHUNK]->(c)",
                    **item,
                ).consume()
            for fact in facts:
                candidates = by_document.get(fact["document_id"], [])
                anchor = normalize(fact["anchor"])
                source = next((item for item in candidates if anchor in normalize(item["text"])), candidates[0] if candidates else None)
                if source is None:
                    raise ValueError(f"Không tìm thấy source chunk cho fact {fact['id']}")
                subject_key = f"{fact['subject']['label']}:{normalize(fact['subject']['name'])}"
                object_key = f"{fact['object']['label']}:{normalize(fact['object']['name'])}"
                session.run(
                    "MERGE (s:Entity {key:$subject_key}) SET s.name=$subject_name, "
                    "s.normalized_name=$subject_normalized, s.kind=$subject_label "
                    "MERGE (o:Entity {key:$object_key}) SET o.name=$object_name, "
                    "o.normalized_name=$object_normalized, o.kind=$object_label "
                    "MERGE (f:Fact {id:$fact_id}) SET f.predicate=$predicate "
                    "MERGE (s)-[:HAS_FACT]->(f) MERGE (f)-[:RELATES_TO]->(o) "
                    "WITH f MATCH (c:DocumentChunk {id:$chunk_id}) MERGE (f)-[:SUPPORTED_BY]->(c)",
                    subject_key=subject_key, subject_name=fact["subject"]["name"],
                    subject_normalized=normalize(fact["subject"]["name"]), subject_label=fact["subject"]["label"],
                    object_key=object_key, object_name=fact["object"]["name"],
                    object_normalized=normalize(fact["object"]["name"]), object_label=fact["object"]["label"],
                    fact_id=fact["id"], predicate=fact["predicate"], chunk_id=source["chunk_id"],
                ).consume()
        return {"status": "completed", "documents": len(documents), "chunks": len(chunks),
                "facts": len(facts), "embedding_model": self.settings.embedding_model, "dimensions": dimension}

    def retrieve(self, question: str, top_k: int = 3) -> dict:
        started = perf_counter()
        driver = self._get_driver()
        encoder = _encoder(self.settings.embedding_model)
        vector = [float(value) for value in encoder.encode([question], normalize_embeddings=True)[0]]
        terms = [term for term in re.findall(r"\w+", normalize(question)) if len(term) >= 3][:12]
        with driver.session(database=self.settings.neo4j_database) as session:
            semantic = [dict(record) for record in session.run(
                "CALL db.index.vector.queryNodes('knowledge_chunk_embeddings', $top_k, $vector) "
                "YIELD node, score RETURN node.id AS chunk_id, node.title AS title, "
                "node.publisher AS publisher, node.url AS url, node.accessed AS accessed, "
                "node.text AS text, score ORDER BY score DESC",
                top_k=max(1, min(top_k, 5)), vector=vector,
            )]
            graph_facts = [dict(record) for record in session.run(
                "MATCH (s:Entity)-[:HAS_FACT]->(f:Fact)-[:RELATES_TO]->(o:Entity) "
                "MATCH (f)-[:SUPPORTED_BY]->(c:DocumentChunk) "
                "WHERE any(term IN $terms WHERE s.normalized_name CONTAINS term OR "
                "o.normalized_name CONTAINS term OR toLower(f.predicate) CONTAINS term) "
                "RETURN f.id AS fact_id, s.name AS subject, f.predicate AS predicate, o.name AS object, "
                "c.id AS chunk_id, c.title AS title, c.publisher AS publisher, c.url AS url, c.accessed AS accessed "
                "LIMIT 10", terms=terms,
            )]
        return {"mode": "neo4j-hybrid", "semantic_evidence": semantic, "graph_facts": graph_facts,
                "duration_ms": round((perf_counter() - started) * 1000, 3)}

    def neighborhood(self, limit: int = 80) -> dict:
        """Return a small provenance-aware subgraph for the React viewer."""
        driver = self._get_driver()
        with driver.session(database=self.settings.neo4j_database) as session:
            records = [dict(record) for record in session.run(
                "MATCH (s:Entity)-[:HAS_FACT]->(f:Fact)-[:RELATES_TO]->(o:Entity) "
                "MATCH (f)-[:SUPPORTED_BY]->(c:DocumentChunk) "
                "RETURN s.key AS source_id, s.name AS source_name, s.kind AS source_kind, "
                "f.id AS fact_id, f.predicate AS predicate, o.key AS target_id, "
                "o.name AS target_name, o.kind AS target_kind, c.id AS chunk_id, "
                "c.title AS title, c.url AS url LIMIT $limit",
                limit=max(1, min(limit, 200)),
            )]
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for item in records:
            nodes[item["source_id"]] = {
                "id": item["source_id"], "label": item["source_name"], "kind": item["source_kind"],
            }
            nodes[item["target_id"]] = {
                "id": item["target_id"], "label": item["target_name"], "kind": item["target_kind"],
            }
            edges.append({
                "id": item["fact_id"], "source": item["source_id"], "target": item["target_id"],
                "label": item["predicate"], "chunk_id": item["chunk_id"],
                "citation": {"title": item["title"], "url": item["url"]},
            })
        return {"nodes": list(nodes.values()), "edges": edges}
