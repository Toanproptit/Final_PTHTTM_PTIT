"""Create Neo4j constraints, vector index, document chunks, entities and sourced facts."""
import argparse

from diabetes_demo.graph_knowledge import KnowledgeGraphService

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    graph = KnowledgeGraphService()
    health = graph.health()
    if not args.force and health.get("ready") and health.get("facts", 0) > 0:
        print({"status": "already_imported", **health})
    else:
        print(graph.import_knowledge())
