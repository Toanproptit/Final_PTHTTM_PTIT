#!/bin/sh
set -eu
export PYTHONPATH=/app

echo "[bootstrap] applying relational migrations"
alembic upgrade head

echo "[bootstrap] preserving and migrating legacy SQLite history when present"
python scripts/migrate_sqlite_to_mysql.py --if-present

echo "[bootstrap] importing the validated dataset into MySQL"
python scripts/import_dataset_mysql.py

echo "[bootstrap] creating the Neo4j graph and vector index"
python scripts/import_knowledge_graph.py

echo "[bootstrap] starting FastAPI"
exec uvicorn diabetes_demo.api:app --host 0.0.0.0 --port 8000
