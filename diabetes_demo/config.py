"""Application settings loaded from environment variables or the local ignored .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from .paths import ROOT


def _env_file_values() -> dict[str, str]:
    path = ROOT / ".env"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value.strip()
    return _env_file_values().get(name, default)


def _boolean(name: str, default: bool = False) -> bool:
    return setting(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    embedding_model: str
    openrouter_api_key: str
    openrouter_model: str
    agent_max_steps: int
    agent_tool_timeout_seconds: float
    allow_sqlite_fallback: bool

    @property
    def mysql_configured(self) -> bool:
        return self.database_url.startswith("mysql+")

    @property
    def neo4j_configured(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_user and self.neo4j_password)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    default_sqlite = f"sqlite:///{(ROOT / 'runtime' / 'demo.sqlite3').as_posix()}"
    return Settings(
        database_url=setting("DATABASE_URL", default_sqlite),
        neo4j_uri=setting("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=setting("NEO4J_USER", "neo4j"),
        neo4j_password=setting("NEO4J_PASSWORD"),
        neo4j_database=setting("NEO4J_DATABASE", "neo4j"),
        embedding_model=setting("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        openrouter_api_key=setting("OPENROUTER_API_KEY"),
        openrouter_model=setting("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        agent_max_steps=max(3, min(30, int(setting("AGENT_MAX_STEPS", "12")))),
        agent_tool_timeout_seconds=max(1.0, min(120.0, float(setting("AGENT_TOOL_TIMEOUT_SECONDS", "30")))),
        allow_sqlite_fallback=_boolean("ALLOW_SQLITE_FALLBACK", False),
    )
