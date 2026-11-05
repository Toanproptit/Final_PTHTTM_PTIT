"""Copy legacy SQLite runs into MySQL without deleting or modifying the SQLite source."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from diabetes_demo.paths import RUNTIME
from diabetes_demo.storage import DemoStore


def migrate(source: Path, if_present: bool = False) -> dict:
    if not source.is_file():
        if if_present:
            return {"status": "source_missing", "source": str(source)}
        raise FileNotFoundError(source)
    destination = DemoStore(create_schema=False)
    copied = skipped = 0
    connection = sqlite3.connect(source)
    try:
        table = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()
        if not table:
            return {"status": "legacy_table_missing"}
        rows = connection.execute("SELECT id, created_at, kind, payload FROM runs").fetchall()
        pending = [row for row in rows if not destination.has_execution(row[0])]
        if not pending:
            return {"status": "already_migrated", "copied": 0, "skipped": len(rows)}
        backup = source.with_name(f"{source.name}.backup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.copy2(source, backup)
        for run_id, _created_at, kind, payload_text in rows:
            if destination.has_execution(run_id):
                skipped += 1
                continue
            destination.save(run_id, kind, json.loads(payload_text), intent=kind)
            copied += 1
    finally:
        connection.close()
    return {"status": "completed", "copied": copied, "skipped": skipped, "backup": str(backup)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=RUNTIME / "demo.sqlite3")
    parser.add_argument("--if-present", action="store_true")
    args = parser.parse_args()
    print(migrate(args.source, args.if_present))
