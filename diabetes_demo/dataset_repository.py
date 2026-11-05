"""Idempotent Kaggle dataset import and training reads through the relational database."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy import select

from .data import DATASET_REF, SOURCE_URL, load_dataset
from .database import DataImportJob, Dataset, DatasetRecord, DatasetVersion
from .paths import RAW
from .storage import DemoStore


def import_dataset(store: DemoStore, path: Path = RAW, batch_size: int = 2_000) -> dict:
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    job_id = str(uuid4())
    frame, stats = load_dataset(path)
    with store.sessions.begin() as session:
        existing = session.scalar(select(DatasetVersion).where(DatasetVersion.version_hash == source_hash))
        if existing:
            session.add(DataImportJob(
                id=job_id, dataset_version_id=existing.id, source_hash=source_hash,
                status="skipped_existing", rows_read=len(frame), rows_inserted=0,
                rows_skipped=len(frame), finished_at=datetime.now(timezone.utc),
            ))
            return {"status": "skipped_existing", "job_id": job_id, "dataset_version_id": existing.id,
                    "version_hash": source_hash, "rows": existing.row_count}
        dataset = session.scalar(select(Dataset).where(Dataset.dataset_ref == DATASET_REF))
        if dataset is None:
            dataset = Dataset(dataset_ref=DATASET_REF, name=stats["name"], source_url=SOURCE_URL, license="CC0: Public Domain")
            session.add(dataset)
            session.flush()
        version = DatasetVersion(
            dataset_id=dataset.id, version_hash=source_hash, source_file=str(path),
            row_count=len(frame), metadata_json=stats,
        )
        session.add(version)
        session.flush()
        job = DataImportJob(id=job_id, dataset_version_id=version.id, source_hash=source_hash,
                            status="running", rows_read=len(frame))
        session.add(job)
        records = []
        for row in frame.itertuples(index=False):
            records.append({
                "dataset_version_id": version.id, "source_row": int(row.source_row),
                "age": float(row.age), "gender": str(row.gender),
                "hypertension": bool(row.hypertension), "heart_disease": bool(row.heart_disease),
                "smoking_history": str(row.smoking_history), "bmi": float(row.bmi),
                "hba1c_level": float(row.hba1c_level),
                "blood_glucose_level": float(row.blood_glucose_level),
                "target": int(row.target), "feature_group": str(row.group),
            })
            if len(records) >= batch_size:
                session.bulk_insert_mappings(DatasetRecord, records)
                records.clear()
        if records:
            session.bulk_insert_mappings(DatasetRecord, records)
        job.status = "completed"
        job.rows_inserted = len(frame)
        job.finished_at = datetime.now(timezone.utc)
        return {"status": "completed", "job_id": job_id, "dataset_version_id": version.id,
                "version_hash": source_hash, "rows": len(frame)}


def load_dataset_from_database(store: DemoStore, version_hash: str | None = None) -> tuple[pd.DataFrame, dict]:
    with store.sessions() as session:
        query = select(DatasetVersion).order_by(DatasetVersion.imported_at.desc())
        if version_hash:
            query = select(DatasetVersion).where(DatasetVersion.version_hash == version_hash)
        version = session.scalar(query.limit(1))
        if version is None:
            raise RuntimeError("MySQL chưa có dataset. Chạy: python scripts/import_dataset_mysql.py")
        rows = session.execute(
            select(
                DatasetRecord.source_row, DatasetRecord.age, DatasetRecord.gender,
                DatasetRecord.hypertension, DatasetRecord.heart_disease, DatasetRecord.smoking_history,
                DatasetRecord.bmi, DatasetRecord.hba1c_level, DatasetRecord.blood_glucose_level,
                DatasetRecord.target, DatasetRecord.feature_group,
            ).where(DatasetRecord.dataset_version_id == version.id).order_by(DatasetRecord.source_row)
        ).all()
        frame = pd.DataFrame(rows, columns=[
            "source_row", "age", "gender", "hypertension", "heart_disease", "smoking_history",
            "bmi", "hba1c_level", "blood_glucose_level", "target", "group",
        ])
        frame["hypertension"] = frame["hypertension"].astype(int)
        frame["heart_disease"] = frame["heart_disease"].astype(int)
        return frame, {**version.metadata_json, "dataset_version_id": version.id, "version_hash": version.version_hash}
