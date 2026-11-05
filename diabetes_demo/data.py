"""Dataset Kaggle do người dùng chọn: download, kiểm tra, chống leakage và tiền xử lý."""
from __future__ import annotations
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .paths import DATA, RAW
from .schema import BINARY_FEATURES, CATEGORY_VALUES, CSV_COLUMNS, EXCLUDED_COLUMNS, FEATURES, NUMERIC_FEATURES

DATASET_REF = "ziya07/diabetes-clinical-dataset100k-rows"
SOURCE_URL = f"https://www.kaggle.com/datasets/{DATASET_REF}"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"

def download_dataset() -> Path:
    """Tải API public; chỉ đọc đúng CSV trong ZIP. Không lưu token/đăng nhập."""
    RAW.parent.mkdir(parents=True, exist_ok=True)
    def fetch(url: str, limit: int) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "StudentResearchDemo/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read(limit + 1)
        if len(payload) > limit:
            raise ValueError("File lớn hơn giới hạn tải; cần kiểm tra phiên bản dataset")
        return payload
    metadata_path = RAW.parent / "metadata.json"
    if not metadata_path.exists():
        metadata_path.write_bytes(fetch(f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_REF}", 5_000_000))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not RAW.exists():
        archive_path = RAW.parent / "dataset.zip"
        if not archive_path.exists():
            archive_path.write_bytes(fetch(DOWNLOAD_URL, 100_000_000))
        with zipfile.ZipFile(archive_path) as archive:
            name = next(member for member in archive.namelist() if Path(member).name == RAW.name)
            payload = archive.read(name)
        candidate = pd.read_csv(io.BytesIO(payload)).rename(columns=CSV_COLUMNS)
        if not set(FEATURES + ["target"]).issubset(candidate.columns):
            raise ValueError("Dataset tải về không khớp schema dự kiến")
        RAW.write_bytes(payload)
    manifest = {
        "source": SOURCE_URL, "download_url": DOWNLOAD_URL, "dataset_ref": DATASET_REF,
        "title": metadata.get("title"), "license": metadata.get("licenseName"),
        "kaggle_id": metadata.get("id"), "last_updated": metadata.get("lastUpdated"),
        "accessed_date": "2026-11-03", "manifest_created_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(), "file": str(RAW.relative_to(DATA)),
        "provenance": "Uploader does not document original collection, institution, patient IDs or label generation. Synthetic/real status not independently verified.",
        "included_features": FEATURES, "excluded_columns": EXCLUDED_COLUMNS,
    }
    (DATA / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return RAW

def encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa kiểu, kiểm tra category. One-hot được fit sau khi chia tập."""
    output = frame[FEATURES].copy()
    for column in NUMERIC_FEATURES + BINARY_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
        if not np.isfinite(output[column].dropna()).all():
            raise ValueError(f"Giá trị không hữu hạn ở cột {column}")
    for column in BINARY_FEATURES:
        if not output[column].dropna().isin([0, 1]).all():
            raise ValueError(f"Cột {column} phải là 0/1")
    for column, categories in CATEGORY_VALUES.items():
        if not output[column].dropna().isin(categories).all():
            raise ValueError(f"Giá trị ngoài schema ở cột {column}")
    return output

def load_dataset(path: Path = RAW) -> tuple[pd.DataFrame, dict]:
    """Deduplicate theo X được học + y; nhóm cùng X kể cả khi khác y."""
    raw = pd.read_csv(path).rename(columns=CSV_COLUMNS)
    if not set(FEATURES + ["target"]).issubset(raw.columns):
        raise ValueError("Dataset thiếu cột bắt buộc")
    selected = raw[FEATURES + ["target"]]
    cleaned = selected.drop_duplicates().reset_index(names="source_row")
    if not cleaned.target.isin([0, 1]).all():
        raise ValueError("Nhãn phải là 0 hoặc 1")
    cleaned["target"] = cleaned.target.astype(int)
    encoded = encode_features(cleaned)
    cleaned["group"] = encoded.fillna("<missing>").astype(str).agg("|".join, axis=1).map(
        lambda text: hashlib.sha256(text.encode()).hexdigest())
    stats = {
        "source": SOURCE_URL, "name": "Kaggle Diabetes Clinical Dataset (100k rows)",
        "raw_rows": len(raw), "raw_columns": len(raw.columns), "unique_rows": len(cleaned),
        "exact_duplicates_all_columns": int(raw.duplicated().sum()),
        "exact_duplicates_removed": len(raw) - len(cleaned),
        "deduplication_basis": "8 selected X + target; excluded notes/demographic extensions do not distinguish model observations",
        "unique_feature_groups": int(cleaned.group.nunique()),
        "conflicting_groups": int((cleaned.groupby("group").target.nunique() > 1).sum()),
        "missing_cells": int(selected.isna().sum().sum()), "excluded_columns": EXCLUDED_COLUMNS,
        "observed_age_min": float(raw.age.min()), "observed_age_max": float(raw.age.max()),
        "numeric_ranges": {name: {"min": float(cleaned[name].min()), "max": float(cleaned[name].max())} for name in NUMERIC_FEATURES},
        "category_counts": {name: {str(k): int(v) for k, v in cleaned[name].value_counts().items()} for name in CATEGORY_VALUES},
        "class_counts": {str(k): int(v) for k, v in cleaned.target.value_counts().items()},
        "provenance_verified": False,
    }
    return cleaned, stats

def split_dataset(frame: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """~60/20/20, giữ mọi X trùng cùng tập; nhãn chỉ dùng để stratify."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    remaining, test = next(outer.split(frame[FEATURES], frame.target, groups=frame.group))
    subset = frame.iloc[remaining]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(inner.split(subset[FEATURES], subset.target, groups=subset.group))
    return {"train": remaining[train_local], "validation": remaining[val_local], "test": test}

def make_preprocessor() -> ColumnTransformer:
    """Median+scaler cho số/binary, one-hot cho category; fit duy nhất train."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(categories=list(CATEGORY_VALUES.values()), sparse_output=False, handle_unknown="error")),
    ])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES + BINARY_FEATURES),
                              ("categorical", categorical, list(CATEGORY_VALUES))])

if __name__ == "__main__":
    print(download_dataset())
