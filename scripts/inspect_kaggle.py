"""Tải metadata và bản ZIP công khai từ Kaggle, không cần token nếu nguồn cho phép."""
from pathlib import Path
import json
import urllib.request
import zipfile
import io
import sys

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1]
destination = root / "data" / "kaggle"
destination.mkdir(parents=True, exist_ok=True)
slug = "ziya07/diabetes-clinical-dataset100k-rows"

def fetch(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "StudentResearchDemo/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Download exceeded configured limit")
    return content

metadata = fetch(f"https://www.kaggle.com/api/v1/datasets/view/{slug}", 5_000_000)
parsed = json.loads(metadata)
(destination / "metadata.json").write_bytes(metadata)
print(json.dumps({key: value for key, value in parsed.items() if key in [
    "title", "subtitle", "description", "licenses", "licenseName", "datasetVersionNumber", "files", "tags", "lastUpdated", "id", "ref"
]}, ensure_ascii=False, indent=2))
archive_path = destination / "dataset.zip"
if not archive_path.exists():
    archive_path.write_bytes(fetch(f"https://www.kaggle.com/api/v1/datasets/download/{slug}", 100_000_000))
with zipfile.ZipFile(archive_path) as archive:
    print("Archive members:", archive.namelist())
    for name in archive.namelist():
        if name.lower().endswith(".csv"):
            content = archive.read(name)
            target = destination / Path(name).name
            target.write_bytes(content)
            import pandas as pd
            frame = pd.read_csv(io.BytesIO(content))
            print("CSV:", target.name, "shape:", frame.shape)
            print("Columns:", list(frame.columns))
            print("Types:", frame.dtypes.to_dict())
            print("Missing:", frame.isna().sum().to_dict())
            for column in frame.columns:
                if frame[column].nunique() <= 15:
                    print(column, frame[column].value_counts(dropna=False).to_dict())
