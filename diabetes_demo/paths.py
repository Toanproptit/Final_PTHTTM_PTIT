"""Đường dẫn tuyệt đối tính từ project, không phụ thuộc thư mục chạy lệnh."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "kaggle" / "diabetes_dataset_with_notes.csv"
ARTIFACTS = ROOT / "artifacts"
ML = ARTIFACTS / "ml"
RL = ARTIFACTS / "rl"
KB = ROOT / "knowledge_base"
WEB = ROOT / "web"
RUNTIME = ROOT / "runtime"
