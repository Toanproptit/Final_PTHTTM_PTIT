"""Specialised agents used by the deterministic multi-agent orchestrator."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import torch

from .data import encode_features
from .model import DiabetesMLP
from .paths import ML
from .schema import BINARY_FEATURES, FEATURES, LABELS, LIMITATION, PatientInput


class ValidationAgent:
    """Validate the model domain after Pydantic validates the API schema."""

    def __init__(self, numeric_ranges: dict | None = None):
        self.numeric_ranges = numeric_ranges or {
            "age": {"min": 0.08, "max": 80.0},
            "bmi": {"min": 10.01, "max": 95.69},
            "hba1c_level": {"min": 3.5, "max": 9.0},
            "blood_glucose_level": {"min": 80, "max": 300},
        }

    def run(self, patient: PatientInput) -> dict:
        outside = [
            name for name, limits in self.numeric_ranges.items()
            if not limits["min"] <= getattr(patient, name) <= limits["max"]
        ]
        warnings: list[str] = []
        if outside:
            warnings.append(
                "Ngoài khoảng quan sát của dữ liệu huấn luyện: "
                + ", ".join(LABELS[name] for name in outside)
                + "; hệ thống sẽ không chạy mô hình cho hồ sơ này."
            )
        if patient.smoking_history == "No Info":
            warnings.append("No Info là thiếu thông tin hút thuốc, không đồng nghĩa với chưa từng hút.")
        if patient.gender == "Other":
            warnings.append("Nhóm Other có rất ít mẫu trong dữ liệu gốc nên độ tin cậy theo nhóm chưa được xác lập.")
        reported = [LABELS[name] for name in BINARY_FEATURES if getattr(patient, name)]
        return {
            "valid_schema": True,
            "within_training_domain": not outside,
            "within_training_age": "age" not in outside,
            "outside_features": outside,
            "matched_rules": [
                "V1_REQUIRED_FIELDS_VALID",
                "V2_IN_DOMAIN" if not outside else "V2_OUTSIDE_DOMAIN",
                "V3_NO_INFO_DISTINCT_FROM_NEVER",
            ],
            "reported_conditions": reported,
            "warnings": warnings,
        }


RuleBasedAgent = ValidationAgent


class SafetyAgent:
    """Independent safety gate; it never turns a threshold into a diagnosis."""

    TESTING_SOURCE = {
        "title": "Diabetes Testing", "publisher": "CDC",
        "url": "https://www.cdc.gov/diabetes/diabetes-testing/",
    }
    DKA_SOURCE = {
        "title": "Diabetic Ketoacidosis", "publisher": "CDC",
        "url": "https://www.cdc.gov/diabetes/about/diabetic-ketoacidosis.html",
    }

    def run(self, patient: PatientInput) -> dict:
        level = "routine"
        rules: list[str] = []
        messages: list[str] = []
        citations = [self.TESTING_SOURCE]

        if patient.hba1c_level >= 6.5:
            level = "clinical_review"
            rules.append("S_HBA1C_GE_6_5")
            messages.append(
                "HbA1c đã nhập nằm trong khoảng CDC dùng cho xét nghiệm đái tháo đường. "
                "Một giá trị trong bản demo không tự xác nhận chẩn đoán; nên trao đổi với nhân viên y tế."
            )
        elif patient.hba1c_level >= 5.7:
            level = "follow_up"
            rules.append("S_HBA1C_5_7_TO_6_4")
            messages.append(
                "HbA1c đã nhập nằm trong khoảng CDC mô tả cho tiền đái tháo đường; "
                "đây là thông tin xét nghiệm chung, không phải kết luận cá nhân."
            )

        if patient.blood_glucose_level >= 300:
            level = "urgent_review"
            rules.append("S_GLUCOSE_GE_300")
            citations.append(self.DKA_SOURCE)
            messages.append(
                "CDC khuyến nghị cấp cứu khi đường huyết duy trì từ 300 mg/dL trở lên, "
                "hoặc có dấu hiệu như nôn, khó thở, hơi thở mùi trái cây hay nhiều triệu chứng DKA. "
                "Hệ thống không biết giá trị này có duy trì hay có triệu chứng kèm theo hay không."
            )
        elif patient.blood_glucose_level >= 250:
            if level in {"routine", "follow_up"}:
                level = "clinical_review"
            rules.append("S_GLUCOSE_GE_250")
            citations.append(self.DKA_SOURCE)
            messages.append(
                "Nếu người đã mắc đái tháo đường đang bị ốm hoặc đường huyết từ 250 mg/dL, "
                "CDC hướng dẫn kiểm tra lại đường huyết và ketone theo kế hoạch chăm sóc."
            )

        if not messages:
            messages.append(
                "Không kích hoạt quy tắc cảnh báo xét nghiệm hiện tại. "
                "Điều này không chứng minh rằng một người không có bệnh."
            )
        return {
            "level": level,
            "requires_knowledge": level != "routine",
            "matched_rules": rules or ["S_NO_THRESHOLD_TRIGGERED"],
            "messages": messages,
            "citations": citations,
            "disclaimer": "SafetyAgent chỉ phân luồng cảnh báo, không chẩn đoán hoặc kê đơn.",
        }


class PredictionAgent:
    """Load a trained local artifact; never train during an API request."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.baseline = None
        self.metadata = None

    @property
    def ready(self) -> bool:
        required = ["mlp.pt", "preprocessor.joblib", "logistic.joblib", "evaluation.json"]
        if not all((ML / name).is_file() for name in required):
            return False
        return json.loads((ML / "evaluation.json").read_text(encoding="utf-8"))["metadata"]["features"] == FEATURES

    def load(self) -> None:
        if not self.ready:
            raise FileNotFoundError("Chưa có artifact. Chạy: python -m diabetes_demo.train")
        self.metadata = json.loads((ML / "evaluation.json").read_text(encoding="utf-8"))["metadata"]
        self.preprocessor = joblib.load(ML / "preprocessor.joblib")
        self.baseline = joblib.load(ML / "logistic.joblib")
        self.model = DiabetesMLP(self.metadata["input_dim"])
        self.model.load_state_dict(torch.load(ML / "mlp.pt", map_location="cpu", weights_only=True))
        self.model.eval()

    def run(self, patient: PatientInput) -> dict:
        if self.model is None:
            self.load()
        encoded = encode_features(pd.DataFrame([patient.model_dump()]))
        inputs = torch.tensor(self.preprocessor.transform(encoded), dtype=torch.float32)
        with torch.inference_mode():
            score = float(torch.sigmoid(self.model(inputs))[0])
        return {
            "model": "MLP",
            "positive_class_score": round(score, 6),
            "model_version": self.metadata.get("model_version", "legacy-artifact"),
            "predicted_dataset_label": "diabetes = 1" if score >= 0.5 else "diabetes = 0",
            "threshold": 0.5,
            "baseline_positive_class_score": round(float(self.baseline.predict_proba(inputs.numpy())[0, 1]), 6),
            "interpretation": "Điểm sigmoid cho nhãn diabetes = 1 của dataset; không phải xác suất mắc bệnh cá nhân.",
            "limitation": LIMITATION,
        }


class ExplanationAgent:
    """Create a traceable report from prior agent outputs."""

    def run(self, prediction: dict | None, validation: dict, safety: dict, knowledge: dict) -> dict:
        if prediction is None:
            model_text = "Không chạy mô hình vì đầu vào nằm ngoài miền số đã quan sát trong dữ liệu huấn luyện."
        else:
            model_text = (
                f"MLP phân loại hồ sơ giả định vào nhãn {prediction['predicted_dataset_label']} "
                "với ngưỡng kỹ thuật 0.5. Kết quả này không phải chẩn đoán y khoa."
            )
        return {
            "model_explanation": model_text,
            "input_summary": validation["reported_conditions"],
            "input_summary_note": "Đây là dữ liệu đã nhập, không phải giải thích quan hệ nhân quả của mô hình.",
            "safety_level": safety["level"],
            "safety_messages": safety["messages"],
            "general_information": knowledge.get("evidence", []) if knowledge else [],
            "rule_messages": validation["warnings"],
            "limitation": LIMITATION,
        }
