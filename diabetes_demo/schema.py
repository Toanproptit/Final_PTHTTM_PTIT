"""Schema Kaggle: 8 đầu vào có cấu trúc; loại nhãn và clinical_notes khỏi X."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool

FEATURES = ["age", "gender", "hypertension", "heart_disease", "smoking_history", "bmi", "hba1c_level", "blood_glucose_level"]
NUMERIC_FEATURES = ["age", "bmi", "hba1c_level", "blood_glucose_level"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORY_VALUES = {"gender": ["Female", "Male", "Other"], "smoking_history": ["No Info", "current", "ever", "former", "never", "not current"]}
LABELS = {
    "age": "Tuổi (năm, có thể là số thập phân)", "gender": "Giới tính ghi trong dataset",
    "hypertension": "Tiền sử tăng huyết áp", "heart_disease": "Tiền sử bệnh tim",
    "smoking_history": "Lịch sử hút thuốc", "bmi": "BMI (kg/m²)",
    "hba1c_level": "HbA1c (%)", "blood_glucose_level": "Glucose máu (mg/dL)",
}
CSV_COLUMNS = {"hbA1c_level": "hba1c_level", "diabetes": "target"}
EXCLUDED_COLUMNS = ["year", "location", "race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other", "clinical_notes"]
LIMITATION = (
    "Đây là phân loại nhãn diabetes của dataset Kaggle, sử dụng cả HbA1c và glucose; nguồn không mô tả khoảng thời gian tiên lượng. "
    "Nguồn gốc thu thập và cách tạo nhãn chưa được mô tả đầy đủ; chưa có kiểm định lâm sàng hoặc hiệu chuẩn "
    "cho quần thể Việt Nam. Điểm mô hình không phải xác suất mắc bệnh trong tương lai và không thay thế khám, "
    "xét nghiệm hay tư vấn chuyên môn. Không dùng kết quả để tự chẩn đoán hoặc thay đổi điều trị."
)

class PatientInput(BaseModel):
    """Mọi trường bắt buộc. Khoảng kiểm tra phần mềm không phải ngưỡng bệnh."""
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=0, le=120, strict=True, allow_inf_nan=False)
    gender: Literal["Female", "Male", "Other"]
    hypertension: StrictBool
    heart_disease: StrictBool
    smoking_history: Literal["No Info", "current", "ever", "former", "never", "not current"]
    bmi: float = Field(gt=0, le=150, strict=True, allow_inf_nan=False)
    hba1c_level: float = Field(gt=0, le=25, strict=True, allow_inf_nan=False)
    blood_glucose_level: float = Field(gt=0, le=1000, strict=True, allow_inf_nan=False)

class QuestionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=1000)

class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class AgentChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)

class PredictionAdviceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1, max_length=64)

DEMO_PATIENT = {"age": 45.0, "gender": "Male", "hypertension": False, "heart_disease": False,
                "smoking_history": "never", "bmi": 27.5, "hba1c_level": 5.8, "blood_glucose_level": 130.0}
