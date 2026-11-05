# Diabetes Agent Lab

## Kiến trúc v3: LangGraph + LangChain + MySQL + Neo4j

Phiên bản hiện tại dùng **LangGraph** để biên dịch state graph của luồng dự đoán. `OrchestratorAgent` giữ `AgentState`, ghi lý do routing và graph chọn/bỏ qua agent theo trạng thái đầu vào:

```text
Patient profile
  -> ValidationAgent
  -> SafetyAgent
  -> PredictionAgent (chỉ khi nằm trong miền huấn luyện)
  -> KnowledgeAgent (khi cảnh báo hoặc nhãn dương cần bằng chứng)
  -> ExplanationAgent

Natural-language chat
  -> LangChain ChatOpenRouter lập kế hoạch
  -> gọi một hoặc nhiều LangChain StructuredTool trong allowlist
  -> tool dự đoán chạy state graph LangGraph ở trên
  -> lưu execution, message và tool call để audit
```

- MySQL là nguồn dữ liệu quan hệ chính khi chạy Docker: dataset/version, bản ghi, training/model/metric, agent execution/step/message/tool call và retrieval log.
- Neo4j lưu `Entity`, `Fact`, `SourceDocument`, `DocumentChunk`, quan hệ provenance và vector index. Khi Neo4j chưa sẵn sàng, KnowledgeAgent ghi trạng thái rõ ràng và fallback về TF-IDF cục bộ.
- React/Vite hiển thị Safety Agent, trạng thái storage và trace routing. FastAPI vẫn phục vụ cùng một URL để demo đơn giản.
- SQLite chỉ còn dùng cho test/local fallback và được sao lưu trước khi script migration sao chép lịch sử cũ sang MySQL.

Chạy toàn bộ stack:

```powershell
Copy-Item .env.example .env  # chỉ khi chưa có .env; sau đó thay mật khẩu/key
docker compose up --build -d
docker compose ps
```

Mở http://127.0.0.1:8000, Neo4j Browser ở http://127.0.0.1:7474 và API docs ở http://127.0.0.1:8000/docs. Lần đầu có thể lâu hơn vì hệ thống nhập 100.000 dòng và tải embedding model. Xem chi tiết tại [docs/05_PRODUCTION_ARCHITECTURE.md](docs/05_PRODUCTION_ARCHITECTURE.md).

Demo LLM tự chọn tool qua OpenRouter: xem hướng dẫn cấu hình tại [docs/LLM_AGENT_SETUP.md](docs/LLM_AGENT_SETUP.md).

Đồ án **hệ thống multi-agent hỗ trợ tìm hiểu bài toán tiểu đường**: phân loại nhãn trên dataset Kaggle, kiểm tra đầu vào bằng luật, truy xuất thông tin WHO/CDC và theo dõi workflow. Module Q-learning kế thừa trực tiếp code AGENT02, hoạt động trong mô phỏng thương mại điện tử độc lập.

Dataset hiện tại do người dùng chỉ định: [ziya07/diabetes-clinical-dataset100k-rows](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows), 100.000 dòng, CC0 theo metadata Kaggle. Nguồn không mô tả đầy đủ quy trình thu thập/tạo nhãn; không khẳng định dữ liệu là bệnh án thật hoặc synthetic. Mô hình có HbA1c/glucose làm đầu vào và **không phải mô hình tiên lượng mắc bệnh tương lai đã kiểm định**. Chỉ dùng hồ sơ giả định cho demo, không dùng để chẩn đoán hoặc thay đổi điều trị.

## Chạy demo đã có artifact trên máy hiện tại

Mở terminal trong thư mục project:

```powershell
.\.venv\Scripts\python.exe -m uvicorn diabetes_demo.api:app --host 127.0.0.1 --port 8000
```

Mở **http://127.0.0.1:8000**. Có thể nhấp đúp `run_demo.bat`. Dừng bằng Ctrl+C. File `index.html` ở thư mục gốc là demo JavaScript thương mại cũ; **giao diện tiểu đường là `web/index.html` và cần server Python**. API docs: http://127.0.0.1:8000/docs.

## Cài từ đầu trên Windows

Python 3.11 trở lên; lần chạy được kiểm chứng dùng Python 3.12.10 trên Windows, CPU. Các workflow nghiên cứu cơ bản không cần GPU, Kaggle token, Ollama hay API trả phí; tab LLM Agent tùy chọn cần OpenRouter API key và có thể phát sinh chi phí theo model.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m diabetes_demo.data
.\.venv\Scripts\python.exe -m diabetes_demo.train
.\.venv\Scripts\python.exe -m diabetes_demo.rl_experiments
.\.venv\Scripts\python.exe -m diabetes_demo.evaluate
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/test_results.xml
.\.venv\Scripts\python.exe scripts\export_report.py
.\.venv\Scripts\python.exe -m uvicorn diabetes_demo.api:app --host 127.0.0.1 --port 8000
```

`requirements-lock.txt` khóa môi trường đã chạy; `requirements.txt` mô tả dải dependency nếu cần giải quyết lại trên nền tảng khác. Nếu dùng artifact joblib đã cung cấp, ưu tiên phiên bản khóa tương ứng. Không nạp checkpoint/joblib từ nguồn lạ. Nếu cổng 8000 đang bận, dừng server cũ hoặc đổi `--port 8001` và dùng đúng URL.

CSV và metadata Kaggle đã nằm trong `data/kaggle/`, nên chạy offline được sau khi cài dependency. `diabetes_demo.data` tái dùng file hiện có và tạo manifest. Nếu Kaggle chặn tải công khai ở máy khác, tải ZIP từ trang nguồn, đặt `diabetes_dataset_with_notes.csv` và metadata ở `data/kaggle/` theo hướng dẫn `docs/03_DATASET.md`; không cần đưa token vào mã nguồn.

Chạy lại mọi thực nghiệm bằng một lệnh (vẫn cần chạy pytest riêng):

```powershell
.\.venv\Scripts\python.exe run_experiments.py
```

`--skip-rl` giữ kết quả RL hiện có khi chỉ thay pipeline phân loại/QA. Thực nghiệm hoàn tất rồi mới khởi động/restart server để nạp model mới; không ghi đè model đang dùng giữa các request.

Frontend React source nằm trong `frontend/`; bundle đã build nằm trong `web/` để FastAPI phục vụ trực tiếp. Khi chỉnh source React, chạy `npm.cmd install` rồi `npm.cmd run build`. Node.js không cần để mở bản demo đã build. Khi server đang mở, `python scripts/smoke_http.py` kiểm tra HTTP/API/ảnh thật và lưu `artifacts/http_smoke.json`; script dùng hồ sơ giả định và tạo hai mục lịch sử demo.

## Chức năng và kết quả đã chạy

- 6 mục: Tổng quan, Dự đoán nghiên cứu, Kho tri thức, LLM Agent, Workflow, Thực nghiệm.
- Dự đoán dùng 8 trường, one-hot thành vector 15 chiều; PyTorch MLP và Logistic Regression được nạp từ artifact.
- 96.146 dòng sau bỏ trùng theo 8 X + y; 91 nhóm X có nhãn mâu thuẫn được giữ trong cùng tập. Train/validation/test = 57.688/19.229/19.229.
- MLP test: Accuracy 97,01%, **Recall 66,16%**, F1 79,63%, ROC-AUC 97,65%. Majority baseline đã đạt Accuracy 91,18% nhưng Recall 0%; phải đọc nhiều chỉ số.
- RL đánh giá riêng, không cập nhật Q-table/khám phá. Trên 5 seed, RL và rule có reward gần nhau; không có bằng chứng RL luôn tốt hơn.
- Hỏi đáp tri thức cơ bản vẫn dùng retrieval-based QA với 5 tài liệu và nguồn, không gọi LLM. Tab LLM Agent là demo tùy chọn qua OpenRouter; cần cấu hình key/model. Bộ retrieval hiện có 14 câu do tác giả soạn: Hit@1 10/11, Hit@3 11/11, từ chối đúng 3/3 câu ngoài phạm vi; bộ này nhỏ và không độc lập với người biên soạn KB.
- SQLite lưu lịch sử/trace cục bộ trong `runtime/`; nút xóa lịch sử ở Workflow. Dữ liệu test dùng database tạm riêng.

Kết quả đầy đủ, đường dẫn hình/CSV và giới hạn được sinh trong `docs/report/03_RESULTS.md`. Báo cáo test: `artifacts/test_results.xml`. Trạng thái kiểm tra giao diện trực quan được ghi ở `docs/04_VERIFICATION.md`.

## Cấu trúc project

```text
ASIGNMENT-AGENT_02/
├── README.md
├── requirements.txt / requirements-lock.txt
├── run_demo.bat / run_demo.ps1
├── run_experiments.py
├── agent02_code.tex                 # Python gốc người dùng cung cấp
├── index.html                       # Demo thương mại JavaScript ban đầu
├── diabetes_demo/
│   ├── paths.py / schema.py          # Đường dẫn và hợp đồng dữ liệu
│   ├── data.py / model.py / train.py # Kaggle -> preprocessing -> MLP/baseline
│   ├── agents.py / knowledge.py     # Agent chuyên trách
│   ├── orchestrator.py / storage.py # Workflow và SQLite
│   ├── api.py                       # FastAPI + phục vụ UI
│   ├── rl_experiments.py            # Gọi code Python AGENT02 thật
│   └── evaluate.py                  # Retrieval + workflow evaluation
├── legacy/agent02_experiment.py     # Bản sao byte-for-byte từ .tex
├── web/index.html / style.css / app.js
├── data/
│   ├── kaggle/                      # CSV, ZIP, metadata nguồn hiện tại
│   ├── source_manifest.json
│   └── raw/                         # CSV UCI thử ban đầu, không dùng hiện tại
├── knowledge_base/                  # 5 Markdown, manifest và câu hỏi đánh giá
├── artifacts/
│   ├── ml/                          # Model, preprocessing, split, metrics, PNG
│   ├── rl/                          # CSV/PNG/Q-table và hash nguồn
│   ├── retrieval/                   # Kết quả QA
│   ├── workflow_evaluation.json
│   ├── test_results.xml
│   └── archive_uci/                 # Kết quả cũ, không trộn vào báo cáo Kaggle
├── tests/                          # Unit + integration tests
├── scripts/                        # Kiểm tra nguồn và xuất báo cáo kết quả
└── docs/
    ├── 01_REQUIREMENTS.md
    ├── 02_ARCHITECTURE.md
    ├── 03_DATASET.md
    ├── 04_VERIFICATION.md
    ├── DEFENSE_QA.md
    ├── learning/                    # 8 bài hướng dẫn đọc code
    └── report/                      # Đề cương, nội dung, kết quả và kịch bản demo
```

## Kịch bản demo 5–7 phút

1. Tổng quan: nêu dataset Kaggle, mục tiêu phân loại nghiên cứu và các agent.
2. Dự đoán: nhấn **Điền hồ sơ giả định**, rồi **Chạy các agent**. Giải thích nhãn dataset, điểm sigmoid và giới hạn.
3. Workflow: mở từng input/output theo thứ tự Rule → Prediction → Knowledge → Explanation.
4. Thử tuổi 100 để thấy Rule chặn ngoài miền; thử bỏ trống trường để thấy validation.
5. Hỏi đáp: “HbA1c phản ánh điều gì?”; chỉ nguồn CDC. Thử “Giá cổ phiếu hôm nay?” hoặc yêu cầu liều thuốc để thấy giới hạn.
6. Thực nghiệm: đọc Recall/F1 cùng Accuracy, confusion matrix và majority baseline. Mở RL, chỉ ra reward lợi nhuận và kết quả nhiều seed.
7. Chỉ đường dẫn model, CSV và test report để chứng minh đây là artifact đã chạy thật.

## Đọc code theo thứ tự

Đọc `docs/learning/01_PROJECT_OVERVIEW.md` đến `08_READING_ORDER.md`. Sau đó theo chuỗi `schema.py → data.py → model.py → train.py → agents.py → knowledge.py → orchestrator.py → storage.py → api.py → web/app.js`, rồi đọc module RL gốc. Câu hỏi bảo vệ ở `docs/DEFENSE_QA.md`.

Chưa triển khai: FAISS, dự báo dọc theo thời gian, kiểm định ngoài dataset/hiệu chuẩn/fairness đầy đủ, tích hợp bệnh viện và triển khai công khai. LLM Agent tùy chọn dùng LangChain + OpenRouter; luồng hồ sơ dùng LangGraph với các nhánh điều kiện có trace và giới hạn bước.
