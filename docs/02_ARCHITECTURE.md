# 02. Kiến trúc và hợp đồng agent

> Cập nhật v3: luồng dự đoán chạy bằng LangGraph; tab LLM Agent dùng LangChain `ChatOpenRouter` và `StructuredTool`. Xem [hướng dẫn cấu hình](LLM_AGENT_SETUP.md).

## Kiến trúc triển khai thật

```mermaid
flowchart LR
  U[Người xem demo] --> UI[HTML/CSS/JavaScript · 5 tab]
  UI --> API[FastAPI · localhost:8000]
  API --> O[LangGraph · OrchestratorAgent]
  O --> R[RuleBasedAgent]
  O --> P[PredictionAgent]
  O --> K[KnowledgeAgent]
  O --> E[ExplanationAgent]
  P --> M[PyTorch MLP + preprocessing đã lưu]
  K --> KB[Neo4j GraphRAG + TF-IDF fallback]
  O --> DB[(MySQL · dữ liệu và audit)]
  API --> LC[LangChain · ChatOpenRouter]
  LC --> O
  T[train.py · offline] --> M
  Kaggle[Kaggle · ziya07 · 100k rows] --> T
  Q[rl_experiments.py · offline] --> L[AGENT02 Python gốc]
  L --> F[CSV/PNG/Q-table]
  API --> F
```

Các agent là component Python có mục tiêu và hợp đồng dữ liệu riêng. LangGraph điều phối state graph có nhánh điều kiện; LangChain cho LLM tự chọn một hoặc nhiều tool đã cho phép. Hệ thống không phải các agent phân tán hoặc tự thương lượng: quyền thực thi vẫn bị giới hạn bởi graph, schema và allowlist phía backend.

## Activity Diagram

```mermaid
flowchart TD
  A[Nhập hồ sơ giả định] --> B{Pydantic: đủ trường và đúng kiểu?}
  B -- Không --> C[HTTP 422 · không gọi model]
  B -- Có --> D[Rule: kiểm tra các khoảng số với miền train]
  D --> E{Đầu vào trong khoảng quan sát?}
  E -- Có --> F[Prediction: mã hóa và transform bằng pipeline train]
  F --> G[MLP eval + inference_mode]
  E -- Không --> H[Prediction skipped · không xuất điểm]
  G --> I[Knowledge: truy xuất thông tin triệu chứng/xét nghiệm]
  H --> I
  I --> J[Explanation: tổng hợp bằng template]
  J --> K[Lưu kết quả và trace vào MySQL]
  K --> L[Hiển thị điểm nghiên cứu, nguồn và giới hạn]
```

Nếu artifact chưa có, API dự đoán trả 503 kèm lệnh train; không thay bằng số ngẫu nhiên. Yêu cầu hỏi đáp đi từ Orchestrator thẳng đến Knowledge Agent; không cần gọi model phân loại.

## Sequence Diagram

```mermaid
sequenceDiagram
  actor User as Người xem demo
  participant UI as Web UI
  participant API as FastAPI/Pydantic
  participant O as Orchestrator
  participant R as Rule Agent
  participant P as Prediction Agent
  participant K as Knowledge Agent
  participant E as Explanation Agent
  participant DB as MySQL
  User->>UI: Điền hồ sơ giả định và gửi
  UI->>API: POST /api/predict (8 trường)
  API->>API: Validate schema
  API->>O: PatientInput
  O->>R: run(patient)
  R-->>O: within_training_domain, warnings, reported_conditions
  alt Các giá trị số trong miền train
    O->>P: run(patient)
    P-->>O: score, nhãn dataset, baseline score
  else Ngoài miền số quan sát
    O->>O: Ghi trace skipped
  end
  opt Safety hoặc nhãn yêu cầu bằng chứng
    O->>K: Truy vấn GraphRAG
    K-->>O: Các chunk, graph fact và nguồn
  end
  O->>E: prediction + rules + knowledge
  E-->>O: Giải thích và giới hạn
  O->>DB: save(run_id, result + trace)
  O-->>API: JSON result
  API-->>UI: JSON response
  UI-->>User: Kết quả + link workflow
```

## Hợp đồng dữ liệu

| Agent | Input | Output | Tài nguyên/quyết định |
|---|---|---|---|
| Orchestrator + LangGraph | `PatientInput` hoặc câu hỏi | `run_id`, kết quả, `trace` | State graph, conditional edge và MySQL |
| RuleBased | Hồ sơ đã đúng schema | `within_training_domain`, luật khớp, cảnh báo, tiền sử nhập Có | Khoảng số đọc từ train artifact; không chẩn đoán từ tiền sử |
| Prediction | 8 trường | Điểm sigmoid MLP, nhãn diabetes=0/1, điểm logistic, giới hạn | Pipeline fit trên train + checkpoint MLP |
| Knowledge | Câu hỏi | Status, mode, evidence chứa chunk/title/url/date/score | Markdown, TF-IDF, cosine, cổng phạm vi và ngưỡng |
| Explanation | Prediction + Rules + Knowledge | Giải thích mô hình, bản tóm tắt input, thông tin chung có nguồn | Template xác định, không LLM |
| QLearning (module riêng) | `(demand, competitor_previous_action)` | Hành động giảm/không giảm giá | Q-table được học trong môi trường thương mại giả lập |

## Endpoint và lưu trữ

- `GET /`, `/assets/*`: UI; `GET /docs`: tài liệu API tự động.
- `GET /api/health`, `/api/schema`: trạng thái và hợp đồng form.
- `POST /api/predict`, `/api/ask`: hai workflow.
- `GET /api/history`, `DELETE /api/history`: lịch sử cục bộ.
- `GET /api/experiments`: đọc artifact thật; `/api/artifact/{category}/{filename}` chỉ phục vụ CSV/PNG/JSON/TXT trong danh mục cho phép.
- Docker dùng MySQL làm system of record cho execution, step, message, tool call, retrieval và prediction. SQLite `runtime/demo.sqlite3` chỉ là fallback local/test. Không có tài khoản/định danh; chỉ dùng hồ sơ giả định.

## Giới hạn của RAG có sinh

LLM có thể gọi tool GraphRAG rồi tổng hợp câu trả lời, nhưng kết quả sinh vẫn cần kiểm tra hallucination, nguồn và câu hỏi ngoài phạm vi. Endpoint `/api/ask` tiếp tục trả retrieval có cấu trúc mà không cần LLM. FAISS, Streamlit, cloud hosting và hệ thống bệnh viện không thuộc triển khai hiện tại.
