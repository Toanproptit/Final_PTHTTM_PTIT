# 03. Các agent nhận gì, quyết định gì và trả gì?

## OrchestratorAgent (`orchestrator.py`)

Constructor nạp `PredictionAgent`, đọc khoảng số từ artifact train để tạo Rule Agent, tạo Knowledge/Explanation và nhận một `DemoStore`. `predict(patient)` là workflow chính. `ask(question)` là workflow QA.

`call(trace, agent, inputs, action)` đo bằng `perf_counter`, gọi hàm và thêm một dictionary vào trace: order, agent, status, input, output, duration_ms. Nếu lỗi, nó ghi trạng thái error rồi phát lại exception; API xử lý trường hợp thiếu model thành HTTP 503. Trace lỗi trong hàm chưa được lưu SQLite vì workflow chưa hoàn tất; đây là giới hạn logging của MVP.

Trong `predict`, output Rule quyết định có gọi Prediction hay ghi skipped. Knowledge nhận câu truy vấn kiến thức xét nghiệm cố định. Explanation nhận **chính các output vừa tính**, không gọi lại model. Cuối cùng `store.save` lưu result cùng trace. Thời gian các bước có thể thay đổi theo máy, không phải chỉ số chất lượng y khoa.

## RuleBasedAgent (`agents.py`)

Input là `PatientInput` đã được Pydantic kiểm tra. Các luật:

- Mọi trường đã đúng schema; không tự thêm thông tin thiếu.
- So 4 biến số với min/max của train; ngoài khoảng → `within_training_domain=False` và nêu `outside_features`.
- `No Info` là category thiếu thông tin hút thuốc, không coi là never.
- gender=Other có ít dữ liệu; cảnh báo hạn chế đại diện.
- Nhắc việc đọc xét nghiệm thuộc chuyên môn y tế; không đưa ngưỡng chẩn đoán.

`reported_conditions` chỉ liệt kê hai tiền sử người dùng chọn Có. Nó không là kết quả suy luận bệnh và không phải tầm quan trọng đặc trưng. Khác `RuleBasedAgent` cũ: cũ trả hành động giảm giá; mới trả quyết định hợp lệ/miền và thông báo.

## PredictionAgent (`agents.py`)

`ready` kiểm tra đủ file và danh sách features khớp schema. `load` đọc `preprocessor.joblib`, `logistic.joblib`, metadata và `mlp.pt`; dựng mạng với `input_dim` đã lưu, chuyển `eval()`.

`run` chuyển input sang DataFrame, gọi `encode_features`, sau đó **transform**, không fit lại. Tensor 1×15 qua mạng cho một logit; sigmoid thành score [0,1]. Ngưỡng 0.5 chọn nhãn diabetes=0/1. Logistic trả score đối chiếu. `torch.inference_mode()` tắt xây đồ thị gradient. Không có `learn()` ở agent này khi nhận request.

## KnowledgeAgent (`knowledge.py`)

Constructor đọc 5 Markdown và manifest, tạo chunks và TF-IDF index trong RAM. `run` kiểm tra câu hỏi có thuộc phạm vi tri thức chung không, tìm cosine similarity và trả các chunk vượt ngưỡng. Mỗi evidence giữ document_id/chunk_id/title/publisher/url/date/text/score. Không có LLM, không tự chế thêm nội dung ngoài chunk.

## ExplanationAgent (`agents.py`)

Input gồm prediction (có thể None), rules và knowledge. `run` tạo văn bản bằng template: diễn giải nhãn dataset; tóm tắt tiền sử đã nhập; giữ evidence có nguồn; ghi giới hạn. Không nói một đặc trưng “gây” điểm cao vì code không tính SHAP/causal attribution. Không đưa đơn thuốc hoặc kế hoạch điều trị.

## Agent Q-learning (module riêng)

`legacy.QLearningAgent` có Q-table, RNG, epsilon, alpha, gamma; quan sát môi trường, chọn hành động và cập nhật qua reward. Nó hoạt động ở một bài toán khác, giúp giải thích learning agent đối chiếu với fixed-rule agent. Không truyền reward hay policy thương mại vào Prediction/Explanation.

Để chứng minh phối hợp khi bảo vệ, mở Workflow và chỉ input của Explanation chứa output Rule/Prediction/Knowledge; đó là dữ liệu trao đổi thật, không phải sơ đồ được vẽ mà chưa có code.
