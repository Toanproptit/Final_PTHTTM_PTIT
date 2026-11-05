# 01. Yêu cầu, phạm vi và giả định

Ngày lập: 02/11/2026; cập nhật dataset theo chỉ định của sinh viên 03/11/2026. Nguồn đầu vào: ảnh bảng, `agent02_code.tex`, yêu cầu đính kèm và link Kaggle của người dùng.

## 1. Đọc ảnh bảng

Nhìn rõ: tiêu đề Project về hệ đa agent cho bệnh tiểu đường; các mục giới thiệu, hệ đa agent/agent, kiến trúc tổng quan, các thành phần hệ thống, Deep Learning → Agent, KB + RAG, Code + Demo. Sơ đồ giữa có DB, Deep Learning, dự đoán/phân loại, KB, RAG, Agent, reinforcement learning và rule.

Chưa chắc chắn: chữ nhỏ/các chú thích và một số mũi tên trong sơ đồ giữa; nội dung đầy đủ ở bảng trái; ý nghĩa chính xác của biểu tượng đầu người. Không suy ra yêu cầu bắt buộc về framework, loại database, dataset hoặc LLM từ ảnh. Stack bên dưới là lựa chọn triển khai, không phải yêu cầu đã xác nhận của giảng viên.

## 2. Đối chiếu code cũ

`agent02_code.tex` chứa Python, không phải LaTeX. Nó mô phỏng hai cửa hàng chọn giảm giá, huấn luyện Q-learning tabular và đánh giá với rule cố định.

| Thành phần | Đã có trong AGENT02 | Hướng kế thừa |
|---|---|---|
| Q-learning, epsilon-greedy, terminal target | Có | Giữ thuật toán Python và dẫn nguồn file gốc |
| RuleBasedAgent | Luật giá | Giữ đối thủ gốc trong thí nghiệm; thêm agent kiểm tra dữ liệu cho bài toán mới |
| Train/evaluate riêng, nhiều seed, alpha/epsilon | Có | Chạy module độc lập và xuất kết quả thực |
| Môi trường thương mại điện tử | Có | Giữ nguyên ngữ nghĩa, không gán reward lợi nhuận thành giá trị y khoa |
| Deep Learning, dữ liệu bệnh tiểu đường | Chưa có | Xây pipeline dữ liệu công khai, MLP và logistic baseline |
| KB, truy xuất, explanation, orchestration, DB | Chưa có | Xây mới với trách nhiệm và dữ liệu vào/ra rõ ràng |
| Web demo | Có `index.html` viết lại bằng JS | Giữ làm demo AGENT02; tạo giao diện hệ thống mới dùng Python thật qua API |

## 3. MVP được chọn

- Một ứng dụng FastAPI phục vụ API và giao diện web 5 tab: tổng quan, dự đoán nghiên cứu, hỏi đáp tri thức, workflow, thực nghiệm.
- Orchestrator → kiểm tra luật → Prediction Agent → Knowledge Agent → Explanation Agent → SQLite. Giao tiếp bằng dictionary/JSON có cấu trúc; trace có input/output và thời gian.
- Dataset **Kaggle Diabetes Clinical Dataset (100k rows)** của `ziya07`, do người dùng chỉ định, giấy phép CC0 theo metadata Kaggle. Chọn 8 đầu vào có cấu trúc; bỏ clinical_notes, year, location và các cột race. UCI thử ban đầu được lưu ở `artifacts/archive_uci/`, không dùng cho mô hình/demo/báo cáo hiện tại.
- Tiền xử lý fit trên train; validation dùng chọn checkpoint MLP; test giữ riêng để so sánh MLP với Logistic Regression. Kiểm tra trùng bản ghi và chia theo nhóm đặc trưng trùng để tránh cùng hồ sơ xuất hiện ở các tập.
- MLP PyTorch nhỏ, chạy CPU. Lưu checkpoint, preprocessing, splits, metrics và biểu đồ. Kết quả UI đọc từ artifact đã chạy.
- KB gồm các bản diễn giải ngắn bằng tiếng Việt, tham chiếu nguồn y tế chính thức, có URL và ngày truy cập. TF-IDF tìm đoạn, trả lời bằng đoạn đã truy xuất; gọi đúng tên **retrieval-based QA**, không có mô hình sinh trong MVP.
- Q-learning là thí nghiệm độc lập kế thừa AGENT02 gốc. Không dùng RL để thay đổi quyết định/chỉ dẫn y khoa. Thử baseline, alpha, epsilon và nhiều seed với đánh giá tắt exploration.
- Unit/integration tests và kiểm thử từ đầu đến cuối; CSV/PNG và hướng dẫn tiếng Việt phục vụ học/bảo vệ.

## 4. Lựa chọn triển khai

Python >=3.11 (máy hiện tại 3.12), FastAPI + Uvicorn, HTML/CSS/JavaScript thuần, PyTorch, scikit-learn, pandas, NumPy, Matplotlib, SQLite, pytest. Một tiến trình server, không microservices và không API trả phí. Giao diện nằm trong `web/`; demo thương mại cũ vẫn ở `index.html`.

Không dùng Streamlit vì FastAPI đã có thể phục vụ cả API và giao diện; không cần hai server. Không dùng Ollama/FAISS ở MVP vì TF-IDF đủ cho KB nhỏ, dễ giải thích và không cần tải LLM. Hướng mở rộng RAG sinh được mô tả riêng, không tuyên bố đã triển khai.

## 5. Ràng buộc dữ liệu và cách diễn giải

Chỉ dùng dữ liệu nghiên cứu công khai và hồ sơ giả định để demo. Không thu thập thông tin định danh. SQLite lưu yêu cầu demo cục bộ; có chức năng xóa lịch sử. Các kiểm tra khoảng tuổi/giá trị là ràng buộc phần mềm và miền dữ liệu, không phải ngưỡng chẩn đoán. Không điều trị, kê đơn hoặc suy ra chẩn đoán từ điểm mô hình. Nguồn Kaggle không mô tả đầy đủ nơi thu thập, mã bệnh nhân, đơn vị và cách tạo nhãn; không tự nhận là hồ sơ bệnh viện thật hoặc là synthetic. HbA1c/glucose có trong X nên bài toán là phân loại nhãn, không chứng minh tiên lượng trước xét nghiệm.

Mọi số liệu báo cáo phải sinh từ script chạy thật. Nếu tải/cài đặt/chạy thất bại, ghi rõ phần chưa kiểm chứng. Không dùng số liệu synthetic làm độ chính xác y khoa. Kết quả test chỉ đại diện file Kaggle và chiến lược chia nhóm đã định nghĩa, hạn chế về nguồn gốc, lệch lớp và khả năng tổng quát. Báo cáo Recall/F1/AP cùng Accuracy và có majority baseline.

## 6. Trình tự thực hiện và nghiệm thu

1. Xác minh dataset và tài liệu chính thức; tạo data pipeline và manifest nguồn.
2. Huấn luyện MLP/baseline, chạy AGENT02, đánh giá retrieval và lưu artifact thực.
3. Tạo các agent, API, SQLite và UI.
4. Chạy unit/integration/end-to-end tests; kiểm tra demo khả dụng.
5. Hoàn thiện sơ đồ, 8 bài học code, báo cáo và >=25 câu hỏi bảo vệ.

Nghiệm thu: lệnh cài đặt/chạy tái lập; prediction đọc model đã lưu; Q&A có nguồn và biết từ chối khi không có đoạn phù hợp; workflow hiển thị trace thật; experiments đọc CSV/JSON/PNG thực; tests kiểm tra dữ liệu, ngăn leakage và xử lý lỗi.
