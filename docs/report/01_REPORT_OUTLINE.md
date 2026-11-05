# 01. Đề cương báo cáo theo nội dung nhìn rõ trên bảng

1. **Giới thiệu:** động cơ hỗ trợ tìm hiểu dữ liệu tiểu đường; mục tiêu nghiên cứu và phạm vi demo.
2. **Hệ đa agent và từng agent:** định nghĩa agent; trách nhiệm Orchestrator/Rule/Prediction/Knowledge/Explanation; module RL độc lập; hợp đồng dữ liệu.
3. **Kiến trúc tổng quan:** sơ đồ thành phần, activity và sequence trong `docs/02_ARCHITECTURE.md`; một tiến trình API, SQLite và artifact offline.
4. **Các thành phần hệ thống:** dataset Kaggle, preprocessing, MLP, baseline, Rule, KB/retrieval; giải thích vì sao chưa có LLM sinh.
5. **Cơ sở thuật toán:** MLP/BCE/backprop/Adam, TF-IDF/cosine, epsilon-greedy và Q-update.
6. **Triển khai:** cấu trúc thư mục và code thực, API, 5 tab demo; SHA-256 chứng minh kế thừa AGENT02.
7. **Thực nghiệm:** các bảng/hình được sinh ở `03_RESULTS.md`, kiểm tra split, test report và workflow thật.
8. **Thảo luận:** lệch lớp/Recall, nguồn gốc Kaggle, thiếu horizon, state RL chưa đầy đủ, benchmark QA nhỏ; không gán ý nghĩa chẩn đoán.
9. **Kết luận và hướng phát triển:** những phần đã chạy; dữ liệu được xác minh, external validation, calibration/fairness, state RL và generator có grounding là bước tiếp theo.

Không trình bày FastAPI/PyTorch/TF-IDF là công nghệ giảng viên bắt buộc. Ảnh chưa xác nhận framework cụ thể. Không đưa Streamlit/FAISS/Ollama vào sơ đồ triển khai thật khi code không sử dụng.

Tài liệu đọc nền: `02_REPORT.md`. Bảng/hình chính xác: `03_RESULTS.md`. Demo trình bày: `04_DEMO_SCRIPT.md`. Câu hỏi bảo vệ: `docs/DEFENSE_QA.md`.
