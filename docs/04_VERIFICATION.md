# 04. Phạm vi đã kiểm chứng

Môi trường thực: Windows, Python 3.12.10 trong `.venv`, CPU; dependencies khóa trong `requirements-lock.txt`. Ngày kiểm tra 03/11/2026 theo Asia/Bangkok. Timestamp artifact là UTC.

## Kiểm chứng dữ liệu và model

- Đã tải metadata/ZIP công khai từ đúng Kaggle URL do người dùng chỉ định. Lưu CSV100.000 dòng,17 cột, metadata và SHA-256; giấy phép metadata ghi CC0.
- Đã chạy preprocessing/group split và huấn luyện MLP/baseline thật. Không dùng số liệu synthetic minh họa làm metric. Source provenance vẫn chưa đủ để xác minh hồ sơ gốc là real/synthetic; không suy từ tên “Clinical”.
- Đã lưu checkpoint, transformer, logistic, split và điểm từng hàng test. Rule inference dùng khoảng số train; dữ liệu nhóm không qua nhiều tập.
- Đã xem trực tiếp PNG confusion matrices và training loss; chữ/axes không bị cắt, số trong confusion matrix khớp JSON.
- Đã chạy nguyên module AGENT02,4 nhóm thí nghiệm, đủ5000 episode train/500 evaluate theo defaults và5 seed. Hash .tex và file Python legacy trùng nhau.
- Đã chạy14 câu QA và workflow với hồ sơ giả định. Sáu điều kiện workflow đạt; JSON/CSV lưu trong artifacts.

## Kiểm thử tự động

`python -m pytest -q --junitxml=artifacts/test_results.xml`: **25 passed**,0 failures/errors/skipped. Kiểm tra schema/unknown category, No Info khác never, nhóm không giao, fit train-only, domain rule, terminal Q target, eval giữ Q, seed tái lập, source evidence, câu ngoài phạm vi, trace/API/SQLite, thiếu model503 và dữ liệu demo thỏa constraint form.

Có một cảnh báo deprecation từ Starlette TestClient về httpx của phiên bản thư viện hiện tại; các test vẫn đạt. Không tắt/cất cảnh báo để trình bày kết quả đẹp hơn. Có thể cập nhật adapter test theo tài liệu thư viện khi nâng dependency; chưa ảnh hưởng request FastAPI trong lần chạy này.

`node --check web/app.js`: đạt kiểm tra cú pháp. `node --test tests/legacy_demo.test.cjs`: **2 passed** với DOM adapter nhỏ, kiểm tra khởi tạo, biên episode, train và reset của demo thương mại. Đây là unit tests, không phải rendering trong trình duyệt. Đã sửa lỗi gọi hàm khởi tạo không tồn tại và bộ đếm episode của bản demo cũ.

## HTTP thật

Đã khởi động Uvicorn trên127.0.0.1:8000 và chạy `scripts/smoke_http.py`. Script kiểm tra HTML/CSS/JS, API docs, demo legacy, health đúng dataset Kaggle/8 trường, constraint hồ sơ demo, prediction4 bước, QA có nguồn, experiments đúng nguồn và6 ảnh PNG. Chứng cứ: `artifacts/http_smoke.json`, `all_passed=true`.

Script tạo hồ sơ giả định trong lịch sử cục bộ để kiểm tra persistence; không truyền hồ sơ thật ra bên ngoài. Server dùng để kiểm tra được dừng khi hoàn tất; mở lại bằng `run_demo.bat` hoặc lệnh trong README.

## Chưa kiểm chứng hoặc chưa triển khai

- **Chưa xác nhận rendering/interaction thật trong Chrome.** Browser access host không khả dụng; đường Computer Use qua Chrome bị công cụ dừng vì không xác định URL hiện tại đủ chắc chắn để thực thi chính sách. Không tiếp tục automation sau thông báo dừng. HTTP/assets/tests đạt không thay thế browser QA. Cần người trình bày mở demo và rà soát5 tab trước buổi bảo vệ.
- Chưa có external validation, calibration hoặc fairness toàn diện; chưa xác minh provenance/label generation với uploader. Không suy ra giá trị lâm sàng.
- Chưa có LLM/Ollama sinh RAG, embeddings/FAISS, prediction horizon theo thời gian, tích hợp bệnh viện hoặc hosting công khai. MVP chọn retrieval-based QA và server cục bộ để dễ học/chạy.
- Chưa sửa state Markov/horizon trong AGENT02 vì giữ bản gốc để đối chiếu; giới hạn được trình bày trong tài liệu Q-learning.

## Chạy lại các kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/test_results.xml
node --check web/app.js
node --test tests/legacy_demo.test.cjs
.\.venv\Scripts\python.exe -m diabetes_demo.evaluate
.\.venv\Scripts\python.exe scripts\export_report.py
```

Chạy server ở terminal riêng trước khi gọi `python scripts/smoke_http.py`. Node chỉ cần cho hai lệnh kiểm tra JavaScript, không cần cho runtime của app. Khi thay model/schema, restart server rồi kiểm tra lại.
