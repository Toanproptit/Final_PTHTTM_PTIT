# 07. Theo một lần chạy từ giao diện đến SQLite

## Chuẩn bị

Chạy server theo README, mở `http://127.0.0.1:8000`, chọn Dự đoán nghiên cứu và “Điền hồ sơ giả định”. Không sử dụng thông tin một người thật. Hồ sơ demo nằm trong `schema.DEMO_PATIENT`, không được sinh bởi model.

## Lần theo từng file

1. `web/app.js` nhận schema từ `/api/schema`, dựng form 8 trường. Category được lấy từ Python; không tự đoán thứ tự biến. Submit đổi các trường số bằng `Number`, bool bằng true/false và gửi JSON.
2. `api.py` nhận `POST /api/predict`. Pydantic dựng `PatientInput`. Thiếu trường, sai category hoặc gửi chuỗi thay bool → HTTP 422 trước khi gọi agent.
3. `OrchestratorAgent.predict` tạo UUID, lấy dict input, gọi Rule qua helper `call`. Helper ghi thời gian và output vào trace.
4. Rule kiểm tra 4 số với khoảng train. Hồ sơ demo nằm trong khoảng nên trả `within_training_domain=True`.
5. Prediction dùng DataFrame 1 hàng → `encode_features` → transformer **đã fit từ train** → tensor 1×15. `model.eval`/`inference_mode` cho một logit rồi sigmoid. Logistic được gọi trên cùng vector để đối chiếu.
6. Knowledge nhận câu hỏi tổng quát về xét nghiệm HbA1c/glucose, tìm chunk và giữ nguồn. Nó không coi input 5.8 hay 130 là căn cứ tự diễn giải ngưỡng bệnh.
7. Explanation nhận ba output và ghép template. Tóm tắt hai tiền sử là thông tin được nhập, không phải feature importance.
8. `DemoStore.save` serialize toàn bộ result + trace vào bảng SQLite. Parameterized SQL tránh ghép dữ liệu người dùng vào câu lệnh.
9. API trả JSON. `web/app.js` hiển thị score, nhãn dataset, cảnh báo và nguồn. Nút “Xem trace” đưa đến đúng run_id.

## Kiểm tra bằng mắt khi bảo vệ

Trong Workflow, mở output Rule để thấy matched_rules; mở input Prediction để thấy đúng 8 trường; mở input Explanation để thấy kết quả truyền từ các agent. `duration_ms` là số đo thật của request đó. UUID khác nhau giữa các lần chạy, còn score cùng hồ sơ và cùng model eval thì ổn định.

Thay tuổi thành 100: schema vẫn cho phép số này nhưng Rule thấy ngoài khoảng train, Prediction có status skipped và không xuất score; Knowledge/Explanation vẫn trả thông tin tổng quát. Đây là minh họa phân biệt “đúng kiểu phần mềm” với “đủ miền mô hình”. Không diễn giải việc bị chặn như một kết luận sức khỏe.

Xóa một trường: trình duyệt nhắc required; nếu gọi API trực tiếp thì nhận 422. Không tự đổi dữ liệu thiếu thành false. Với smoking, lựa chọn No Info đã tồn tại trong nguồn và được giữ như một category riêng.

## Luồng hỏi đáp

Chọn Hỏi đáp → “HbA1c phản ánh điều gì?” → `/api/ask` → Orchestrator.ask → Knowledge.run → lưu SQLite. Trace chỉ có một agent chuyên trách vì không cần Rule/Prediction/Explanation cho yêu cầu này. Điều phối gọi đúng phần cần thiết, không gọi đủ tên agent chỉ để tăng số bước.

## Chứng cứ có thể mở trực tiếp

- `artifacts/workflow_evaluation.json`: một lần chạy mẫu thật dùng hồ sơ giả định và sáu điều kiện kiểm tra.
- `artifacts/ml/test_predictions.csv`: điểm trên tập test nghiên cứu, khác dữ liệu demo.
- `artifacts/test_results.xml`: unit/integration tests.
- `runtime/demo.sqlite3`: lịch sử app hiện tại; không đưa dữ liệu người thật vào file này.
