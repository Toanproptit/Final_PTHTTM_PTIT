# 04. Kịch bản demo và lời trình bày

## Trước buổi demo

Chạy tests và kiểm tra artifact trước, không train giữa buổi. Khởi động `run_demo.bat`, mở localhost:8000. Đọc `03_RESULTS.md` mới nhất. Chỉ dùng hồ sơ giả định. Đảm bảo hình trong tab Thực nghiệm mở được ở trình duyệt đang dùng.

## Phút 0–1: mục tiêu và phạm vi

“Đồ án tổ chức nhiều agent chuyên trách: kiểm tra, suy luận, truy xuất, giải thích và điều phối. Dataset Kaggle có100.000 dòng. Mô hình phân loại nhãn trong dataset có HbA1c/glucose, không phải công cụ chẩn đoán hoặc dự báo tương lai đã thẩm định.”

Chỉ Overview và luồng 4 bước. Nêu rõ FastAPI/PyTorch là lựa chọn triển khai, ảnh bảng không bắt buộc các framework đó.

## Phút 1–3: một hồ sơ và trace thật

Vào Dự đoán → Điền hồ sơ giả định → Chạy. Chỉ điểm sigmoid, label và baseline. “Ngưỡng0.5 là lựa chọn kỹ thuật. Dữ liệu nhập không được tự coi là feature importance.”

Mở Workflow từ nút trace. Đi qua output Rule, input/output Prediction, evidence Knowledge và input Explanation. “Các dictionary này được truyền thật giữa các class, có thời gian xử lý, lưu SQLite theo run_id.”

Đổi age100: model bị skipped vì ngoài khoảng train. Bỏ trống một trường: validation. Chọn smoking No Info để giải thích khác never.

## Phút 3–4: hỏi đáp có nguồn

Hỏi “HbA1c phản ánh điều gì?”; mở link Vinmec nếu có mạng. “Hệ thống đang trả các đoạn KB, chưa dùng LLM sinh.” Thử “Giá cổ phiếu hôm nay?” hoặc câu hỏi liều thuốc; giải thích giới hạn thay vì tạo câu trả lời không có căn cứ.

## Phút 4–6: thực nghiệm

Chỉ bảng MLP/Logistic/Majority: “Accuracy MLP97,01% nhưng Recall66,16%, còn574 nhãn1 bị bỏ sót. Majority Accuracy91,18% nhưng Recall0, nên Accuracy riêng không đủ.” Mở confusion matrix, loss train/validation và CSV. Chỉ split group giữ91 nhóm mâu thuẫn nhãn cùng tập.

Mở kết quả RL: “Đây là mô phỏng thương mại gốc độc lập. Hai action giảm/không giảm, reward lợi nhuận. Evaluate tắt khám phá, không updateQ.” Chỉ mean5seed gần nhau, không nói RL luôn thắng. Nêu state cũ có giới hạn Markov nếu giảng viên hỏi sâu.

## Phút 6–7: chứng cứ và hạn chế

Mở source `rl_experiments.py` để chỉ import legacy, `tests/test_rl.py` để chỉ so hash, và `artifacts/test_results.xml`. Nêu source Kaggle chưa đủ provenance, chưa external validation/calibration, KBbenchmark nhỏ, UI kiểm tra trực quan cần người trình bày rà soát trước buổi thật.

Kết thúc bằng những gì mình đã hiểu: input/output agent, tách train/validation/test, forward/loss/backprop, Q-update và retrieval/source. Không cần hứa về tích hợp bệnh viện hoặc chẩn đoán.
