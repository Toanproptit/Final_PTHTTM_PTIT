# 01. Hiểu toàn bộ project trước khi đọc từng hàm

## Bài toán thực tế được code giải quyết

Tiêu đề đồ án là hỗ trợ đánh giá nguy cơ tiểu đường. Trong phiên bản nghiên cứu này, bài toán học máy cụ thể là: nhận 8 trường có cấu trúc và dự đoán **nhãn diabetes=0/1 trong file Kaggle**. Có HbA1c và glucose trong đầu vào; file không định nghĩa một khoảng thời gian theo dõi. Vì vậy không nói “người này sẽ mắc bệnh sau X năm”.

Tách ba loại đầu ra khi trình bày:

1. **Điểm mô hình**: sigmoid của MLP, học từ mối liên hệ trong dataset.
2. **Thông báo luật**: đầu vào đúng schema, có nằm trong khoảng số đã quan sát không, No Info có ý nghĩa gì.
3. **Kiến thức chung**: đoạn được biên soạn từ các bài sức khỏe Vinmec, kèm URL. Đoạn này không biến điểm mô hình thành chẩn đoán.

## Vì sao dùng nhiều agent?

Một hàm lớn có thể làm hết công việc. Chia agent ở đây giúp mỗi phần có trách nhiệm, input/output và kiểm thử riêng: Rule kiểm tra dữ liệu; Prediction suy luận; Knowledge tìm nguồn; Explanation ghép thông tin; Orchestrator quyết định thứ tự và ghi trace. Các agent phối hợp để tạo một kết quả chung.

Đây là các component chuyên trách được LangGraph điều phối bằng state graph và nhánh điều kiện. Tab LLM dùng một model qua LangChain để tự chọn tool trong allowlist; đây không phải nhiều LLM tự nói chuyện. Q-learning là một agent học qua tương tác ở module thí nghiệm độc lập.

## Hai luồng cần nhớ

**Offline:** CSV → bỏ trùng/chia nhóm → fit preprocessing trên train → huấn luyện MLP/baseline → chọn checkpoint bằng validation → đánh giá test → lưu artifact. Luồng này nằm ở `data.py`, `train.py`.

**Online:** trình duyệt → API validate → LangGraph → Validation/Safety → Prediction (hoặc skipped) → Knowledge (khi cần) → Explanation → MySQL → kết quả giao diện. Luồng này không huấn luyện thêm MLP và không dùng nhãn test; local/test có thể fallback sang SQLite.

Hỏi đáp chỉ gọi Knowledge Agent vì câu hỏi kiến thức không cần điểm model. RL chạy offline trong môi trường thương mại gốc; reward của nó không điều khiển y tế.

## Ví dụ dễ trình bày

Nút “Điền hồ sơ giả định” tạo: tuổi 45, Male, chưa ghi nhận tăng huyết áp/bệnh tim, never smoking, BMI 27.5, HbA1c 5.8, glucose 130. Đây là ví dụ kỹ thuật. Pydantic xác nhận kiểu; Rule kiểm tra khoảng số; pipeline biến thành 15 số; MLP trả một logit rồi sigmoid; Knowledge tìm thông tin xét nghiệm; Explanation ghi rõ giới hạn. Xem giá trị thực trong UI hoặc `artifacts/workflow_evaluation.json`, không tự viết số minh họa thành kết quả model.

## Những gì không suy ra được

Dataset lớn hơn không tự làm nguồn đáng tin hơn. Metadata Kaggle chưa mô tả đủ nguồn gốc/cách tạo nhãn. Accuracy cao không đồng nghĩa sàng lọc tốt: Recall của MLP ở ngưỡng 0.5 chỉ khoảng 66%. Một hồ sơ nằm trong khoảng min/max cũng chưa chắc giống phân bố train. Các hạn chế này là phần của lập luận nghiên cứu, không phải lỗi giao diện.
