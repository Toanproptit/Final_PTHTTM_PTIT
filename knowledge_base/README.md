# Kho tri thức của demo

Năm tài liệu Markdown là **bản diễn giải ngắn tự biên soạn bằng tiếng Việt**, không phải bản dịch chính thức. `manifest.json` lưu nguồn và ngày truy cập 02/11/2026. Chỉ lưu thông tin chung, không có liều thuốc hoặc quy tắc chẩn đoán cá nhân. Tài liệu gốc được liên kết ở từng kết quả và có thể đọc miễn phí.

Nguồn: WHO và CDC. Việc dùng nguồn/liên kết không hàm ý WHO, CDC, ATSDR, HHS hoặc Chính phủ Hoa Kỳ bảo trợ, phê duyệt hay chứng nhận ứng dụng. Không sử dụng logo/hình ảnh của các tổ chức. Chính sách CDC: https://www.cdc.gov/other/agencymaterials.html. Ưu tiên liên kết nội dung gốc; nội dung có thể thay đổi và cần được rà soát khi mở rộng hoặc dùng sau thời điểm truy cập.

MVP chạy **retrieval-based QA**: chunk Markdown theo đoạn, TF-IDF, cosine similarity, lọc ngưỡng rồi hiển thị nguyên các đoạn tiếng Việt đã biên soạn. Không có LLM sinh câu trả lời. Không gọi đây là RAG có mô hình sinh.

`evaluation_questions.json` là bộ câu hỏi nhỏ do người làm đồ án tự xây; không phải benchmark độc lập. Ngưỡng truy xuất là tham số thiết kế, không được tối ưu bằng cách nhìn kết quả test y khoa. Muốn chứng minh khả năng tổng quát cần một tập câu hỏi độc lập lớn hơn, người đánh giá chuyên môn và kiểm tra nguồn cập nhật.
