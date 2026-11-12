# Kho tri thức của demo

Năm tài liệu Markdown là các bản diễn giải ngắn tự biên soạn bằng tiếng Việt. `manifest.json` ghi tiêu đề, nhà xuất bản, liên kết gốc và ngày truy cập. Nguồn hiện dùng cho nội dung RAG là các bài sức khỏe của Vinmec. Nội dung không phải bản sao, tư vấn cá nhân hay nội dung được Vinmec bảo trợ.

MVP dùng retrieval-based QA: chia Markdown thành các đoạn, xếp hạng bằng TF-IDF/cosine và hiển thị evidence cùng đường dẫn nguồn. Khi Neo4j được nhập bằng `scripts/import_knowledge_graph.py`, ứng dụng có thể thêm truy xuất vector và facts của đồ thị. Không có mô hình ngôn ngữ sinh câu trả lời trong luồng QA cơ bản.

`evaluation_questions.json` là bộ câu hỏi demo do nhóm tự xây, không phải benchmark độc lập. Nội dung y tế cần được rà soát định kỳ; các đoạn này chỉ cung cấp thông tin chung và không thay thế đánh giá của nhân viên y tế.
