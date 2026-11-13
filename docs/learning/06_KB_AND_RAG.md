# 06. Từ tài liệu đến câu trả lời có nguồn

## KB khác RAG ở đâu?

Knowledge Base là phần nội dung: ở đây gồm 5 tệp Markdown và `manifest.json`. Retrieval là bước tìm đoạn liên quan. RAG có mô hình sinh còn dùng các đoạn tìm được làm ngữ cảnh cho generator. MVP **chỉ triển khai retrieval-based QA**, không có generator hoặc Ollama. Không nói “LLM đã đọc hồ sơ” vì code không làm việc đó.

Các bản diễn giải hiện được biên soạn từ các bài sức khỏe Vinmec, kèm liên kết và ngày truy cập trong `manifest.json`. Đây không phải nội dung Vinmec bảo trợ. `clinical_notes` trong dataset không phải nguồn tri thức y khoa và không đưa vào KB.

## Pipeline trong `KnowledgeAgent.__init__`

1. Đọc `manifest.json` để có document_id, title, file, publisher, URL, accessed.
2. Đọc Markdown UTF-8 và gọi `chunk_markdown`.
3. Mỗi đoạn giữ `chunk_id=document_id:number` cùng metadata. Đoạn dài hơn 90 từ dùng cửa sổ, overlap 15 từ; đoạn ngắn giữ nguyên. Heading đi cùng nội dung của đoạn.
4. `TfidfVectorizer` học vocabulary/IDF từ các chunk KB, không từ test bệnh tiểu đường hoặc câu hỏi người dùng.
5. Ma trận tài liệu nằm trong RAM; hash index lưu ở artifact đánh giá để truy nguyên nội dung.

## TF-IDF và cosine dễ hiểu

TF cho biết từ xuất hiện trong một đoạn; IDF giảm trọng số từ quá phổ biến ở nhiều đoạn. Code dùng word unigram và bigram, `sublinear_tf=True`, bỏ một số từ chức năng tiếng Việt. `normalize` chuyển chữ thường, bỏ dấu và đổi đ→d để câu có/không dấu gần nhau.

Một query trở thành vector với cùng vocabulary. Cosine similarity là tích vô hướng chia cho độ dài hai vector, đo mức trùng về hướng/từ vựng. Nó không phải xác suất thông tin đúng hoặc độ chắc chắn y khoa. Từ đồng nghĩa không xuất hiện trong KB có thể bị bỏ sót; đây không phải hiểu ngôn ngữ sâu.

## `run(question)` quyết định thế nào?

Kiểm tra rỗng; chặn yêu cầu liều thuốc/chẩn đoán cá nhân bằng heuristic; kiểm tra có chủ đề tiểu đường/xét nghiệm/triệu chứng trong từ điển phạm vi; tính cosine; lấy tối đa 3 chunk có score>=0.08. Không có chunk → `no_evidence`. Yêu cầu ngoài phạm vi hành động cá nhân → `outside_scope`. Có evidence → `answered` cùng các đoạn đã lưu.

Ngưỡng 0.08 và từ điển là lựa chọn kỹ thuật của KB nhỏ, không phải tiêu chuẩn lâm sàng. Cổng phạm vi giúp tránh từ tình cờ như “phiếu” làm câu tài chính khớp tài liệu y tế. Heuristic có thể bỏ sót hoặc chặn nhầm cách diễn đạt mới; cần mở rộng benchmark trước khi khẳng định chất lượng.

## Ví dụ đường đi của câu hỏi

“HbA1c phản ánh điều gì?” → normalize chứa hba1c → query vector → chunk từ `04_testing.md` có điểm cao → output gồm text, URL Vinmec, ngày truy cập và score → UI hiển thị đoạn và link. Không tạo đáp án mới bằng sampling LLM. Người xem có thể mở nguồn để kiểm chứng.

“Giá cổ phiếu hôm nay?” không qua cổng phạm vi. “Tôi nên uống metformin liều bao nhiêu?” thuộc yêu cầu ngoài phạm vi. Không lấy điểm phân loại của một hồ sơ để trả lời chẩn đoán cá nhân.

## Đánh giá và cách đọc kết quả

`evaluation_questions.json` có 11 câu trong phạm vi gắn expected_document và 3 câu ngoài phạm vi. `evaluate_retrieval` chạy từng câu, so ID tài liệu, lưu CSV. Hit@1 đếm nguồn đúng đứng đầu; Hit@3 đếm nguồn đúng trong các chunk top-3 sau gộp document_id. Rejection accuracy đếm câu ngoài phạm vi không có evidence.

Lần chạy hiện tại Hit@1=10/11, Hit@3=11/11 và rejection=3/3. Một câu top-1 sai vẫn được giữ trong CSV để thảo luận. Đây là bộ do cùng người xây KB biên soạn, có thể lạc quan và **không độc lập**. Không dùng con số này để khẳng định khả năng trả lời mọi câu hỏi y khoa.

## Nếu mở rộng Ollama sau này

Đặt generator sau bước truy xuất, buộc trả lời chỉ dựa trên evidence và trích chunk ID, giữ cơ chế không có bằng chứng. Cần kiểm tra faithful citation, hallucination, câu hỏi ngoài phạm vi và thời gian chạy. Đây là hướng phát triển chưa có trong code hiện tại; không liệt kê Ollama/embeddings/FAISS là công nghệ đã sử dụng.
