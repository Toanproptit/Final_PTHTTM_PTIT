# 02. Nội dung báo cáo nghiên cứu

## 1. Giới thiệu và động cơ

Đồ án xây dựng một workflow gồm nhiều agent chuyên trách để minh họa cách kết hợp mô hình phân loại, kiểm tra luật và thông tin tham khảo. Tiểu đường là chủ đề thực hành; dữ liệu và kết quả trong project phục vụ học thuật. Hệ thống giúp người học theo dõi từng bước thay vì chỉ nhìn một nhãn đầu ra.

Phạm vi phân loại cụ thể là nhãn `diabetes` trong dataset Kaggle do sinh viên chọn. Có HbA1c và glucose làm input; không có thông tin theo dõi tương lai. Vì vậy kết quả không chứng minh dự báo nguy cơ trước xét nghiệm hoặc giá trị chẩn đoán lâm sàng.

## 2. Cơ sở lý thuyết

Agent có thể nhận thông tin từ môi trường, áp dụng chính sách/tri thức và tạo hành động/đầu ra nhằm mục tiêu. Hệ nhiều agent phân chia trách nhiệm và trao đổi dữ liệu để tạo kết quả chung. Trong MVP, điều phối được lập trình tuần tự; không có LLM tự lập kế hoạch hoặc thương lượng.

MLP là mạng feed-forward nhiều lớp, học trọng số từ ví dụ có nhãn bằng gradient. Logistic Regression cung cấp baseline tuyến tính. Rule Agent thực hiện các điều kiện xác định, minh bạch; kiểm tra ngoài miền không tự học từ reward.

Q-learning học giá trị hành động từ reward trong môi trường tương tác. KB lưu tri thức; retrieval tìm đoạn; RAG có generator còn tổng hợp văn bản theo evidence. MVP dùng retrieval-based QA và gọi đúng tên khả năng đó.

## 3. Phân tích yêu cầu và thiết kế

Ảnh bảng nhìn rõ DB, Deep Learning/phân loại, agent, KB/RAG, rule, reinforcement learning, code/demo và các mục báo cáo. Một số chữ/mũi tên không rõ nên không suy ra yêu cầu công nghệ. MVP chọn Python/FastAPI/HTML để chạy một server; PyTorch và sklearn để học/đánh giá; SQLite để lưu trace; Markdown/TF-IDF cho KB nhỏ.

Nguồn và schema Kaggle đã được kiểm tra qua metadata/CSV thực. Loại notes vì mô tả là tóm tắt thuộc tính và có thể lặp nội dung liên quan nhãn; không coi notes là KB độc lập. Loại năm/địa điểm/race để MVP đơn giản, hạn chế học proxy chưa được xác minh. Quyết định này không chứng minh các biến còn lại đã hết thiên lệch.

## 4. Kiến trúc và vai trò agent

Ba sơ đồ Mermaid và bảng input/output nằm ở [kiến trúc](../02_ARCHITECTURE.md). Orchestrator gọi Rule, Prediction, Knowledge, Explanation rồi lưu SQLite. Pydantic validate trước workflow. Prediction sử dụng transformer/checkpoint offline; Knowledge trả đoạn có nguồn; Explanation dùng template. RL độc lập, không ảnh hưởng hướng dẫn y tế.

## 5. Thuật toán

Chia tập theo group của 8 X, giữ nhóm trùng không qua nhiều tập. Median/scaler/one-hot chỉ fit train. MLP 15→16→8→1 với ReLU, dropout0.1; BCEWithLogitsLoss, Adam lr0.001, weight_decay1e-4, batch512, max100, patience12. Validation chọn checkpoint; test đánh giá cuối; ngưỡng0.5 định trước. Chi tiết giải thích từng bước ở `docs/learning/04_DEEP_LEARNING.md`.

TF-IDF unigram/bigram trên chunk Markdown, cosine similarity, ngưỡng0.08 và cổng phạm vi heuristic. Không có mô hình sinh. QA trả evidence gốc đã biên soạn với URL/date; không kê liều hay chẩn đoán cá nhân.

Q-learning dùng `Q←Q+alpha*(r+gamma*maxQnext−Q)` khi chưa terminal; terminal chỉ dùng r. Epsilon giảm theo episode. Code Python gốc được bảo toàn, wrapper chỉ thay output path và thêm manifest. Giới hạn Markov/horizon của state cũ được nêu trong bài học Q-learning.

## 6. Triển khai

API và UI thực nằm trong `diabetes_demo/` và `web/`. Giao tiếp bằng dictionary/JSON có schema. Lịch sử lưu SQLite với UUID và timestamp UTC; UI trình bày thời gian theo trình duyệt. Server bind localhost; không triển khai cloud hoặc tích hợp bệnh viện.

Source data, split, checkpoint, preprocessing và version đều được lưu. Tests kiểm tra schema, chống leakage, frozen Q, provenance evidence, xử lý lỗi, persistence và workflow. Kết nối SQLite được đóng tường minh để hoạt động đúng trên Windows.

## 7. Thực nghiệm

Bảng số/hình tự động lấy từ artifact trong [kết quả](03_RESULTS.md): MLP trên test và so baseline/majority; Q-learning vs rule và alpha/epsilon/nhiều seed; retrieval14 câu; workflow bằng hồ sơ giả định và unit/integration tests. Không trộn số liệu UCI lưu trữ với mô hình Kaggle hiện tại.

## 8. Đánh giá và hạn chế

Accuracy khoảng97% của MLP cần đọc cùng Recall khoảng66%; phần nhãn1 bỏ sót còn đáng kể ở ngưỡng0.5. Majority baseline minh họa lệch lớp. Không có external validation, hiệu chuẩn theo quần thể hay đánh giá fairness đầy đủ. Nguồn Kaggle chưa mô tả đầy đủ quá trình thu thập/tạo nhãn; đơn vị trên form là quy ước demo cần xác minh. Trùng nhóm X được xử lý nhưng thiếu patient ID nên không đảm bảo độc lập cấp con người.

RL và rule có reward gần nhau; không có bằng chứng ưu thế tổng quát. State gốc thiếu own_previous_action và time_remaining nên không viện dẫn bảo đảm hội tụ tối ưu. Môi trường giả lập không mô tả khách hàng độc quyền hoặc tác động y tế.

KB nhỏ, câu hỏi do cùng tác giả soạn và heuristic từ vựng có thể thất bại ở diễn đạt mới. UI chưa được xác nhận bằng kiểm tra trực quan vì công cụ Chrome dừng do không xác định được URL đủ chắc chắn; HTTP/API/asset đã được kiểm tra riêng, xem tài liệu verification.

## 9. Kết luận và hướng phát triển

Đồ án có pipeline học có giám sát, agent kiểm tra/giải thích/truy xuất, workflow quan sát được và mô phỏng RL kế thừa thực tế. Bước tiếp theo là xác minh provenance dữ liệu, xác định rõ bài toán prospective nếu cần, đánh giá ngoài mẫu/calibration/subgroup; mở rộng KB có đánh giá độc lập; thiết kế state RL đầy đủ hơn. Nếu thêm generator, phải đo tính trung thành với nguồn trước khi sử dụng.

## Nguồn tham khảo chính

- [Kaggle dataset do người dùng chỉ định](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows), CC0 theo metadata; truy cập03/11/2026.
- [WHO Diabetes fact sheet](https://www.who.int/news-room/fact-sheets/detail/diabetes), nguồn tổng quan KB.
- [CDC Symptoms](https://www.cdc.gov/diabetes/signs-symptoms/index.html), [CDC Testing](https://www.cdc.gov/diabetes/diabetes-testing/index.html), [CDC Type2](https://www.cdc.gov/diabetes/about/about-type-2-diabetes.html), [CDC Prevention](https://www.cdc.gov/diabetes/prevention-type-2/index.html), truy cập02/11/2026.
- [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) và [sklearn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- Code `AGENT02: Reinforcement-Learning-Based Agent vs Rule-Based Agent` do sinh viên cung cấp, `agent02_code.tex`.
