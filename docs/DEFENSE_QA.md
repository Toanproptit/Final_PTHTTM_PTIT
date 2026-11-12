# Câu hỏi bảo vệ và câu trả lời ngắn

1. **Agent là gì?** Một thành phần nhận thông tin, áp dụng chính sách/tri thức và tạo hành động/đầu ra để đạt mục tiêu. Trong project vai trò và input/output được định nghĩa rõ.
2. **Hệ này có phải nhiều LLM nói chuyện không?** Không. Một LLM qua LangChain chọn tool; các agent chuyên trách là component Python. LangGraph điều phối state và các nhánh điều kiện, không phải nhiều LLM tự tranh luận.
3. **Multi-agent khác một hàm dự đoán ở đâu?** Ngoài dự đoán còn kiểm tra luật, truy xuất nguồn, giải thích, điều phối và lưu trace; các phần trao đổi dữ liệu có cấu trúc.
4. **Tại sao cần nhiều agent?** Để tách trách nhiệm, kiểm thử riêng và quan sát lỗi từng bước. Một ứng dụng nhỏ không nhất thiết cần framework agent phức tạp.
5. **Orchestrator quyết định gì?** Thứ tự gọi agent và có bỏ qua Prediction khi ngoài miền không; đồng thời tổng hợp và lưu trace.
6. **Agent nào thực sự có học?** MLP/Logistic học offline bằng nhãn; Q-learning học trong môi trường RL. Rule/Explanation không học; TF-IDF fit chỉ số tri thức.
7. **Vì sao dùng Deep Learning?** Để thực hành mô hình phi tuyến theo yêu cầu đồ án và so với baseline; không mặc định deep learning luôn tốt hơn trên dữ liệu bảng.
8. **Mạng có bao nhiêu lớp/tham số?** 15 đầu vào đã preprocessing, hidden16 và8, một output logit; 401 tham số trainable. ReLU và dropout0.1.
9. **Tại sao 8 trường thành15?** Sáu biến số/binary cộng3 one-hot giới tính và6 one-hot hút thuốc.
10. **Loss là gì?** BCEWithLogitsLoss đo sai lệch phân loại nhị phân từ logit, ổn định hơn tự tính sigmoid+log.
11. **Backprop làm gì?** Autograd tính gradient loss theo trọng số; Adam dùng gradient để cập nhật sau mỗi minibatch.
12. **Validation khác test?** Validation chọn checkpoint; test chỉ đánh giá cuối. Không chọn epoch hay threshold bằng test.
13. **Data leakage được giảm thế nào?** Bỏ trùng theo X+y, nhóm cùngX ở cùng tập, fit preprocessing chỉ train, loại target/notes khỏiX.
14. **Có chắc chống lặp bệnh nhân không?** Không có patient ID nên chỉ chống trùng quan sát nhìn thấy, chưa chứng minh độc lập ở cấp người bệnh.
15. **Tại sao dùng đúng dataset này?** Người dùng chỉ định Kaggle ziya07/diabetes-clinical-dataset100k-rows; có100k dòng và các biến cấu trúc phù hợp MLP.
16. **Nguồn có phải bệnh viện thật không?** Metadata chưa mô tả đủ cơ sở thu thập và cách tạo nhãn; project không khẳng định real/synthetic đã xác minh.
17. **Giấy phép dữ liệu?** Metadata Kaggle ghi CC0: Public Domain; URL, metadata và hash được lưu để truy nguyên.
18. **Tại sao không dùng clinical_notes?** Nguồn nói notes là tóm tắt thuộc tính; có nội dung diễn giải xét nghiệm, có thể trùng/proxy thông tin nhãn và không là nguồn KB độc lập.
19. **Tại sao bỏ năm/địa điểm/race?** Để MVP dễ học, tránh thêm proxy chưa xác minh và schema lớn; việc bỏ không tự chứng minh fairness.
20. **Có bao nhiêu dòng sau làm sạch?** 96146 theo8X+y;3854 dòng trùng bị bỏ.91 nhóm cùngX khácnhãn được giữ cùng tập.
21. **Tại sao không sửa91 nhóm nhãn mâu thuẫn?** Không có căn cứ biết nhãn nào đúng. Giữ dữ liệu, nhóm để tránh leakage và báo cáo hạn chế.
22. **No Info có phải never?** Không. No Info là thiếu thông tin hút thuốc, được one-hot riêng.
23. **Vì sao không xem Accuracy riêng?** Dataset lệch lớp: majority đã~91% Accuracy nhưng Recall0. MLP~97% vẫn bỏ sót574 nhãn1 ở test.
24. **Precision100% có nghĩa không bao giờ sai dương không?** Chỉ nghĩa FP=0 trên split test hiện tại ở ngưỡng0.5; không bảo đảm với dữ liệu ngoài mẫu.
25. **ROC-AUC khác AP?** ROC-AUC đánh giá ranking theo TPR/FPR; AP tổng hợp precision-recall và hữu ích khi lớp dương hiếm. Cả hai không thay thế calibration.
26. **Score3% nghĩa là người đó có3% nguy cơ bệnh không?** Không. Đó là sigmoid cho nhãn dataset, chưa hiệu chuẩn cho cá nhân/quần thể hoặc một khoảng thời gian tương lai.
27. **Vì sao có HbA1c/glucose nhưng vẫn gọi bài toán nguy cơ?** Tiêu đề đề tài rộng; triển khai cụ thể là phân loại nhãn với dữ liệu xét nghiệm. Không tuyên bố dự đoán trước xét nghiệm hoặc prospective risk.
28. **Rule Agent dùng ngưỡng lâm sàng nào?** Không dùng ngưỡng chẩn đoán. Nó kiểm tra schema, khoảng train và ý nghĩa category; thông tin chung dẫn nguồn CDC.
29. **Nếu đầu vào trong min/max thì chắc chắn hợp lệ với model?** Không. Min/max chỉ bắt một dạng OOD; tổ hợp chưa từng gặp vẫn có thể ngoài phân bố.
30. **KB khác RAG?** KB là nội dung lưu; RAG thường kết hợp truy xuất với generator. MVP chỉ retrieval-based QA, không generator.
31. **TF-IDF/cosine làm gì?** Chuyển query/chunk sang vector có trọng số từ vựng, xếp hạng theo cosine. Score không phải độ đúng y khoa.
32. **Hệ thống trả lời khi thiếu bằng chứng thế nào?** Trả no_evidence hoặc outside_scope, không tạo thêm kết luận/đơn thuốc.
33. **Tài liệu KB lấy từ đâu?** Bản diễn giải ngắn từ các bài sức khỏe Vinmec, có URL/ngày truy cập, không dùng clinical_notes. Việc dẫn nguồn không hàm ý Vinmec bảo trợ ứng dụng.
34. **Hit@3=100% có chứng minh QA tốt mọi câu không?** Không. Chỉ11 câu có nguồn trong bộ nhỏ do tác giả biên soạn; có thiên lệch và chưa độc lập.
35. **Q-learning hoạt động thế nào?** Lưu Q(s,a), chọn epsilon-greedy, cập nhật theo TD error giữa target r+gamma*maxQnext và Q cũ.
36. **State/action/reward RL là gì?** State=(nhu cầu, hành động trước của đối thủ), action=giảm/không giảm, reward=lợi nhuận mô phỏng.
37. **Terminal target khác gì?** Terminal chỉ dùng reward hiện tại; chưa terminal cộng giá trị tương lai chiết khấu.
38. **Tại sao tách evaluation RL?** Để đo chính sách đã học khi không khám phá và không cập nhậtQ, dùng seed môi trường khác.
39. **RL có luôn tốt hơn Rule không?** Không. Kết quả nhiều seed hiện tại gần nhau, RL trung bình hơi thấp hơn; không tuyên bố ưu thế.
40. **State RL gốc có đầy đủ Markov không?** Có giới hạn: thiếu own_previous_action dù đối thủ phụ thuộc nó, và thiếu time_remaining. Không viện dẫn bảo đảm hội tụ tối ưu đầy đủ.
41. **Những gì kế thừa AGENT02?** Import trực tiếp Config/env/QLearningAgent/RuleBasedAgent/train/evaluate và4thínghiệm từ bản sao nguyênbyte; hash và tests kiểm chứng.
42. **Nếu bỏ Knowledge Agent?** Vẫn có score model nhưng thiếu thông tin chung có nguồn; Explanation không còn evidence để hiển thị.
43. **Nếu bỏ Rule Agent?** Schema vẫn bắt kiểu nhưng mất kiểm tra miền train và các cảnh báo category; dễ xuất score cho đầu vào ngoài khoảng quan sát.
44. **Lưu model để làm gì?** Tránh train mỗi request và đảm bảo dùng cùng trọng số/transformer. Restart server sau train lại.
45. **Đã kiểm thử gì?** Schema, nhóm không giao, fit train-only, category, frozenQ, terminalupdate, nguồnQA, API/trace/SQLite và thiếumodel; xem JUnit XML và verification.
46. **Đã dùng Streamlit/Ollama/FAISS chưa?** Chưa. FastAPI+HTML đủ demo, TF-IDF đủ KB nhỏ; không liệt kê công nghệ chưa dùng vào phần triển khai.
47. **Có được dùng hệ này cho bệnh nhân thật không?** Chưa có kiểm định lâm sàng/nguồn dữ liệu/hiệu chuẩn đầy đủ; phạm vi hiện tại là đồ án và hồ sơ giả định.
48. **Hướng cải tiến quan trọng nhất?** Xác minh nguồn/nhãn, định nghĩa thời điểm tiên lượng, external validation và calibration; mở rộng QA đánh giá độc lập và sửa stateRL trước khi tăng độ phức tạp.
