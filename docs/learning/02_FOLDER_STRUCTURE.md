# 02. Cấu trúc thư mục và lý do tồn tại từng phần

| Đường dẫn | Dùng để làm gì | Vì sao cần tách |
|---|---|---|
| `diabetes_demo/schema.py` | Danh sách 8 biến, kiểu, category, hồ sơ demo | UI/API/training dùng chung một hợp đồng |
| `paths.py` | Tính đường dẫn từ project root | Không phụ thuộc terminal đang ở thư mục nào |
| `data.py` | Tải Kaggle, kiểm tra, bỏ trùng, chia tập, preprocessing | Không lẫn xử lý dữ liệu với học mạng |
| `model.py` | Class `DiabetesMLP` | Kiến trúc dùng giống nhau khi train và load |
| `train.py` | Huấn luyện, chọn checkpoint, đánh giá, vẽ hình | Tách offline khỏi API |
| `agents.py` | Rule, Prediction, Explanation | Nhiệm vụ chuyên trách; dễ test bằng input nhỏ |
| `knowledge.py` | Chunking, TF-IDF và truy xuất | QA không phụ thuộc mô hình phân loại |
| `orchestrator.py` | Thứ tự gọi, trace và lưu kết quả | Có nơi duy nhất mô tả workflow |
| `storage.py` | SQLite | Tách chi tiết kết nối SQL khỏi agent |
| `api.py` | HTTP endpoints và static UI | Chuyển yêu cầu web thành gọi Python |
| `rl_experiments.py` | Chạy các hàm AGENT02 thật | Kế thừa code cũ có thể kiểm chứng bằng hash |
| `evaluate.py` | Đánh giá KB/workflow | Lưu kết quả thay vì chỉ nói “đã chạy” |
| `legacy/agent02_experiment.py` | Bản sao nguyên byte từ `.tex` | Mã gốc giữ nguyên, dễ đối chiếu |
| `web/` | 5 tab HTML/CSS/JS gọi API | Giao diện không tự tính điểm bệnh bằng JavaScript |
| `data/kaggle/` | CSV gốc, ZIP, metadata | Giữ dữ liệu nguồn và truy nguyên |
| `knowledge_base/` | Markdown, metadata và câu hỏi mẫu | Nội dung y khoa có nguồn độc lập với clinical_notes |
| `artifacts/ml/` | Model, transformer, metrics, CSV, PNG | Dùng chung cho suy luận và báo cáo |
| `artifacts/rl/` | Reward, policy, Q-table và so sánh seed | Tách kết quả RL khỏi chỉ số phân loại |
| `artifacts/retrieval/` | Kết quả từng câu và aggregate | Có thể xem câu nào truy xuất sai |
| `runtime/` | SQLite khi chạy demo | Dữ liệu sinh khi dùng app, được gitignore |
| `tests/` | Các kiểm tra dữ liệu, QA, RL và API | Phát hiện lỗi ảnh hưởng thuật toán/luồng |
| `scripts/export_report.py` | Sinh bảng kết quả từ artifact | Tránh chép nhầm/sửa số liệu thủ công |
| `docs/learning/` | Hướng dẫn code | Để học độc lập từng phần |
| `docs/report/` | Đề cương, nội dung và kịch bản bảo vệ | Tách phần trình bày khỏi hướng dẫn lập trình |

`index.html` gốc là demo thương mại JavaScript trước đây. `web/index.html` là UI mới cần FastAPI. `artifacts/archive_uci/` và `data/raw/` giữ lịch sử thử UCI; không dùng chúng khi báo cáo mô hình Kaggle.

Thư viện nằm trong `.venv/`, không phải mã nguồn của sinh viên. `requirements-lock.txt` ghi phiên bản đã kiểm chứng. Không cần học toàn bộ code của thư viện; cần hiểu những API và giả định mà project dùng.
