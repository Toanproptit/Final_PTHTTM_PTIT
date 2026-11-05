# 03. Dataset Kaggle do người dùng chỉ định

[Diabetes Clinical Dataset (100k rows)](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows), uploader `ziya07`. Metadata public Kaggle: ID 6620808, cập nhật 07/02/2025, giấy phép **CC0: Public Domain**. Truy cập 03/11/2026. `data/kaggle/metadata.json` giữ mô tả nguồn; `data/source_manifest.json` giữ URL và SHA-256. CSV thực: `data/kaggle/diabetes_dataset_with_notes.csv`.

Mô tả nguồn không cung cấp cơ sở thu thập ban đầu, patient ID, quy trình tạo nhãn, tình trạng synthetic/real đã được xác minh hoặc từ điển đơn vị hoàn chỉnh. Vì vậy project không khẳng định đây là bệnh án thật, không tự gắn nhãn synthetic và không dùng độ chính xác để tuyên bố giá trị y khoa. Không suy ra tính đại diện cho Việt Nam từ tên “Clinical”.

## Toàn bộ 17 cột và lựa chọn MVP

| Cột gốc | Vai trò trong project | Ý nghĩa/cách xử lý |
|---|---|---|
| year | Bỏ | Năm ghi trong file; không đủ thông tin để chứng minh thời điểm đo/tiên lượng |
| gender | Đầu vào | Female/Male/Other, one-hot 3 cột |
| age | Đầu vào | Tuổi số, có thập phân, khoảng quan sát 0.08–80 |
| location | Bỏ | Địa điểm dạng chuỗi; giảm độ phức tạp và tránh học proxy địa lý chưa được xác minh |
| race:AfricanAmerican | Bỏ | Biến chủng tộc đã mã hóa |
| race:Asian | Bỏ | Như trên |
| race:Caucasian | Bỏ | Như trên |
| race:Hispanic | Bỏ | Như trên |
| race:Other | Bỏ | Như trên; 5 cột race có tổng mỗi hàng bằng 1 trong file |
| hypertension | Đầu vào | 0/1, API dùng bool; ghi nhận tăng huyết áp |
| heart_disease | Đầu vào | 0/1, API dùng bool; ghi nhận bệnh tim |
| smoking_history | Đầu vào | No Info/current/ever/former/never/not current; giữ nguyên 6 category |
| bmi | Đầu vào | Số, 10.01–95.69 quan sát; quy ước demo kg/m² |
| hbA1c_level | Đầu vào | Đổi tên API thành `hba1c_level`, 3.5–9; quy ước demo % |
| blood_glucose_level | Đầu vào | Số, 80–300; quy ước demo mg/dL, không tự đoán xét nghiệm lúc đói |
| diabetes | Target | 0/1; tuyệt đối không là đầu vào |
| clinical_notes | Bỏ | Nguồn mô tả là tóm tắt dựa trên thuộc tính; chứa nhận xét từ BMI/HbA1c/glucose, không phải KB độc lập |

Các đơn vị quy ước trên cần xác minh với bên cung cấp trước khi sử dụng ngoài demo. Cột notes có 761 chuỗi khác nhau ở lần kiểm tra này. Không dùng notes để tăng điểm mô hình hoặc làm tài liệu y khoa cho QA. HbA1c/glucose vẫn là biến liên quan mạnh đến nhãn: bài toán là **phân loại nhãn diabetes trong file**, không phải dự báo mắc bệnh tương lai trước khi xét nghiệm.

## Kiểm tra file thực tế

- 100.000 dòng × 17 cột. Nhãn gốc: 91.500 dòng 0 và 8.500 dòng 1; không có ô NaN.
- `No Info` là thông tin thiếu về hút thuốc dưới dạng category, không phải “never”. `bmi=27.32` xuất hiện nhiều nhưng nguồn không giải thích cách xử lý trước; không tự coi là giá trị thật hoặc tự thay bằng NaN.
- Có 14 bản ghi trùng trên toàn bộ 17 cột. Khi chỉ xét đúng 8 X được học + y, có **3.854 dòng trùng**, giữ 96.146 dòng. Khác biệt các cột đã loại không biến hai X giống nhau thành hai quan sát độc lập đối với model.
- 96.055 nhóm X duy nhất, **91 nhóm cùng X nhưng khác y**. Giữ cả hai nhãn và luôn đặt nhóm trong cùng một tập, không tự sửa nhãn.
- Sau bỏ trùng: 87.664 nhãn 0, 8.482 nhãn 1. Nhóm gender=Other chỉ 18 dòng, không đủ để tuyên bố hiệu năng nhóm.

## Chống rò rỉ và tiền xử lý

1. Chọn 8 X + y, bỏ trùng đúng tập cột này.
2. Group SHA-256 từ toàn bộ X đã chuẩn hóa kiểu, không đưa y vào hash.
3. Nested `StratifiedGroupKFold`: outer 5 folds lấy test, inner 4 folds lấy validation. Seed 42 và 43. Mọi nhóm cùng X chỉ ở một tập.
4. Kết quả: **57.688 train / 19.229 validation / 19.229 test**. Test có 1.696 nhãn 1 và 17.533 nhãn 0. Xem `artifacts/ml/splits.csv` để kiểm chứng source row và group.
5. Nhánh số: age, bmi, hba1c, glucose, hypertension, heart_disease → median imputer → StandardScaler. Chỉ fit trên train.
6. Nhánh category: imputer most frequent → one-hot (3 gender + 6 smoking). Danh sách category là schema cố định, không chọn bằng nhãn/test. Tổng vector **15 chiều**.
7. Validation chọn checkpoint; test đánh giá cuối, ngưỡng 0.5 đã định trước. Không tối ưu ngưỡng bằng test.
8. Khoảng kiểm tra số cho inference lấy từ train. Đây là heuristic phát hiện một dạng ngoài miền, không chứng minh một hồ sơ trong khoảng là in-distribution.

Không có patient ID để chống lặp ở cấp con người; hash cùng X chỉ xử lý trùng quan sát nhìn thấy. Random group split không chứng minh khả năng tổng quát qua bệnh viện/thời gian/quần thể. Cần nguồn được xác minh, kiểm định ngoài mẫu, hiệu chuẩn và đánh giá theo nhóm trước mọi ứng dụng thực tế.

## Dữ liệu UCI cũ

Thử nghiệm UCI ban đầu được lưu ở `artifacts/archive_uci/`; CSV cũ còn trong `data/raw/` để truy nguyên. API và báo cáo hiện dùng **Kaggle** ở `artifacts/ml/`. Không trộn số liệu UCI vào bảng đánh giá Kaggle.
