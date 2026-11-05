# 04. MLP hoạt động như thế nào trong code?

## Từ 8 trường thành 15 số

`data.make_preprocessor` là ColumnTransformer. Sáu biến số/binary được impute median và chuẩn hóa z-score: `(x - mean_train) / std_train`. Hai category được one-hot thành 3 + 6 cột. Tổng 6 + 3 + 6 = 15. `No Info` có cột riêng. Các transformer chỉ fit trên train; cùng transformer được lưu để inference giống huấn luyện.

Giới tính không được mã hóa thành thứ tự 0/1/2 vì không có quan hệ khoảng cách như vậy. One-hot biểu diễn một category bằng một cột 1 và các cột còn lại 0. Trong MVP danh sách category được định nghĩa từ schema nguồn; unknown category bị từ chối thay vì âm thầm đoán.

## Forward propagation (`model.py`)

Mạng `DiabetesMLP`:

```text
15 đầu vào → Linear(15,16) → ReLU → Dropout(0.1)
           → Linear(16,8) → ReLU → Linear(8,1) → logit z
```

Mỗi Linear tính `Wx+b`. ReLU lấy `max(0,x)`, tạo khả năng biểu diễn quan hệ phi tuyến. Dropout ngẫu nhiên tắt 10% activation khi train để giảm phụ thuộc vào một đường kết nối; tắt trong eval. Mạng nhỏ có 401 tham số, đủ để minh họa deep learning trên dữ liệu bảng mà không làm kiến trúc khó giải thích.

Logit là số thực bất kỳ. Khi inference, `sigmoid(z)=1/(1+exp(-z))` đưa về [0,1]. Giá trị 0.5 tương ứng logit 0. Điểm này là đầu ra phân loại đã học; không tự bảo đảm xác suất được hiệu chuẩn trên người bệnh hoặc tương lai.

## Loss và backpropagation (`train.py`)

Với nhãn y=0/1, binary cross entropy phạt dự đoán chắc chắn nhưng sai. Code dùng `BCEWithLogitsLoss`, kết hợp sigmoid và BCE theo cách ổn định số học; không đặt sigmoid trong `forward` rồi gọi loss này lần nữa.

Mỗi minibatch:

```python
optimizer.zero_grad()           # xóa gradient của batch trước
loss = loss_fn(model(inputs), labels)
loss.backward()                # autograd tính đạo hàm qua các lớp
optimizer.step()               # Adam cập nhật trọng số
```

Gradient cho biết thay đổi trọng số theo hướng nào làm loss giảm. Adam dùng các trung bình động gradient để điều chỉnh cập nhật. Khác Q-learning, MLP có nhãn thật trong dataset và tối ưu supervised loss, không tương tác môi trường để nhận reward.

## Vì sao chọn tham số hiện tại?

| Tham số | Giá trị | Lý do thiết kế |
|---|---|---|
| Hidden layers | 16 và 8 | Mạng nhỏ dễ giải thích, CPU chạy được |
| Dropout | 0.1 | Regularization nhẹ |
| Adam learning rate | 0.001 | Điểm bắt đầu thông dụng; không tuyên bố là tối ưu |
| Weight decay | 0.0001 | Hạn chế trọng số quá lớn |
| Batch size | 512 | Dataset ~96k sau lọc, giảm số bước và chạy CPU nhanh |
| Max epochs | 100 | Chặn thời gian huấn luyện |
| Patience | 12 | Dừng nếu validation BCE không cải thiện |
| Seed | 42 | Tái lập split/khởi tạo/shuffle |
| Threshold | 0.5 | Quy ước định trước, không chỉnh theo test |

Checkpoint tốt nhất chọn bằng validation loss, không phải train loss hay test accuracy. Lần chạy hiện tại chạy 90 epoch và chọn epoch 78. Early stopping giữ bản copy trọng số tốt nhất; sau vòng lặp load lại bản đó trước khi đánh giá.

## Baseline và đọc chỉ số

Logistic Regression là mô hình tuyến tính trên cùng 15 đầu vào, cùng train/test, C=1.0 và max_iter=1000. Majority baseline luôn đoán lớp đa số. Dataset lệch lớp nên majority đã đạt hơn 91% Accuracy nhưng Recall=0. Đây là lý do cần Precision, Recall, F1, ROC-AUC, average precision (AP) và confusion matrix.

MLP ở lần chạy hiện tại: Accuracy ~97%, Recall ~66%, tức còn 574/1696 nhãn 1 trong test bị bỏ sót ở ngưỡng 0.5. Không trình bày 97% như “chẩn đoán đúng cho 97% dân số”. Muốn chọn ngưỡng cho mục tiêu sử dụng phải xác định trade-off trước, dùng validation để chọn và kiểm định test mới; việc này chưa được triển khai.

`torch.save(state_dict)` lưu trọng số; `joblib.dump` lưu preprocessing và logistic. `PredictionAgent.load` dựng lại đúng mạng rồi load. Sau khi train lại cần restart server. Xem bảng số chính xác và hình trong `artifacts/ml/`, không suy ra số liệu từ một ví dụ demo.
