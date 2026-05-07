# Báo cáo ngắn

## 1. Bộ siêu tham số đã chọn và lý do

Sau khi chạy nhiều lần huấn luyện và so sánh trên MLflow ở Bước 1, em chọn cấu hình sau cho mô hình chính:

- `model_type: random_forest`
- `n_estimators: 300`
- `max_depth: 20`
- `min_samples_split: 2`

Lý do lựa chọn:

- Đây là cấu hình cho kết quả tốt nhất trong các lần thử nghiệm của em ở giai đoạn đầu.
- Trên MLflow, cấu hình này đạt `accuracy` xấp xỉ `0.678` và `f1_score` xấp xỉ `0.677` với tập dữ liệu huấn luyện ban đầu.
- Khi bổ sung dữ liệu mới ở Bước 3, cùng hướng cấu hình này tiếp tục cho kết quả tốt và đạt `accuracy` xấp xỉ `0.756`, đủ để vượt eval gate và triển khai.

Vì vậy, em giữ `RandomForest` làm mô hình chính vì cho kết quả ổn định hơn các lựa chọn khác đã thử như `logistic_regression` và `gradient_boosting`.

## 2. Khó khăn gặp phải và cách giải quyết

Trong quá trình làm bài, em gặp một số khó khăn chính:

- **Lỗi xác thực DVC với GCS (`401 Invalid Credentials`)**  
  Nguyên nhân là đường dẫn `credentialpath` và secret trên GitHub Actions chưa đồng nhất với key thực tế. Em đã sửa lại cấu hình DVC, tách local credential khỏi file config commit chung, và cập nhật workflow để runner dùng đúng file `/tmp/sa-key.json`.

- **Dữ liệu bị cộng thêm sớm trước khi dựng lại flow**  
  Em đã từng chạy `add_new_data.py` sớm, làm `train_phase1.csv` chuyển sang trạng thái Bước 3. Em tạo thêm script `restore_phase1.py` để khôi phục về bộ dữ liệu gốc, sau đó mới dựng lại đúng thứ tự `Bước 1 -> Bước 2 -> Bước 3`.

- **Lỗi phiên bản thư viện trên VM**  
  Lúc đầu VM dùng phiên bản `scikit-learn` khác với phiên bản khi train model, gây warning khi load model. Em tạo virtual environment riêng trên VM và cài đúng các version thư viện phù hợp với môi trường huấn luyện.

- **Không truy cập được API từ bên ngoài VM**  
  Dịch vụ chạy được trên `localhost` nhưng không truy cập được bằng IP public. Em đã thêm firewall rule mở cổng `8000` và gắn đúng `network tag` cho VM, sau đó endpoint `/health` và `/predict` đều truy cập thành công.
