# Mô hình và báo cáo chỉ dùng train

Theo yêu cầu mới nhất, không đọc hay dùng `test_blind.csv`, kể cả để dự đoán
hoặc kiểm tra định dạng. Không tạo file submission mới trong chế độ này.

`train_only.py` chặn thao tác mở file có tên đó trước khi đọc dữ liệu. Mọi lệnh
bên dưới chạy trong chế độ này. Notebook cũ và các lệnh dự đoán blind trong tài
liệu trước đây không được chạy ở lượt thử hiện tại.

## Cài đặt

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Giữ `train.csv` gốc ở một thư mục riêng. Gói mô hình không chứa dữ liệu gốc.

## Đánh giá lại trên train

Sau khi giải nén gói, chạy:

```bash
python train_only.py evaluate_train_only.py --model-dir models/stacked --train /duong/dan/train.csv --out-dir validation_review
```

Lệnh mặc định tái tính bộ kết hợp và chỉ số từ đầu ra encoder được lưu trên
938 mẫu validation của train; nó không huấn luyện lại encoder. ID, thứ tự,
hash dữ liệu và trọng số được kiểm tra trước khi tính điểm.

Để chạy lại encoder thực sự trên chính 938 mẫu này:

```bash
python train_only.py evaluate_train_only.py --model-dir models/stacked --train /duong/dan/train.csv --out-dir validation_fresh --fresh-encoder-scores --threads 2 --batch-size 8
```

Lần chạy đầu cần tải các backbone pretrained công khai, đúng phiên bản được
ghim trong cấu hình. Tệp adapter chứa các tham số đã học. RAM dùng tuần tự từng
encoder; chạy trên CPU được, nhưng chậm hơn việc đọc đầu ra validation đã lưu.

Kết quả gồm metrics, dự đoán trên validation và năm ví dụ sai lấy từ train.
Những ví dụ này dành cho báo cáo riêng; không đưa nội dung đánh giá gốc lên
repository công khai.

## Cách hiểu điểm

ABSA-F1 là trung bình F1 của 14 nhãn khen/chê. Điểm validation không thay thế
điểm công khai. Validation đã được xem qua nhiều lượt phát triển; khoảng tin
cậy bootstrap không tính hết sai số do chọn phương pháp qua nhiều lần thử.

Các refit `outer80` chỉ học fit+tune (3.750 mẫu); 938 mẫu validation vẫn nằm
ngoài mọi bước huấn luyện. Meta và ngưỡng giữ nguyên khi thay encoder bằng bản
refit. Không học meta mới từ dự đoán in-sample của các encoder này.

Các script huấn luyện vẫn được kèm theo. Khi chạy bất kỳ script nào, đặt
`python train_only.py` trước tên script. Các mô hình học trên toàn bộ train
không được báo cáo như có một validation độc lập sau refit.
