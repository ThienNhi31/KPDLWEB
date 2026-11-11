# Hướng dẫn tái lập ABSA

Python đã dùng: 3.12.14. Mô hình chạy CPU, không cần CUDA. Seed: 2026.

## Chuẩn bị

Giải nén vào thư mục riêng. Đặt các tệp giảng viên cung cấp vào `data/`:
`train.csv`, `test_blind.csv`, `submission_demo.csv`. `score_absa.py` đi kèm
là mã chấm gốc, không sửa. Không đính kèm dữ liệu gốc trong gói bài nộp.
Phiên bản dữ liệu tham chiếu: repository ThienNhi31/KPDLWEB, commit
`7c052a71010eac9361ea8803b9e8bdd6cfc99be8`; SHA-256 trong `results/run_info.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Trên Windows dùng `.venv\Scripts\activate`. Không cần tải mô hình ngoài.

## Chạy notebook

Mở `submission.ipynb` bằng Jupyter, chọn kernel của môi trường trên và Run All.
Nếu notebook nằm cùng `pipeline.py`, nó tự đọc `data/` và ghi `results/`.
Có thể dùng biến môi trường `ABSA_DATA_DIR` để chỉ định vị trí dữ liệu.

Notebook thực sự huấn luyện cả ba mô hình, xuất bảng so sánh và toàn bộ các
chỉ số, đọc ví dụ lỗi từ validation, huấn luyện lại trên toàn bộ train rồi tạo
`results/submission.csv`. Không hiển thị văn bản test trong notebook.

Chạy tương đương từ dòng lệnh:

```bash
python pipeline.py --data-dir data --out-dir reproduced
python score_absa.py --kiem-tra reproduced/submission.csv data/test_blind.csv
python -c "from pathlib import Path; assert Path('results/submission.csv').read_bytes() == Path('reproduced/submission.csv').read_bytes(); print('Exact match')"
```

Ước lượng thời gian: khoảng 1–4 phút trên CPU của môi trường đã chạy; phụ thuộc
máy. Số đo thực tế và cấu hình phần cứng nằm trong `results/run_info.json`;
thời gian huấn luyện từng mô hình nằm trong `results/comparison.csv`.

## Chỉ chạy suy luận bằng mô hình đã lưu

```bash
python predict.py --model results/model.joblib --test data/test_blind.csv --output submission_reproduced.csv
python -c "from pathlib import Path; assert Path('results/submission.csv').read_bytes() == Path('submission_reproduced.csv').read_bytes(); print('Exact match')"
```

Giữ `pipeline.py` cùng thư mục khi nạp `model.joblib` vì mô hình NB-SVM dùng
lớp `NBSVM` trong module đó. Chỉ nạp file joblib tin cậy.

## Giao thức đánh giá và giới hạn

Chuẩn hóa Unicode NFC, chữ thường và khoảng trắng; không xóa dấu tiếng Việt,
phủ định hoặc các placeholder. Các văn bản giống nhau sau chuẩn hóa nằm cùng
một nhóm. Tách khoảng 60% fit / 20% tuning / 20% validation bằng GroupShuffleSplit.
Vectorizer, tỷ số NB và trọng số classifier chỉ fit trên tập fit khi so sánh.
Ngưỡng cho từng nhãn và lựa chọn mô hình dựa trên tuning. Cả ba mô hình dùng
cùng validation. Chỉ dùng train có nhãn; không sinh thêm dữ liệu.

Ban đầu đã xem kết quả hai baseline rồi bổ sung NB-SVM. Vì quá trình phát triển
đã tham khảo điểm validation ở mức tổng hợp, kết quả validation không nên được
coi là đánh giá xác nhận hoàn toàn độc lập của một nghiên cứu đã đăng ký trước.
Không chỉnh ngưỡng theo validation, không dùng điểm leaderboard hay nhãn test.
Bootstrap 1.000 lần lấy mẫu theo đánh giá, theo đúng hàm của mã chấm; CI này là
độ bất định lấy mẫu có điều kiện với mô hình cố định, không gồm biến thiên giữa
các lần huấn luyện hay dịch chuyển phân bố train/test.

Sau lựa chọn, fit lại vectorizer và mô hình trên toàn bộ train, giữ các ngưỡng
đã học từ tuning. Tái fit làm phân bố margin có thể thay đổi; đây là một giới hạn
của chiến lược ngưỡng này. Không thể suy ra điểm private từ điểm validation.

`nhac = nhac OR khen OR che` bảo đảm ràng buộc nhãn. Có thể đồng thời khen và
chê. Nhãn nhắc vẫn được học riêng để giữ các trường hợp trung tính.

Đây là lời giải cổ điển có thể tái lập, chưa đạt mốc validation 0,80. Không có
đáp án test để khẳng định đạt ngưỡng private. Một hướng cải thiện tiếp theo là
tinh chỉnh mô hình tiếng Việt với cùng nguyên tắc tách dữ liệu và đánh giá trên
một holdout mới hoặc nested CV; phần đó chưa được huấn luyện trong gói này.

Không dùng LLM/API dự đoán hay gán nhãn test, không gán nhãn thủ công test.
LLM chỉ hỗ trợ viết mã và diễn giải lỗi từ dữ liệu train. Không có
`du_lieu_them.csv` vì không sinh dữ liệu huấn luyện bổ sung.

## Sản phẩm

- `submission.ipynb`: notebook đã chạy, giữ output.
- `pipeline.py`, `predict.py`, `score_absa.py`: mã nguồn.
- `requirements.txt`: phiên bản thư viện.
- `BAO_CAO.md`: kết quả thực nghiệm và phân tích lỗi.
- `results/submission.csv`: dự đoán từ mô hình tự huấn luyện, 732 dòng, 22 cột.
- `results/model.joblib`: vectorizer, mô hình, ngưỡng.
- `results/comparison.csv`, `metrics.json`: chỉ số và CI.
- `results/validation_*.csv`, `validation_truth.csv`: nhãn để đối chiếu, không có text.
- `results/validation_errors.json`: ví dụ lỗi từ train, không phải test.
- `results/split.csv`, `label_counts.csv`, `thresholds.json`, `run_info.json`: tái lập.

Không có kết quả chấm công khai trong phiên này, nên chưa thể xác nhận đây là
file có điểm công khai cao nhất. Nếu nộp nhiều lần, lưu đúng file tốt nhất theo
trang chấm và cập nhật gói bài tương ứng. Đổi tên ZIP theo mã nhóm trước khi nộp.
