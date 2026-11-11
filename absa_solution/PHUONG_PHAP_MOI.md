# Phương pháp mới cho máy không có GPU

Mục tiêu người dùng đề nghị là ABSA-F1 0,98. Chưa có thực nghiệm chứng minh đạt
mức đó. Không hứa điểm private và không dùng Accuracy thay cho ABSA-F1.

## Các hướng đã bổ sung

1. `cpu_ensemble.py`: chuẩn hóa viết tắt, tách từ bằng PyVi, word 1–3 gram và
   character 3–6 gram; so sánh NB-SVM, NB-logistic và Linear SVM. Chọn C, tỷ số
   NB, ngưỡng và ensemble trên tuning. Validation chỉ chấm sau khi chọn xong.
2. `scoped_cpu.py`: chia đánh giá thành mệnh đề; dùng từ khóa khía cạnh để lấy
   phần ngữ cảnh liên quan; kết hợp TF-IDF cục bộ và toàn văn. Nhãn vẫn do mô hình
   huấn luyện dự đoán, không phải luật gán nhãn test. Học ba classifier cho mỗi
   khía cạnh; chọn mức kết hợp local/global và C bằng tuning.
3. `phobert_absa.py`: hướng học sâu dùng PhoBERT-base-v2, PyVi, sliding windows
   và max pooling theo đánh giá cho 21 nhãn. Chọn epoch/ngưỡng trên tuning.
   Có lệnh smoke, train và predict; chưa huấn luyện đủ để báo ABSA-F1.

Các hướng giữ cùng split 2.812 fit / 938 tuning / 938 validation với baseline,
seed 2026, nhóm theo văn bản chuẩn hóa. Từ khóa/ngưỡng không được tạo từ test.
Test chỉ đi qua mã tiền xử lý và mô hình cục bộ để dự đoán.

## Chạy phương pháp CPU

Trong thư mục `absa_solution/`, sau khi tải toàn repository:

```bash
python -m pip install -r requirements_cpu.txt
python cpu_ensemble.py train --data-dir .. --out-dir cpu_results
python scoped_cpu.py train --data-dir .. --out-dir scoped_results
python score_absa.py --kiem-tra scoped_results/submission.csv ../test_blind.csv
```

Hoặc mở `submission_cpu.ipynb` và Run All. Notebook chạy lại phương pháp theo
khía cạnh, báo chỉ số, CI, thời gian và kiểm tra nạp mô hình. Các CSV nằm ở thư
mục cha; nếu lưu dữ liệu nơi khác, thay `..` bằng đường dẫn phù hợp.

Tái tạo suy luận mà không huấn luyện lại:

```bash
python cpu_ensemble.py predict --model cpu_results/model.joblib --test ../test_blind.csv --output ensemble_reproduced.csv
python scoped_cpu.py predict --model scoped_results/model.joblib --test ../test_blind.csv --output scoped_reproduced.csv
```

Giữ `pipeline.py`, `cpu_ensemble.py`, `scoped_cpu.py` và `score_absa.py` cùng
thư mục để nạp các class được lưu bằng joblib. Không cần torch cho hai hướng CPU.
Chỉ nạp model joblib tin cậy. Có thể mất vài phút tùy CPU; xem thời gian đo thực
tế trong `run_info.json` và bảng thực nghiệm đi kèm.

## Hướng PhoBERT, chưa có kết quả huấn luyện đầy đủ

Đã tải và chạy backward một batch thật bằng checkpoint
`vinai/phobert-base-v2`, revision `86cd7fd4c148980922ac11a2cf5e257f2ba639e1`.
Smoke test trên CPU, hai đánh giá ngắn, chỉ mở hai tầng cuối và head:
khoảng 1,67 giây/batch ở lần đầu và 9,25 giây/batch khi chạy đồng thời với mô hình CPU khác. Đây không phải điểm F1 và không chứng minh chất lượng.
Huấn luyện đủ trên CPU sẽ lâu, không có số đo tổng thời gian để cam kết.

Nếu vẫn muốn chạy trên CPU:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_phobert.txt
python phobert_absa.py smoke --data-dir .. --unfreeze-layers 2 --max-length 160 --stride 32
python phobert_absa.py train --data-dir .. --out-dir phobert_results --unfreeze-layers 2 --epochs 5 --batch-size 2 --accumulation 8 --max-length 160 --stride 32
```

Không cần đăng nhập API để dự đoán: chỉ tải checkpoint công khai một lần rồi
huấn luyện/suy luận cục bộ. Cần mạng khi tải pretrained model; dung lượng weights
khoảng 540 MB. Nhãn học từ train. Không dùng mô hình sinh để gán nhãn test.
Nếu có GPU về sau, bỏ `--unfreeze-layers 2` để fine-tune toàn encoder.

Trình train tạo model.pt, tokenizer, config, chỉ số, lịch sử epoch, lỗi validation,
thông tin runtime và submission.csv. So sánh các lần chạy chỉ dùng tuning trước
khi chọn; không dùng điểm test để quyết định nhãn từng dòng.

## Diễn giải kết quả

Chỉ số chính: ABSA-F1 = macro F1 của 14 nhãn khen/chê. Bootstrap 1.000 lần theo
đánh giá, dùng mã chấm gốc. Các báo cáo cũng gồm Precision, Recall, Accuracy,
kappa, MCC, balanced accuracy và F1 từng khía cạnh.

Validation này đã được xem trong quá trình phát triển trước đó. Dù cấu hình
mới chỉ chọn bằng tuning, điểm validation lặp lại không phải một phép xác nhận
độc lập hoàn toàn. Để kết luận mạnh cần holdout mới hoặc nested CV. Không đổi
split để tìm điểm đẹp, không học từ validation/test rồi báo đó là validation.

Sau chọn mô hình, refit trên toàn train và giữ ngưỡng tuning, có thể thay đổi
phân bố margin. Train có dư lời chê so với test nên điểm private chưa biết.
Không có bằng chứng rằng tăng độ phức tạp tự động đạt 0,98.

Không tạo dữ liệu huấn luyện thêm; không có du_lieu_them.csv. Các bản công khai
không chứa nguyên văn ví dụ lỗi, mô hình hoặc file dự đoán; chạy cục bộ sẽ tạo.
