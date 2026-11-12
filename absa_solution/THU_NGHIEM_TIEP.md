# Các thử nghiệm tiếp theo trên CPU

Điểm công khai được người dùng xác nhận bằng ảnh trang chấm là **ABSA-F1 0,8840**.
Chưa có kết quả xác nhận đạt 0,96. Accuracy 0,967 trong ảnh là chỉ số khác.

Giữ nguyên phép chia nhóm theo văn bản chuẩn hóa và seed 2026: 2.812 mẫu fit,
938 mẫu chọn mô hình/ngưỡng, 938 mẫu validation. Các kết quả bên dưới đã được
xem trong quá trình phát triển, không phải một holdout xác nhận mới.

| Phương án | ABSA-F1 validation | Điểm công khai đã xác nhận |
|---|---:|---:|
| Bốn encoder + meta logistic có cross-fitting | 0,880738 | 0,8840 |
| Thêm encoder có attention và chuẩn hóa viết tắt | 0,882568 | Chưa có |
| Cùng năm encoder, ổn định ngưỡng bằng bootstrap | 0,882755 | Chưa có |
| Năm encoder + mô hình TF-IDF/NB | 0,878694 | Chưa có |
| Encoder toàn văn thứ nhất học thêm trên outer80 | 0,856092 | Chưa có |
| Encoder toàn văn cân bằng loss học thêm trên outer80 | 0,860196 | Chưa có |

Mức tăng 0,882568 → 0,882755 rất nhỏ; chưa dùng nó để khẳng định một file nộp
tốt hơn trên tập công khai. `bootstrap_thresholds.py` tái tạo đúng điểm OOF của
các meta estimator đã chọn, sau đó lấy trung vị ngưỡng tối ưu qua 1.000 Bayesian
bootstrap trên OOF của train. Hệ số meta được giữ nguyên; không dùng nhãn
validation để chọn ngưỡng và không đọc test.

`refit_cached_absa.py --scope outer80 --loss preserve` học thêm với lịch hai
epoch cố định trên fit+tune (3.750 mẫu). `fixed_refit_stack.py` kiểm tra nguồn
khởi tạo, giữ nguyên hệ số meta và ngưỡng cũ, rồi đánh giá trên 938 mẫu validation
vẫn nằm ngoài mọi bước huấn luyện. Không huấn luyện meta mới bằng dự đoán
in-sample của encoder vừa refit.

`deeper_cached_absa.py` mở thêm bốn tầng PhoBERT để học tám tầng cuối. Các tầng
vừa mở bắt đầu từ trọng số pretrained đã ghim phiên bản; bốn tầng trên và head
giữ khởi tạo đã học từ fit. Epoch 0 vẫn được chọn nếu các epoch mới không cải
thiện tuning. Đã kiểm tra thực tế trên hai mẫu fit: đầu ra khởi tạo trùng mô hình
cũ, đầu ra cache trùng chạy encoder gốc, tầng vừa mở có gradient hữu hạn khác
không và tầng vẫn đóng không có gradient. Điểm của lượt huấn luyện đầy đủ sẽ
được ghi vào `run_info.json` khi hoàn tất.

Các encoder PhoBERT-large, mDeBERTa và đầu phân loại NLI đang được thử riêng
trên CPU. Dạng NLI ở đây là bộ phân loại xác suất cục bộ, được fine-tune bằng
nhãn train; không sinh văn bản và không gọi LLM/API để gán nhãn test.

Lượt production học thêm trên toàn bộ 4.688 mẫu train không còn một tập
validation độc lập cho trọng số sau refit. Báo cáo trước refit được lưu riêng;
không dùng điểm đó làm điểm của mô hình mới.

Mọi dự đoán blind được tạo bằng mã mô hình trong workspace riêng. Không đưa
văn bản blind vào prompt, không sửa nhãn blind bằng tay, không tìm lại đánh giá
gốc hay dùng điểm sao. Các gói nộp loại trừ train.csv và test_blind.csv gốc.
