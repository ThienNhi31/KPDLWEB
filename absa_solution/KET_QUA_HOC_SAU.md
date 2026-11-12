# Kết quả học sâu đã đo

**Mục tiêu 0,96 chưa đạt. Người dùng đã chấm file ứng viên stacking và gửi ảnh
ABSA-F1 công khai 0,8840 cho nhóm KPW2-NHOM3.**

Ảnh kết quả: F1 khen 0,929; F1 chê 0,839; Precision 0,913; Recall 0,861;
Accuracy 0,967; kappa 0,861. Đây là điểm trên 220 mẫu công khai, không phải 512
mẫu riêng tư. File ứng viên được gửi ngay trước ảnh có SHA256
`6b0bd799cb3270ce87d3f2cbc6828fc50a08b261299b1dd0ceacc76db8af735d`;
ảnh không hiển thị checksum, việc gắn phiên bản dựa trên phản hồi trong hội thoại.
Mức yếu trong ảnh gồm vị trí–chê 0,700, phòng–khen 0,831 và tiện nghi–chê 0,792.
Không sửa nhãn từng mẫu test; phản hồi tổng hợp chỉ định hướng thử nghiệm trên train.

Cùng split review 2.812 fit /938 tuning /938 validation, seed2026. Chọn epoch và ngưỡng bằng tuning. Bảng dưới là validation, không phải tuning hay điểm trên trang chấm.

| Phương pháp | ABSA-F1 validation | CI95% bootstrap |
|---|---:|---|
| PhoBERT 4 tầng cuối | 0.8538 | [0.8379; 0.8680] |
| PhoBERT + loss cân bằng | 0.8577 | [0.8424; 0.8714] |
| PhoBERT 12 tầng | 0.8129 | [0.7952; 0.8286] |
| ViSoBERT 12 tầng | 0.8321 | [0.8175; 0.8459] |
| PhoBERT aspect-pair, shared head | 0.8666 | [0.8522; 0.8795] |
| Trộn xác suất 2 joint + aspect-pair | 0.8777 | [0.8634; 0.8906] |
| Stacking 4 encoder, meta 5-fold trên tuning | 0.8807 | [0.8655; 0.8938] |
| PhoBERT label-specific token attention | 0.8659 | [0.8507; 0.8791] |
| Stacking thêm token-attention, 5 encoder | 0.8804 | [0.8657; 0.8937] |

Trọng số trộn xác suất chọn theo từng khía cạnh bằng tuning. Stacking dùng dự
đoán của 4 mô hình fit trên 2.812 review; meta logistic học trên 938 review tuning,
chọn regularization/ngưỡng bằng dự đoán 5-fold out-of-fold trong tuning. Validation
938 review không huấn luyện encoder hay meta. Vì meta dùng thêm nhãn tuning,
stacking có 3.750 review đóng góp huấn luyện; các hàng encoder đơn chỉ có 2.812.
Không coi số tầng lớn hơn là bằng chứng điểm tốt hơn.

Token-attention chọn epoch 5, tuning F1 0,8703. Khi thêm vào stacking, meta OOF
tuning giảm từ 0,892105 xuống 0,891557; validation giảm từ 0,880738 xuống 0,880376.
Giữ bộ 4 encoder. Đang thử refit aspect-pair trên 80% review gốc với lịch/ngưỡng
cố định, PhoBERT-large 24 tầng giữ cố định 20 tầng đầu và mDeBERTa dùng tokenizer
native, cùng shared head khía cạnh. Chưa có kết quả hoàn tất cho các lượt này;
không ghi điểm dự kiến thành điểm đo.

Bản loss cân bằng warm-start từ checkpoint tốt nhất của bản 4 tầng cuối, rồi tinh chỉnh thêm 4 epoch. Head phụ nhắc được giảm trọng số; loss dương/âm cân theo từng nhãn. Chọn checkpoint/threshold vẫn trên tuning.

Tất cả mô hình học từ train có nhãn. Không thêm đánh giá/nhãn tổng hợp. Không đưa văn bản test vào LLM/API và không gán nhãn test thủ công. Validation đã được dùng làm phản hồi phát triển ở mức tổng hợp nên không phải phép xác nhận hoàn toàn độc lập.

Export model mặc định không kèm submission.csv. Việc công bố dự đoán trên test là thao tác riêng, không phải đăng mã nguồn.
