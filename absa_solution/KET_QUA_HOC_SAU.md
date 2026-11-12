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
| Token-attention + 1.400 mẫu train tổng hợp trọng số 0,4 | 0.8684 | [0.8534; 0.8810] |
| Stacking thêm encoder đã học mẫu tổng hợp | 0.8804 | [0.8651; 0.8934] |
| Token-attention + chuẩn hóa viết tắt, không thêm mẫu | 0.8701 | [0.8551; 0.8846] |
| Stacking 5 encoder có chuẩn hóa viết tắt | 0.8826 | [0.8685; 0.8960] |

Bản chuẩn hóa dùng đúng split gốc; số review trùng sau chuẩn hóa giữa từng cặp
fit/tuning/validation đều bằng 0. Meta OOF tuning 0,892538, validation 0,882568.
File đã tạo có SHA256 e38970af7b0519f21a41c0b07fd8cb545f0e86bc71730ee331168ce21d02db02,
chưa có điểm công khai. Bộ 0,8840 công khai vẫn được giữ riêng.

Refit aspect-pair dùng 3.750 review thật, lịch hai epoch cố định, ngưỡng giữ từ
lựa chọn cũ: validation 0,867488, CI95% [0,853582; 0,879729]. Thay duy nhất encoder
cặp trong bộ kết hợp, giữ nguyên meta/ngưỡng: validation 0,882001, CI95%
[0,867786; 0,894539]. Các encoder và meta vẫn chưa học 938 review validation ở
phép đo này. Đây là biến thể khác, chưa có điểm công khai và chưa thay file 0,8840.

Trọng số trộn xác suất chọn theo từng khía cạnh bằng tuning. Stacking dùng dự
đoán của 4 mô hình fit trên 2.812 review; meta logistic học trên 938 review tuning,
chọn regularization/ngưỡng bằng dự đoán 5-fold out-of-fold trong tuning. Validation
938 review không huấn luyện encoder hay meta. Vì meta dùng thêm nhãn tuning,
stacking có 3.750 review đóng góp huấn luyện; các hàng encoder đơn chỉ có 2.812.
Không coi số tầng lớn hơn là bằng chứng điểm tốt hơn.

Token-attention chọn epoch 5, tuning F1 0,8703. Khi thêm vào stacking, meta OOF
tuning giảm từ 0,892105 xuống 0,891557; validation giảm từ 0,880738 xuống 0,880376.
Giữ bộ 4 encoder cho file đã chấm. Refit aspect-pair 80% đã có số đo riêng ở trên.
Đang thử chuẩn hóa viết tắt, PhoBERT-large 24 tầng giữ cố định 20 tầng đầu và
mDeBERTa dùng tokenizer native (gồm cả khởi tạo từ mô hình NLI), cùng shared head
khía cạnh. Chưa có kết quả hoàn tất cho các lượt này; không ghi điểm dự kiến
thành điểm đo.

Bản loss cân bằng warm-start từ checkpoint tốt nhất của bản 4 tầng cuối, rồi tinh chỉnh thêm 4 epoch. Head phụ nhắc được giảm trọng số; loss dương/âm cân theo từng nhãn. Chọn checkpoint/threshold vẫn trên tuning.

Các model trong bộ chính học từ train có nhãn, không dùng review tổng hợp.
Thử nghiệm 1.400 mẫu bổ sung do trợ lý soạn được báo cáo riêng, chưa chọn vào bộ
chính; xem THU_NGHIEM_DU_LIEU_THEM.md. Không đưa văn bản test vào LLM/API và không
gán nhãn test thủ công. Validation đã được dùng làm phản hồi phát triển ở mức
tổng hợp nên không phải phép xác nhận hoàn toàn độc lập.

Export model mặc định không kèm submission.csv. Việc công bố dự đoán trên test là thao tác riêng, không phải đăng mã nguồn.
