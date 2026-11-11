# Kết quả học sâu đã đo

**Chưa xác nhận đạt 0,90. Chưa có điểm công khai cho file ứng viên.**

Cùng split review 2.812 fit /938 tuning /938 validation, seed2026. Chọn epoch và ngưỡng bằng tuning. Bảng dưới là validation, không phải tuning hay điểm trên trang chấm.

| Phương pháp | ABSA-F1 validation | CI95% bootstrap |
|---|---:|---|
| PhoBERT 4 tầng cuối | 0.8538 | [0.8379; 0.8680] |
| PhoBERT + loss cân bằng | 0.8577 | [0.8424; 0.8714] |
| PhoBERT 12 tầng | 0.8129 | [0.7952; 0.8286] |

ViSoBERT còn đang chạy. Phương pháp aspect-pair là thử nghiệm mới chưa có điểm. Không coi số tầng lớn hơn là bằng chứng điểm tốt hơn.

Bản loss cân bằng warm-start từ checkpoint tốt nhất của bản 4 tầng cuối, rồi tinh chỉnh thêm 4 epoch. Head phụ nhắc được giảm trọng số; loss dương/âm cân theo từng nhãn. Chọn checkpoint/threshold vẫn trên tuning.

Tất cả mô hình học từ train có nhãn. Không thêm đánh giá/nhãn tổng hợp. Không đưa văn bản test vào LLM/API và không gán nhãn test thủ công. Validation đã được dùng làm phản hồi phát triển ở mức tổng hợp nên không phải phép xác nhận hoàn toàn độc lập.

Export model mặc định không kèm submission.csv. Việc công bố dự đoán trên test là thao tác riêng, không phải đăng mã nguồn.
