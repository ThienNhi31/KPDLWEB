# Kết quả phương pháp CPU mới

Mục tiêu đề nghị: ABSA-F1 0,98. **Chưa đạt mục tiêu đó.** Tất cả điểm dưới đây là validation, không phải điểm private. Không có nhãn test.

Cùng split 2.812 fit / 938 tuning / 938 validation, seed 2026. Cấu hình và ngưỡng chọn trên tuning; mỗi phương pháp mới chỉ đánh giá validation sau lựa chọn. Validation đã từng được tham khảo trong phát triển, nên không phải phép xác nhận độc lập hoàn toàn.

| Phương pháp | ABSA-F1 | Bootstrap CI95% |
|---|---:|---|
| NB-SVM ban đầu | 0.7561 | [0.7394; 0.7718] |
| Tách từ + ensemble | 0.7656 | [0.7489; 0.7806] |
| Ngữ cảnh theo khía cạnh | 0.7815 | [0.7645; 0.7971] |

## Chỉ số ngữ cảnh theo khía cạnh

| Chỉ số | Giá trị |
|---|---:|
| precision_TB | 0.8092 |
| recall_TB | 0.7681 |
| accuracy_TB | 0.9348 |
| kappa_TB | 0.7385 |
| mcc_TB | 0.7423 |
| bal_acc_TB | 0.8635 |

| Khía cạnh | F1 khen | F1 chê |
|---|---:|---:|
| nhan_vien | 0.9360 | 0.7045 |
| sach_se | 0.9759 | 0.8252 |
| vi_tri | 0.9010 | 0.6055 |
| phong | 0.8365 | 0.8164 |
| gia_tri | 0.8963 | 0.4500 |
| an_uong | 0.8532 | 0.8048 |
| tien_nghi | 0.6472 | 0.6886 |

Khía cạnh khó nhất theo F1 trung bình khen/chê: tien_nghi. Nhãn yếu nhất: {'nhan': 'gia_tri_che', 'f1': 0.45}. Bootstrap 1.000 lần theo đánh giá; không bao gồm dịch chuyển phân bố train/test hoặc bất định do lựa chọn mô hình.

## Lựa chọn trên tuning

Ensemble toàn văn: [['svm', 1, 0.5], ['linear', 1, 1]], tune F1=0.7808. Ngữ cảnh theo khía cạnh: tune F1=0.8099. Không chọn bằng điểm validation.

- nhan_vien: C=2, trọng số local=0.5.
- sach_se: C=2, trọng số local=0.5.
- vi_tri: C=0.5, trọng số local=0.
- phong: C=2, trọng số local=0.5.
- gia_tri: C=0.5, trọng số local=0.5.
- an_uong: C=0.5, trọng số local=0.5.
- tien_nghi: C=0.5, trọng số local=0.5.

## Thời gian và phần cứng

INTEL(R) XEON(R) PLATINUM 8573C; RAM báo bởi hệ thống: MemTotal:       10207480 kB; Python 3.12.14; CPU, một thread tính toán, không dùng GPU. Tách từ/ensemble: 135.24 giây trước suy luận test; refit toàn train 36.92 giây; dự đoán validation, không gồm vectorize, 0.223 giây. Theo khía cạnh: 508.51 giây trước suy luận test, gồm tìm cấu hình và refit.

Mã tìm cấu hình có một cảnh báo hội tụ ở một ứng viên C lớn. Các C được chọn ghi ở phần trên; số vòng lặp của mô hình cuối được kiểm tra riêng trong phiên chạy. Không dùng một điểm Accuracy cao để tuyên bố ABSA-F1 0,98.

## Phân tích lỗi và giới hạn

Notebook cục bộ tạo tối thiểu năm ví dụ lỗi từ validation, in nhãn thật/dự đoán và loại lỗi bỏ sót/thừa. Bản công khai bỏ nguyên văn đánh giá; chạy tại máy để xem. Các nhận xét tự động là mô tả loại lỗi, không thay cho đọc và phân tích thủ công. Không dùng ví dụ test để sửa nhãn.

Từ khóa theo khía cạnh có thể không nhận ra diễn đạt gián tiếp hoặc tách sai mệnh đề; vì thế trộn đặc trưng toàn văn. Việc tối ưu nhiều ngưỡng/cấu hình trên tuning vẫn có nguy cơ quá khớp tuning. Cần holdout mới hoặc nested CV khi so sánh thêm. Refit toàn train giữ ngưỡng tuning có thể làm margin thay đổi. Train lấy dư lời chê so với test.

PhoBERT đã chạy backward trên CPU, nhưng chưa train/evaluate đầy đủ; không có điểm F1 PhoBERT để công bố. Không có cơ sở cam kết 0,98. Không tạo dữ liệu huấn luyện thêm, không dùng API/LLM sinh nhãn test.

Xem PHUONG_PHAP_MOI.md để cài và chạy.

## Nhận xét thủ công năm lỗi validation

- `train_0008`: đã nhận đúng khen nhân viên và lời chê sạch sẽ, nhưng dự đoán thừa nhắc phòng. Từ phòng nằm trong mô tả vệ sinh; ranh giới nhãn khía cạnh cần theo quy ước của giảng viên.
- `train_0011`: mô hình dự đoán thêm nhắc/chê phòng ở nội dung mùi hôi nhà vệ sinh. Nhãn gốc chỉ tính sạch sẽ ở phần đó; từ khóa nhà vệ sinh giao nhau giữa hai khía cạnh.
- `train_0012`: lời mô tả đường sửa khiến mô hình dự đoán chê vị trí, trong khi nhãn gốc bằng 0. Cần đối chiếu quy ước ranh giới, không tự sửa nhãn hay cho rằng nhãn gốc sai.
- `train_0013`: bỏ sót khen sạch sẽ khi lời khen diễn đạt gián tiếp qua mùi thơm; vẫn nhận được nhắc sạch sẽ.
- `train_0022`: dự đoán thừa khen ăn uống ở gợi ý cần tốt hơn. Dù đã tách lời khen phòng, từ tích cực trong cấu trúc yêu cầu cải thiện vẫn gây nhiễu.

Các nhận xét này chỉ dùng để giải thích lỗi, không tạo luật sửa nhãn từng dòng validation/test. Nguyên văn đánh giá chỉ có ở notebook chạy cục bộ.
