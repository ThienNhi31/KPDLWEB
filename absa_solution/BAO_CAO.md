# Báo cáo thực nghiệm ABSA

Dữ liệu: 4.688 đánh giá train, 732 đánh giá test không nhãn. Không dùng dữ liệu bổ sung. Phân chia: 2.812 fit, 938 tuning, 938 validation; seed 2026. Không đưa văn bản test vào LLM.

## So sánh mô hình

| Mô hình | Tuning F1 | Validation F1 | CI 95% | Fit (giây) |
|---|---:|---:|---|---:|
| logistic | 0.7648 | 0.7323 | [0.7133, 0.7490] | 4.76 |
| svm | 0.7709 | 0.7411 | [0.7231, 0.7570] | 2.91 |
| nbsvm | 0.7809 | 0.7561 | [0.7394, 0.7718] | 3.77 |

Chọn **nbsvm** theo tuning. Vectorizer của cả ba mô hình giống nhau: TF-IDF word 1–2 gram và char 3–5 gram. NB-SVM thêm tỷ số log-count theo từng nhãn, tính từ fit, trước LinearSVC. Các mô hình học độc lập 21 nhãn. Ngưỡng từng nhãn chọn từ 17 giá trị trong [-1,4; 0,2] trên tuning; đồng hạng ưu tiên gần 0.

Đã thêm NB-SVM sau khi xem điểm tổng hợp của baseline. Vì vậy validation có vai trò phản hồi phát triển ở mức mô hình; không nên coi đây là holdout xác nhận hoàn toàn độc lập. Không chọn ngưỡng hay lựa chọn mô hình bằng điểm validation. Không biết nhãn/điểm test.

## Chỉ số mô hình được chọn

| Chỉ số | Giá trị |
|---|---:|
| ABSA_F1 | 0.7561 |
| precision_TB | 0.7677 |
| recall_TB | 0.7501 |
| accuracy_TB | 0.9208 |
| kappa_TB | 0.7027 |
| mcc_TB | 0.7044 |
| bal_acc_TB | 0.8478 |
| F1_phat_hien_khia_canh_TB | 0.8885 |

CI95% ABSA-F1: [0.7393548036272827, 0.7717808702972868]; bootstrap 1.000 lần theo đánh giá. CI có điều kiện với mô hình đã fit, không gồm bất định huấn luyện và dịch chuyển phân bố. Accuracy cao không đồng nghĩa mô hình tốt, do nhiều nhãn âm.

## Khía cạnh khó

| Khía cạnh | F1 khen | F1 chê | Trung bình |
|---|---:|---:|---:|
| nhan_vien | 0.9127 | 0.6540 | 0.7834 |
| sach_se | 0.9712 | 0.8289 | 0.9001 |
| vi_tri | 0.8937 | 0.5688 | 0.7312 |
| phong | 0.7676 | 0.7911 | 0.7794 |
| gia_tri | 0.8464 | 0.5714 | 0.7089 |
| an_uong | 0.8146 | 0.7197 | 0.7671 |
| tien_nghi | 0.6163 | 0.6291 | 0.6227 |

Khía cạnh khó nhất: **tien_nghi**. Nhãn yếu nhất: **vi_tri_che**, F1=0.5688. Nhãn hiếm có ít mẫu dương để học và ước lượng ngưỡng; không thể suy ra nguyên nhân chỉ từ F1. Xem label_counts.csv để đối chiếu.

## Năm ví dụ lỗi

Các ví dụ sau thuộc train được tách sang validation. Không sửa nhãn gốc. Nhận xét là giả thuyết giải thích lỗi.

### train_0008


Các nhãn sai (thật → dự đoán): `nhan_vien_khen`: 1 → 0, `sach_se_khen`: 0 → 1, `phong_nhac`: 0 → 1.

Viết tắt nv/dthuong làm mất lời khen nhân viên. Cụm chưa sạch chứa từ sạch nhưng mang phủ định. Chỉ nhắc tới phòng trong nội dung sạch sẽ không tự động thuộc khía cạnh phòng theo nhãn gốc.

### train_0011


Các nhãn sai (thật → dự đoán): `nhan_vien_nhac`: 0 → 1, `phong_nhac`: 0 → 1, `phong_che`: 0 → 1.

Mùi hôi nhà vệ sinh thuộc sạch sẽ theo nhãn gốc; mô hình lan nhãn sang phòng/nhân viên vì tương quan từ vựng.

### train_0012


Các nhãn sai (thật → dự đoán): `vi_tri_che`: 0 → 1, `an_uong_che`: 1 → 0, `tien_nghi_che`: 0 → 1.

Đoạn dài có nhiều khía cạnh và tương phản; mô hình bỏ sót chê buffet, đồng thời lan lời chê sang vị trí/tiện nghi so với nhãn gốc.

### train_0013


Các nhãn sai (thật → dự đoán): `sach_se_nhac`: 1 → 0, `sach_se_khen`: 1 → 0.

Mùi thơm là tín hiệu gián tiếp về sạch sẽ. Mô hình từ vựng bỏ sót nhắc và khen sạch sẽ.

### train_0022


Các nhãn sai (thật → dự đoán): `an_uong_khen`: 0 → 1.

Cần tốt hơn là gợi ý cải thiện, thuộc lời chê nhẹ; từ tốt làm phát sinh dự đoán khen sai.

## Chi phí, tái lập và giới hạn

Python 3.12.14; Linux-6.18.57-x86_64-with-glibc2.41; CPU INTEL(R) XEON(R) PLATINUM 8573C; 3 logical CPU; MemTotal:       10207480 kB; không dùng GPU/CUDA. Giới hạn thread tính toán: 1. Fit vectorizer khi đánh giá: 4.31 giây; fit cuối trên toàn train, gồm vectorizer: 12.49 giây; suy luận 732 test, gồm vectorize: 0.57 giây; tổng pipeline: 37.01 giây.

Đã chạy notebook đầy đủ, kiểm tra định dạng bằng code giảng viên, nạp lại mô hình và so toàn bộ dự đoán. Không kèm train.csv/test_blind.csv gốc trong ZIP. requirements và hướng dẫn chạy được cung cấp.

**Kết quả validation chưa đạt 0,80.** Không có nhãn test, không biết điểm công khai/private và không khẳng định đạt mốc chấm. Sau khi fit lại toàn train, giữ nguyên ngưỡng tuning có thể bị thay đổi phân bố margin. Train lấy dư lời chê so với test là rủi ro dịch chuyển phân bố chưa lượng hóa.

Hướng tiếp theo: thử PhoBERT/ViSoBERT tinh chỉnh trên train, chuẩn hóa viết tắt, tăng khả năng xử lý phủ định và liên kết khía cạnh–cảm xúc; đánh giá các lựa chọn mới bằng holdout mới hoặc nested CV. Chưa thực hiện các thử nghiệm đó. Không dùng các lỗi validation để viết luật sửa từng dòng.

Không có thí nghiệm dữ liệu thêm vì không tạo dữ liệu bổ sung. Không dùng điểm sao, không tìm đánh giá gốc, không gán nhãn thủ công test.

SHA-256 submission: `15f913519d5ec57d1de452709c0766278d889337b3d3e7c7c3b3b2ba53c17ce2`.
Bản GitHub đã bỏ nguyên văn đánh giá. Chạy notebook trên máy sẽ tạo lại ví dụ lỗi từ dữ liệu train của bạn.
