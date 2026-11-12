# Hướng dẫn bản chính: ABSA trên CPU

File đã được người dùng chấm có **ABSA-F1 công khai 0,8840**. Mục tiêu 0,96 chưa đạt. Notebook chính tái lập bộ stacking 4 encoder, validation **0,880738** trên 938 review, CI95% **[0,865538; 0,893802]**. Các biến thể đang thử được báo cáo riêng trong KET_QUA_HOC_SAU.md; không gán điểm public cũ cho file từ mô hình khác.

SHA256 của file đã gửi trước ảnh chấm:

`6b0bd799cb3270ce87d3f2cbc6828fc50a08b261299b1dd0ceacc76db8af735d`

public_score_report.json ghi nguồn là ảnh người dùng cung cấp. Ảnh không hiển thị hash; việc liên kết phiên bản dựa trên phản hồi trong hội thoại. Điểm private 512 mẫu chưa có. Accuracy 0,967 trên ảnh không thay cho ABSA-F1.

## File và môi trường

Python 3.12.14 tại cloud, 3.12.16 trên runner GitHub. CPU, không dùng GPU/CUDA. Cài trước bản PyTorch CPU rồi các requirements có phiên bản đã ghim:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Đặt train.csv, test_blind.csv và submission_demo.csv ở thư mục cha; source và score_absa.py ở thư mục chứa submission.ipynb. Có thể đặt ABSA_DATA_DIR trỏ tới thư mục dữ liệu khi chạy notebook. Không kèm các CSV gốc trong gói nộp.

Đặt checkpoint đã chọn vào model_bundle với các thư mục cached_phobert, balanced_phobert, aspect_pair, visobert và stacked. Adapter chỉ bỏ những tensor đã kiểm tra khớp chính xác pretrained base ghim revision. Không cần GPU để nạp. Base weights cần tải lần đầu; cache chúng để dùng offline sau đó.

## Chạy lại file dự đoán

```bash
python stacked_absa.py predict --model-dir model_bundle/stacked --test ../test_blind.csv --output submission.csv --threads 2 --batch-size 8
python score_absa.py --kiem-tra submission.csv ../test_blind.csv
```

Notebook submission.ipynb mặc định dùng checkpoint và các xác suất tuning đã lưu từ quá trình train, fit lại meta, báo cáo validation và tạo CSV. Nó kiểm tra hash cấu hình/trọng số, thứ tự model/cột và đối chiếu với file tham chiếu có trước. Đặt RETRAIN_BASES=True để train từ đầu; RECOMPUTE_BASE_VALIDATION=True để chạy forward lại cả tuning/validation. Đọc HUONG_DAN_STACKING.md cho tất cả lệnh train, export adapter và refit. Không ghi thời gian nạp checkpoint thành thời gian train.

Lần suy luận lạnh đã đo: 952,50 giây cho 732 review và 4 encoder trên quota 2 CPU, trong lúc có một job train khác cùng chạy. Đây là số đo cho toàn pipeline, không phải tốc độ riêng của mỗi encoder. inference_cost.json lưu số đo và hash CSV. Tái huấn luyện toàn bộ từ đầu có thể mất 4–8 giờ trên 2 CPU; nên có 16 GB RAM khi train model theo cặp vì cache lớn. Xem lịch sử và run_info cho từng số đo.

Các workflow mới trên GitHub chỉ xuất model và thống kê từ train. Artifact có hạn lưu, cần tải checkpoint về trước khi hết hạn. Bundle tái lập trên GitHub:

https://github.com/ThienNhi31/KPDLWEB/actions/runs/43158772777

Đối chiếu đúng hash CSV khi dùng một bundle khác; số liệu validation khớp không tự chứng minh file dự đoán trùng. Bundle compact tại cloud đã được chạy suy luận và xác nhận khớp byte với file có hash nêu trên.

## Đánh giá và phân tích

Chia group theo review chuẩn hóa, seed2026: 2.812 fit /938 tuning /938 validation. Encoder chỉ học fit; meta học tuning và chọn tham số/ngưỡng bằng 5-fold out-of-fold. Stacking dùng tổng cộng 3.750 review có đóng góp huấn luyện. So sánh TF-IDF và encoder trên cùng validation; báo cáo F1/P/R/accuracy/kappa/MCC/balanced accuracy, bootstrap 1.000 lần và 5 lỗi train được giữ làm validation.

Bản nộp riêng giữ đủ output của 5 lỗi. Bản notebook công khai xóa output nguyên văn ở cell này để không công bố review gốc. Validation đã được xem trong các vòng phát triển; CI không tính bất định do chọn mô hình hay dịch chuyển train/test. Không chấm mô hình refit toàn train như thể validation vẫn chưa được học.

Bộ 4 encoder chính không sử dụng dữ liệu tổng hợp. Thử nghiệm 1.400 mẫu bổ sung được tách riêng, có du_lieu_them.csv, JSON công cụ/đặc tả và ablation; kết quả chưa vượt bộ chính. Xem THU_NGHIEM_DU_LIEU_THEM.md. LLM chỉ hỗ trợ code, phân tích lỗi train và soạn mẫu train bổ sung; không nhận văn bản test hoặc tạo nhãn test.

Bản TF-IDF cũ: HUONG_DAN_CO_DIEN.md. Các file .zip cũ chỉ là phiên bản lịch sử; file nộp cuối phải khớp đúng mô hình và CSV có điểm công khai tốt nhất đã xác nhận.
