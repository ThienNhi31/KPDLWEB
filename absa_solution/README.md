# Lời giải ABSA tiếng Việt

Bản mã nguồn để tải về và tự chạy; không kèm mô hình hoặc dữ liệu.

1. Tải repository bằng **Code → Download ZIP**, giải nén.
2. Mở terminal trong thư mục `absa_solution/`.
3. Cài thư viện: `python -m pip install -r requirements.txt`. Python đã dùng: 3.12.14.
4. Các CSV hiện nằm ở thư mục cha của `absa_solution/`; chạy:

```bash
python pipeline.py --data-dir .. --out-dir results
python score_absa.py --kiem-tra results/submission.csv ../test_blind.csv
```

Hoặc đặt `ABSA_DATA_DIR=..` trước khi mở `submission.ipynb` rồi Run All.
Nếu dùng dữ liệu giảng viên lưu riêng, thay `..` bằng đường dẫn đó.

Chương trình huấn luyện ba mô hình, tính chỉ số, tạo `results/model.joblib` và
`results/submission.csv`. Xem HUONG_DAN.md và BAO_CAO.md.

**ABSA-F1 validation tốt nhất: 0,7561; chưa đạt mốc 0,80. Chưa có điểm private.**

Bản công khai bỏ output có nguyên văn đánh giá trong notebook và báo cáo.
Chạy notebook tại máy sẽ tạo lại output đó. Không tải dữ liệu lên dịch vụ ngoài
và không dùng LLM để gán nhãn test. Các tệp sinh ra được bỏ qua bằng .gitignore.
Không đăng dữ liệu lớp hoặc ví dụ nguyên văn lên repository công khai.

Gói này không chứa file submission hoặc mô hình đã huấn luyện; chạy lại mã để tạo.
