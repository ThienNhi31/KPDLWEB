# Lời giải ABSA tiếng Việt — phương pháp CPU mới

**Chưa đạt ABSA-F1 0,98.** Điểm validation:

| Phương pháp | ABSA-F1 |
|---|---:|
| NB-SVM ban đầu | 0.7561 |
| Tách từ + ensemble | 0.7656 |
| Ngữ cảnh theo khía cạnh | 0.7815 |

Không biết điểm test/private. Xem `KET_QUA_CPU.md` và `PHUONG_PHAP_MOI.md`.

## Chạy trên máy không có GPU

1. Tải repository: **Code → Download ZIP**, giải nén.
2. Mở terminal trong `absa_solution/`.
3. Python đã dùng: 3.12.14. Chạy:

```bash
python -m pip install -r requirements_cpu.txt
python scoped_cpu.py train --data-dir .. --out-dir scoped_results
python score_absa.py --kiem-tra scoped_results/submission.csv ../test_blind.csv
```

Các CSV hiện ở thư mục cha. Nếu đặt dữ liệu riêng nơi khác, thay `..` bằng
đường dẫn đó. Hoặc mở `submission_cpu.ipynb`, chọn kernel Python đã cài PyVi,
và Run All. Chưa có kết quả thì notebook tự huấn luyện. Để bắt buộc fit lại,
đặt FORCE_TRAIN=True.

Chương trình tạo `scoped_results/model.joblib`, báo cáo chỉ số và
`scoped_results/submission.csv`. Không kèm mô hình/file dự đoán trong bản công khai.
Xem KET_QUA_CPU.md để biết thời gian đo trên CPU; không cần cài torch cho hướng này.

Hướng tách từ + ensemble khác:

```bash
python cpu_ensemble.py train --data-dir .. --out-dir cpu_results
```

Mã PhoBERT nằm trong `phobert_absa.py`, đã kiểm tra backward nhưng **chưa có
kết quả train đầy đủ**; chạy CPU chậm. Không coi đó là mô hình đã đạt 0,98.

Bản baseline cũ vẫn nằm trong `pipeline.py` và `submission.ipynb`.
Không đăng dữ liệu lớp hay output chứa nguyên văn đánh giá lên GitHub công khai.
Các notebook công khai đã bỏ output có nguyên văn; chạy cục bộ sẽ tạo lại.
Không sinh dữ liệu thêm; không dùng LLM/API gán nhãn test.

## Huấn luyện PhoBERT đang chạy

Đã bắt đầu hai thực nghiệm PhoBERT có huấn luyện thực sự: bản cache 4 tầng cuối trên cloud và 12 tầng encoder trên GitHub Actions. **Chưa có điểm validation hoàn tất cho các lượt này**, không tuyên bố đạt 0,90.

- [Theo dõi lượt GitHub Actions](https://github.com/ThienNhi31/KPDLWEB/actions/runs/43050114260).
- Xem `HUAN_LUYEN_PHO_BERT.md` để chạy trên CPU, resume, và tải mô hình gọn.
- `finetune_cpu_cached.py` giữ cache riêng tư, kiểm tra đường tính cache khớp encoder đầy đủ trước khi học.
