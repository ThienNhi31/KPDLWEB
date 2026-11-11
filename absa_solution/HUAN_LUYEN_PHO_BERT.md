# Huấn luyện PhoBERT trên CPU

Mục tiêu: ABSA-F1 >= 0,90. Đây là mục tiêu, không phải một điểm đã đạt.
Xem metrics.json của lượt đã hoàn tất để biết điểm validation thực tế.

## Lượt chạy hiện tại

- Cloud: fine-tune 4 tầng encoder cuối, giữ cố định 8 tầng đầu và embeddings.
- GitHub Actions: fine-tune cả 12 tầng encoder, giữ cố định embeddings.
- Workflow: https://github.com/ThienNhi31/KPDLWEB/actions/workflows/train-phobert-cpu.yml
- Cùng seed và split baseline: 2.812 fit / 938 tuning / 938 validation.
- Epoch và ngưỡng từng nhãn chọn bằng tuning. Validation chấm sau khi chọn xong.
- Validation đã được xem trong các vòng phát triển trước, không phải holdout xác
  nhận mới hoàn toàn. Không điều chỉnh từng nhãn dự đoán theo validation/test.

Không dùng API/LLM để tạo nhãn test. PhoBERT ở đây là encoder được tinh chỉnh
bằng các nhãn train, theo hướng được đề cho phép. Không có dữ liệu bổ sung.

## Cài và chạy bản cache 4 tầng

Chạy trong thư mục absa_solution, với CSV ở thư mục cha:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_phobert.txt
python finetune_cpu_cached.py --data-dir .. --out-dir cached_phobert_results --cache private_cache/phobert_frozen8_160.pt --epochs 12 --layers 4 --threads 2 --batch-size 8 --accumulation 2 --log-steps 25
```

Tải checkpoint công khai PhoBERT-base-v2 một lần, revision được ghim trong code.
PyVi tách từ, Unicode NFC, sliding window 160 token với overlap 32; không cắt bỏ
phần đuôi đánh giá dài. Phân loại 21 nhãn bằng max pooling logits theo cửa sổ.
Các nhãn của cả đánh giá không bị sao chép để huấn luyện riêng từng cửa sổ.

Các tầng giữ cố định được tính trước một lần ở chế độ eval. Cache chỉ gồm train
có nhãn; tính cả phần validation bằng encoder pretrained cố định không cập nhật
bất kỳ trọng số/vocabulary nào từ validation. Sau đó chỉ gradient trên fit.
Batch train nhóm theo độ dài và shuffle thứ tự batch để giảm tính toán padding.

Code kiểm tra logits của đường tính cache khớp đường tính encoder đầy đủ trước
huấn luyện. Cache mặc định float32, không lượng tử hóa. Code lưu best.pt, latest.pt,
history.csv. best.pt chọn bằng macro F1 14 nhãn khen/chê trên tuning.

**Không đẩy private_cache lên GitHub**: nó chứa biểu diễn của dữ liệu lớp.
.gitignore đã loại cache, checkpoint và output cục bộ.

Tiếp tục sau gián đoạn, giữ nguyên tham số của lượt trước và thêm --resume:

```bash
python finetune_cpu_cached.py --data-dir .. --out-dir cached_phobert_results --cache private_cache/phobert_frozen8_160.pt --epochs 12 --layers 4 --threads 2 --batch-size 8 --accumulation 2 --log-steps 25 --resume
```

Checkpoint resume khôi phục optimizer, scheduler, Torch RNG và RNG sampler ở ranh
giới epoch. Phần epoch đang dở sẽ chạy lại. Không đổi lịch epochs khi resume.
Nếu đổi số tầng/độ dài/dữ liệu, dùng cache mới; code kiểm tra hash/config cache.

Ở lượt cloud hiện tại, chuẩn bị cache khoảng 7 phút, epoch đầu khoảng 4,5 phút
gồm tuning. Đây là số đo của lượt đang chạy, không cam kết thời gian trên máy khác.

## Sản phẩm sau huấn luyện

cached_phobert_results chứa model.pt, tokenizer/, config.json, metrics.json,
history.csv, run_info.json và nhãn/probability validation để kiểm tra nội bộ.
Model cuối của bản cache là checkpoint chọn từ fit/tuning, không refit toàn train;
config ghi production_refit=false. Muốn tăng dữ liệu fit cần một thí nghiệm và
đánh giá riêng, không thay đổi rồi giữ nguyên báo cáo điểm cũ.

Mặc định không chạy test trong script cache. Sau khi đã chọn mô hình, chạy:

```bash
python phobert_absa.py predict --model-dir cached_phobert_results --test ../test_blind.csv --output submission.csv --cpu-threads 2
python score_absa.py --kiem-tra submission.csv ../test_blind.csv
```

## Mô hình gọn và GitHub Actions

```bash
python export_phobert_adapter.py export --model-dir cached_phobert_results --adapter-dir phobert_adapter
python export_phobert_adapter.py predict --adapter-dir phobert_adapter --test ../test_blind.csv --output submission_reproduced.csv --threads 2
```

Adapter chỉ lưu các tham số đã học; phần còn lại lấy từ checkpoint công khai đã
ghim revision. Export kiểm tra từng tensor bị bỏ phải khớp checkpoint pretrained
chính xác. Giữ tokenizer, config và mã nguồn cùng với adapter để tái lập.

Workflow GitHub chạy CPU, tối đa 8 epoch và 330 phút/job; chỉ giữ output mô hình,
submission và chỉ số tổng hợp dưới mục Artifacts nếu huấn luyện hoàn tất thành công.
Artifact hết hạn sau 3 ngày, cần tải về. Không upload cache, văn bản train/test,
hoặc file ví dụ lỗi chứa text. Không dùng điểm test để chọn epoch/ngưỡng.

Mọi F1 trong report là validation; chưa có căn cứ suy ra điểm public/private.
Bootstrap 1.000 lần theo đánh giá, cùng mã chấm gốc, không bao gồm dịch chuyển
phân bố giữa train và test hoặc bất định giữa các seed.
