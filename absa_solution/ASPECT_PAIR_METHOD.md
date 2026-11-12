# Phân loại theo cặp khía cạnh – đánh giá

Mô hình base đã hoàn tất: ABSA-F1 validation **0,8666**, CI95%
**[0,8522; 0,8795]**, checkpoint epoch 4 được chọn bằng tuning F1 0,8733.
Lượt chạy CPU mất 4.956 giây, gồm cả cache và đánh giá. Điểm công khai chưa có.
Mục tiêu mới 0,96 chưa được xác nhận.

Mô hình joint trước dùng một vector CLS chung để dự đoán 21 nhãn. Phương pháp
này đưa tên khía cạnh vào encoder cùng đánh giá, và dùng chung head ba nhãn
nhắc/khen/chê cho tất cả khía cạnh. Vì vậy lời chê học từ khía cạnh thường gặp
có thể giúp khía cạnh hiếm, đồng thời input khía cạnh buộc encoder phân biệt
phạm vi cảm xúc trong đánh giá có nhiều mặt.

Quy trình:

1. Chia các đánh giá gốc thành fit/tuning/validation trước khi tạo cặp; giữ đúng
   split 2.812/938/938, seed2026, nhóm theo văn bản chuẩn hóa. Không để các cặp
   của một đánh giá rơi vào nhiều tập.
2. Trên fit, dùng tất cả khía cạnh có nhắc hoặc có cực tính và tối đa hai khía cạnh
   vắng mặt chọn ngẫu nhiên theo seed. Trên tuning/validation, đánh giá đủ cả 7.
3. Input PhoBERT là cặp tên khía cạnh và review; PyVi tách từ. Mọi cửa sổ dài đều
   lặp lại tên khía cạnh. Đầu ra mỗi cặp là 3 sigmoid, cho phép vừa khen vừa chê.
4. Giữ cố định 8 tầng encoder đầu, cache float32; học 4 tầng cuối và head 3 nhãn.
5. Gom 7 cặp về đúng 21 cột của đánh giá gốc, chọn epoch/ngưỡng bằng **macro F1
   của 14 nhãn khen/chê**. Không dùng F1 gộp ba nhãn để thay thế điểm chính.
6. Chấm validation sau khi chọn mô hình. Cache, input text và nhãn gốc không được
   đưa vào artifact công khai. Workflow chỉ lưu model và thống kê, không đọc test.

Các cặp dùng đúng nhãn có sẵn từ train.csv. Không tạo đánh giá hoặc nhãn mới,
không dùng LLM/API để gán nhãn. Tên khía cạnh là metadata cố định, không phải
prompt gửi cho mô hình sinh. Không có du_lieu_them.csv.

Chạy trong absa_solution, CSV ở thư mục cha:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_phobert.txt
python aspect_pair_absa.py --data-dir .. --out-dir aspect_pair_results --cache private_cache/aspect_pair.pt --epochs 6 --patience 2 --layers 4 --threads 4 --batch-size 16 --accumulation 1 --lr 0.00003 --max-length 160 --stride 32 --negatives 2
python export_phobert_adapter.py export --model-dir aspect_pair_results --adapter-dir aspect_pair_adapter
```

Cache có thể lớn vài GB và phải được giữ riêng. Thread tự giới hạn theo quota CPU.
Mô hình gọn chỉ lưu các tầng đã học, còn base public checkpoint ghim revision.
Export kiểm tra phần bị bỏ khớp chính xác base. `export` mặc định không chép
submission.csv; xuất file suy ra từ test cần lựa chọn rõ ràng riêng.

Sau khi chọn mô hình, có thể dự đoán cục bộ (không tự công bố file):

```bash
python export_phobert_adapter.py predict --adapter-dir aspect_pair_adapter --test ../test_blind.csv --output submission.csv --threads 2 --batch-size 16
```

Validation đã được xem qua nhiều vòng phát triển; đây không phải một holdout
xác nhận mới hoàn toàn. Không thay split/chỉ số để tạo điểm đẹp và không suy ra
điểm public/private từ điểm validation. CI bootstrap dùng 1.000 lần theo review.
