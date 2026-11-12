# Thử nghiệm bổ sung dữ liệu train

Đây là thử nghiệm riêng, chưa được chọn vào bộ stacking 4 encoder đạt validation
0,8807 và công khai 0,8840. Không gán kết quả của mô hình đối chứng cho thử nghiệm.

generate_training_augmentation.py chứa 140 mẫu câu theo 7 khía cạnh và các câu
nhận xét chung. Mẫu được trợ lý ChatGPT/Codex soạn theo đặc tả AUTHORING_SPEC
trong mã; Python ghép theo seed2026 thành 1.400 review mới. Không gọi API sinh
dữ liệu riêng, không dùng hoặc viết lại review test, không lấy review gốc làm
mẫu để paraphrase. Loại mọi review tổng hợp trùng văn bản chuẩn hóa với train gốc.
CSV có id, text và đúng 21 cột nhãn; nhãn ghép là OR nhãn của các câu. Lời chê
nhẹ, bình thường, tạm được và đề nghị cải thiện được gán chê theo quy ước đề.

Lượt tạo đã thực hiện:

- File: augmentation_trial/du_lieu_them.csv, 1.400 dòng.
- SHA256: 79dc8ad6655d8c20f86175dd8e57b68bb94588a479612e306f5b37b7d4bbc063.
- augmentation_trial/du_lieu_them.generation.json ghi công cụ, toàn bộ đặc tả,
  seed, hash train, hash CSV và phân bố nhãn.

Chia train gốc trước khi thêm mẫu, giữ 2.812 fit /938 tuning /938 validation.
Chỉ thêm review tổng hợp vào fit, trọng số mỗi mẫu 0,4 so với mẫu thật. Warm-start
model token-attention chỉ đã học fit gốc; cấm dùng checkpoint refit toàn train.
Tuning chọn checkpoint/ngưỡng, giữ epoch 0 làm đối chứng nếu học thêm không tốt
hơn. Validation giữ nguyên 938 review thật. augmentation_ablation.csv đo cả có
và không có dữ liệu thêm trên cùng validation, không lấy F1 train để thay thế.

```bash
python generate_training_augmentation.py --train ../train.csv --output augmentation_trial/du_lieu_them.csv --rows 1400 --seed 2026
python augment_cached_absa.py --data-dir .. --init-dir label_attention_results --augmentation augmentation_trial/du_lieu_them.csv --out-dir augmentation_results --cache private_cache/phobert_frozen8_160.pt --synthetic-cache private_cache/synthetic_phi_79dc.pt --synthetic-weight 0.4 --epochs 4 --patience 2 --threads 2 --batch-size 8 --accumulation 2 --lr 0.00001
```

Kết quả đã đo: mô hình đơn từ validation 0,865875 tăng lên 0,868441,
CI95% [0,853413; 0,880977], chọn epoch 3 bằng tuning F1 0,875941.
Khi thêm vào stacking, validation 0,880435 và meta OOF tuning 0,891934, đều chưa
vượt bộ 4 encoder (0,880738 và 0,892105). Không chọn augmentation vào bộ chính.

Nếu chọn mô hình này cho bản nộp, phải
đưa du_lieu_them.csv và JSON đặc tả vào gói nộp, ghi công cụ/đặc tả trong notebook,
báo cáo ablation cùng thời gian/hardware. Không tự đưa CSV bổ sung hoặc dự đoán
test lên GitHub. Model chỉ lưu các tham số đã học; cache original train giữ riêng.

Một hướng khác đang thử là mDeBERTa với encoder đã học NLI (MNLI/XNLI), sau đó
tinh chỉnh từ nhãn train của bài ABSA bằng head nhị phân 3 nhãn theo từng khía
cạnh. Không dùng zero-shot hoặc API để tạo nhãn test. Checkpoint public được
ghim revision 8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c; workflow CPU tối đa 4 epoch.
