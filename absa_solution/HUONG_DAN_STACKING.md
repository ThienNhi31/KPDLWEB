# ABSA học có giám sát, CPU

Kết quả đã đo: **ABSA-F1 validation 0,880738**, CI95% bootstrap 1.000 lần
**[0,865538; 0,893802]**. Mục tiêu người dùng 0,96 chưa đạt trên validation;
chưa có kết quả nộp công khai. Accuracy không phải điểm chính.

Pipeline dùng 4 encoder: PhoBERT 4 tầng cuối, bản tinh chỉnh loss cân bằng,
PhoBERT aspect-pair và ViSoBERT. Một meta logistic cho mỗi nhãn học trên dự đoán
của encoder. Meta chọn C, tập feature và ngưỡng bằng dự đoán 5-fold trong tuning.
Feature của một nhãn là logit 3 nhãn của khía cạnh đó hoặc đủ 21 nhãn của mỗi
encoder. Fit scaler ở từng fold. Không đưa nhãn validation vào encoder/meta.

## Cài đặt và đặt file

Python 3.12, CPU x86_64. Không cần GPU. Đặt train.csv, test_blind.csv,
submission_demo.csv ở thư mục cha của absa_solution. score_absa.py nằm trong
absa_solution; file này giữ nguyên logic chấm của giảng viên.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Lần đầu cần Internet để tải pretrained checkpoint ghim revision. Chỉ tải model
public; chương trình không gọi API dự đoán/gán nhãn. Bộ dữ liệu gốc không đi
ra ngoài qua pipeline. Sau khi cache đủ base weights/tokenizer, có thể đặt
HF_HUB_OFFLINE=1 và TRANSFORMERS_OFFLINE=1.

## Huấn luyện từ đầu

Chạy các lệnh sau trong absa_solution. 2.812 review fit, 938 tuning và 938
validation, seed2026. Group theo review chuẩn hóa trước khi chia. 7 cặp của cùng
review luôn ở cùng tập. Các lệnh dưới chỉ đọc train; không tạo dự đoán test.

```bash
python finetune_cpu_cached.py --data-dir .. --out-dir cached_phobert_results --cache private_cache/phobert_frozen8_160.pt --epochs 12 --layers 4 --threads 2 --batch-size 8 --accumulation 2 --lr 0.00003
python finetune_cpu_cached.py --data-dir .. --out-dir balanced_phobert_results --cache private_cache/phobert_frozen8_160.pt --epochs 4 --layers 4 --threads 2 --batch-size 8 --accumulation 2 --lr 0.00001 --init-dir cached_phobert_results --balanced-loss
python aspect_pair_absa.py --data-dir .. --out-dir aspect_pair_results --cache private_cache/aspect_pair.pt --epochs 6 --patience 2 --layers 4 --threads 2 --batch-size 16 --accumulation 1 --lr 0.00003 --max-length 160 --stride 32 --negatives 2
python phobert_absa.py train --data-dir .. --out-dir visobert_results --model-name uitnlp/visobert --revision 196a62afad9cbe4f52a54aabad828b13f0eec59a --fast-tokenizer --no-segmentation --unfreeze-layers 12 --epochs 8 --patience 2 --batch-size 8 --accumulation 2 --cpu-threads 2 --lr 0.00002 --max-length 160 --stride 32 --no-refit --skip-test
python stacked_absa.py fit --data-dir .. --models cached_phobert_results balanced_phobert_results aspect_pair_results visobert_results --out-dir stacked_results
```

Lưu cache train riêng, không công bố hoặc đưa vào gói nộp. Các CSV/JSON lỗi train
cũng chỉ dùng cục bộ. Không có dữ liệu huấn luyện tổng hợp, không cần
du_lieu_them.csv. LLM hỗ trợ viết code và diễn giải lỗi train, không nhận text test.

Tái lập từ đầu có thể mất khoảng 4–8 giờ trên máy 2 CPU; phụ thuộc độ dài review,
CPU, RAM, tải máy và tải model. Thời gian này là dự tính, không phải số đo mới.
Xem history.csv/run_info.json/hardware.json của từng model cho số đo thực tế.
Nếu giới hạn RAM/disk, huấn luyện tuần tự và bỏ best.pt/latest.pt sau khi đã
có model.pt, config.json và metrics.json hoàn tất; không bỏ model.pt.

## Checkpoint gọn và chạy dự đoán

export_phobert_adapter.py chỉ giữ tham số đã học. Nó kiểm tra mọi tensor bị bỏ
khớp đúng pretrained base trước khi xuất. Export mặc định không kèm CSV test.

```bash
python export_phobert_adapter.py export --model-dir cached_phobert_results --adapter-dir model_bundle/cached_phobert
python export_phobert_adapter.py export --model-dir balanced_phobert_results --adapter-dir model_bundle/balanced_phobert
python export_phobert_adapter.py export --model-dir aspect_pair_results --adapter-dir model_bundle/aspect_pair
python export_phobert_adapter.py export --model-dir visobert_results --adapter-dir model_bundle/visobert
python stacked_absa.py fit --data-dir .. --models model_bundle/cached_phobert model_bundle/balanced_phobert model_bundle/aspect_pair model_bundle/visobert --out-dir model_bundle/stacked
python stacked_absa.py predict --model-dir model_bundle/stacked --test ../test_blind.csv --output submission.csv --threads 2 --batch-size 8 --score-cache-dir private_predictions
python score_absa.py --kiem-tra submission.csv ../test_blind.csv
```

Phải fit meta lại sau khi đổi full checkpoint sang adapter: byte hash của hai
định dạng khác nhau dù tensor cho cùng dự đoán. Predictor kiểm tra hash cấu hình
và checkpoint, thứ tự base encoder và cột. Dự đoán encoder chạy lần lượt bằng CPU;
không in review test. Cache private_predictions chỉ được lưu cục bộ.

Lưu SHA256 của CSV và chạy predict lần hai để đối chiếu byte. Notebook đã chạy
phải chỉ rõ dùng checkpoint sẵn hay train từ đầu; không ghi việc nạp checkpoint
thành thời gian huấn luyện mới. Sai khác phần cứng có thể ảnh hưởng tái huấn luyện;
tái tạo file nộp nên dùng đúng checkpoint đã chọn.

## Giới hạn của đánh giá

Cả 4 encoder fit trên 2.812 review. Meta dùng thêm nhãn của 938 review tuning;
do đó stacking có 3.750 review góp phần huấn luyện, encoder đơn có 2.812.
Validation 938 review giữ nguyên để so sánh. Validation đã được xem trong nhiều
vòng phát triển; đây không phải một phép xác nhận hoàn toàn mới. CI bootstrap
không tính bất định do chọn mô hình hoặc dịch chuyển phân bố train/test.

Không dùng mô hình production_refit=true để báo cáo held-out validation. Không
đổi split, lấy accuracy hay tuning F1 thay cho điểm ABSA-F1 validation/public.
Điểm public/private chỉ được xác nhận qua hệ thống chấm với tài khoản nhóm.

GitHub Actions mới chỉ xuất checkpoint và thống kê từ train, không xuất CSV dự
đoán test. Artifact có hạn lưu: tải model về trước khi hết hạn; source luôn có
lệnh huấn luyện lại. Gói nộp riêng gồm code, notebook có output, requirements,
model/link model, hướng dẫn và CSV được chấm; không kèm train/test gốc.

## Bản refit để nộp sau khi cố định lựa chọn

Đây là bước bổ sung đang triển khai, chưa có điểm công khai. Refit dùng toàn bộ
4.688 review có nhãn, hai epoch cố định với learning rate 1e-5. Không dùng F1
trên train hoặc validation đã được học để báo thành điểm held-out. Lịch/ngưỡng
và meta được giữ từ lựa chọn trước refit; có thể không cải thiện điểm test.

```bash
python refit_cached_absa.py --data-dir .. --init-dir model_bundle/cached_phobert --out-dir production_models/cached_phobert --cache private_cache/phobert_frozen8_160.pt --epochs 2 --threads 2
python refit_cached_absa.py --data-dir .. --init-dir model_bundle/balanced_phobert --out-dir production_models/balanced_phobert --cache private_cache/phobert_frozen8_160.pt --epochs 2 --threads 2
python refit_pair_absa.py --data-dir .. --init-dir model_bundle/aspect_pair --out-dir production_pair_full --cache private_cache/production_pair.pt --scope all --epochs 2 --threads 2
python export_phobert_adapter.py export --model-dir production_pair_full --adapter-dir production_models/aspect_pair
python stacked_absa.py promote --model-dir model_bundle/stacked --models production_models/cached_phobert production_models/balanced_phobert production_models/aspect_pair model_bundle/visobert --out-dir production_models/stacked
python stacked_absa.py predict --model-dir production_models/stacked --test ../test_blind.csv --output submission_production.csv --threads 2 --batch-size 8
```

promote chỉ đổi checkpoint theo đúng provenance, không học lại meta, không dùng
nhãn validation để hiệu chỉnh thêm. selection_metrics.json là điểm của bộ trước
refit. Ba PhoBERT học toàn train; ViSoBERT trong cấu hình này giữ nguyên bản fit
2.812 review. Chưa thể gán điểm 0,96 cho bản này khi chưa qua trang chấm.
