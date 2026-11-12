# Chế độ chỉ dùng train

Theo yêu cầu mới nhất của người dùng ngày 12/11/2026, không sử dụng
`test_blind.csv`, kể cả để suy luận, kiểm tra định dạng, tính hash hay xem nội
dung. Chỉ tiếp tục huấn luyện và đánh giá từ `train.csv`.

Giữ nguyên split theo nhóm và seed 2026: 2.812 fit, 938 tune, 938 validation.
Không đổi seed, không chuyển validation vào train rồi vẫn gọi điểm đó là
held-out. Điểm validation không thay thế điểm công khai trên trang chấm.

Không chạy các lệnh `predict`, các cell dự đoán blind trong notebook cũ, các
watcher tự tạo submission hoặc lệnh kiểm tra submission với file blind.
Các tệp kết quả blind đã có trước yêu cầu này không được dùng cho những lượt
thử tiếp theo. Mã suy luận có thể được giữ để tham khảo, nhưng không được gọi
trong chế độ hiện tại.

Việc học thêm trên toàn bộ train không tạo ra điểm validation độc lập cho
chính các trọng số đó. Các báo cáo phải phân biệt kết quả trước và sau refit.
