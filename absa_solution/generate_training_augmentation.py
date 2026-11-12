"""Deterministic synthetic TRAIN data from disclosed, LLM-authored templates.

Never reads blind test. No original review is paraphrased or copied. Generated
reviews duplicate neither fit nor held-out original train reviews exactly.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import score_absa as scorer

AUTHORING_SPEC = '''Soạn các mẫu câu tiếng Việt mới để bổ sung dữ liệu HUẤN LUYỆN
ABSA khách sạn. Chỉ dựa trên 7 định nghĩa khía cạnh trong đề, không dùng văn bản
test hoặc viết lại các review gốc. Mỗi mẫu chỉ đánh giá một khía cạnh rõ ràng.
Khen/chê chung cả khách sạn không thuộc khía cạnh nào. Bình thường, tạm được,
đề nghị cải thiện là chê. Tiếng ồn từ đường phố thuộc vị trí; cách âm giữa phòng,
thiết bị trong phòng thuộc phòng; WiFi, gym, thang máy, hồ bơi chung thuộc tiện
nghi. Quán ăn xung quanh thuộc vị trí, nhà hàng trong khách sạn thuộc ăn uống.
Ghép các mẫu theo seed, cho phép cùng một khía cạnh vừa khen vừa chê; nhãn của
review ghép là OR nhãn của các câu. Lưu đầy đủ du_lieu_them.csv và mô tả công cụ.'''

# The assistant authored these clauses. Combining them is ordinary seeded code,
# not an external generation API. The CSV is disclosed as synthetic LLM-assisted data.
PHRASES = {
 'nhan_vien': [
  ['Nhân viên đón tiếp rất niềm nở','Các bạn lễ tân hỗ trợ tận tình',
   'Nhân viên giải đáp câu hỏi rất rõ ràng','Thái độ phục vụ thân thiện',
   'Lễ tân xử lý yêu cầu nhanh chóng','Nhân viên luôn sẵn sàng giúp đỡ',
   'Thủ tục nhận phòng nhanh gọn','Nhân viên hỗ trợ với thái độ lịch sự',
   'Dịch vụ khiến tôi rất hài lòng','Nhân viên chủ động hướng dẫn khách'],
  ['Nhân viên để khách chờ quá lâu','Lễ tân phản hồi yêu cầu khá chậm',
   'Thái độ phục vụ còn cần cải thiện','Nhân viên nói chuyện thiếu lịch sự',
   'Thủ tục nhận phòng mất nhiều thời gian','Dịch vụ chỉ ở mức tạm được',
   'Nhân viên xử lý sự cố chưa tốt','Lễ tân không nhiệt tình hỗ trợ',
   'Cách phục vụ chỉ ở mức bình thường','Nhân viên ít quan tâm đến yêu cầu của khách']],
 'sach_se': [
  ['Vệ sinh được chăm chút rất kỹ','Ga gối không có vết bẩn',
   'Nhà vệ sinh được lau rửa sạch','Khu vực sinh hoạt không có rác',
   'Dọn dẹp kỹ nên mọi thứ rất sạch','Khăn tắm đã được giặt sạch sẽ',
   'Sàn nhà được lau sạch','Công tác vệ sinh rất tốt',
   'Tôi hài lòng về sự sạch sẽ','Các bề mặt đều sạch bụi'],
  ['Ga giường còn dính vết bẩn','Tôi thấy nhiều côn trùng quanh giường',
   'Mùi ẩm mốc rất khó chịu','Công tác vệ sinh mới ở mức bình thường',
   'Dọn dẹp chưa sạch các góc khuất','Sàn nhà vẫn còn nhiều bụi',
   'Vệ sinh phòng tắm cần được cải thiện','Khăn còn mùi khó chịu',
   'Rác vẫn còn sót lại sau khi dọn','Mùi hôi làm tôi không hài lòng']],
 'vi_tri': [
  ['Đi bộ vài phút là tới bãi biển','Xung quanh có nhiều quán ăn để lựa chọn',
   'Có nhiều cửa hàng tiện lợi gần đây','Nằm ngay khu trung tâm nên đi đâu cũng dễ',
   'Đường tới khách sạn dễ tìm','Chỗ này gọi taxi khá tiện',
   'Gần bến xe nên di chuyển thuận lợi','Khu vực xung quanh yên tĩnh',
   'Vị trí thuận tiện để đi tham quan','Từ đây đi ra biển rất gần'],
  ['Muốn ra bãi biển phải đi quá xa','Khu vực này quá ồn bởi xe cộ ngoài đường',
   'Đường vào khách sạn rất khó tìm','Gọi taxi ở đây khó khăn',
   'Xung quanh rất ít chỗ ăn uống','Vị trí hơi bất tiện cho việc đi lại',
   'Đi vào trung tâm mất quá nhiều thời gian','Lối vào nằm sâu trong hẻm, di chuyển bất tiện',
   'Vị trí chỉ ở mức tạm được','Tiếng xe ngoài phố quá ồn ào']],
 'phong': [
  ['Căn phòng rộng nên để đồ thoải mái','Giường có đệm êm, nằm dễ chịu',
   'Ban công của phòng có tầm nhìn đẹp','Nội thất phòng được bố trí hợp lý',
   'Máy lạnh trong phòng chạy êm và mát','Phòng tắm bên trong phòng bố trí tiện dụng',
   'Các thiết bị trong phòng đều hoạt động tốt','Không gian phòng sáng và dễ chịu',
   'Cửa sổ trong phòng đón nhiều ánh sáng','Cách âm giữa các phòng rất tốt'],
  ['Phòng khá chật nên khó sắp đồ','Nệm trên giường quá cứng',
   'Máy lạnh trong phòng làm mát kém','Phòng thiếu chỗ để hành lý',
   'Nội thất trong phòng đã xuống cấp','Cách âm giữa hai phòng quá kém',
   'Bố trí phòng tắm bên trong phòng khá bất tiện','Thiết bị trong phòng cần được bảo dưỡng',
   'Phòng chỉ ở mức bình thường','Ánh sáng trong phòng quá tối']],
 'gia_tri': [
  ['Giá phòng như vậy rất đáng tiền','Chi phí bỏ ra tương xứng với những gì nhận được',
   'Mức giá khá hợp lý','Với số tiền đã trả tôi thấy rất xứng đáng',
   'Giá phù hợp với ví tiền của tôi','Giá khá dễ chịu so với nơi tương tự',
   'Đây là lựa chọn hợp túi tiền','Giá tương đối mềm',
   'Đáng tiền với ngân sách chuyến đi','Tôi hài lòng về giá trị nhận được so với chi phí'],
  ['Giá cao hơn chất lượng nhận được','Số tiền bỏ ra không xứng đáng',
   'Mức giá này hơi đắt','Có quá nhiều khoản phụ thu',
   'Chi phí bị đội lên bởi phụ phí không hợp lý','Tôi thấy không đáng với số tiền đã trả',
   'Giá chưa tương xứng với trải nghiệm','Phí phát sinh khiến tổng tiền quá cao',
   'Giá trị so với giá chỉ bình thường','Cần cải thiện chất lượng để xứng với giá']],
 'an_uong': [
  ['Bữa sáng có nhiều món ngon','Đồ uống tại quầy bar pha vừa miệng',
   'Món ăn ở nhà hàng khách sạn hợp khẩu vị','Thực đơn bữa sáng thay đổi đa dạng',
   'Bữa sáng chuẩn bị hấp dẫn','Đồ ăn được nêm nếm rất vừa',
   'Nước uống của khách sạn khá ngon','Bữa sáng đủ món để lựa chọn',
   'Phần ăn tại nhà hàng rất ngon','Thức uống ở bar khách sạn khá phong phú'],
  ['Bữa sáng ít lựa chọn','Đồ ăn ở nhà hàng quá mặn',
   'Bữa sáng chỉ ở mức bình thường','Thực đơn cần bổ sung thêm món',
   'Đồ uống ở bar quá nhạt','Bữa sáng nên được cải thiện',
   'Các món ăn không hợp khẩu vị','Đồ ăn dọn ra bị nguội',
   'Ăn sáng chỉ tạm được','Bữa sáng lặp lại mãi vài món']],
 'tien_nghi': [
  ['Wifi của khách sạn ổn định','Hồ bơi chung được thiết kế rộng rãi',
   'Chỗ đỗ xe thuận tiện','Thang máy di chuyển êm',
   'Khuôn viên có nhiều góc thư giãn đẹp','Phòng gym có đủ máy để tập',
   'Dịch vụ spa của khách sạn rất thư giãn','Có nhiều tiện ích chung để sử dụng',
   'Wifi mạnh cả khi có nhiều người truy cập','Sảnh chung rộng, ngồi chờ rất thoải mái'],
  ['Wifi chập chờn nên khó sử dụng','Thang máy thường xuyên trục trặc',
   'Bãi đỗ xe quá nhỏ','Hồ bơi chung đóng cửa nên không sử dụng được',
   'Khuôn viên thiếu chỗ ngồi nghỉ','Phòng gym có ít dụng cụ hoạt động',
   'Các tiện ích chung cần được nâng cấp','Sảnh chung quá chật',
   'Tốc độ wifi quá yếu','Spa của khách sạn chưa mang lại cảm giác thư giãn']]
}
GENERIC = ['Tôi khá thích kỳ nghỉ này','Đây là chuyến đi đáng nhớ',
 'Khách sạn để lại ấn tượng tốt','Tôi sẽ cân nhắc quay lại lần sau',
 'Trải nghiệm chung không như mong đợi','Tôi chưa hài lòng về chuyến đi này',
 'Khó có thể giới thiệu nơi này với mọi người','Tôi không muốn quay lại nữa']


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(text)).lower()).strip()


def generate(train_path, output, n=1400, seed=2026):
    train_path, output = Path(train_path), Path(output)
    train = pd.read_csv(train_path)
    assert list(train.columns) == ['id', 'text'] + scorer.COLS
    blocked = set(train.text.map(normalize))
    seen, rows = set(), []
    rng = np.random.default_rng(seed)
    aspects = list(scorer.ASPECTS)
    assert list(PHRASES) == aspects
    for attempt in range(n * 30):
        if len(rows) == n:
            break
        y = np.zeros(21, dtype=int)
        phrases = []
        if attempt % 12 == 0:
            phrases = [str(rng.choice(GENERIC))]
        else:
            count = int(rng.choice([1, 2, 3], p=[.25, .55, .20]))
            chosen = rng.choice(7, size=count, replace=False).tolist()
            if count >= 2 and rng.random() < .12:
                chosen[1] = chosen[0]
            for j, a in enumerate(chosen):
                polarity = int(rng.integers(0, 2))
                if j == 1 and chosen[0] == a:
                    polarity = 1 - int(y[a*3+2] == 1)
                phrases.append(str(rng.choice(PHRASES[aspects[a]][polarity])))
                y[a*3] = 1
                y[a*3+1+polarity] = 1
            if rng.random() < .2:
                phrases.append(str(rng.choice(GENERIC)))
        text = '. '.join(phrases) + '.'
        key = normalize(text)
        if key in blocked or key in seen:
            continue
        seen.add(key)
        rows.append([f'aug_{seed}_{len(rows)+1:05d}', text, *y.tolist()])
    assert len(rows) == n
    result = pd.DataFrame(rows, columns=['id', 'text'] + scorer.COLS)
    assert result.id.is_unique and not result.text.map(normalize).isin(blocked).any()
    assert result[scorer.COLS].isin([0, 1]).all().all()
    for a in aspects:
        assert ((result[a+'_khen'] | result[a+'_che']) <= result[a+'_nhac']).all()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    info = {'tool': 'ChatGPT/Codex assistant authored fixed templates; seeded Python combines them; no separate generation API',
        'authoring_specification': AUTHORING_SPEC, 'seed': seed, 'rows': len(result),
        'train_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
        'augmentation_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'no_original_review_copied_or_paraphrased': True,
        'blind_test_read': False, 'exact_original_train_duplicates_removed': True,
        'label_counts': result[scorer.COLS].sum().to_dict()}
    output.with_suffix('.generation.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print('Synthetic training reviews:', len(result), 'SHA256:', info['augmentation_sha256'])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='../train.csv')
    p.add_argument('--output', default='du_lieu_them.csv')
    p.add_argument('--rows', type=int, default=1400)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    generate(a.train, a.output, a.rows, a.seed)


if __name__ == '__main__':
    main()
