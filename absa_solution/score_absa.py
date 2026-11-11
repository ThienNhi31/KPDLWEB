"""Chấm bài ABSA 7 khía cạnh, bản 2 (2026-11-06).

Dùng:  python score_absa.py ground_truth.csv submission.csv [--ids ids_private.csv] [--train train.csv]
                                [--boot 1000] [--json out.json] [--so-sanh submission_khac.csv]
Kiểm tra định dạng file nộp trước khi lên trang chấm (không cần đáp án):
        python score_absa.py --kiem-tra submission.csv test_blind.csv

Mỗi khía cạnh <kc> có 3 cột nhị phân:
    <kc>_nhac  = 1 nếu đánh giá có nhắc tới khía cạnh
    <kc>_khen  = 1 nếu có khen (+1)
    <kc>_che   = 1 nếu có chê, kể cả chê nhẹ (-1)
Trung tính = nhắc nhưng không khen, không chê. Vừa khen vừa chê: khen = che = 1.
Ràng buộc: khen = 1 hoặc che = 1 thì nhac = 1. Bộ chấm tự đặt nhac = 1 khi vi phạm và báo số ô đã sửa.

Điểm xếp hạng chính: ABSA-F1 = trung bình F1 của 14 nhãn nhị phân (7 khía cạnh x {khen, chê}).
Chỉ số độ tin cậy: F1/Precision/Recall theo cực, Accuracy, kappa, MCC, balanced accuracy, bootstrap.
Chỉ số phụ: F1 trung tính, F1 phát hiện khía cạnh (nhắc/không), F1 vi mô kiểu VLSP...
(Trung tính rất hiếm trong dữ liệu nên không đưa vào điểm chính; quyết định ngày 2026-11-06.)
Chỉ dùng numpy + pandas.
"""
import argparse, json, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean khi nhãn không có dự đoán dương
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ASPECTS = ["nhan_vien", "sach_se", "vi_tri", "phong", "gia_tri", "an_uong", "tien_nghi"]
COLS = [f"{a}_{s}" for a in ASPECTS for s in ("nhac", "khen", "che")]
POL = ("khen", "che", "trung_tinh")


def to_arrays(df):
    """X: (n, 7, 3) theo [nhac, khen, che]; P: (n, 7, 3) bool theo [khen, che, trung_tinh]."""
    X = df[COLS].to_numpy(int).reshape(len(df), 7, 3)
    nhac, khen, che = X[..., 0], X[..., 1], X[..., 2]
    tt = (nhac == 1) & (khen == 0) & (che == 0)
    return X, np.stack([khen == 1, che == 1, tt], -1)


# ---------- chỉ số nhị phân ----------
def f1(t, p):
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    if tp + fp + fn == 0:
        return float("nan")          # nhãn không xuất hiện ở cả nhãn thật lẫn dự đoán: bỏ qua
    return 2 * tp / (2 * tp + fp + fn)


def prec(t, p):
    return float((t & p).sum() / p.sum()) if p.sum() else float("nan")


def rec(t, p):
    return float((t & p).sum() / t.sum()) if t.sum() else float("nan")


def acc(t, p):
    return float((t == p).mean())


def kappa(t, p):
    po = (t == p).mean()
    pe = t.mean() * p.mean() + (1 - t.mean()) * (1 - p.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def mcc(t, p):
    tp = float((t & p).sum()); tn = float((~t & ~p).sum()); fp = float((~t & p).sum()); fn = float((t & ~p).sum())
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / d) if d > 0 else 0.0


def bal_acc(t, p):
    return float(((t & p).sum() / max(t.sum(), 1) + (~t & ~p).sum() / max((~t).sum(), 1)) / 2)


MAIN = (0, 1)   # chỉ số cực dùng cho điểm chính: khen, chê


def core_score(Pt, Pp):
    return float(np.nanmean([f1(Pt[:, k, j], Pp[:, k, j]) for k in range(7) for j in MAIN]))


def bootstrap(Pt, Pp, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    n = len(Pt)
    s = [core_score(Pt[i], Pp[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired(Pt, Pa, Pb, n_boot, seed=0):
    """Bootstrap ghép cặp: xác suất bài A > bài B và CI 95% của hiệu ABSA-F1."""
    rng = np.random.default_rng(seed)
    n = len(Pt)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(core_score(Pt[i], Pa[i]) - core_score(Pt[i], Pb[i]))
    d = np.array(d)
    return float((d > 0).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def baseline_majority(n, train_path):
    """Mốc: mỗi khía cạnh luôn đoán tổ hợp (nhac, khen, che) phổ biến nhất trong train."""
    Xs = to_arrays(pd.read_csv(train_path))[0]
    out = np.zeros((n, 7, 3), bool)
    for k in range(7):
        combos, cnt = np.unique(Xs[:, k, :], axis=0, return_counts=True)
        nh, kh, ch = combos[cnt.argmax()]
        out[:, k] = [kh == 1, ch == 1, nh == 1 and kh == 0 and ch == 0]
    return out


# ---------- nạp và kiểm tra ----------
def load(gt_path, sub_path, ids_path=None):
    gt = pd.read_csv(gt_path).set_index("id")
    if ids_path:
        gt = gt.loc[gt.index.isin(set(pd.read_csv(ids_path)["id"]))]
    sb = pd.read_csv(sub_path)
    miss = [c for c in ["id"] + COLS if c not in sb.columns]
    if miss:
        sys.exit(f"Thiếu cột: {miss}")
    if sb["id"].duplicated().any():
        sys.exit("File nộp có id trùng lặp")
    sb = sb.set_index("id")
    if ids_path:
        sb = sb.loc[sb.index.isin(gt.index)]
    if set(sb.index) != set(gt.index):
        sys.exit(f"Tập id không khớp: thiếu {len(set(gt.index) - set(sb.index))}, thừa {len(set(sb.index) - set(gt.index))}")
    sb = sb.loc[gt.index].copy()
    if not sb[COLS].isin([0, 1]).all().all():
        sys.exit("Giá trị phải là 0 hoặc 1")
    sua = 0
    for a in ASPECTS:
        bad = ((sb[f"{a}_khen"] == 1) | (sb[f"{a}_che"] == 1)) & (sb[f"{a}_nhac"] == 0)
        sua += int(bad.sum())
        sb.loc[bad, f"{a}_nhac"] = 1
    return gt.reset_index(), sb.reset_index(), sua


# ---------- báo cáo ----------
def report(gt, sb, n_boot=1000, train_path=None):
    Xt, Pt = to_arrays(gt)
    Xp, Pp = to_arrays(sb)
    per = {}
    for k, a in enumerate(ASPECTS):
        d = {}
        for j, s in enumerate(POL):
            t, p = Pt[:, k, j], Pp[:, k, j]
            d[s] = {"f1": f1(t, p), "precision": prec(t, p), "recall": rec(t, p), "accuracy": acc(t, p), "kappa": kappa(t, p), "mcc": mcc(t, p), "bal_acc": bal_acc(t, p), "ho_tro": int(t.sum())}
        t, p = Xt[:, k, 0] == 1, Xp[:, k, 0] == 1
        d["nhac"] = {"f1": f1(t, p), "kappa": kappa(t, p), "ho_tro": int(t.sum())}
        per[a] = d
    labels = [(a, s) for a in ASPECTS for s in ("khen", "che")]
    f1s = np.array([per[a][s]["f1"] for a, s in labels])
    lo, hi = bootstrap(Pt, Pp, n_boot)
    score = core_score(Pt, Pp)
    base = core_score(Pt, baseline_majority(len(gt), train_path)) if train_path else float("nan")
    return {
        "n_mau": int(len(gt)),
        "ABSA_F1": score,
        "ABSA_F1_CI95": [lo, hi],
        "F1_nhan_thap_nhat": {"nhan": "%s_%s" % labels[int(np.nanargmin(f1s))], "f1": float(np.nanmin(f1s))},
        "do_lech_chuan_F1_14_nhan": float(np.nanstd(f1s)),
        "F1_theo_cuc": {s: float(np.nanmean([per[a][s]["f1"] for a in ASPECTS])) for s in POL},
        "kappa_TB": float(np.nanmean([per[a][s]["kappa"] for a, s in labels])),
        # Precision, Recall, Accuracy: trung bình 14 nhãn; accuracy cao giả do phần lớn ô bằng 0, chỉ để đối chiếu
        "precision_TB": float(np.nanmean([per[a][s]["precision"] for a, s in labels])),
        "recall_TB": float(np.nanmean([per[a][s]["recall"] for a, s in labels])),
        "accuracy_TB": float(np.nanmean([per[a][s]["accuracy"] for a, s in labels])),
        "P_R_theo_cuc": {s: {"precision": float(np.nanmean([per[a][s]["precision"] for a in ASPECTS])),
                             "recall": float(np.nanmean([per[a][s]["recall"] for a in ASPECTS]))} for s in ("khen", "che")},
        "F1_phat_hien_khia_canh_TB": float(np.nanmean([per[a]["nhac"]["f1"] for a in ASPECTS])),
        "mcc_TB": float(np.nanmean([per[a][s]["mcc"] for a, s in labels])),
        "bal_acc_TB": float(np.nanmean([per[a][s]["bal_acc"] for a, s in labels])),
        # chuẩn thường dùng trong ABSA tiếng Việt (VLSP 2018): F1 vi mô của khía cạnh và của cặp (khía cạnh, cực)
        "F1_vi_mo_khia_canh": f1((Xt[..., 0] == 1).ravel(), (Xp[..., 0] == 1).ravel()),
        "F1_vi_mo_khia_canh_cuc": f1(Pt.ravel(), Pp.ravel()),
        "khop_dung_ca_danh_gia": float((Xt == Xp).all((1, 2)).mean()),
        "moc_lop_da_so": base,
        "cai_thien_so_voi_moc": (score - base) / (1 - base) if base == base and base < 1 else float("nan"),
        "theo_khia_canh": per,
    }


def show(r):
    f = lambda x: "  nan" if x != x else f"{x:.3f}"
    print(f"Số mẫu: {r['n_mau']}")
    print(f"ABSA-F1 (điểm xếp hạng) = {f(r['ABSA_F1'])}   CI95% [{f(r['ABSA_F1_CI95'][0])}, {f(r['ABSA_F1_CI95'][1])}]")
    c = r["F1_theo_cuc"]
    print(f"F1 theo cực: khen {f(c['khen'])} | chê {f(c['che'])} | trung tính (phụ) {f(c['trung_tinh'])} | phát hiện khía cạnh (phụ) {f(r['F1_phat_hien_khia_canh_TB'])}")
    print(f"Nhãn yếu nhất: {r['F1_nhan_thap_nhat']['nhan']} F1 = {f(r['F1_nhan_thap_nhat']['f1'])}; độ lệch chuẩn F1 14 nhãn = {f(r['do_lech_chuan_F1_14_nhan'])}")
    q = r["P_R_theo_cuc"]
    print(f"Precision TB = {f(r['precision_TB'])} | Recall TB = {f(r['recall_TB'])} | Accuracy TB = {f(r['accuracy_TB'])} (14 nhãn)")
    print(f"  khen: P {f(q['khen']['precision'])} R {f(q['khen']['recall'])} | chê: P {f(q['che']['precision'])} R {f(q['che']['recall'])}")
    print(f"kappa TB = {f(r['kappa_TB'])} | MCC TB = {f(r['mcc_TB'])} | Balanced acc TB = {f(r['bal_acc_TB'])}")
    print(f"F1 vi mô: khía cạnh = {f(r['F1_vi_mo_khia_canh'])} | (khía cạnh, cực) = {f(r['F1_vi_mo_khia_canh_cuc'])} | khớp đúng cả đánh giá = {f(r['khop_dung_ca_danh_gia'])}")
    print(f"Mốc lớp đa số = {f(r['moc_lop_da_so'])}; cải thiện tương đối = {f(r['cai_thien_so_voi_moc'])}")
    print("\nKhía cạnh      Nhắc F1 | Khen F1     P     R     κ     n | Chê F1      P     R     κ     n | Trung tính F1  n")
    for a, d in r["theo_khia_canh"].items():
        k, c = d["khen"], d["che"]
        print(f"{a:<14} {f(d['nhac']['f1'])}  | {f(k['f1'])} {f(k['precision'])} {f(k['recall'])} {f(k['kappa'])} {k['ho_tro']:>5} | "
              f"{f(c['f1'])}  {f(c['precision'])} {f(c['recall'])} {f(c['kappa'])} {c['ho_tro']:>5} | {f(d['trung_tinh']['f1'])}    {d['trung_tinh']['ho_tro']:>4}")


def kiem_tra(sub_path, test_path):
    """Kiểm tra file nộp theo đúng các quy tắc của trang chấm; in lỗi và trả về số lỗi."""
    loi = []
    try:
        sb = pd.read_csv(sub_path, dtype=str, keep_default_na=False)   # đọc dạng chuỗi: kiểm tra đúng ký tự 0/1
    except Exception as e:
        print(f"Không đọc được file CSV: {e}")
        return 1
    ids = pd.read_csv(test_path)["id"]
    if list(sb.columns) != ["id"] + COLS:
        thieu = [c for c in ["id"] + COLS if c not in sb.columns]
        thua = [c for c in sb.columns if c not in ["id"] + COLS]
        loi.append(f"Cột phải đúng 22 cột theo thứ tự id, {COLS[0]}, ..., {COLS[-1]}. Thiếu: {thieu}; thừa: {thua}")
    if "id" in sb.columns:
        if sb["id"].duplicated().any():
            loi.append(f"Có {int(sb['id'].duplicated().sum())} id trùng")
        t, s_ = set(ids), set(sb["id"])
        if t != s_:
            loi.append(f"id không khớp test_blind.csv: thiếu {len(t - s_)}, thừa {len(s_ - t)}")
    co = [c for c in COLS if c in sb.columns]
    if co and not sb[co].apply(lambda c: c.str.strip()).isin(["0", "1"]).all().all():
        loi.append("Giá trị nhãn phải là số 0 hoặc 1 (không để trống, không dùng True/False, 0.0)")
    if all(c in sb.columns for c in COLS):
        X = sb[COLS].apply(lambda c: c.str.strip()).to_numpy().reshape(len(sb), 7, 3)
        vp = int((((X[..., 1] == "1") | (X[..., 2] == "1")) & (X[..., 0] == "0")).sum())
        if vp:
            print(f"Cảnh báo: {vp} ô có khen/chê = 1 nhưng nhac = 0 (khi chấm sẽ tự đặt nhac = 1)")
    for l in loi:
        print("LỖI:", l)
    print("File hợp lệ, có thể nộp." if not loi else f"Có {len(loi)} lỗi, sửa trước khi nộp.")
    return len(loi)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--kiem-tra":
        sys.exit(1 if kiem_tra(sys.argv[2], sys.argv[3]) else 0)
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth")
    ap.add_argument("submission")
    ap.add_argument("--ids", help="CSV cột id: chỉ chấm các mẫu này (public/private)")
    ap.add_argument("--train", help="train.csv để tính mốc lớp đa số")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--so-sanh", help="bài nộp thứ hai: bootstrap ghép cặp, xác suất bài 1 > bài 2")
    ap.add_argument("--json")
    a = ap.parse_args()
    gt, sb, sua = load(a.ground_truth, a.submission, a.ids)
    if sua:
        print(f"Lưu ý: đã đặt nhac = 1 ở {sua} ô có khen/chê nhưng nhac = 0")
    r = report(gt, sb, a.boot, a.train)
    show(r)
    if a.so_sanh:
        _, sb2, _ = load(a.ground_truth, a.so_sanh, a.ids)
        pr, lo, hi = paired(to_arrays(gt)[1], to_arrays(sb)[1], to_arrays(sb2)[1], a.boot)
        print(f"\nBootstrap ghép cặp: P(bài 1 > bài 2) = {pr:.3f}; hiệu ABSA-F1 CI95% [{lo:.3f}, {hi:.3f}]")
        r["so_sanh"] = {"p_hon": pr, "ci95_hieu": [lo, hi]}
    if a.json:
        json.dump(r, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
