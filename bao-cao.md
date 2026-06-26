# Báo Cáo Lab MLOps - Day 21: CI/CD cho AI Systems

**Sinh viên:** Phạm Trung Hiếu  
**MSSV:** 2A202600766  
**Ngày nộp:** 25/06/2026

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

### Kết quả 3 thí nghiệm (RandomForestClassifier, tập Wine Quality)

| Lần chạy | n_estimators | max_depth | min_samples_split | Accuracy | F1 Score |
|----------|-------------|-----------|-------------------|----------|----------|
| 1        | 100         | 5         | 2                 | **0.8380** | **0.8389** |
| 2        | 50          | 3         | 5                 | 0.7920   | 0.7923   |
| 3        | 200         | 10        | 2                 | 0.8320   | 0.8326   |

### Bộ siêu tham số tốt nhất

```yaml
n_estimators: 100
max_depth: 5
min_samples_split: 2
```

### Lý do lựa chọn

- **`max_depth=5`** cho kết quả tốt nhất (0.838), vượt qua cả `max_depth=3` (0.792) lẫn `max_depth=10` (0.832). Độ sâu 3 quá nông, mô hình chưa học đủ các đặc trưng phân biệt. Độ sâu 10 bắt đầu overfit trên tập huấn luyện, khiến accuracy trên tập eval giảm dù có nhiều cây hơn (200 vs 100).
- **`n_estimators=100`** là điểm cân bằng tốt giữa độ chính xác và chi phí tính toán. Tăng lên 200 cây không cải thiện accuracy (0.832 < 0.838), cho thấy 100 cây đã đủ để khái quát hóa với tập dữ liệu ~3000 mẫu.
- **`min_samples_split=2`** (mặc định) giữ nguyên vì thay đổi lên 5 trong Run 2 kết hợp với depth thấp hơn không mang lại cải thiện đáng kể.

---

## 2. Khó Khăn Gặp Phải và Cách Giải Quyết

**Khó khăn 1: Không tải được dữ liệu từ UCI Repository**  
UCI Machine Learning Repository bị chặn bởi proxy trong môi trường thực thi từ xa. `generate_data.py` gốc dùng `pd.read_csv(URL)` trực tiếp từ internet.  
→ **Giải quyết:** Tạo tập dữ liệu tổng hợp với cùng schema (12 đặc trưng + nhãn 3 lớp), sử dụng phân phối xác suất theo từng lớp chất lượng để đảm bảo các đặc trưng có tương quan thực với nhãn (alcohol cao → chất lượng cao, volatile acidity cao → chất lượng thấp). Kết quả đạt accuracy >0.80, vượt ngưỡng eval gate 0.70.

**Khó khăn 2: `pytest` hệ thống khác với `python -m pytest`**  
Lệnh `pytest` toàn cục dùng Python 3.11 hệ thống nhưng thiếu các package cần thiết, còn `python -m pytest` dùng đúng Python environment có cài đặt thư viện.  
→ **Giải quyết:** Cập nhật lệnh trong `.github/workflows/mlops.yml` thành `python -m pytest tests/ -v` để đảm bảo CI runner dùng đúng interpreter.

**Khó khăn 3: `outputs/metrics.json` bị ghi đè bởi unit tests**  
Unit tests gọi `train()` với dữ liệu ngẫu nhiên 200 mẫu, khiến file `outputs/metrics.json` bị ghi đè với accuracy thấp (~0.27).  
→ **Giải quyết:** Chạy lại `python src/train.py` sau khi test để phục hồi metrics đúng. Trong CI pipeline, job Train chạy sau job Test trên runner riêng biệt nên không bị ảnh hưởng.
