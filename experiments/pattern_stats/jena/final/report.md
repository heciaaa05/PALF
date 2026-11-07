# PALF — Thống kê Pattern Spec v1 (Bản đóng băng)

> Căn cứ: `PALF_Pattern_Spec_v1.md` và Prompt 4.3.
> Ngưỡng và khí hậu month_hour được tính toán **hoàn toàn từ tập Train sau purge gap**.
> Đặc trưng nhãn chỉ dùng cửa sổ đầu vào X, hoàn toàn không chạm vào y.

## 1. Bộ Pattern Spec v1 (4 nhãn)

Thứ tự ưu tiên cố định: `pressure_drop` > `humidity_net_rise` > `temp_anomaly` > `normal`.

| Thứ tự | Tên nhãn | Biến đặc trưng | Phân vị | Ngưỡng tính từ Train |
|:---:|---|---|:---:|:---:|
| 1 | `pressure_drop` | Delta p (resid) | p10 | -12.2190 mbar |
| 2 | `humidity_net_rise` | Delta rh (resid) | p90 | +20.4280 % |
| 3 | `temp_anomaly` | A_T | p90 / p10 | +4.7405 °C / -4.6102 °C |
| 4 | `normal` | không thỏa luật nào | — | — |

## 2. Phân bố số lượng và tỷ lệ nhãn (Cửa sổ)

| label | n_train | share_train | n_val | share_val | n_test | share_test |
|---|---|---|---|---|---|---|
| pressure_drop | 4899 | 0.1000 | 1071 | 0.1032 | 929 | 0.0895 |
| humidity_net_rise | 4378 | 0.0894 | 879 | 0.0847 | 776 | 0.0748 |
| temp_anomaly | 7801 | 0.1593 | 1337 | 0.1288 | 1730 | 0.1667 |
| normal | 31903 | 0.6513 | 7090 | 0.6832 | 6942 | 0.6690 |

## 3. Thống kê sự kiện (Runs) và độ dài run

| label | split | n_windows | share | n_runs | mean_run_length | max_run_length |
|---|---|---|---|---|---|---|
| pressure_drop | train | 4899 | 0.1000 | 185 | 26.4811 | 97 |
| humidity_net_rise | train | 4378 | 0.0894 | 879 | 4.9807 | 40 |
| temp_anomaly | train | 7801 | 0.1593 | 287 | 27.1812 | 286 |
| normal | train | 31903 | 0.6513 | 900 | 35.4478 | 273 |
| pressure_drop | val | 1071 | 0.1032 | 36 | 29.7500 | 102 |
| humidity_net_rise | val | 879 | 0.0847 | 190 | 4.6263 | 33 |
| temp_anomaly | val | 1337 | 0.1288 | 44 | 30.3864 | 192 |
| normal | val | 7090 | 0.6832 | 197 | 35.9898 | 377 |
| pressure_drop | test | 929 | 0.0895 | 36 | 25.8056 | 111 |
| humidity_net_rise | test | 776 | 0.0748 | 171 | 4.5380 | 24 |
| temp_anomaly | test | 1730 | 0.1667 | 54 | 32.0370 | 228 |
| normal | test | 6942 | 0.6690 | 177 | 39.2203 | 298 |

## 4. Kiểm tra tiêu chí C1–C7 theo Spec v1

### 4.1. Tiêu chí C2 (Kích thước & Số sự kiện)
- Ngưỡng yêu cầu: Tỷ lệ train >= 2%, số cửa sổ mỗi tập >= 300, số run >= (20, 5, 5).

| label | share_train | n_train | n_val | n_test | min_n | runs_train | runs_val | runs_test | min_runs | C2_cua_so | C2_su_kien | C2_toan_dien |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pressure_drop | 0.1000 | 4899 | 1071 | 929 | 929 | 185 | 36 | 36 | 36 | True | True | True |
| humidity_net_rise | 0.0894 | 4378 | 879 | 776 | 776 | 879 | 190 | 171 | 171 | True | True | True |
| temp_anomaly | 0.1593 | 7801 | 1337 | 1730 | 1337 | 287 | 44 | 54 | 44 | True | True | True |

**Kết luận C2:** ĐẠT toàn diện cho cả 3 nhãn khác normal.

### 4.2. Tiêu chí C3 (Tỷ lệ Normal <= 70%)

| normal_train | normal_val | normal_test | c3_max_allowed | C3_pass |
|---|---|---|---|---|
| 0.6513 | 0.6832 | 0.6690 | 0.7000 | True |

**Kết luận C3:** ĐẠT trên cả 3 tập (Train: 65.13%, Val: 68.32%, Test: 66.90% <= 70%).

### 4.3. Tiêu chí C4 (Độ ổn định tỷ lệ giữa các tập)
- Ngưỡng yêu cầu: Độ lệch tương đối <= 0.50.

| label | share_train | share_val | share_test | rel_dev_val | rel_dev_test | C4_pass |
|---|---|---|---|---|---|---|
| pressure_drop | 0.1000 | 0.1032 | 0.0895 | 0.0319 | 0.1049 | True |
| humidity_net_rise | 0.0894 | 0.0847 | 0.0748 | 0.0523 | 0.1634 | True |
| temp_anomaly | 0.1593 | 0.1288 | 0.1667 | 0.1910 | 0.0468 | True |
| normal | 0.6513 | 0.6832 | 0.6690 | 0.0490 | 0.0271 | True |

**Kết luận C4:** ĐẠT cho toàn bộ 4 nhãn.

### 4.4. Phân tích Dấu của `temp_anomaly` (Heat vs Cold)
> **Lưu ý bắt buộc (Mục 5 Spec v1):** Dấu của dị thường nhiệt độ được ghi nhận dưới dạng siêu dữ liệu (`temp_sign` = +1 hoặc -1).

| split | n_temp_anomaly | share_in_split | n_heat | pct_heat | n_cold | pct_cold |
|---|---|---|---|---|---|---|
| train | 7801 | 0.1593 | 3749 | 48.0579 | 4052 | 51.9421 |
| val | 1337 | 0.1288 | 1058 | 79.1324 | 279 | 20.8676 |
| test | 1730 | 0.1667 | 1511 | 87.3410 | 219 | 12.6590 |

**Nhận xét:**
- Ở Train: Heat và Cold cân bằng (~48% vs 52%).
- Ở Val và Test: Heat chiếm đa số (Val: 79.1%, Test: 87.3%) do biến động nhiệt độ thực tế năm 2015–2016.

### 4.5. Ma trận chồng lấn các luật trên Train

| rule | pressure_drop | humidity_net_rise | temp_anomaly |
|---|---|---|---|
| pressure_drop | 4899 | 521 | 1191 |
| humidity_net_rise | 521 | 4899 | 943 |
| temp_anomaly | 1191 | 943 | 9798 |

### 4.6. Kiểm tra trực quan C5 cho `humidity_net_rise`
- Đã xuất biểu đồ trực quan 16 cửa sổ ngẫu nhiên (seed=0) cho cả 4 kênh tại: `figures/samples_humidity_net_rise.png`.
- Danh sách chỉ số cửa sổ lưu tại: `sample_indices_humidity_net_rise.csv`.

---

## 5. Kết luận
- Toàn bộ quy trình gán nhãn đã được chuẩn hóa thành module [`src/palf/data/patterns_weather.py`](file:///d:/Github/PALF/src/palf/data/patterns_weather.py).
- Loader [`prepare_weather_data`](file:///d:/Github/PALF/src/palf/data/weather.py) đã hỗ trợ trực tiếp `pattern_scheme='weather_v1'` với purge gap và siêu dữ liệu `temp_sign`.
- Tất cả tiêu chí C1, C2, C3, C4, C6 đều ĐẠT 100%.
