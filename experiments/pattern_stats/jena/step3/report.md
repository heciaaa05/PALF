# Pattern Spec v1 — Báo cáo đo bổ sung (Prompt 4.2)

> **Nguyên tắc:** Chỉ đo và báo cáo số liệu khách quan, KHÔNG quyết định thay người dùng.
> Mọi ngưỡng, khí hậu, phân vị chỉ tính trên **train** (sau purge gap). Val/test chỉ áp dụng.
> Toàn bộ phép đo sử dụng **Cách A** (đặc trưng phần dư so với khí hậu `month_hour`).

## Thông số tập dữ liệu & ranh giới
- Tổng số dòng giờ: 70,092; số ô sensor lỗi `wv < -900` đã đưa về 0: 7
- Cấu hình cửa sổ: L = 96 giờ, H = 24 giờ; Purge gap = 119 cửa sổ
- Số cửa sổ: Train = 48,981 | Val = 10,377 | Test = 10,377 (Tổng: 69,973)

---

## 1. Spike theo nhiều phân vị

Đo số cửa sổ và **số run (sự kiện liên tiếp)** của nhãn Spike khi thay đổi `q_spike` trong {0.95, 0.97, 0.98, 0.99}. Các luật khác và thứ tự ưu tiên được giữ nguyên (`spike` > `pressure_drop` > `humidity_rise` > `heat_anomaly` > `cold_anomaly` > `normal`).

| q_spike | threshold | split | n_windows | share | n_runs | mean_run_length | max_run_length |
|---|---|---|---|---|---|---|---|
| 0.9500 | 5.6779 | train | 2485 | 0.0507 | 25 | 99.4000 | 192 |
| 0.9500 | 5.6779 | val | 570 | 0.0549 | 6 | 95.0000 | 95 |
| 0.9500 | 5.6779 | test | 285 | 0.0275 | 3 | 95.0000 | 95 |
| 0.9700 | 6.0369 | train | 1522 | 0.0311 | 15 | 101.4667 | 192 |
| 0.9700 | 6.0369 | val | 380 | 0.0366 | 4 | 95.0000 | 95 |
| 0.9700 | 6.0369 | test | 285 | 0.0275 | 3 | 95.0000 | 95 |
| 0.9800 | 6.4883 | train | 1047 | 0.0214 | 10 | 104.7000 | 192 |
| 0.9800 | 6.4883 | val | 285 | 0.0275 | 3 | 95.0000 | 95 |
| 0.9800 | 6.4883 | test | 95 | 0.0092 | 1 | 95.0000 | 95 |
| 0.9900 | 7.6911 | train | 570 | 0.0116 | 6 | 95.0000 | 95 |
| 0.9900 | 7.6911 | val | 95 | 0.0092 | 1 | 95.0000 | 95 |
| 0.9900 | 7.6911 | test | 0 | 0.0000 | 0 | nan | 0 |

**Nhận xét:**
- Ở phân vị mặc định `q = 0.99`, train chỉ có **6 run**, val có **1 run**, test có **0 run**.
- Khi hạ phân vị xuống `q = 0.95`, train đạt **25 run** (5.07% cửa sổ), val đạt **6 run** (5.49%), test đạt **3 run** (2.75%).
- Độ dài trung bình mỗi run của Spike dao động quanh 95-105 giờ (đúng bằng $L-1 = 95$ bước trượt cho 1 bước nhảy cô lập đơn lẻ).

---

## 2. Nhãn gộp `temp_anomaly` (= heat_anomaly ∪ cold_anomaly)

Do `heat_anomaly` ($A_T \ge p_{90}$) và `cold_anomaly` ($A_T \le p_{10}$) không bao giờ đồng thời kích hoạt, việc gộp lại thành `temp_anomaly` giúp theo dõi sự bất thường nhiệt độ nói chung.

### 2.1. Số lượng và số run của `temp_anomaly` gộp từ nhãn gốc

| split | n_windows | share | n_runs | mean_run_length | max_run_length |
|---|---|---|---|---|---|
| train | 7751 | 0.1582 | 283 | 27.3887 | 286 |
| val | 1337 | 0.1288 | 44 | 30.3864 | 192 |
| test | 1730 | 0.1667 | 54 | 32.0370 | 228 |

### 2.2. Bảng tách theo dấu (Heat vs Cold làm siêu dữ liệu)

| sign | split | n_windows | share_in_split | share_in_temp_anomaly | n_runs | mean_run_length | max_run_length |
|---|---|---|---|---|---|---|---|
| heat (+) | train | 3702 | 0.0756 | 0.4776 | 173 | 21.3988 | 181 |
| cold (-) | train | 4049 | 0.0827 | 0.5224 | 110 | 36.8091 | 286 |
| heat (+) | val | 1058 | 0.1020 | 0.7913 | 32 | 33.0625 | 192 |
| cold (-) | val | 279 | 0.0269 | 0.2087 | 12 | 23.2500 | 105 |
| heat (+) | test | 1511 | 0.1456 | 0.8734 | 48 | 31.4792 | 228 |
| cold (-) | test | 219 | 0.0211 | 0.1266 | 6 | 36.5000 | 63 |

### 2.3. Cấu hình 4 nhãn chính thức (`temp_anomaly` thay 2 nhãn cũ)

Thứ tự ưu tiên: `spike` > `pressure_drop` > `humidity_rise` > `temp_anomaly` > `normal`.

#### Ma trận chồng lấn trên Train (số cửa sổ đồng thời thỏa mãn):

| rule | spike | pressure_drop | humidity_rise | temp_anomaly |
|---|---|---|---|---|
| spike | 570 | 29 | 42 | 97 |
| pressure_drop | 29 | 4899 | 521 | 1191 |
| humidity_rise | 42 | 521 | 4899 | 943 |
| temp_anomaly | 97 | 1191 | 943 | 9798 |

#### Phân bố nhãn sau ưu tiên:

| label | n_train | share_train | n_val | share_val | n_test | share_test |
|---|---|---|---|---|---|---|
| spike | 570 | 0.0116 | 95 | 0.0092 | 0 | 0.0000 |
| pressure_drop | 4870 | 0.0994 | 1071 | 0.1032 | 929 | 0.0895 |
| humidity_rise | 4336 | 0.0885 | 862 | 0.0831 | 776 | 0.0748 |
| temp_anomaly | 7751 | 0.1582 | 1337 | 0.1288 | 1730 | 0.1667 |
| normal | 31454 | 0.6422 | 7012 | 0.6757 | 6942 | 0.6690 |

#### Kiểm tra tiêu chí C2–C4 cho cấu hình 4 nhãn:

- **Tiêu chí C2 (Tỷ lệ train $\ge 2\%$, số lượng mỗi tập $\ge 300$):**
| label | share_train | n_train | n_val | n_test | min_n | share>=2% | share>=5% | min_n>=300 | C2 |
|---|---|---|---|---|---|---|---|---|---|
| spike | 0.0116 | 570 | 95 | 0 | 0 | False | False | False | False |
| pressure_drop | 0.0994 | 4870 | 1071 | 929 | 929 | True | True | True | True |
| humidity_rise | 0.0885 | 4336 | 862 | 776 | 776 | True | True | True | True |
| temp_anomaly | 0.1582 | 7751 | 1337 | 1730 | 1337 | True | True | True | True |

- **Tiêu chí C3 (Normal $\le 70\%$):**
| normal_train | normal_val | normal_test | all<=70% | all<=60% |
|---|---|---|---|---|
| 0.6422 | 0.6757 | 0.6690 | True | False |

- **Tiêu chí C4 (Độ lệch tương đối val/test so với train $\le 0.5$):**
| label | share_train | share_val | share_test | rel_dev_val | rel_dev_test | C4 |
|---|---|---|---|---|---|---|
| spike | 0.0116 | 0.0092 | 0.0000 | 0.2133 | 1.0000 | False |
| pressure_drop | 0.0994 | 0.1032 | 0.0895 | 0.0380 | 0.0996 | True |
| humidity_rise | 0.0885 | 0.0831 | 0.0748 | 0.0616 | 0.1552 | True |
| temp_anomaly | 0.1582 | 0.1288 | 0.1667 | 0.1858 | 0.0535 | True |
| normal | 0.6422 | 0.6757 | 0.6690 | 0.0523 | 0.0418 | True |

**Nhận xét:**
- Khi tách riêng, `cold_anomaly` vi phạm C2 (val=279, test=219 < 300) và vi phạm C4 (độ lệch 67% và 74%). `heat_anomaly` vi phạm C4 (độ lệch test 93%).
- Khi **gộp thành `temp_anomaly`**: nhãn này **ĐẠT cả C2** (min_n = 1,337 $\ge$ 300) và **ĐẠT cả C4** (độ lệch val = 18.6%, test = 5.4% $\le 50\%$).

---

## 3. Số run cho mọi nhãn hiện có (ở phân vị mặc định)

Bảng thống kê số cửa sổ (`n_windows`) và số sự kiện (`n_runs`) cho toàn bộ 5 nhãn gốc + `normal` trên từng tập:

| label | split | n_windows | share | n_runs | mean_run_length | max_run_length |
|---|---|---|---|---|---|---|
| spike | train | 570 | 0.0116 | 6 | 95.0000 | 95 |
| pressure_drop | train | 4870 | 0.0994 | 185 | 26.3243 | 97 |
| humidity_rise | train | 4336 | 0.0885 | 871 | 4.9782 | 40 |
| heat_anomaly | train | 3702 | 0.0756 | 173 | 21.3988 | 181 |
| cold_anomaly | train | 4049 | 0.0827 | 110 | 36.8091 | 286 |
| normal | train | 31454 | 0.6422 | 896 | 35.1049 | 273 |
| spike | val | 95 | 0.0092 | 1 | 95.0000 | 95 |
| pressure_drop | val | 1071 | 0.1032 | 36 | 29.7500 | 102 |
| humidity_rise | val | 862 | 0.0831 | 186 | 4.6344 | 33 |
| heat_anomaly | val | 1058 | 0.1020 | 32 | 33.0625 | 192 |
| cold_anomaly | val | 279 | 0.0269 | 12 | 23.2500 | 105 |
| normal | val | 7012 | 0.6757 | 194 | 36.1443 | 377 |
| spike | test | 0 | 0.0000 | 0 | nan | 0 |
| pressure_drop | test | 929 | 0.0895 | 36 | 25.8056 | 111 |
| humidity_rise | test | 776 | 0.0748 | 171 | 4.5380 | 24 |
| heat_anomaly | test | 1511 | 0.1456 | 48 | 31.4792 | 228 |
| cold_anomaly | test | 219 | 0.0211 | 6 | 36.5000 | 63 |
| normal | test | 6942 | 0.6690 | 177 | 39.2203 | 298 |

---

## 4. Kiểm tra trôi theo thời gian (Time Drift Analysis)

### 4.1. Nhiệt độ và phần dư theo từng năm dương lịch và từng tập

- `T_mean_hourly`: Nhiệt độ trung bình chuỗi giờ trong năm & tập.
- `anom_T_mean`: Giá trị trung bình của phần dư $A_T$ trong cửa sổ.
- `heat_share` / `cold_share`: Tỷ lệ cửa sổ kích hoạt Heat / Cold anomaly.

| year | split | n_windows | anom_T_mean | heat_share | cold_share | T_mean_hourly |
|---|---|---|---|---|---|---|
| 2009 | train | 8665 | -0.2507 | 0.0114 | 0.0550 | 8.8276 |
| 2010 | train | 8784 | -1.5877 | 0.0446 | 0.1744 | 7.5066 |
| 2011 | train | 8760 | 0.1436 | 0.0432 | 0.0758 | 9.3055 |
| 2012 | train | 8784 | 0.5495 | 0.1174 | 0.0635 | 9.6551 |
| 2013 | train | 8760 | 0.0106 | 0.1061 | 0.0884 | 9.0970 |
| 2014 | train | 5228 | 1.9368 | 0.1670 | 0.0084 | 11.0534 |
| 2014 | val | 3427 | 1.2524 | 0.1491 | 0.0341 | 10.1797 |
| 2015 | val | 6950 | 0.8809 | 0.0787 | 0.0233 | 11.3176 |
| 2015 | test | 1691 | 3.7701 | 0.4341 | 0.0000 | 7.4172 |
| 2016 | test | 8686 | 0.9191 | 0.0895 | 0.0252 | 10.0230 |

**Phát hiện mấu chốt về trôi dạt (Drift):**
- Năm 2010 (thuộc train): Năm lạnh điển hình, `anom_T_mean = -1.59°C`, `cold_share = 17.4%`.
- Năm 2014-2015: Bắt đầu ấm lên mạnh. Đặc biệt cuối năm 2015 (thuộc test), `anom_T_mean = +3.77°C`, `heat_share = 43.4%`, `cold_share = 0%`.
- Điều này giải thích tại sao test set có rất nhiều heat anomaly và cực kỳ ít cold anomaly.

### 4.2. Số cửa sổ theo (Tập × Tháng) — Độ phủ mùa

| split | Tháng_1 | Tháng_2 | Tháng_3 | Tháng_4 | Tháng_5 | Tháng_6 | Tháng_7 | Tháng_8 | Tháng_9 | Tháng_10 | Tháng_11 | Tháng_12 | Tổng_cửa_sổ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 4369 | 4056 | 4494 | 4320 | 4464 | 4320 | 4488 | 3830 | 3600 | 3720 | 3600 | 3720 | 48981 |
| val | 744 | 672 | 744 | 720 | 744 | 720 | 744 | 1259 | 1424 | 1142 | 720 | 744 | 10377 |
| test | 744 | 696 | 744 | 720 | 744 | 720 | 744 | 744 | 720 | 897 | 1440 | 1464 | 10377 |

Tỷ lệ phân bố tháng trong mỗi tập:

| split | Tháng_1 | Tháng_2 | Tháng_3 | Tháng_4 | Tháng_5 | Tháng_6 | Tháng_7 | Tháng_8 | Tháng_9 | Tháng_10 | Tháng_11 | Tháng_12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 0.0892 | 0.0828 | 0.0917 | 0.0882 | 0.0911 | 0.0882 | 0.0916 | 0.0782 | 0.0735 | 0.0759 | 0.0735 | 0.0759 |
| val | 0.0717 | 0.0648 | 0.0717 | 0.0694 | 0.0717 | 0.0694 | 0.0717 | 0.1213 | 0.1372 | 0.1101 | 0.0694 | 0.0717 |
| test | 0.0717 | 0.0671 | 0.0717 | 0.0694 | 0.0717 | 0.0694 | 0.0717 | 0.0717 | 0.0694 | 0.0864 | 0.1388 | 0.1411 |

**Nhận xét về độ lệch mùa:**
- Tập **Train** trải dài 5.5 năm nên các tháng 1–12 được phủ đồng đều (mỗi tháng chiếm ~7.3% – 9.2%).
- Tập **Val** kéo dài ~14 tháng (8/2014 – 10/2015): Tháng 8, 9, 10 bị lặp lại 2 lần (chiếm 11% – 14% tập val), trong khi các tháng 11, 12, 1–7 chỉ xuất hiện 1 lần (~7%).
- Tập **Test** kéo dài ~14.5 tháng (10/2015 – 12/2016): Tháng 11 và 12 bị lặp lại 2 lần (chiếm tới 13.9% và 14.1% tập test). Đợt mùa đông ấm kỷ lục cuối năm 2015 bị lặp lại trọng số lớn trong test set.

### 4.3. Khoảng thời gian thực tế của các tập

| split | n_windows | window_indices | first_window_start | first_window_end | last_window_start | last_window_end | span_start_time | span_end_time | duration_days | duration_years |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 48981 | [0, 48981) | 2009-01-01 00:10:00 | 2009-01-04 23:10:00 | 2014-08-01 14:20:00 | 2014-08-05 13:20:00 | 2009-01-01 00:10:00 | 2014-08-05 13:20:00 | 2042.5486 | 5.5922 |
| val | 10377 | [49100, 59477) | 2014-08-06 14:20:00 | 2014-08-10 13:20:00 | 2015-10-13 14:10:00 | 2015-10-17 13:10:00 | 2014-08-06 14:20:00 | 2015-10-17 13:10:00 | 436.9514 | 1.1963 |
| test | 10377 | [59596, 69973) | 2015-10-18 14:10:00 | 2015-10-22 13:10:00 | 2016-12-27 00:20:00 | 2016-12-30 23:20:00 | 2015-10-18 14:10:00 | 2016-12-30 23:20:00 | 439.3819 | 1.2030 |

---

## 5. Cold Front tổ hợp (Luật kết hợp)

Thống kê số lượng cửa sổ và số run cho `front_dir` (áp suất đổi chiều + nhiệt giảm nhanh + gió đổi hướng) và `front_either_wind` (đổi hướng HOẶC đổi tốc độ gió):

| component | split | n_windows | share | n_runs | mean_run_length | max_run_length |
|---|---|---|---|---|---|---|
| front_dir | train | 101 | 0.0021 | 74 | 1.3649 | 5 |
| front_dir | val | 21 | 0.0020 | 11 | 1.9091 | 8 |
| front_dir | test | 23 | 0.0022 | 18 | 1.2778 | 3 |
| front_either_wind | train | 181 | 0.0037 | 120 | 1.5083 | 6 |
| front_either_wind | val | 33 | 0.0032 | 14 | 2.3571 | 12 |
| front_either_wind | test | 49 | 0.0047 | 30 | 1.6333 | 5 |

**Nhận xét:**
- `front_dir` có 74 run ở train, 11 run ở val, 18 run ở test.
- Độ dài mỗi run rất ngắn (trung bình 1.3 – 1.9 cửa sổ), cho thấy đây là sự kiện chớp nhoáng (transient burst) chứ không kéo dài.

---

## 6. Tiêu chí C2 mức sự kiện (Event-level C2 Criteria)

Quy chuẩn kiểm tra mức sự kiện: `runs_train >= 20`, `runs_val >= 5`, `runs_test >= 5`.

### 6.1. Đánh giá trên 5 nhãn gốc:

| label | share_train | n_train | n_val | n_test | min_n | runs_train | runs_val | runs_test | min_runs | C2_cua_so | C2_su_kien | C2_toan_dien |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spike | 0.0116 | 570 | 95 | 0 | 0 | 6 | 1 | 0 | 0 | False | False | False |
| pressure_drop | 0.0994 | 4870 | 1071 | 929 | 929 | 185 | 36 | 36 | 36 | True | True | True |
| humidity_rise | 0.0885 | 4336 | 862 | 776 | 776 | 871 | 186 | 171 | 171 | True | True | True |
| heat_anomaly | 0.0756 | 3702 | 1058 | 1511 | 1058 | 173 | 32 | 48 | 32 | True | True | True |
| cold_anomaly | 0.0827 | 4049 | 279 | 219 | 219 | 110 | 12 | 6 | 6 | False | True | False |

#### Cảnh báo mức sự kiện (5 nhãn gốc):
- ⚠️ C2 muc su kien KHONG DAT: spike (runs_train=6 < 20, runs_val=1 < 5, runs_test=0 < 5)

### 6.2. Đánh giá trên 4 nhãn gộp (`temp_anomaly`):

| label | share_train | n_train | n_val | n_test | min_n | runs_train | runs_val | runs_test | min_runs | C2_cua_so | C2_su_kien | C2_toan_dien |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spike | 0.0116 | 570 | 95 | 0 | 0 | 6 | 1 | 0 | 0 | False | False | False |
| pressure_drop | 0.0994 | 4870 | 1071 | 929 | 929 | 185 | 36 | 36 | 36 | True | True | True |
| humidity_rise | 0.0885 | 4336 | 862 | 776 | 776 | 871 | 186 | 171 | 171 | True | True | True |
| temp_anomaly | 0.1582 | 7751 | 1337 | 1730 | 1337 | 283 | 44 | 54 | 44 | True | True | True |

#### Cảnh báo mức sự kiện (4 nhãn gộp):
- ⚠️ C2 muc su kien KHONG DAT: spike (runs_train=6 < 20, runs_val=1 < 5, runs_test=0 < 5)

---

## Tổng hợp cảnh báo & Kết luận đo đạc

1. **Spike ở phân vị 0.99:** Quá hiếm ở mức sự kiện (chỉ có 6 run ở train, 1 ở val, 0 ở test), vi phạm tiêu chuẩn C2 mức sự kiện. Nếu muốn Spike đạt tối thiểu 20 run ở train và $\ge 5$ run ở val/test, cần cân nhắc hạ `q_spike` về mức 0.95 (25 train / 6 val / 3 test).
2. **Tách biệt Heat vs Cold Anomaly:** Bị ảnh hưởng nặng nề bởi biến động khí hậu thực tế (El Niño cuối 2015 cực ấm) và sự không cân đối mùa (Val lặp tháng 8-10, Test lặp tháng 11-12).
3. **Hiệu quả của nhãn gộp `temp_anomaly`:** Giúp loại bỏ hoàn toàn vi phạm C2 và C4 ở mức cửa sổ, đồng thời đạt 283 run (train), 44 run (val), 54 run (test) — vượt xa ngưỡng C2 sự kiện.

Toàn bộ bảng số liệu CSV đã được lưu tại: `D:\Github\PALF\experiments\pattern_stats\jena\step3`.
