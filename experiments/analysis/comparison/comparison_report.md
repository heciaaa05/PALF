# Báo cáo So sánh Giao thức Công bằng: Baseline vs PALF (Spec v1 — Mục 7)

> **Quy tắc thống kê & đồng nhất (Spec v1 Mục 7):**
> 1. Mọi lần chạy trong bảng bắt buộc có cùng `n_train`, `n_val`, `n_test`, `epochs`, `batch`, `lr` và cùng danh sách `seeds` (các lần chạy thử nhỏ lưu riêng ở `experiments/results/smoke/`).
> 2. Với cấu hình có ít hơn 5 seed (`n_seeds < 5`), báo cáo **chỉ mô tả xu hướng** và hiệu số ghép theo từng seed, tuyệt đối **không tuyên bố ý nghĩa thống kê**.

## 1. Bảng tổng hợp toàn cục theo Cấu hình & Backbone

| Backbone | Mã | Config Tag | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Mẫu tới `val_target` | Giây tới `val_target` | AUC Val (mẫu) | Epoch time (s) | Overhead PALF (s) | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | 3 (0,1,2) | 1.7872 ± 0.0335 | 2.3517 ± 0.0411 | 0.0818 ± 0.0003 | 163270 (3/3) | 159.1s | 0.1121 | 50.40s | 0.0000s | trend_only (<5 seeds) |
| `GRU` | **A0r** | `uniform_b0.7` | 3 (0,1,2) | 1.7692 ± 0.0111 | 2.3334 ± 0.0173 | 0.0808 ± 0.0005 | 171435 (3/3) | 274.6s | 0.1014 | 58.14s | 0.0045s | trend_only (<5 seeds) |
| `GRU` | **A1** | `loss_b1.0_a1.0` | 3 (0,1,2) | 1.7734 ± 0.0246 | 2.3392 ± 0.0371 | 0.0814 ± 0.0020 | 212251 (3/3) | 332.0s | 0.1110 | 77.70s | 0.0062s | trend_only (<5 seeds) |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | 3 (0,1,2) | 1.7704 ± 0.0156 | 2.3309 ± 0.0195 | 0.0802 ± 0.0020 | 212251 (3/3) | 334.4s | 0.1103 | 77.91s | 0.0064s | trend_only (<5 seeds) |
| `GRU` | **A3r** | `predictive_b0.7_a0.5_w2` | 3 (0,1,2) | 1.7701 ± 0.0071 | 2.3319 ± 0.0114 | 0.0801 ± 0.0008 | 160006 (3/3) | 267.4s | 0.1036 | 60.06s | 0.0050s | trend_only (<5 seeds) |

## 2. Bảng MAE và RMSE (°C) theo từng Pattern (kể cả tách `temp_anomaly` nóng/lạnh)

| Backbone | Mã | Nhóm | Pattern | Số mẫu test | MAE (°C) (Mean ± Std) | RMSE (°C) (Mean ± Std) | Chênh lệch MAE so với `normal` (°C) |
|---|---|---|---|---:|---:|---:|---:|
| `GRU` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.7411 ± 0.0599 | 2.3123 ± 0.0866 | -0.0132 |
| `GRU` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.6973 ± 0.0532 | 2.2208 ± 0.0628 | -0.0569 |
| `GRU` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 1.9848 ± 0.0474 | 2.6021 ± 0.0689 | +0.2305 |
| `GRU` | **A0** | `main_pattern` | `normal` | 6942 | 1.7542 ± 0.0264 | 2.3048 ± 0.0302 | +0.0000 |
| `GRU` | **A0r** | `main_pattern` | `pressure_drop` | 929 | 1.7006 ± 0.0399 | 2.2981 ± 0.0605 | -0.0362 |
| `GRU` | **A0r** | `main_pattern` | `humidity_net_rise` | 776 | 1.6767 ± 0.0186 | 2.2078 ± 0.0243 | -0.0601 |
| `GRU` | **A0r** | `main_pattern` | `temp_anomaly` | 1730 | 1.9772 ± 0.0154 | 2.5789 ± 0.0122 | +0.2404 |
| `GRU` | **A0r** | `main_pattern` | `normal` | 6942 | 1.7368 ± 0.0056 | 2.2869 ± 0.0124 | +0.0000 |
| `GRU` | **A0r** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8963 ± 0.0182 | 2.4625 ± 0.0262 | +0.1595 |
| `GRU` | **A0r** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.5354 ± 0.1950 | 3.2664 ± 0.1998 | +0.7986 |
| `GRU` | **A1** | `main_pattern` | `pressure_drop` | 929 | 1.6926 ± 0.0435 | 2.2892 ± 0.0730 | -0.0530 |
| `GRU` | **A1** | `main_pattern` | `humidity_net_rise` | 776 | 1.6757 ± 0.0245 | 2.2048 ± 0.0502 | -0.0698 |
| `GRU` | **A1** | `main_pattern` | `temp_anomaly` | 1730 | 1.9722 ± 0.0493 | 2.5735 ± 0.0808 | +0.2266 |
| `GRU` | **A1** | `main_pattern` | `normal` | 6942 | 1.7455 ± 0.0168 | 2.2987 ± 0.0198 | +0.0000 |
| `GRU` | **A1** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8927 ± 0.0227 | 2.4651 ± 0.0407 | +0.1471 |
| `GRU` | **A1** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.5206 ± 0.2445 | 3.2184 ± 0.2913 | +0.7751 |
| `GRU` | **A3** | `main_pattern` | `pressure_drop` | 929 | 1.6881 ± 0.0418 | 2.2723 ± 0.0410 | -0.0532 |
| `GRU` | **A3** | `main_pattern` | `humidity_net_rise` | 776 | 1.7064 ± 0.0200 | 2.2347 ± 0.0305 | -0.0349 |
| `GRU` | **A3** | `main_pattern` | `temp_anomaly` | 1730 | 1.9600 ± 0.0114 | 2.5540 ± 0.0103 | +0.2186 |
| `GRU` | **A3** | `main_pattern` | `normal` | 6942 | 1.7413 ± 0.0225 | 2.2906 ± 0.0246 | +0.0000 |
| `GRU` | **A3** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8997 ± 0.0207 | 2.4717 ± 0.0256 | +0.1584 |
| `GRU` | **A3** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.3760 ± 0.0660 | 3.0614 ± 0.0801 | +0.6347 |
| `GRU` | **A3r** | `main_pattern` | `pressure_drop` | 929 | 1.6930 ± 0.0255 | 2.2745 ± 0.0303 | -0.0431 |
| `GRU` | **A3r** | `main_pattern` | `humidity_net_rise` | 776 | 1.6962 ± 0.0091 | 2.2314 ± 0.0053 | -0.0398 |
| `GRU` | **A3r** | `main_pattern` | `temp_anomaly` | 1730 | 1.9815 ± 0.0218 | 2.5765 ± 0.0240 | +0.2455 |
| `GRU` | **A3r** | `main_pattern` | `normal` | 6942 | 1.7360 ± 0.0084 | 2.2858 ± 0.0173 | +0.0000 |
| `GRU` | **A3r** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.9221 ± 0.0210 | 2.4940 ± 0.0233 | +0.1861 |
| `GRU` | **A3r** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.3918 ± 0.0638 | 3.0849 ± 0.1099 | +0.6558 |

## 3. So sánh ghép cặp theo Seed (Paired Differences vs A0 / A0r / A1)

| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |
|---|---|---|---|---:|---:|---|---|
| `GRU` | **A0r** | **A0** | `mae_degC` | 3 | -0.0181 ± 0.0228 | `s0:-0.0035; s1:-0.0444; s2:-0.0064` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `rmse_degC` | 3 | -0.0183 ± 0.0273 | `s0:+0.0068; s1:-0.0474; s2:-0.0143` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `best_val_loss` | 3 | -0.0010 ± 0.0007 | `s0:-0.0008; s1:-0.0004; s2:-0.0018` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `samples_to_target` | 3 | +8165.0000 ± 60454.3026 | `s0:+58779.0000; s1:-58776.0000; s2:+24492.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `time_sec_to_target` | 3 | +115.5221 ± 60.9306 | `s0:+166.4882; s1:+48.0346; s2:+132.0436` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `val_auc_samples` | 3 | -0.0107 ± 0.0031 | `s0:-0.0072; s1:-0.0118; s2:-0.0130` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `mae_degC` | 3 | -0.0139 ± 0.0090 | `s0:-0.0085; s1:-0.0243; s2:-0.0089` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `rmse_degC` | 3 | -0.0125 ± 0.0070 | `s0:-0.0045; s1:-0.0170; s2:-0.0160` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `best_val_loss` | 3 | -0.0005 ± 0.0021 | `s0:-0.0028; s1:+0.0014; s2:-0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `samples_to_target` | 3 | +48981.0000 ± 48981.0000 | `s0:+97962.0000; s1:+48981.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `time_sec_to_target` | 3 | +172.9315 ± 73.9092 | `s0:+233.0684; s1:+195.3058; s2:+90.4204` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `val_auc_samples` | 3 | -0.0011 ± 0.0035 | `s0:+0.0016; s1:+0.0002; s2:-0.0050` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `mae_degC` | 3 | -0.0029 ± 0.0108 | `s0:-0.0030; s1:-0.0137; s2:+0.0079` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `rmse_degC` | 3 | -0.0083 ± 0.0230 | `s0:-0.0084; s1:-0.0312; s2:+0.0147` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `best_val_loss` | 3 | -0.0012 ± 0.0008 | `s0:-0.0012; s1:-0.0020; s2:-0.0003` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `time_sec_to_target` | 3 | +2.3661 ± 2.6587 | `s0:-0.6867; s1:+3.6122; s2:+4.1729` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `val_auc_samples` | 3 | -0.0007 ± 0.0004 | `s0:-0.0012; s1:-0.0004; s2:-0.0006` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `mae_degC` | 3 | -0.0168 ± 0.0191 | `s0:-0.0114; s1:-0.0380; s2:-0.0010` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `rmse_degC` | 3 | -0.0208 ± 0.0245 | `s0:-0.0129; s1:-0.0483; s2:-0.0012` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `best_val_loss` | 3 | -0.0017 ± 0.0020 | `s0:-0.0040; s1:-0.0006; s2:-0.0004` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `samples_to_target` | 3 | +48981.0000 ± 48981.0000 | `s0:+97962.0000; s1:+48981.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `time_sec_to_target` | 3 | +175.2976 ± 71.8669 | `s0:+232.3816; s1:+198.9181; s2:+94.5932` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `val_auc_samples` | 3 | -0.0018 ± 0.0033 | `s0:+0.0003; s1:-0.0002; s2:-0.0056` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `mae_degC` | 3 | +0.0010 ± 0.0048 | `s0:-0.0001; s1:-0.0032; s2:+0.0062` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `rmse_degC` | 3 | -0.0016 ± 0.0086 | `s0:-0.0064; s1:-0.0066; s2:+0.0083` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `best_val_loss` | 3 | -0.0008 ± 0.0011 | `s0:-0.0020; s1:-0.0005; s2:+0.0001` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `samples_to_target` | 3 | -11429.0000 ± 19795.6087 | `s0:-34287.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `time_sec_to_target` | 3 | -7.2640 ± 32.6292 | `s0:-43.5654; s1:+2.1502; s2:+19.6231` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `val_auc_samples` | 3 | +0.0021 ± 0.0018 | `s0:+0.0002; s1:+0.0024; s2:+0.0038` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `mae_degC` | 3 | -0.0171 ± 0.0264 | `s0:-0.0036; s1:-0.0476; s2:-0.0002` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `rmse_degC` | 3 | -0.0199 ± 0.0297 | `s0:+0.0004; s1:-0.0540; s2:-0.0059` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `best_val_loss` | 3 | -0.0018 ± 0.0010 | `s0:-0.0028; s1:-0.0009; s2:-0.0016` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `samples_to_target` | 3 | -3264.0000 ± 48074.8022 | `s0:+24492.0000; s1:-58776.0000; s2:+24492.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `time_sec_to_target` | 3 | +108.2581 ± 52.3062 | `s0:+122.9229; s1:+50.1848; s2:+151.6667` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `val_auc_samples` | 3 | -0.0085 ± 0.0013 | `s0:-0.0070; s1:-0.0094; s2:-0.0092` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |

## 4. Báo cáo các cấu hình còn thiếu (Missing Configurations)

Tất cả các cấu hình và seed trong danh sách kiểm tra đều đã có đủ trên đĩa.
