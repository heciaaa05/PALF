# Báo cáo So sánh Giao thức Công bằng: Baseline vs PALF (Mục 7)

> **Quy tắc thống kê (Spec Mục 7):** Với cấu hình có ít hơn 5 seed (`n_seeds < 5`), báo cáo **chỉ mô tả xu hướng** và hiệu số ghép theo từng seed, tuyệt đối **không tuyên bố ý nghĩa thống kê**.

## 1. Bảng tổng hợp toàn cục theo Cấu hình & Backbone

| Backbone | Mã | Config Tag | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Mẫu tới `val_target` | Giây tới `val_target` | AUC Val (mẫu) | Epoch time (s) | Overhead PALF (s) | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | 5 (0,1,2,3,4) | 1.7836 ± 0.0255 | 2.3446 ± 0.0326 | 0.0809 ± 0.0017 | 166535 (5/5) | 156.5s | 0.1092 | 47.37s | 0.0000s | 5_seeds_complete |
| `GRU` | **A0r** | `uniform_b0.7` | 3 (0,1,2) | 3.4064 ± 0.0597 | 4.3694 ± 0.0633 | 0.1809 ± 0.0119 | Chưa đạt (0/3) | Chưa đạt | 0.2212 | 1.98s | 0.0006s | trend_only (<5 seeds) |
| `GRU` | **A1** | `loss_b1.0_a1.0` | 3 (0,1,2) | 3.4071 ± 0.0402 | 4.3700 ± 0.0412 | 0.1814 ± 0.0116 | Chưa đạt (0/3) | Chưa đạt | 0.2128 | 2.56s | 0.0009s | trend_only (<5 seeds) |
| `GRU` | **A2** | `state_b1.0_a1.0` | 3 (0,1,2) | 3.4071 ± 0.0402 | 4.3700 ± 0.0412 | 0.1814 ± 0.0116 | Chưa đạt (0/3) | Chưa đạt | 0.2128 | 2.62s | 0.0008s | trend_only (<5 seeds) |
| `GRU` | **A2r** | `state_b0.7_a1.0` | 3 (0,1,2) | 3.4102 ± 0.0312 | 4.3775 ± 0.0478 | 0.1756 ± 0.0132 | Chưa đạt (0/3) | Chưa đạt | 0.2197 | 1.96s | 0.0008s | trend_only (<5 seeds) |
| `TCN` | **A0** | `baseline` | 5 (0,1,2,3,4) | 1.7751 ± 0.0180 | 2.3369 ± 0.0221 | 0.0820 ± 0.0011 | 146943 (4/5) | 79.9s | 0.1010 | 26.64s | 0.0000s | 5_seeds_complete |
| `TRANSFORMER` | **A0** | `baseline` | 5 (0,1,2,3,4) | 1.8952 ± 0.0443 | 2.4752 ± 0.0495 | 0.0946 ± 0.0040 | 97962 (4/5) | 81.1s | 0.1344 | 41.35s | 0.0000s | 5_seeds_complete |
| `PATCHTST` | **A0** | `baseline` | 5 (0,1,2,3,4) | 1.9148 ± 0.0193 | 2.5185 ± 0.0304 | 0.0902 ± 0.0023 | 404093 (4/5) | 1344.9s | 0.1069 | 166.95s | 0.0000s | 5_seeds_complete |

## 2. Bảng MAE và RMSE (°C) theo từng Pattern (kể cả tách `temp_anomaly` nóng/lạnh)

| Backbone | Mã | Nhóm | Pattern | Số mẫu test | MAE (°C) (Mean ± Std) | RMSE (°C) (Mean ± Std) | Chênh lệch MAE so với `normal` (°C) |
|---|---|---|---|---:|---:|---:|---:|
| `GRU` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.7223 ± 0.0537 | 2.2890 ± 0.0702 | -0.0335 |
| `GRU` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.7083 ± 0.0413 | 2.2380 ± 0.0512 | -0.0474 |
| `GRU` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 1.9623 ± 0.0459 | 2.5615 ± 0.0773 | +0.2065 |
| `GRU` | **A0** | `main_pattern` | `normal` | 6942 | 1.7558 ± 0.0200 | 2.3066 ± 0.0230 | +0.0000 |
| `GRU` | **A0r** | `main_pattern` | `humidity_net_rise` | 9 | 3.5429 ± 0.1366 | 4.2814 ± 0.1599 | -0.3278 |
| `GRU` | **A0r** | `main_pattern` | `temp_anomaly` | 132 | 2.0919 ± 0.1142 | 2.6123 ± 0.1100 | -1.7789 |
| `GRU` | **A0r** | `main_pattern` | `normal` | 371 | 3.8708 ± 0.0387 | 4.8449 ± 0.0547 | +0.0000 |
| `GRU` | **A0r** | `temp_sign_split` | `temp_anomaly_heat` | 132 | 2.0919 ± 0.1142 | 2.6123 ± 0.1100 | -1.7789 |
| `GRU` | **A1** | `main_pattern` | `humidity_net_rise` | 9 | 3.5098 ± 0.0504 | 4.2478 ± 0.0541 | -0.3641 |
| `GRU` | **A1** | `main_pattern` | `temp_anomaly` | 132 | 2.0882 ± 0.0756 | 2.6118 ± 0.0708 | -1.7856 |
| `GRU` | **A1** | `main_pattern` | `normal` | 371 | 3.8739 ± 0.0279 | 4.8465 ± 0.0377 | +0.0000 |
| `GRU` | **A1** | `temp_sign_split` | `temp_anomaly_heat` | 132 | 2.0882 ± 0.0756 | 2.6118 ± 0.0708 | -1.7856 |
| `GRU` | **A2** | `main_pattern` | `humidity_net_rise` | 9 | 3.5098 ± 0.0504 | 4.2478 ± 0.0541 | -0.3641 |
| `GRU` | **A2** | `main_pattern` | `temp_anomaly` | 132 | 2.0882 ± 0.0756 | 2.6118 ± 0.0708 | -1.7856 |
| `GRU` | **A2** | `main_pattern` | `normal` | 371 | 3.8739 ± 0.0279 | 4.8465 ± 0.0377 | +0.0000 |
| `GRU` | **A2** | `temp_sign_split` | `temp_anomaly_heat` | 132 | 2.0882 ± 0.0756 | 2.6118 ± 0.0708 | -1.7856 |
| `GRU` | **A2r** | `main_pattern` | `humidity_net_rise` | 9 | 3.3939 ± 0.0685 | 4.0864 ± 0.0756 | -0.5040 |
| `GRU` | **A2r** | `main_pattern` | `temp_anomaly` | 132 | 2.0405 ± 0.0517 | 2.5600 ± 0.0384 | -1.8574 |
| `GRU` | **A2r** | `main_pattern` | `normal` | 371 | 3.8979 ± 0.0315 | 4.8692 ± 0.0524 | +0.0000 |
| `GRU` | **A2r** | `temp_sign_split` | `temp_anomaly_heat` | 132 | 2.0405 ± 0.0517 | 2.5600 ± 0.0384 | -1.8574 |
| `TCN` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.7520 ± 0.0295 | 2.3724 ± 0.0285 | -0.0009 |
| `TCN` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.6455 ± 0.0232 | 2.1505 ± 0.0195 | -0.1073 |
| `TCN` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 1.9352 ± 0.0418 | 2.5345 ± 0.0624 | +0.1824 |
| `TCN` | **A0** | `main_pattern` | `normal` | 6942 | 1.7528 ± 0.0221 | 2.3004 ± 0.0260 | +0.0000 |
| `TRANSFORMER` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.7331 ± 0.0419 | 2.3303 ± 0.0446 | -0.1467 |
| `TRANSFORMER` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.8184 ± 0.0537 | 2.4005 ± 0.0576 | -0.0613 |
| `TRANSFORMER` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 2.0784 ± 0.0552 | 2.6784 ± 0.0697 | +0.1987 |
| `TRANSFORMER` | **A0** | `main_pattern` | `normal` | 6942 | 1.8798 ± 0.0451 | 2.4494 ± 0.0477 | +0.0000 |
| `PATCHTST` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.9543 ± 0.0212 | 2.6586 ± 0.0435 | +0.0975 |
| `PATCHTST` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.8763 ± 0.0487 | 2.4264 ± 0.0563 | +0.0196 |
| `PATCHTST` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 2.1437 ± 0.0203 | 2.8062 ± 0.0241 | +0.2870 |
| `PATCHTST` | **A0** | `main_pattern` | `normal` | 6942 | 1.8567 ± 0.0215 | 2.4320 ± 0.0353 | +0.0000 |

## 3. So sánh ghép cặp theo Seed (Paired Differences vs A0 / A0r)

| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |
|---|---|---|---|---:|---:|---|---|
| `GRU` | **A0r** | **A0** | `mae_degC` | 3 | +1.6191 ± 0.0334 | `s0:+1.6364; s1:+1.6404; s2:+1.5807` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `rmse_degC` | 3 | +2.0177 ± 0.0335 | `s0:+2.0369; s1:+2.0372; s2:+1.9790` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `best_val_loss` | 3 | +0.0990 ± 0.0119 | `s0:+0.1100; s1:+0.1008; s2:+0.0863` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `val_auc_samples` | 3 | +0.1090 ± 0.0261 | `s0:+0.1371; s1:+0.1045; s2:+0.0855` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `mae_degC` | 3 | +1.6199 ± 0.0164 | `s0:+1.6346; s1:+1.6228; s2:+1.6022` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `rmse_degC` | 3 | +2.0182 ± 0.0148 | `s0:+2.0337; s1:+2.0167; s2:+2.0043` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `best_val_loss` | 3 | +0.0995 ± 0.0117 | `s0:+0.1096; s1:+0.1024; s2:+0.0867` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `val_auc_samples` | 3 | +0.1007 ± 0.0220 | `s0:+0.1210; s1:+0.1038; s2:+0.0773` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2** | **A0** | `mae_degC` | 3 | +1.6199 ± 0.0164 | `s0:+1.6346; s1:+1.6228; s2:+1.6022` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2** | **A0** | `rmse_degC` | 3 | +2.0182 ± 0.0148 | `s0:+2.0337; s1:+2.0167; s2:+2.0043` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2** | **A0** | `best_val_loss` | 3 | +0.0995 ± 0.0117 | `s0:+0.1096; s1:+0.1024; s2:+0.0867` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2** | **A0** | `val_auc_samples` | 3 | +0.1007 ± 0.0220 | `s0:+0.1210; s1:+0.1038; s2:+0.0773` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2r** | **A0r** | `mae_degC` | 3 | +0.0038 ± 0.0418 | `s0:-0.0197; s1:-0.0209; s2:+0.0521` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2r** | **A0r** | `rmse_degC` | 3 | +0.0081 ± 0.0520 | `s0:-0.0318; s1:-0.0108; s2:+0.0670` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2r** | **A0r** | `best_val_loss` | 3 | -0.0052 ± 0.0017 | `s0:-0.0049; s1:-0.0037; s2:-0.0071` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2r** | **A0r** | `val_auc_samples` | 3 | -0.0014 ± 0.0006 | `s0:-0.0021; s1:-0.0013; s2:-0.0009` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2r** | **A0** | `mae_degC` | 3 | +1.6230 ± 0.0086 | `s0:+1.6167; s1:+1.6195; s2:+1.6328` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2r** | **A0** | `rmse_degC` | 3 | +2.0258 ± 0.0205 | `s0:+2.0051; s1:+2.0264; s2:+2.0460` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2r** | **A0** | `best_val_loss` | 3 | +0.0938 ± 0.0133 | `s0:+0.1051; s1:+0.0971; s2:+0.0792` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A2r** | **A0** | `val_auc_samples` | 3 | +0.1076 ± 0.0255 | `s0:+0.1350; s1:+0.1032; s2:+0.0846` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |

## 4. Báo cáo các cấu hình còn thiếu (Missing Configurations)

| Backbone | Mã | Config Tag | Mô tả | Trạng thái | Seed đã có | Seed còn thiếu |
|---|---|---|---|---|---|---|
| `GRU` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | `PARTIAL` | `0,1,2` (3) | `3,4` (2) |
| `GRU` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | `PARTIAL` | `0,1,2` (3) | `3,4` (2) |
| `GRU` | **A2** | `state_b1.0_a1.0` | PALF Learning State (budget 1.0) | `PARTIAL` | `0,1,2` (3) | `3,4` (2) |
| `GRU` | **A2r** | `state_b0.7_a1.0` | PALF Learning State giam ngan sach (budget 0.7) | `PARTIAL` | `0,1,2` (3) | `3,4` (2) |
| `TCN` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `TCN` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `TCN` | **A2** | `state_b1.0_a1.0` | PALF Learning State (budget 1.0) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `TCN` | **A2r** | `state_b0.7_a1.0` | PALF Learning State giam ngan sach (budget 0.7) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `TRANSFORMER` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `TRANSFORMER` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `TRANSFORMER` | **A2** | `state_b1.0_a1.0` | PALF Learning State (budget 1.0) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `TRANSFORMER` | **A2r** | `state_b0.7_a1.0` | PALF Learning State giam ngan sach (budget 0.7) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `PATCHTST` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `PATCHTST` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `PATCHTST` | **A2** | `state_b1.0_a1.0` | PALF Learning State (budget 1.0) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
| `PATCHTST` | **A2r** | `state_b0.7_a1.0` | PALF Learning State giam ngan sach (budget 0.7) | `MISSING_ALL` | `none` (0) | `0,1,2,3,4` (5) |
