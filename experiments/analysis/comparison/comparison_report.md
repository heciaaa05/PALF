# Báo cáo So sánh Giao thức Công bằng: Baseline vs PALF (Spec v1 — Mục 7 & Prompt 8)

> **Quy tắc thống kê, đồng nhất & định danh G3 (Spec v1 Mục 7 & Prompt 8):**
> 1. Mọi lần chạy trong bảng bắt buộc có cùng `n_train`, `n_val`, `n_test`, `epochs`, `batch`, `lr` và cùng danh sách `seeds` (các lần chạy thử nhỏ lưu riêng ở `experiments/results/smoke/`).
> 2. Tiêu chí G3 đặt trước cho kết quả **"lui về phương án dự phòng"** (chỉ 2/4 backbone đạt $\Delta R^2 \ge 0.02$ sau kiểm soát epoch và dấu $\beta$ không cùng chiều ở PatchTST). Do đó, cấu hình **A3** và **A3r** được gọi đúng tên là **`predictive (không đạt ngưỡng G3)`**.
> 3. **Ghi chú về thời gian chạy:** Thời gian chạy (`epoch_time_sec`, `time_sec_to_target`, `overhead_sec`) không so sánh được giữa các máy/lần chạy khác nhau và đường mã lấy mẫu khác nhau, không dùng để kết luận và được chuyển xuống **Phụ lục** ở cuối báo cáo.

## 1. Bảng tổng hợp chính theo Cấu hình & Backbone

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | Baseline uniform (budget 1.0) | 5 (0,1,2,3,4) | 1.7836 ± 0.0255 | 2.3446 ± 0.0326 | 0.0809 ± 0.0017 | 5.6 | 166535 (5/5) | 5_seeds_complete |
| `GRU` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | 5 (0,1,2,3,4) | 1.7678 ± 0.0118 | 2.3331 ± 0.0148 | 0.0805 ± 0.0009 | 6.4 | 185150 (5/5) | 5_seeds_complete |
| `GRU` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | 5 (0,1,2,3,4) | 1.7618 ± 0.0238 | 2.3321 ± 0.0312 | 0.0811 ± 0.0019 | 5.8 | 205720 (5/5) | 5_seeds_complete |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | predictive (khong dat nguong G3, budget 1.0) | 5 (0,1,2,3,4) | 1.7682 ± 0.0124 | 2.3308 ± 0.0140 | 0.0801 ± 0.0015 | 5.2 | 205720 (5/5) | 5_seeds_complete |
| `GRU` | **A3r** | `predictive_b0.7_a0.5_w2` | predictive (khong dat nguong G3, budget 0.7) | 5 (0,1,2,3,4) | 1.7699 ± 0.0118 | 2.3323 ± 0.0120 | 0.0799 ± 0.0013 | 6.2 | 171435 (5/5) | 5_seeds_complete |

### 1b. Bảng đối chiếu nguyên vẹn trên 3 Seed gốc (`seeds 0, 1, 2` — theo Mục 8.3)

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | Baseline uniform (budget 1.0) | 3 (0,1,2) | 1.7872 ± 0.0335 | 2.3517 ± 0.0411 | 0.0818 ± 0.0003 | 5.7 | 179597 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | 3 (0,1,2) | 1.7692 ± 0.0111 | 2.3334 ± 0.0173 | 0.0808 ± 0.0005 | 6.0 | 194293 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | 3 (0,1,2) | 1.7734 ± 0.0246 | 2.3392 ± 0.0371 | 0.0814 ± 0.0020 | 5.7 | 212251 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | predictive (khong dat nguong G3, budget 1.0) | 3 (0,1,2) | 1.7704 ± 0.0156 | 2.3309 ± 0.0195 | 0.0802 ± 0.0020 | 5.3 | 212251 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A3r** | `predictive_b0.7_a0.5_w2` | predictive (khong dat nguong G3, budget 0.7) | 3 (0,1,2) | 1.7701 ± 0.0071 | 2.3319 ± 0.0114 | 0.0801 ± 0.0008 | 6.3 | 171435 (3/3) | trend_only (<5 seeds) |

### 1c. Bảng trung bình 4 Seed độc lập (`seeds 1, 2, 3, 4` — tách khỏi `seed 0` đã dùng chọn siêu tham số ở Bước 5.1)

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | Baseline uniform (budget 1.0) | 4 (1,2,3,4) | 1.7872 ± 0.0280 | 2.3496 ± 0.0354 | 0.0807 ± 0.0018 | 5.5 | 159188 (4/4) | trend_only (<5 seeds) |
| `GRU` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | 4 (1,2,3,4) | 1.7683 ± 0.0135 | 2.3334 ± 0.0171 | 0.0804 ± 0.0009 | 6.5 | 180007 (4/4) | trend_only (<5 seeds) |
| `GRU` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | 4 (1,2,3,4) | 1.7621 ± 0.0275 | 2.3350 ± 0.0352 | 0.0816 ± 0.0018 | 5.8 | 195924 (4/4) | trend_only (<5 seeds) |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | predictive (khong dat nguong G3, budget 1.0) | 4 (1,2,3,4) | 1.7707 ± 0.0128 | 2.3356 ± 0.0106 | 0.0806 ± 0.0010 | 5.2 | 195924 (4/4) | trend_only (<5 seeds) |
| `GRU` | **A3r** | `predictive_b0.7_a0.5_w2` | predictive (khong dat nguong G3, budget 0.7) | 4 (1,2,3,4) | 1.7709 ± 0.0134 | 2.3341 ± 0.0131 | 0.0801 ± 0.0014 | 6.2 | 171435 (4/4) | trend_only (<5 seeds) |

## 2. Bảng MAE và RMSE (°C) theo từng Pattern (kể cả tách `temp_anomaly` nóng/lạnh)

| Backbone | Mã | Nhóm | Pattern | Số mẫu test | MAE (°C) (Mean ± Std) | RMSE (°C) (Mean ± Std) | Chênh lệch MAE so với `normal` (°C) |
|---|---|---|---|---:|---:|---:|---:|
| `GRU` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.7223 ± 0.0537 | 2.2890 ± 0.0702 | -0.0335 |
| `GRU` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.7083 ± 0.0413 | 2.2380 ± 0.0512 | -0.0474 |
| `GRU` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 1.9623 ± 0.0459 | 2.5615 ± 0.0773 | +0.2065 |
| `GRU` | **A0** | `main_pattern` | `normal` | 6942 | 1.7558 ± 0.0200 | 2.3066 ± 0.0230 | +0.0000 |
| `GRU` | **A0r** | `main_pattern` | `pressure_drop` | 929 | 1.6969 ± 0.0333 | 2.2840 ± 0.0520 | -0.0422 |
| `GRU` | **A0r** | `main_pattern` | `humidity_net_rise` | 776 | 1.6679 ± 0.0243 | 2.2039 ± 0.0242 | -0.0712 |
| `GRU` | **A0r** | `main_pattern` | `temp_anomaly` | 1730 | 1.9659 ± 0.0209 | 2.5664 ± 0.0255 | +0.2268 |
| `GRU` | **A0r** | `main_pattern` | `normal` | 6942 | 1.7391 ± 0.0083 | 2.2921 ± 0.0118 | +0.0000 |
| `GRU` | **A0r** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8812 ± 0.0247 | 2.4421 ± 0.0340 | +0.1421 |
| `GRU` | **A0r** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.5505 ± 0.1456 | 3.2951 ± 0.1638 | +0.8114 |
| `GRU` | **A1** | `main_pattern` | `pressure_drop` | 929 | 1.6893 ± 0.0313 | 2.2862 ± 0.0518 | -0.0391 |
| `GRU` | **A1** | `main_pattern` | `humidity_net_rise` | 776 | 1.6689 ± 0.0252 | 2.2024 ± 0.0379 | -0.0595 |
| `GRU` | **A1** | `main_pattern` | `temp_anomaly` | 1730 | 1.9766 ± 0.0439 | 2.5870 ± 0.0734 | +0.2482 |
| `GRU` | **A1** | `main_pattern` | `normal` | 6942 | 1.7284 ± 0.0263 | 2.2846 ± 0.0260 | +0.0000 |
| `GRU` | **A1** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8919 ± 0.0205 | 2.4672 ± 0.0362 | +0.1635 |
| `GRU` | **A1** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.5612 ± 0.2162 | 3.2922 ± 0.2723 | +0.8328 |
| `GRU` | **A3** | `main_pattern` | `pressure_drop` | 929 | 1.6889 ± 0.0339 | 2.2680 ± 0.0313 | -0.0470 |
| `GRU` | **A3** | `main_pattern` | `humidity_net_rise` | 776 | 1.6982 ± 0.0221 | 2.2297 ± 0.0231 | -0.0377 |
| `GRU` | **A3** | `main_pattern` | `temp_anomaly` | 1730 | 1.9713 ± 0.0185 | 2.5700 ± 0.0272 | +0.2354 |
| `GRU` | **A3** | `main_pattern` | `normal` | 6942 | 1.7360 ± 0.0176 | 2.2870 ± 0.0181 | +0.0000 |
| `GRU` | **A3** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.9083 ± 0.0200 | 2.4852 ± 0.0318 | +0.1724 |
| `GRU` | **A3** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4060 ± 0.0622 | 3.0918 ± 0.0707 | +0.6700 |
| `GRU` | **A3r** | `main_pattern` | `pressure_drop` | 929 | 1.6855 ± 0.0283 | 2.2674 ± 0.0306 | -0.0512 |
| `GRU` | **A3r** | `main_pattern` | `humidity_net_rise` | 776 | 1.6986 ± 0.0077 | 2.2342 ± 0.0105 | -0.0381 |
| `GRU` | **A3r** | `main_pattern` | `temp_anomaly` | 1730 | 1.9806 ± 0.0155 | 2.5741 ± 0.0175 | +0.2439 |
| `GRU` | **A3r** | `main_pattern` | `normal` | 6942 | 1.7367 ± 0.0141 | 2.2877 ± 0.0177 | +0.0000 |
| `GRU` | **A3r** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.9195 ± 0.0165 | 2.4907 ± 0.0237 | +0.1829 |
| `GRU` | **A3r** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4016 ± 0.0709 | 3.0871 ± 0.1081 | +0.6650 |

## 3. So sánh ghép cặp theo Seed (Paired Differences vs A0 / A0r / A1)

| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |
|---|---|---|---|---:|---:|---|---|
| `GRU` | **A0r** | **A0** | `mae_degC` | 5 | -0.0158 ± 0.0164 | `s0:-0.0035; s1:-0.0444; s2:-0.0064; s3:-0.0133; s4:-0.0116` | Du 5 seed ghep cap |
| `GRU` | **A0r** | **A0** | `rmse_degC` | 5 | -0.0116 ± 0.0216 | `s0:+0.0068; s1:-0.0474; s2:-0.0143; s3:+0.0026; s4:-0.0057` | Du 5 seed ghep cap |
| `GRU` | **A0r** | **A0** | `best_val_loss` | 5 | -0.0004 ± 0.0010 | `s0:-0.0008; s1:-0.0004; s2:-0.0018; s3:+0.0010; s4:-0.0001` | Du 5 seed ghep cap |
| `GRU` | **A0r** | **A0** | `samples_to_target` | 5 | +18614.4000 ± 25545.5138 | `s0:+9798.0000; s1:+9798.0000; s2:+24492.0000; s3:-9795.0000; s4:+58779.0000` | Du 5 seed ghep cap |
| `GRU` | **A0r** | **A0** | `val_auc_samples` | 5 | -0.0147 ± 0.0061 | `s0:-0.0072; s1:-0.0118; s2:-0.0130; s3:-0.0228; s4:-0.0185` | Du 5 seed ghep cap |
| `GRU` | **A1** | **A0** | `mae_degC` | 5 | -0.0218 ± 0.0131 | `s0:-0.0085; s1:-0.0243; s2:-0.0089; s3:-0.0284; s4:-0.0390` | Du 5 seed ghep cap |
| `GRU` | **A1** | **A0** | `rmse_degC` | 5 | -0.0126 ± 0.0055 | `s0:-0.0045; s1:-0.0170; s2:-0.0160; s3:-0.0162; s4:-0.0092` | Du 5 seed ghep cap |
| `GRU` | **A1** | **A0** | `best_val_loss` | 5 | +0.0002 ± 0.0017 | `s0:-0.0028; s1:+0.0014; s2:-0.0000; s3:+0.0009; s4:+0.0013` | Du 5 seed ghep cap |
| `GRU` | **A1** | **A0** | `samples_to_target` | 5 | +39184.8000 ± 21904.9691 | `s0:+48981.0000; s1:+48981.0000; s2:+0.0000; s3:+48981.0000; s4:+48981.0000` | Du 5 seed ghep cap |
| `GRU` | **A1** | **A0** | `val_auc_samples` | 5 | -0.0080 ± 0.0099 | `s0:+0.0016; s1:+0.0002; s2:-0.0050; s3:-0.0206; s4:-0.0161` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A1** | `mae_degC` | 5 | +0.0063 ± 0.0148 | `s0:-0.0030; s1:-0.0137; s2:+0.0079; s3:+0.0194; s4:+0.0210` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A1** | `rmse_degC` | 5 | -0.0012 ± 0.0219 | `s0:-0.0084; s1:-0.0312; s2:+0.0147; s3:+0.0250; s4:-0.0062` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A1** | `best_val_loss` | 5 | -0.0010 ± 0.0010 | `s0:-0.0012; s1:-0.0020; s2:-0.0003; s3:+0.0003; s4:-0.0018` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A1** | `samples_to_target` | 5 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000; s3:+0.0000; s4:+0.0000` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A1** | `val_auc_samples` | 5 | -0.0008 ± 0.0004 | `s0:-0.0012; s1:-0.0004; s2:-0.0006; s3:-0.0008; s4:-0.0012` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A0** | `mae_degC` | 5 | -0.0155 ± 0.0140 | `s0:-0.0114; s1:-0.0380; s2:-0.0010; s3:-0.0090; s4:-0.0179` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A0** | `rmse_degC` | 5 | -0.0138 ± 0.0216 | `s0:-0.0129; s1:-0.0483; s2:-0.0012; s3:+0.0088; s4:-0.0154` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A0** | `best_val_loss` | 5 | -0.0009 ± 0.0019 | `s0:-0.0040; s1:-0.0006; s2:-0.0004; s3:+0.0012; s4:-0.0005` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A0** | `samples_to_target` | 5 | +39184.8000 ± 21904.9691 | `s0:+48981.0000; s1:+48981.0000; s2:+0.0000; s3:+48981.0000; s4:+48981.0000` | Du 5 seed ghep cap |
| `GRU` | **A3** | **A0** | `val_auc_samples` | 5 | -0.0088 ± 0.0100 | `s0:+0.0003; s1:-0.0002; s2:-0.0056; s3:-0.0214; s4:-0.0173` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0r** | `mae_degC` | 5 | +0.0021 ± 0.0043 | `s0:-0.0001; s1:-0.0032; s2:+0.0062; s3:+0.0008; s4:+0.0067` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0r** | `rmse_degC` | 5 | -0.0008 ± 0.0062 | `s0:-0.0064; s1:-0.0066; s2:+0.0083; s3:-0.0004; s4:+0.0012` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0r** | `best_val_loss` | 5 | -0.0007 ± 0.0009 | `s0:-0.0020; s1:-0.0005; s2:+0.0001; s3:-0.0011; s4:+0.0002` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0r** | `samples_to_target` | 5 | -13714.8000 ± 18779.7633 | `s0:-34287.0000; s1:-34287.0000; s2:+0.0000; s3:+0.0000; s4:+0.0000` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0r** | `val_auc_samples` | 5 | +0.0013 ± 0.0017 | `s0:+0.0002; s1:+0.0024; s2:+0.0038; s3:-0.0001; s4:+0.0002` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0** | `mae_degC` | 5 | -0.0138 ± 0.0194 | `s0:-0.0036; s1:-0.0476; s2:-0.0002; s3:-0.0125; s4:-0.0049` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0** | `rmse_degC` | 5 | -0.0124 ± 0.0235 | `s0:+0.0004; s1:-0.0540; s2:-0.0059; s3:+0.0022; s4:-0.0045` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0** | `best_val_loss` | 5 | -0.0011 ± 0.0012 | `s0:-0.0028; s1:-0.0009; s2:-0.0016; s3:-0.0001; s4:+0.0001` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0** | `samples_to_target` | 5 | +4899.6000 ± 36159.9321 | `s0:-24489.0000; s1:-24489.0000; s2:+24492.0000; s3:-9795.0000; s4:+58779.0000` | Du 5 seed ghep cap |
| `GRU` | **A3r** | **A0** | `val_auc_samples` | 5 | -0.0134 ± 0.0069 | `s0:-0.0070; s1:-0.0094; s2:-0.0092; s3:-0.0229; s4:-0.0183` | Du 5 seed ghep cap |

### 3b. Hiệu số ghép cặp trên 3 Seed gốc (`seeds 0, 1, 2`)

| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |
|---|---|---|---|---:|---:|---|---|
| `GRU` | **A0r** | **A0** | `mae_degC` | 3 | -0.0181 ± 0.0228 | `s0:-0.0035; s1:-0.0444; s2:-0.0064` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `rmse_degC` | 3 | -0.0183 ± 0.0273 | `s0:+0.0068; s1:-0.0474; s2:-0.0143` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `best_val_loss` | 3 | -0.0010 ± 0.0007 | `s0:-0.0008; s1:-0.0004; s2:-0.0018` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `samples_to_target` | 3 | +14696.0000 ± 8483.5849 | `s0:+9798.0000; s1:+9798.0000; s2:+24492.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `val_auc_samples` | 3 | -0.0107 ± 0.0031 | `s0:-0.0072; s1:-0.0118; s2:-0.0130` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `mae_degC` | 3 | -0.0139 ± 0.0090 | `s0:-0.0085; s1:-0.0243; s2:-0.0089` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `rmse_degC` | 3 | -0.0125 ± 0.0070 | `s0:-0.0045; s1:-0.0170; s2:-0.0160` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `best_val_loss` | 3 | -0.0005 ± 0.0021 | `s0:-0.0028; s1:+0.0014; s2:-0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `samples_to_target` | 3 | +32654.0000 ± 28279.1935 | `s0:+48981.0000; s1:+48981.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `val_auc_samples` | 3 | -0.0011 ± 0.0035 | `s0:+0.0016; s1:+0.0002; s2:-0.0050` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `mae_degC` | 3 | -0.0029 ± 0.0108 | `s0:-0.0030; s1:-0.0137; s2:+0.0079` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `rmse_degC` | 3 | -0.0083 ± 0.0230 | `s0:-0.0084; s1:-0.0312; s2:+0.0147` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `best_val_loss` | 3 | -0.0012 ± 0.0008 | `s0:-0.0012; s1:-0.0020; s2:-0.0003` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `val_auc_samples` | 3 | -0.0007 ± 0.0004 | `s0:-0.0012; s1:-0.0004; s2:-0.0006` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `mae_degC` | 3 | -0.0168 ± 0.0191 | `s0:-0.0114; s1:-0.0380; s2:-0.0010` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `rmse_degC` | 3 | -0.0208 ± 0.0245 | `s0:-0.0129; s1:-0.0483; s2:-0.0012` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `best_val_loss` | 3 | -0.0017 ± 0.0020 | `s0:-0.0040; s1:-0.0006; s2:-0.0004` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `samples_to_target` | 3 | +32654.0000 ± 28279.1935 | `s0:+48981.0000; s1:+48981.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `val_auc_samples` | 3 | -0.0018 ± 0.0033 | `s0:+0.0003; s1:-0.0002; s2:-0.0056` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `mae_degC` | 3 | +0.0010 ± 0.0048 | `s0:-0.0001; s1:-0.0032; s2:+0.0062` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `rmse_degC` | 3 | -0.0016 ± 0.0086 | `s0:-0.0064; s1:-0.0066; s2:+0.0083` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `best_val_loss` | 3 | -0.0008 ± 0.0011 | `s0:-0.0020; s1:-0.0005; s2:+0.0001` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `samples_to_target` | 3 | -22858.0000 ± 19795.6087 | `s0:-34287.0000; s1:-34287.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0r** | `val_auc_samples` | 3 | +0.0021 ± 0.0018 | `s0:+0.0002; s1:+0.0024; s2:+0.0038` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `mae_degC` | 3 | -0.0171 ± 0.0264 | `s0:-0.0036; s1:-0.0476; s2:-0.0002` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `rmse_degC` | 3 | -0.0199 ± 0.0297 | `s0:+0.0004; s1:-0.0540; s2:-0.0059` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `best_val_loss` | 3 | -0.0018 ± 0.0010 | `s0:-0.0028; s1:-0.0009; s2:-0.0016` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `samples_to_target` | 3 | -8162.0000 ± 28279.1935 | `s0:-24489.0000; s1:-24489.0000; s2:+24492.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3r** | **A0** | `val_auc_samples` | 3 | -0.0085 ± 0.0013 | `s0:-0.0070; s1:-0.0094; s2:-0.0092` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |

## 4. Hai thí nghiệm khám phá đặt trước trên GRU (`E1: warmup_epochs=1` & `E2: policy=loss_abs`, `seeds 0, 1, 2`)

> **Ghi chú tiền đăng ký (Prompt 8 — Mục 8.4):** Cả hai thí nghiệm **E1** (`policy=predictive`, `warmup_epochs=1`, `alpha=0.5`, `w_max=2.0`) và **E2** (`policy=loss_abs`, `warmup_epochs=3`, `alpha=0.5`, `w_max=2.0`) được báo cáo đầy đủ bất kể kết quả cải thiện hay không, đặt cạnh `A0`, `A0r`, `A1`, `A3` trên cùng `seeds 0, 1, 2`.

### 4a. Bảng tổng hợp toàn cục (`E1`, `E2` đặt cạnh `A0`, `A0r`, `A1`, `A3` — `seeds 0, 1, 2`)

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | Baseline uniform (budget 1.0) | 3 (0,1,2) | 1.7872 ± 0.0335 | 2.3517 ± 0.0411 | 0.0818 ± 0.0003 | 5.7 | 163270 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | 3 (0,1,2) | 1.7692 ± 0.0111 | 2.3334 ± 0.0173 | 0.0808 ± 0.0005 | 6.0 | 171435 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | 3 (0,1,2) | 1.7734 ± 0.0246 | 2.3392 ± 0.0371 | 0.0814 ± 0.0020 | 5.7 | 212251 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | predictive (khong dat nguong G3, budget 1.0) | 3 (0,1,2) | 1.7704 ± 0.0156 | 2.3309 ± 0.0195 | 0.0802 ± 0.0020 | 5.3 | 212251 (3/3) | trend_only (<5 seeds) |
| `GRU` | **E1** | `e1_warmup1_b1.0_a0.5_w2` | [Kham pha E1] predictive (khong dat nguong G3, warmup=1) | 3 (0,1,2) | 1.7873 ± 0.0247 | 2.3577 ± 0.0204 | 0.0811 ± 0.0013 | 6.0 | 212251 (3/3) | trend_only (<5 seeds) |
| `GRU` | **E2** | `loss_abs_b1.0_a0.5_w2` | [Kham pha E2] loss_abs (loss tuyet doi theo Pattern, warmup=3) | 3 (0,1,2) | 1.7756 ± 0.0224 | 2.3417 ± 0.0280 | 0.0816 ± 0.0017 | 4.3 | 212251 (3/3) | trend_only (<5 seeds) |

### 4b. Bảng Test MAE & RMSE (°C) theo từng Pattern (`E1`, `E2` — `seeds 0, 1, 2`)

| Backbone | Mã | Nhóm | Pattern | Số mẫu test | MAE (°C) (Mean ± Std) | RMSE (°C) (Mean ± Std) | Chênh lệch MAE so với `normal` (°C) |
|---|---|---|---|---:|---:|---:|---:|
| `GRU` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.7411 ± 0.0599 | 2.3123 ± 0.0866 | -0.0132 |
| `GRU` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.6973 ± 0.0532 | 2.2208 ± 0.0628 | -0.0569 |
| `GRU` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 1.9848 ± 0.0474 | 2.6021 ± 0.0689 | +0.2305 |
| `GRU` | **A0** | `main_pattern` | `normal` | 6942 | 1.7542 ± 0.0264 | 2.3048 ± 0.0302 | +0.0000 |
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
| `GRU` | **E1** | `main_pattern` | `pressure_drop` | 929 | 1.7177 ± 0.0261 | 2.3046 ± 0.0374 | -0.0374 |
| `GRU` | **E1** | `main_pattern` | `humidity_net_rise` | 776 | 1.7023 ± 0.0360 | 2.2391 ± 0.0456 | -0.0527 |
| `GRU` | **E1** | `main_pattern` | `temp_anomaly` | 1730 | 1.9923 ± 0.0327 | 2.5811 ± 0.0268 | +0.2373 |
| `GRU` | **E1** | `main_pattern` | `normal` | 6942 | 1.7550 ± 0.0311 | 2.3189 ± 0.0218 | +0.0000 |
| `GRU` | **E1** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.9219 ± 0.0459 | 2.4848 ± 0.0412 | +0.1668 |
| `GRU` | **E1** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4786 ± 0.1111 | 3.1649 ± 0.1132 | +0.7236 |
| `GRU` | **E2** | `main_pattern` | `pressure_drop` | 929 | 1.7013 ± 0.0377 | 2.2841 ± 0.0572 | -0.0435 |
| `GRU` | **E2** | `main_pattern` | `humidity_net_rise` | 776 | 1.6601 ± 0.0017 | 2.2004 ± 0.0214 | -0.0846 |
| `GRU` | **E2** | `main_pattern` | `temp_anomaly` | 1730 | 1.9911 ± 0.0223 | 2.5814 ± 0.0262 | +0.2464 |
| `GRU` | **E2** | `main_pattern` | `normal` | 6942 | 1.7447 ± 0.0230 | 2.3015 ± 0.0286 | +0.0000 |
| `GRU` | **E2** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.9181 ± 0.0134 | 2.4821 ± 0.0275 | +0.1734 |
| `GRU` | **E2** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4945 ± 0.1244 | 3.1809 ± 0.1328 | +0.7497 |

### 4c. Hiệu số ghép cặp theo Seed của `E1` và `E2` so với `A0`, `A1`, `A3` (`seeds 0, 1, 2`)

| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |
|---|---|---|---|---:|---:|---|---|
| `GRU` | **E1** | **A3** | `mae_degC` | 3 | +0.0169 ± 0.0091 | `s0:+0.0102; s1:+0.0273; s2:+0.0132` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A3** | `rmse_degC` | 3 | +0.0268 ± 0.0037 | `s0:+0.0284; s1:+0.0294; s2:+0.0225` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A3** | `best_val_loss` | 3 | +0.0010 ± 0.0008 | `s0:+0.0019; s1:+0.0004; s2:+0.0007` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A3** | `samples_to_target` | 3 | +0.0000 ± 129591.5450 | `s0:-97962.0000; s1:-48981.0000; s2:+146943.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A3** | `val_auc_samples` | 3 | -0.0234 ± 0.0023 | `s0:-0.0238; s1:-0.0255; s2:-0.0210` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `mae_degC` | 3 | +0.0139 ± 0.0069 | `s0:+0.0072; s1:+0.0136; s2:+0.0210` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `rmse_degC` | 3 | +0.0185 ± 0.0196 | `s0:+0.0200; s1:-0.0018; s2:+0.0372` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `best_val_loss` | 3 | -0.0002 ± 0.0012 | `s0:+0.0006; s1:-0.0016; s2:+0.0003` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `samples_to_target` | 3 | +0.0000 ± 129591.5450 | `s0:-97962.0000; s1:-48981.0000; s2:+146943.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `val_auc_samples` | 3 | -0.0241 ± 0.0023 | `s0:-0.0250; s1:-0.0259; s2:-0.0216` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `mae_degC` | 3 | +0.0001 ± 0.0115 | `s0:-0.0012; s1:-0.0107; s2:+0.0121` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `rmse_degC` | 3 | +0.0060 ± 0.0217 | `s0:+0.0155; s1:-0.0189; s2:+0.0213` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `best_val_loss` | 3 | -0.0007 ± 0.0013 | `s0:-0.0022; s1:-0.0002; s2:+0.0003` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `samples_to_target` | 3 | +48981.0000 ± 84837.5806 | `s0:+0.0000; s1:+0.0000; s2:+146943.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `val_auc_samples` | 3 | -0.0252 ± 0.0016 | `s0:-0.0234; s1:-0.0256; s2:-0.0266` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `mae_degC` | 3 | +0.0052 ± 0.0070 | `s0:+0.0012; s1:+0.0132; s2:+0.0011` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `rmse_degC` | 3 | +0.0108 ± 0.0124 | `s0:+0.0110; s1:+0.0231; s2:-0.0016` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `best_val_loss` | 3 | +0.0015 ± 0.0005 | `s0:+0.0017; s1:+0.0018; s2:+0.0009` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `val_auc_samples` | 3 | -0.0239 ± 0.0007 | `s0:-0.0242; s1:-0.0244; s2:-0.0231` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `mae_degC` | 3 | +0.0022 ± 0.0059 | `s0:-0.0018; s1:-0.0005; s2:+0.0089` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `rmse_degC` | 3 | +0.0025 ± 0.0106 | `s0:+0.0025; s1:-0.0081; s2:+0.0131` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `best_val_loss` | 3 | +0.0003 ± 0.0004 | `s0:+0.0005; s1:-0.0002; s2:+0.0005` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `val_auc_samples` | 3 | -0.0246 ± 0.0009 | `s0:-0.0254; s1:-0.0248; s2:-0.0237` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `mae_degC` | 3 | -0.0117 ± 0.0125 | `s0:-0.0102; s1:-0.0248; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `rmse_degC` | 3 | -0.0100 ± 0.0131 | `s0:-0.0019; s1:-0.0251; s2:-0.0028` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `best_val_loss` | 3 | -0.0002 ± 0.0018 | `s0:-0.0023; s1:+0.0012; s2:+0.0005` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `samples_to_target` | 3 | +48981.0000 ± 48981.0000 | `s0:+97962.0000; s1:+48981.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `val_auc_samples` | 3 | -0.0257 ± 0.0027 | `s0:-0.0238; s1:-0.0245; s2:-0.0287` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |

## 5. Báo cáo các cấu hình còn thiếu (Missing Configurations)

Tất cả các cấu hình và seed trong danh sách kiểm tra đều đã có đủ trên đĩa.

## Phụ lục: Thời gian chạy (Chỉ tham khảo kỹ thuật — Không dùng để so sánh/kết luận)

> **Lưu ý:** Thời gian chạy mỗi epoch (`Epoch time`), thời gian đạt ngưỡng (`Giây tới val_target`) và `Overhead PALF` phụ thuộc vào tải CPU tại thời điểm chạy cũng như sự khác biệt đường mã giữa `DataLoader(train)` chuẩn và `DataLoader(Subset(train, indices))`. Các số liệu này **không dùng để kết luận hiệu năng thuật toán**.

| Backbone | Mã | Config Tag | Số seed | Giây tới `val_target` | Epoch time (s) | Overhead PALF (s) |
|---|---|---|---:|---:|---:|---:|
| `GRU` | **A0** | `baseline` | 5 (0,1,2,3,4) | 156.5s | 47.37s | 0.0000s |
| `GRU` | **A0r** | `uniform_b0.7` | 5 (0,1,2,3,4) | 280.6s | 53.55s | 0.0043s |
| `GRU` | **A1** | `loss_b1.0_a1.0` | 5 (0,1,2,3,4) | 305.3s | 71.23s | 0.0057s |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | 5 (0,1,2,3,4) | 309.0s | 72.84s | 0.0060s |
| `GRU` | **A3r** | `predictive_b0.7_a0.5_w2` | 5 (0,1,2,3,4) | 270.0s | 55.31s | 0.0045s |
