# Báo cáo So sánh Giao thức Công bằng: Baseline vs PALF (Spec v1 — Mục 7 & Prompt 8)

> **Quy tắc thống kê, đồng nhất & định danh G3 (Spec v1 Mục 7 & Prompt 8):**
> 1. Mọi lần chạy trong bảng bắt buộc có cùng `n_train`, `n_val`, `n_test`, `epochs`, `batch`, `lr` và cùng danh sách `seeds` (các lần chạy thử nhỏ lưu riêng ở `experiments/results/smoke/`).
> 2. Tiêu chí G3 đặt trước cho kết quả **"lui về phương án dự phòng"** (chỉ 2/4 backbone đạt $\Delta R^2 \ge 0.02$ sau kiểm soát epoch và dấu $\beta$ không cùng chiều ở PatchTST). Do đó, cấu hình **A3** và **A3r** được gọi đúng tên là **`predictive (không đạt ngưỡng G3)`**.
> 3. **Ghi chú về thời gian chạy:** Thời gian chạy (`epoch_time_sec`, `time_sec_to_target`, `overhead_sec`) không so sánh được giữa các máy/lần chạy khác nhau và đường mã lấy mẫu khác nhau, không dùng để kết luận và được chuyển xuống **Phụ lục** ở cuối báo cáo.

## 1. Bảng tổng hợp chính theo Cấu hình & Backbone

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | Baseline uniform (DataLoader chuan, budget 1.0) | 10 (0,1,2,3,4,5,6,7,8,9) | 1.7701 ± 0.0240 | 2.3329 ± 0.0265 | 0.0804 ± 0.0014 | 5.6 | 195924 (10/10) | 10_seeds_complete |
| `GRU` | **A0s** | `uniform_b1.0` | Baseline uniform qua Scheduler (budget 1.0) | 10 (0,1,2,3,4,5,6,7,8,9) | 1.7685 ± 0.0190 | 2.3340 ± 0.0186 | 0.0807 ± 0.0010 | 5.4 | 215516 (10/10) | 10_seeds_complete |
| `GRU` | **A0w** | `loss_shuffled_b1.0_a1.0` | Doi chung A0w trong so xao tron (budget 1.0) | 10 (0,1,2,3,4,5,6,7,8,9) | 1.7646 ± 0.0291 | 2.3309 ± 0.0257 | 0.0808 ± 0.0015 | 5.3 | 206809 (9/10) | 10_seeds_complete |
| `GRU` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | 10 (0,1,2,3,4,5,6,7,8,9) | 1.7646 ± 0.0109 | 2.3328 ± 0.0109 | 0.0804 ± 0.0007 | 7.6 | 202293 (10/10) | 10_seeds_complete |
| `GRU` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | 10 (0,1,2,3,4,5,6,7,8,9) | 1.7618 ± 0.0164 | 2.3304 ± 0.0221 | 0.0805 ± 0.0015 | 6.0 | 208169 (8/10) | 10_seeds_complete |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | predictive (khong dat nguong G3, budget 1.0) | 10 (0,1,2,3,4,5,6,7,8,9) | 1.7693 ± 0.0185 | 2.3340 ± 0.0182 | 0.0802 ± 0.0015 | 5.4 | 212251 (9/10) | 10_seeds_complete |

### 1b. Bảng đối chiếu nguyên vẹn trên 3 Seed gốc (`seeds 0, 1, 2` — theo Mục 8.3)

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | Baseline uniform (DataLoader chuan, budget 1.0) | 3 (0,1,2) | 1.7872 ± 0.0335 | 2.3517 ± 0.0411 | 0.0818 ± 0.0003 | 5.7 | 212251 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A0s** | `uniform_b1.0` | Baseline uniform qua Scheduler (budget 1.0) | 3 (0,1,2) | 1.7883 ± 0.0164 | 2.3492 ± 0.0184 | 0.0813 ± 0.0007 | 5.3 | 261232 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A0w** | `loss_shuffled_b1.0_a1.0` | Doi chung A0w trong so xao tron (budget 1.0) | 3 (0,1,2) | 1.7824 ± 0.0469 | 2.3485 ± 0.0425 | 0.0817 ± 0.0015 | 5.0 | 244905 (2/3) | trend_only (<5 seeds) |
| `GRU` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | 3 (0,1,2) | 1.7692 ± 0.0111 | 2.3334 ± 0.0173 | 0.0808 ± 0.0005 | 6.0 | 205722 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | 3 (0,1,2) | 1.7734 ± 0.0246 | 2.3392 ± 0.0371 | 0.0814 ± 0.0020 | 5.7 | 269396 (2/3) | trend_only (<5 seeds) |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | predictive (khong dat nguong G3, budget 1.0) | 3 (0,1,2) | 1.7704 ± 0.0156 | 2.3309 ± 0.0195 | 0.0802 ± 0.0020 | 5.3 | 261232 (3/3) | trend_only (<5 seeds) |

### 1c. Bảng trung bình 4 Seed độc lập (`seeds 1, 2, 3, 4` — tách khỏi `seed 0` đã dùng chọn siêu tham số ở Bước 5.1)

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | Baseline uniform (DataLoader chuan, budget 1.0) | 4 (1,2,3,4) | 1.7872 ± 0.0280 | 2.3496 ± 0.0354 | 0.0807 ± 0.0018 | 5.5 | 183679 (4/4) | trend_only (<5 seeds) |
| `GRU` | **A0s** | `uniform_b1.0` | Baseline uniform qua Scheduler (budget 1.0) | 4 (1,2,3,4) | 1.7807 ± 0.0177 | 2.3423 ± 0.0219 | 0.0810 ± 0.0014 | 5.5 | 244905 (4/4) | trend_only (<5 seeds) |
| `GRU` | **A0w** | `loss_shuffled_b1.0_a1.0` | Doi chung A0w trong so xao tron (budget 1.0) | 4 (1,2,3,4) | 1.7793 ± 0.0387 | 2.3449 ± 0.0350 | 0.0810 ± 0.0024 | 5.2 | 228578 (3/4) | trend_only (<5 seeds) |
| `GRU` | **A0r** | `uniform_b0.7` | Baseline uniform qua Scheduler (budget 0.7) | 4 (1,2,3,4) | 1.7683 ± 0.0135 | 2.3334 ± 0.0171 | 0.0804 ± 0.0009 | 6.5 | 188578 (4/4) | trend_only (<5 seeds) |
| `GRU` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | 4 (1,2,3,4) | 1.7621 ± 0.0275 | 2.3350 ± 0.0352 | 0.0816 ± 0.0018 | 5.8 | 244905 (2/4) | trend_only (<5 seeds) |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | predictive (khong dat nguong G3, budget 1.0) | 4 (1,2,3,4) | 1.7707 ± 0.0128 | 2.3356 ± 0.0106 | 0.0806 ± 0.0010 | 5.2 | 244905 (4/4) | trend_only (<5 seeds) |

## 2. Bảng MAE và RMSE (°C) theo từng Pattern (kể cả tách `temp_anomaly` nóng/lạnh)

| Backbone | Mã | Nhóm | Pattern | Số mẫu test | MAE (°C) (Mean ± Std) | RMSE (°C) (Mean ± Std) | Chênh lệch MAE so với `normal` (°C) |
|---|---|---|---|---:|---:|---:|---:|
| `GRU` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.7017 ± 0.0489 | 2.2772 ± 0.0624 | -0.0398 |
| `GRU` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.6919 ± 0.0390 | 2.2217 ± 0.0487 | -0.0496 |
| `GRU` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 1.9563 ± 0.0336 | 2.5530 ± 0.0540 | +0.2147 |
| `GRU` | **A0** | `main_pattern` | `normal` | 6942 | 1.7415 ± 0.0233 | 2.2944 ± 0.0229 | +0.0000 |
| `GRU` | **A0** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8809 ± 0.0099 | 2.4464 ± 0.0123 | +0.1536 |
| `GRU` | **A0** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4287 ± 0.0983 | 3.1374 ± 0.1005 | +0.7014 |
| `GRU` | **A0s** | `main_pattern` | `pressure_drop` | 929 | 1.6840 ± 0.0363 | 2.2687 ± 0.0530 | -0.0595 |
| `GRU` | **A0s** | `main_pattern` | `humidity_net_rise` | 776 | 1.6763 ± 0.0472 | 2.2022 ± 0.0558 | -0.0672 |
| `GRU` | **A0s** | `main_pattern` | `temp_anomaly` | 1730 | 1.9553 ± 0.0369 | 2.5530 ± 0.0510 | +0.2118 |
| `GRU` | **A0s** | `main_pattern` | `normal` | 6942 | 1.7436 ± 0.0233 | 2.2993 ± 0.0212 | +0.0000 |
| `GRU` | **A0s** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8836 ± 0.0333 | 2.4522 ± 0.0423 | +0.1401 |
| `GRU` | **A0s** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4502 ± 0.1429 | 3.1581 ± 0.1899 | +0.7067 |
| `GRU` | **A0w** | `main_pattern` | `pressure_drop` | 929 | 1.6777 ± 0.0376 | 2.2528 ± 0.0469 | -0.0584 |
| `GRU` | **A0w** | `main_pattern` | `humidity_net_rise` | 776 | 1.6613 ± 0.0454 | 2.1893 ± 0.0473 | -0.0747 |
| `GRU` | **A0w** | `main_pattern` | `temp_anomaly` | 1730 | 1.9719 ± 0.0324 | 2.5679 ± 0.0509 | +0.2358 |
| `GRU` | **A0w** | `main_pattern` | `normal` | 6942 | 1.7361 ± 0.0356 | 2.2940 ± 0.0325 | +0.0000 |
| `GRU` | **A0w** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.9058 ± 0.0287 | 2.4759 ± 0.0453 | +0.1697 |
| `GRU` | **A0w** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4280 ± 0.1524 | 3.1243 ± 0.2111 | +0.6920 |
| `GRU` | **A0r** | `main_pattern` | `pressure_drop` | 929 | 1.6990 ± 0.0364 | 2.2875 ± 0.0444 | -0.0353 |
| `GRU` | **A0r** | `main_pattern` | `humidity_net_rise` | 776 | 1.6679 ± 0.0248 | 2.2000 ± 0.0241 | -0.0665 |
| `GRU` | **A0r** | `main_pattern` | `temp_anomaly` | 1730 | 1.9642 ± 0.0179 | 2.5740 ± 0.0320 | +0.2298 |
| `GRU` | **A0r** | `main_pattern` | `normal` | 6942 | 1.7344 ± 0.0132 | 2.2894 ± 0.0148 | +0.0000 |
| `GRU` | **A0r** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8840 ± 0.0203 | 2.4591 ± 0.0354 | +0.1496 |
| `GRU` | **A0r** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.5175 ± 0.1561 | 3.2530 ± 0.1949 | +0.7831 |
| `GRU` | **A1** | `main_pattern` | `pressure_drop` | 929 | 1.7032 ± 0.0380 | 2.3057 ± 0.0615 | -0.0269 |
| `GRU` | **A1** | `main_pattern` | `humidity_net_rise` | 776 | 1.6621 ± 0.0212 | 2.1857 ± 0.0320 | -0.0680 |
| `GRU` | **A1** | `main_pattern` | `temp_anomaly` | 1730 | 1.9653 ± 0.0414 | 2.5731 ± 0.0612 | +0.2352 |
| `GRU` | **A1** | `main_pattern` | `normal` | 6942 | 1.7301 ± 0.0181 | 2.2850 ± 0.0183 | +0.0000 |
| `GRU` | **A1** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8871 ± 0.0370 | 2.4643 ± 0.0509 | +0.1569 |
| `GRU` | **A1** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.5054 ± 0.1691 | 3.2208 ± 0.2117 | +0.7753 |
| `GRU` | **A3** | `main_pattern` | `pressure_drop` | 929 | 1.6923 ± 0.0277 | 2.2741 ± 0.0330 | -0.0467 |
| `GRU` | **A3** | `main_pattern` | `humidity_net_rise` | 776 | 1.6824 ± 0.0341 | 2.2120 ± 0.0373 | -0.0566 |
| `GRU` | **A3** | `main_pattern` | `temp_anomaly` | 1730 | 1.9715 ± 0.0314 | 2.5715 ± 0.0455 | +0.2325 |
| `GRU` | **A3** | `main_pattern` | `normal` | 6942 | 1.7390 ± 0.0253 | 2.2924 ± 0.0234 | +0.0000 |
| `GRU` | **A3** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.9071 ± 0.0334 | 2.4838 ± 0.0506 | +0.1681 |
| `GRU` | **A3** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4157 ± 0.0872 | 3.1079 ± 0.1117 | +0.6768 |

## 3. So sánh ghép cặp theo Seed (Paired Differences vs A0s / A0 / A0r / A1 — A0s là đối chứng chính của A1, A3 theo Mục 9.1)

> **Cách đọc đăng ký trước đối với `A0s` (Prompt 9 — Mục 9.1):**
> - **Kết quả thực nghiệm trên 5 seed GRU:** `A0s - A0 = -0.0007 ± 0.0164 °C` (`|hiệu| = 0.0007 °C` nhỏ hơn nhiều so với độ lệch chuẩn giữa các seed của `A0` là `0.0255 °C` và của `A0s` là `0.0161 °C` $\Rightarrow$ **`A0s ≈ A0`**).
> - **Theo quy tắc đăng ký trước (Nhánh 1 — `A0s ≈ A0`):** Khi giữ nguyên lấy mẫu đều 100% không hoàn lại mỗi epoch (`uniform, budget=1.0`), đường mã `AdaptiveScheduler` (`A0s: 1.7829 ± 0.0161 °C`) cho kết quả trùng khít với `DataLoader` chuẩn (`A0: 1.7836 ± 0.0255 °C`). Do đó, phần cải thiện của `A1` (`-0.0211 ± 0.0142 °C` so với `A0s`), `A3` (`-0.0148 ± 0.0118 °C` so với `A0s`) và `A0r` (`-0.0151 ± 0.0079 °C` so với `A0s`) **không phải do bản thân đường mã `AdaptiveScheduler` khi lấy đủ 100% mẫu**, mà xuất hiện khi thay đổi cơ chế lấy mẫu sau warmup (giảm mẫu ngẫu nhiên 30% ở `A0r` hoặc lấy mẫu có hoàn lại theo trọng số ở `A1/A3`).
> - **Theo quy tắc đăng ký trước (Nhánh 3 & tổng hợp):** Độ giảm của `A1` (`-0.0211 °C`) và `A3` (`-0.0148 °C`) so với `A0s` nhỉnh hơn/xấp xỉ 1 độ lệch chuẩn liên-seed (`0.0161 °C`), nên được ghi nhận là **xu hướng kèm cảnh báo $n=5$ nhỏ và không hiệu chỉnh đa so sánh**. Tuy nhiên, vì `A3` (`1.7682 °C`) và `A1` (`1.7618 °C`) **vẫn chưa phân biệt được với đối chứng giảm mẫu đồng đều không dùng Pattern (`A0r: 1.7678 ± 0.0118 °C`)** và `A3 - A1 = +0.0063 ± 0.0148 °C`, `A3r - A0r = +0.0021 ± 0.0043 °C`, nên **không có bằng chứng cho thấy trọng số theo Pattern hay công thức `predictive` mang lại lợi ích vượt trên việc giảm mẫu / lấy mẫu có hoàn lại**.

| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |
|---|---|---|---|---:|---:|---|---|
| `GRU` | **A0s** | **A0** | `mae_degC` | 10 | -0.0015 ± 0.0143 | `s0:+0.0224; s1:-0.0233; s2:+0.0041; s3:-0.0041; s4:-0.0027; s5:+0.0026; s6:+0.0103; s7:-0.0014; s8:+0.0023; s9:-0.0257` | Du 10 seed ghep cap |
| `GRU` | **A0s** | **A0** | `rmse_degC` | 10 | +0.0011 ± 0.0171 | `s0:+0.0212; s1:-0.0300; s2:+0.0012; s3:+0.0001; s4:-0.0004; s5:+0.0246; s6:+0.0117; s7:-0.0071; s8:+0.0097; s9:-0.0201` | Du 10 seed ghep cap |
| `GRU` | **A0s** | **A0** | `best_val_loss` | 10 | +0.0003 ± 0.0009 | `s0:-0.0014; s1:-0.0000; s2:-0.0002; s3:+0.0008; s4:+0.0006; s5:+0.0010; s6:+0.0019; s7:-0.0007; s8:+0.0007; s9:+0.0000` | Du 10 seed ghep cap |
| `GRU` | **A0s** | **A0** | `samples_to_target` | 10 | +19592.4000 ± 57493.2998 | `s0:+48981.0000; s1:+48981.0000; s2:+48981.0000; s3:+48981.0000; s4:+97962.0000; s5:-48981.0000; s6:-97962.0000; s7:+0.0000; s8:+0.0000; s9:+48981.0000` | Du 10 seed ghep cap |
| `GRU` | **A0s** | **A0** | `val_auc_samples` | 10 | -0.0115 ± 0.0126 | `s0:-0.0242; s1:-0.0253; s2:-0.0269; s3:-0.0222; s4:-0.0176; s5:-0.0002; s6:-0.0004; s7:-0.0006; s8:-0.0005; s9:+0.0028` | Du 10 seed ghep cap |
| `GRU` | **A0w** | **A0s** | `mae_degC` | 10 | -0.0040 ± 0.0243 | `s0:-0.0413; s1:+0.0336; s2:-0.0101; s3:+0.0063; s4:-0.0355; s5:+0.0003; s6:-0.0116; s7:-0.0146; s8:+0.0298; s9:+0.0037` | Du 10 seed ghep cap |
| `GRU` | **A0w** | **A0s** | `rmse_degC` | 10 | -0.0030 ± 0.0182 | `s0:-0.0249; s1:+0.0284; s2:-0.0056; s3:+0.0077; s4:-0.0200; s5:-0.0037; s6:-0.0295; s7:+0.0053; s8:+0.0149; s9:-0.0027` | Du 10 seed ghep cap |
| `GRU` | **A0w** | **A0s** | `best_val_loss` | 10 | +0.0001 ± 0.0009 | `s0:-0.0002; s1:+0.0018; s2:-0.0004; s3:-0.0012; s4:+0.0000; s5:+0.0007; s6:-0.0008; s7:+0.0006; s8:+0.0001; s9:+0.0009` | Du 10 seed ghep cap |
| `GRU` | **A0w** | **A0s** | `samples_to_target` | 9 | -5442.3333 ± 16327.0000 | `s0:+0.0000; s2:-48981.0000; s3:+0.0000; s4:+0.0000; s5:+0.0000; s6:+0.0000; s7:+0.0000; s8:+0.0000; s9:+0.0000` | Du 9 seed ghep cap |
| `GRU` | **A0w** | **A0s** | `val_auc_samples` | 10 | +0.0008 ± 0.0012 | `s0:+0.0026; s1:+0.0006; s2:-0.0011; s3:+0.0006; s4:+0.0000; s5:+0.0005; s6:-0.0004; s7:+0.0027; s8:+0.0015; s9:+0.0012` | Du 10 seed ghep cap |
| `GRU` | **A0w** | **A0** | `mae_degC` | 10 | -0.0055 ± 0.0196 | `s0:-0.0189; s1:+0.0103; s2:-0.0060; s3:+0.0022; s4:-0.0383; s5:+0.0029; s6:-0.0013; s7:-0.0160; s8:+0.0321; s9:-0.0220` | Du 10 seed ghep cap |
| `GRU` | **A0w** | **A0** | `rmse_degC` | 10 | -0.0020 ± 0.0162 | `s0:-0.0037; s1:-0.0016; s2:-0.0044; s3:+0.0078; s4:-0.0205; s5:+0.0209; s6:-0.0178; s7:-0.0019; s8:+0.0246; s9:-0.0228` | Du 10 seed ghep cap |
| `GRU` | **A0w** | **A0** | `best_val_loss` | 10 | +0.0004 ± 0.0011 | `s0:-0.0017; s1:+0.0018; s2:-0.0006; s3:-0.0003; s4:+0.0006; s5:+0.0017; s6:+0.0011; s7:-0.0001; s8:+0.0008; s9:+0.0009` | Du 10 seed ghep cap |
| `GRU` | **A0w** | **A0** | `samples_to_target` | 9 | +10884.6667 ± 58867.8357 | `s0:+48981.0000; s2:+0.0000; s3:+48981.0000; s4:+97962.0000; s5:-48981.0000; s6:-97962.0000; s7:+0.0000; s8:+0.0000; s9:+48981.0000` | Du 9 seed ghep cap |
| `GRU` | **A0w** | **A0** | `val_auc_samples` | 10 | -0.0107 ± 0.0130 | `s0:-0.0215; s1:-0.0247; s2:-0.0279; s3:-0.0216; s4:-0.0176; s5:+0.0003; s6:-0.0007; s7:+0.0021; s8:+0.0010; s9:+0.0039` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0s** | `mae_degC` | 10 | -0.0040 ± 0.0145 | `s0:-0.0259; s1:-0.0211; s2:-0.0104; s3:-0.0092; s4:-0.0089; s5:-0.0102; s6:+0.0118; s7:+0.0158; s8:+0.0089; s9:+0.0099` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0s** | `rmse_degC` | 10 | -0.0012 ± 0.0162 | `s0:-0.0144; s1:-0.0174; s2:-0.0154; s3:+0.0025; s4:-0.0052; s5:-0.0112; s6:+0.0006; s7:+0.0372; s8:+0.0031; s9:+0.0079` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0s** | `best_val_loss` | 10 | -0.0003 ± 0.0009 | `s0:+0.0007; s1:-0.0004; s2:-0.0016; s3:+0.0002; s4:-0.0006; s5:-0.0008; s6:-0.0002; s7:+0.0010; s8:+0.0002; s9:-0.0015` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0s** | `samples_to_target` | 10 | -13223.1000 ± 54691.9268 | `s0:-39183.0000; s1:-39183.0000; s2:-88164.0000; s3:-58776.0000; s4:-39183.0000; s5:+9798.0000; s6:+93066.0000; s7:+44085.0000; s8:+24492.0000; s9:-39183.0000` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0s** | `val_auc_samples` | 10 | +0.0043 ± 0.0074 | `s0:+0.0170; s1:+0.0135; s2:+0.0139; s3:-0.0006; s4:-0.0008; s5:+0.0008; s6:+0.0003; s7:+0.0009; s8:+0.0006; s9:-0.0030` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0** | `mae_degC` | 10 | -0.0055 ± 0.0187 | `s0:-0.0035; s1:-0.0444; s2:-0.0064; s3:-0.0133; s4:-0.0116; s5:-0.0076; s6:+0.0221; s7:+0.0144; s8:+0.0112; s9:-0.0158` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0** | `rmse_degC` | 10 | -0.0001 ± 0.0213 | `s0:+0.0068; s1:-0.0474; s2:-0.0143; s3:+0.0026; s4:-0.0057; s5:+0.0134; s6:+0.0123; s7:+0.0301; s8:+0.0128; s9:-0.0122` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0** | `best_val_loss` | 10 | -0.0000 ± 0.0011 | `s0:-0.0008; s1:-0.0004; s2:-0.0018; s3:+0.0010; s4:-0.0001; s5:+0.0002; s6:+0.0017; s7:+0.0003; s8:+0.0009; s9:-0.0015` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0** | `samples_to_target` | 10 | +6369.3000 ± 31747.5573 | `s0:+9798.0000; s1:+9798.0000; s2:-39183.0000; s3:-9795.0000; s4:+58779.0000; s5:-39183.0000; s6:-4896.0000; s7:+44085.0000; s8:+24492.0000; s9:+9798.0000` | Du 10 seed ghep cap |
| `GRU` | **A0r** | **A0** | `val_auc_samples` | 10 | -0.0073 ± 0.0088 | `s0:-0.0072; s1:-0.0118; s2:-0.0130; s3:-0.0228; s4:-0.0185; s5:+0.0007; s6:-0.0000; s7:+0.0003; s8:+0.0001; s9:-0.0002` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0s** | `mae_degC` | 10 | -0.0067 ± 0.0191 | `s0:-0.0309; s1:-0.0010; s2:-0.0130; s3:-0.0243; s4:-0.0362; s5:+0.0142; s6:-0.0071; s7:+0.0179; s8:+0.0042; s9:+0.0096` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0s** | `rmse_degC` | 10 | -0.0036 ± 0.0188 | `s0:-0.0256; s1:+0.0130; s2:-0.0171; s3:-0.0163; s4:-0.0087; s5:+0.0066; s6:-0.0279; s7:+0.0302; s8:-0.0008; s9:+0.0109` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0s** | `best_val_loss` | 10 | -0.0001 ± 0.0010 | `s0:-0.0013; s1:+0.0014; s2:+0.0002; s3:+0.0001; s4:+0.0007; s5:-0.0014; s6:-0.0012; s7:+0.0004; s8:-0.0008; s9:+0.0005` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0s** | `samples_to_target` | 8 | +0.0000 ± 0.0000 | `s0:+0.0000; s2:+0.0000; s3:+0.0000; s5:+0.0000; s6:+0.0000; s7:+0.0000; s8:+0.0000; s9:+0.0000` | Du 8 seed ghep cap |
| `GRU` | **A1** | **A0s** | `val_auc_samples` | 10 | +0.0079 ± 0.0115 | `s0:+0.0257; s1:+0.0255; s2:+0.0218; s3:+0.0016; s4:+0.0015; s5:-0.0004; s6:+0.0003; s7:+0.0003; s8:+0.0006; s9:+0.0015` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0** | `mae_degC` | 10 | -0.0082 ± 0.0190 | `s0:-0.0085; s1:-0.0243; s2:-0.0089; s3:-0.0284; s4:-0.0390; s5:+0.0168; s6:+0.0032; s7:+0.0165; s8:+0.0065; s9:-0.0161` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0** | `rmse_degC` | 10 | -0.0025 ± 0.0176 | `s0:-0.0045; s1:-0.0170; s2:-0.0160; s3:-0.0162; s4:-0.0092; s5:+0.0312; s6:-0.0163; s7:+0.0230; s8:+0.0089; s9:-0.0092` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0** | `best_val_loss` | 10 | +0.0001 ± 0.0012 | `s0:-0.0028; s1:+0.0014; s2:-0.0000; s3:+0.0009; s4:+0.0013; s5:-0.0003; s6:+0.0007; s7:-0.0003; s8:-0.0001; s9:+0.0005` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0** | `samples_to_target` | 8 | +6122.6250 ± 55152.1959 | `s0:+48981.0000; s2:+48981.0000; s3:+48981.0000; s5:-48981.0000; s6:-97962.0000; s7:+0.0000; s8:+0.0000; s9:+48981.0000` | Du 8 seed ghep cap |
| `GRU` | **A1** | **A0** | `val_auc_samples` | 10 | -0.0037 ± 0.0082 | `s0:+0.0016; s1:+0.0002; s2:-0.0050; s3:-0.0206; s4:-0.0161; s5:-0.0006; s6:-0.0001; s7:-0.0003; s8:+0.0001; s9:+0.0042` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0w** | `mae_degC` | 10 | -0.0027 ± 0.0214 | `s0:+0.0104; s1:-0.0346; s2:-0.0029; s3:-0.0306; s4:-0.0007; s5:+0.0139; s6:+0.0045; s7:+0.0325; s8:-0.0256; s9:+0.0059` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0w** | `rmse_degC` | 10 | -0.0006 ± 0.0158 | `s0:-0.0007; s1:-0.0154; s2:-0.0116; s3:-0.0240; s4:+0.0113; s5:+0.0104; s6:+0.0016; s7:+0.0249; s8:-0.0157; s9:+0.0136` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0w** | `best_val_loss` | 10 | -0.0003 ± 0.0010 | `s0:-0.0011; s1:-0.0004; s2:+0.0005; s3:+0.0012; s4:+0.0007; s5:-0.0021; s6:-0.0004; s7:-0.0002; s8:-0.0009; s9:-0.0004` | Du 10 seed ghep cap |
| `GRU` | **A1** | **A0w** | `samples_to_target` | 8 | +6122.6250 ± 17317.3986 | `s0:+0.0000; s2:+48981.0000; s3:+0.0000; s5:+0.0000; s6:+0.0000; s7:+0.0000; s8:+0.0000; s9:+0.0000` | Du 8 seed ghep cap |
| `GRU` | **A1** | **A0w** | `val_auc_samples` | 10 | +0.0070 ± 0.0115 | `s0:+0.0231; s1:+0.0250; s2:+0.0229; s3:+0.0009; s4:+0.0015; s5:-0.0009; s6:+0.0007; s7:-0.0024; s8:-0.0008; s9:+0.0003` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A0s** | `mae_degC` | 10 | +0.0008 ± 0.0248 | `s0:-0.0339; s1:-0.0147; s2:-0.0051; s3:-0.0049; s4:-0.0152; s5:+0.0040; s6:+0.0026; s7:+0.0152; s8:-0.0003; s9:+0.0602` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A0s** | `rmse_degC` | 10 | +0.0000 ± 0.0263 | `s0:-0.0340; s1:-0.0182; s2:-0.0024; s3:+0.0087; s4:-0.0150; s5:-0.0005; s6:-0.0227; s7:+0.0279; s8:+0.0006; s9:+0.0558` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A0s** | `best_val_loss` | 10 | -0.0005 ± 0.0013 | `s0:-0.0026; s1:-0.0006; s2:-0.0002; s3:+0.0004; s4:-0.0011; s5:-0.0008; s6:-0.0010; s7:+0.0002; s8:-0.0015; s9:+0.0023` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A0s** | `samples_to_target` | 9 | +0.0000 ± 24490.5000 | `s0:+0.0000; s1:+48981.0000; s2:-48981.0000; s3:+0.0000; s4:+0.0000; s5:+0.0000; s6:+0.0000; s7:+0.0000; s8:+0.0000` | Du 9 seed ghep cap |
| `GRU` | **A3** | **A0s** | `val_auc_samples` | 10 | +0.0071 ± 0.0115 | `s0:+0.0245; s1:+0.0252; s2:+0.0213; s3:+0.0008; s4:+0.0004; s5:+0.0000; s6:-0.0007; s7:+0.0010; s8:+0.0005; s9:-0.0018` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A0** | `mae_degC` | 10 | -0.0007 ± 0.0199 | `s0:-0.0114; s1:-0.0380; s2:-0.0010; s3:-0.0090; s4:-0.0179; s5:+0.0066; s6:+0.0130; s7:+0.0138; s8:+0.0020; s9:+0.0346` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A0** | `rmse_degC` | 10 | +0.0011 ± 0.0243 | `s0:-0.0129; s1:-0.0483; s2:-0.0012; s3:+0.0088; s4:-0.0154; s5:+0.0242; s6:-0.0110; s7:+0.0208; s8:+0.0103; s9:+0.0357` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A0** | `best_val_loss` | 10 | -0.0002 ± 0.0017 | `s0:-0.0040; s1:-0.0006; s2:-0.0004; s3:+0.0012; s4:-0.0005; s5:+0.0002; s6:+0.0009; s7:-0.0005; s8:-0.0008; s9:+0.0023` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A0** | `samples_to_target` | 9 | +16327.0000 ± 64795.7725 | `s0:+48981.0000; s1:+97962.0000; s2:+0.0000; s3:+48981.0000; s4:+97962.0000; s5:-48981.0000; s6:-97962.0000; s7:+0.0000; s8:+0.0000` | Du 9 seed ghep cap |
| `GRU` | **A3** | **A0** | `val_auc_samples` | 10 | -0.0044 ± 0.0082 | `s0:+0.0003; s1:-0.0002; s2:-0.0056; s3:-0.0214; s4:-0.0173; s5:-0.0001; s6:-0.0011; s7:+0.0005; s8:-0.0000; s9:+0.0010` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A1** | `mae_degC` | 10 | +0.0075 ± 0.0192 | `s0:-0.0030; s1:-0.0137; s2:+0.0079; s3:+0.0194; s4:+0.0210; s5:-0.0102; s6:+0.0098; s7:-0.0027; s8:-0.0045; s9:+0.0506` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A1** | `rmse_degC` | 10 | +0.0036 ± 0.0208 | `s0:-0.0084; s1:-0.0312; s2:+0.0147; s3:+0.0250; s4:-0.0062; s5:-0.0071; s6:+0.0053; s7:-0.0023; s8:+0.0014; s9:+0.0449` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A1** | `best_val_loss` | 10 | -0.0003 ± 0.0011 | `s0:-0.0012; s1:-0.0020; s2:-0.0003; s3:+0.0003; s4:-0.0018; s5:+0.0005; s6:+0.0002; s7:-0.0002; s8:-0.0007; s9:+0.0018` | Du 10 seed ghep cap |
| `GRU` | **A3** | **A1** | `samples_to_target` | 7 | -6997.2857 ± 18513.0779 | `s0:+0.0000; s2:-48981.0000; s3:+0.0000; s5:+0.0000; s6:+0.0000; s7:+0.0000; s8:+0.0000` | Du 7 seed ghep cap |
| `GRU` | **A3** | **A1** | `val_auc_samples` | 10 | -0.0007 ± 0.0011 | `s0:-0.0012; s1:-0.0004; s2:-0.0006; s3:-0.0008; s4:-0.0012; s5:+0.0004; s6:-0.0010; s7:+0.0007; s8:-0.0002; s9:-0.0033` | Du 10 seed ghep cap |

### 3c. Ba đối chiếu chính Prompt 10 (Hiệu chỉnh Bonferroni: ngưỡng $\alpha_{\text{Bonf}} = 0.05 / 3 = 0.0167$, kiểm định t ghép cặp hai phía & Wilcoxon)

| Backbone | Mã | Đối chiếu | Số seed | Hiệu MAE (Mean ± Std) (°C) | Số seed âm/dương | t-stat | p-value (t) | p-Bonferroni (×3) | Đạt ngưỡng 0.0167? | Wilcoxon (W, p) | Hiệu số từng seed |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---|---|
| `GRU` | **C1** | **A1 − A0s** | 10 | -0.0067 ± 0.0191 | 6/10 âm (4 dương) | -1.106 | 0.2974 | 0.8922 | Không đạt | W=19.0, p=0.4316 | `s0:-0.0309; s1:-0.0010; s2:-0.0130; s3:-0.0243; s4:-0.0362; s5:+0.0142; s6:-0.0071; s7:+0.0179; s8:+0.0042; s9:+0.0096` |
| `GRU` | **C2** | **A3 − A1** | 10 | +0.0075 ± 0.0192 | 5/10 âm (5 dương) | +1.230 | 0.2500 | 0.7501 | Không đạt | W=19.0, p=0.4316 | `s0:-0.0030; s1:-0.0137; s2:+0.0079; s3:+0.0194; s4:+0.0210; s5:-0.0102; s6:+0.0098; s7:-0.0027; s8:-0.0045; s9:+0.0506` |
| `GRU` | **C3** | **A1 − A0w** | 10 | -0.0027 ± 0.0214 | 5/10 âm (5 dương) | -0.400 | 0.6983 | 1.0000 | Không đạt | W=27.0, p=1.0000 | `s0:+0.0104; s1:-0.0346; s2:-0.0029; s3:-0.0306; s4:-0.0007; s5:+0.0139; s6:+0.0045; s7:+0.0325; s8:-0.0256; s9:+0.0059` |

> **Kết luận theo đúng bảng Cách đọc đăng ký trước (Prompt 10):**
> - **C1 không đạt (`p_t = 0.2974 ≥ 0.0167`):** Không có hiệu ứng ổn định của trọng số theo loss (`A1`) so với lấy đều (`A0s`) ở 10 seed; hiệu ứng ở 5 seed là biến động ngẫu nhiên.
> - **C2 không đạt (`A3 − A1 = +0.0075 °C`, `p_t = 0.2500`):** **Learning State đa biến không hơn loss đơn thuần.** (Kết quả chính của đề tài).

### 3b. Hiệu số ghép cặp trên 3 Seed gốc (`seeds 0, 1, 2`)

| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |
|---|---|---|---|---:|---:|---|---|
| `GRU` | **A0s** | **A0** | `mae_degC` | 3 | +0.0011 ± 0.0230 | `s0:+0.0224; s1:-0.0233; s2:+0.0041` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0s** | **A0** | `rmse_degC` | 3 | -0.0026 ± 0.0258 | `s0:+0.0212; s1:-0.0300; s2:+0.0012` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0s** | **A0** | `best_val_loss` | 3 | -0.0005 ± 0.0008 | `s0:-0.0014; s1:-0.0000; s2:-0.0002` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0s** | **A0** | `samples_to_target` | 3 | +48981.0000 ± 0.0000 | `s0:+48981.0000; s1:+48981.0000; s2:+48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0s** | **A0** | `val_auc_samples` | 3 | -0.0254 ± 0.0014 | `s0:-0.0242; s1:-0.0253; s2:-0.0269` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0s** | `mae_degC` | 3 | -0.0059 ± 0.0376 | `s0:-0.0413; s1:+0.0336; s2:-0.0101` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0s** | `rmse_degC` | 3 | -0.0007 ± 0.0270 | `s0:-0.0249; s1:+0.0284; s2:-0.0056` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0s** | `best_val_loss` | 3 | +0.0004 ± 0.0012 | `s0:-0.0002; s1:+0.0018; s2:-0.0004` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0s** | `samples_to_target` | 2 | -24490.5000 ± 34634.7972 | `s0:+0.0000; s2:-48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0s** | `val_auc_samples` | 3 | +0.0007 ± 0.0019 | `s0:+0.0026; s1:+0.0006; s2:-0.0011` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0** | `mae_degC` | 3 | -0.0048 ± 0.0146 | `s0:-0.0189; s1:+0.0103; s2:-0.0060` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0** | `rmse_degC` | 3 | -0.0033 ± 0.0014 | `s0:-0.0037; s1:-0.0016; s2:-0.0044` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0** | `best_val_loss` | 3 | -0.0002 ± 0.0018 | `s0:-0.0017; s1:+0.0018; s2:-0.0006` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0** | `samples_to_target` | 2 | +24490.5000 ± 34634.7972 | `s0:+48981.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0w** | **A0** | `val_auc_samples` | 3 | -0.0247 ± 0.0032 | `s0:-0.0215; s1:-0.0247; s2:-0.0279` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0s** | `mae_degC` | 3 | -0.0192 ± 0.0079 | `s0:-0.0259; s1:-0.0211; s2:-0.0104` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0s** | `rmse_degC` | 3 | -0.0157 ± 0.0015 | `s0:-0.0144; s1:-0.0174; s2:-0.0154` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0s** | `best_val_loss` | 3 | -0.0004 ± 0.0011 | `s0:+0.0007; s1:-0.0004; s2:-0.0016` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0s** | `samples_to_target` | 3 | -55510.0000 ± 28279.1935 | `s0:-39183.0000; s1:-39183.0000; s2:-88164.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0s** | `val_auc_samples` | 3 | +0.0148 ± 0.0019 | `s0:+0.0170; s1:+0.0135; s2:+0.0139` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `mae_degC` | 3 | -0.0181 ± 0.0228 | `s0:-0.0035; s1:-0.0444; s2:-0.0064` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `rmse_degC` | 3 | -0.0183 ± 0.0273 | `s0:+0.0068; s1:-0.0474; s2:-0.0143` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `best_val_loss` | 3 | -0.0010 ± 0.0007 | `s0:-0.0008; s1:-0.0004; s2:-0.0018` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `samples_to_target` | 3 | -6529.0000 ± 28279.1935 | `s0:+9798.0000; s1:+9798.0000; s2:-39183.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A0r** | **A0** | `val_auc_samples` | 3 | -0.0107 ± 0.0031 | `s0:-0.0072; s1:-0.0118; s2:-0.0130` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0s** | `mae_degC` | 3 | -0.0150 ± 0.0150 | `s0:-0.0309; s1:-0.0010; s2:-0.0130` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0s** | `rmse_degC` | 3 | -0.0099 ± 0.0203 | `s0:-0.0256; s1:+0.0130; s2:-0.0171` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0s** | `best_val_loss` | 3 | +0.0001 ± 0.0014 | `s0:-0.0013; s1:+0.0014; s2:+0.0002` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0s** | `samples_to_target` | 2 | +0.0000 ± 0.0000 | `s0:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0s** | `val_auc_samples` | 3 | +0.0244 ± 0.0022 | `s0:+0.0257; s1:+0.0255; s2:+0.0218` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `mae_degC` | 3 | -0.0139 ± 0.0090 | `s0:-0.0085; s1:-0.0243; s2:-0.0089` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `rmse_degC` | 3 | -0.0125 ± 0.0070 | `s0:-0.0045; s1:-0.0170; s2:-0.0160` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `best_val_loss` | 3 | -0.0005 ± 0.0021 | `s0:-0.0028; s1:+0.0014; s2:-0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `samples_to_target` | 2 | +48981.0000 ± 0.0000 | `s0:+48981.0000; s2:+48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0** | `val_auc_samples` | 3 | -0.0011 ± 0.0035 | `s0:+0.0016; s1:+0.0002; s2:-0.0050` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0w** | `mae_degC` | 3 | -0.0090 ± 0.0231 | `s0:+0.0104; s1:-0.0346; s2:-0.0029` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0w** | `rmse_degC` | 3 | -0.0092 ± 0.0076 | `s0:-0.0007; s1:-0.0154; s2:-0.0116` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0w** | `best_val_loss` | 3 | -0.0003 ± 0.0008 | `s0:-0.0011; s1:-0.0004; s2:+0.0005` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0w** | `samples_to_target` | 2 | +24490.5000 ± 34634.7972 | `s0:+0.0000; s2:+48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A1** | **A0w** | `val_auc_samples` | 3 | +0.0236 ± 0.0011 | `s0:+0.0231; s1:+0.0250; s2:+0.0229` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0s** | `mae_degC` | 3 | -0.0179 ± 0.0146 | `s0:-0.0339; s1:-0.0147; s2:-0.0051` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0s** | `rmse_degC` | 3 | -0.0182 ± 0.0158 | `s0:-0.0340; s1:-0.0182; s2:-0.0024` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0s** | `best_val_loss` | 3 | -0.0011 ± 0.0013 | `s0:-0.0026; s1:-0.0006; s2:-0.0002` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0s** | `samples_to_target` | 3 | +0.0000 ± 48981.0000 | `s0:+0.0000; s1:+48981.0000; s2:-48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0s** | `val_auc_samples` | 3 | +0.0236 ± 0.0021 | `s0:+0.0245; s1:+0.0252; s2:+0.0213` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `mae_degC` | 3 | -0.0168 ± 0.0191 | `s0:-0.0114; s1:-0.0380; s2:-0.0010` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `rmse_degC` | 3 | -0.0208 ± 0.0245 | `s0:-0.0129; s1:-0.0483; s2:-0.0012` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `best_val_loss` | 3 | -0.0017 ± 0.0020 | `s0:-0.0040; s1:-0.0006; s2:-0.0004` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `samples_to_target` | 3 | +48981.0000 ± 48981.0000 | `s0:+48981.0000; s1:+97962.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A0** | `val_auc_samples` | 3 | -0.0018 ± 0.0033 | `s0:+0.0003; s1:-0.0002; s2:-0.0056` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `mae_degC` | 3 | -0.0029 ± 0.0108 | `s0:-0.0030; s1:-0.0137; s2:+0.0079` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `rmse_degC` | 3 | -0.0083 ± 0.0230 | `s0:-0.0084; s1:-0.0312; s2:+0.0147` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `best_val_loss` | 3 | -0.0012 ± 0.0008 | `s0:-0.0012; s1:-0.0020; s2:-0.0003` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `samples_to_target` | 2 | -24490.5000 ± 34634.7972 | `s0:+0.0000; s2:-48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **A3** | **A1** | `val_auc_samples` | 3 | -0.0007 ± 0.0004 | `s0:-0.0012; s1:-0.0004; s2:-0.0006` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |

## 4. Hai thí nghiệm khám phá đặt trước trên GRU (`E1: warmup_epochs=1` & `E2: policy=loss_abs`, `seeds 0, 1, 2`)

> **Ghi chú tiền đăng ký (Prompt 8 — Mục 8.4):** Cả hai thí nghiệm **E1** (`policy=predictive`, `warmup_epochs=1`, `alpha=0.5`, `w_max=2.0`) và **E2** (`policy=loss_abs`, `warmup_epochs=3`, `alpha=0.5`, `w_max=2.0`) được báo cáo đầy đủ bất kể kết quả cải thiện hay không, đặt cạnh `A0`, `A0r`, `A1`, `A3` trên cùng `seeds 0, 1, 2`.

### 4a. Bảng tổng hợp toàn cục (`E1`, `E2` đặt cạnh `A0`, `A0r`, `A1`, `A3` — `seeds 0, 1, 2`)

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `GRU` | **A0** | `baseline` | Baseline uniform (DataLoader chuan, budget 1.0) | 3 (0,1,2) | 1.7872 ± 0.0335 | 2.3517 ± 0.0411 | 0.0818 ± 0.0003 | 5.7 | 163270 (3/3) | trend_only (<5 seeds) |
| `GRU` | **A0s** | `uniform_b1.0` | Baseline uniform qua Scheduler (budget 1.0) | 3 (0,1,2) | 1.7883 ± 0.0164 | 2.3492 ± 0.0184 | 0.0813 ± 0.0007 | 5.3 | 212251 (3/3) | trend_only (<5 seeds) |
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
| `GRU` | **A0s** | `main_pattern` | `pressure_drop` | 929 | 1.6948 ± 0.0536 | 2.2736 ± 0.0815 | -0.0728 |
| `GRU` | **A0s** | `main_pattern` | `humidity_net_rise` | 776 | 1.7204 ± 0.0210 | 2.2485 ± 0.0099 | -0.0473 |
| `GRU` | **A0s** | `main_pattern` | `temp_anomaly` | 1730 | 1.9519 ± 0.0223 | 2.5508 ± 0.0350 | +0.1842 |
| `GRU` | **A0s** | `main_pattern` | `normal` | 6942 | 1.7677 ± 0.0197 | 2.3173 ± 0.0195 | +0.0000 |
| `GRU` | **A0s** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8898 ± 0.0065 | 2.4649 ± 0.0057 | +0.1221 |
| `GRU` | **A0s** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.3803 ± 0.1587 | 3.0757 ± 0.1955 | +0.6126 |
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
| `GRU` | **E1** | **A0s** | `mae_degC` | 3 | -0.0010 ± 0.0197 | `s0:-0.0237; s1:+0.0126; s2:+0.0081` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0s** | `rmse_degC` | 3 | +0.0085 ± 0.0131 | `s0:-0.0056; s1:+0.0111; s2:+0.0201` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0s** | `best_val_loss` | 3 | -0.0002 ± 0.0006 | `s0:-0.0007; s1:-0.0002; s2:+0.0005` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0s** | `samples_to_target` | 3 | +0.0000 ± 129591.5450 | `s0:-97962.0000; s1:-48981.0000; s2:+146943.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0s** | `val_auc_samples` | 3 | +0.0002 ± 0.0005 | `s0:+0.0007; s1:-0.0003; s2:+0.0003` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `mae_degC` | 3 | +0.0001 ± 0.0115 | `s0:-0.0012; s1:-0.0107; s2:+0.0121` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `rmse_degC` | 3 | +0.0060 ± 0.0217 | `s0:+0.0155; s1:-0.0189; s2:+0.0213` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `best_val_loss` | 3 | -0.0007 ± 0.0013 | `s0:-0.0022; s1:-0.0002; s2:+0.0003` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `samples_to_target` | 3 | +48981.0000 ± 84837.5806 | `s0:+0.0000; s1:+0.0000; s2:+146943.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A0** | `val_auc_samples` | 3 | -0.0252 ± 0.0016 | `s0:-0.0234; s1:-0.0256; s2:-0.0266` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `mae_degC` | 3 | +0.0139 ± 0.0069 | `s0:+0.0072; s1:+0.0136; s2:+0.0210` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `rmse_degC` | 3 | +0.0185 ± 0.0196 | `s0:+0.0200; s1:-0.0018; s2:+0.0372` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `best_val_loss` | 3 | -0.0002 ± 0.0012 | `s0:+0.0006; s1:-0.0016; s2:+0.0003` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `samples_to_target` | 3 | +0.0000 ± 129591.5450 | `s0:-97962.0000; s1:-48981.0000; s2:+146943.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E1** | **A1** | `val_auc_samples` | 3 | -0.0241 ± 0.0023 | `s0:-0.0250; s1:-0.0259; s2:-0.0216` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `mae_degC` | 3 | +0.0052 ± 0.0070 | `s0:+0.0012; s1:+0.0132; s2:+0.0011` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `rmse_degC` | 3 | +0.0108 ± 0.0124 | `s0:+0.0110; s1:+0.0231; s2:-0.0016` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `best_val_loss` | 3 | +0.0015 ± 0.0005 | `s0:+0.0017; s1:+0.0018; s2:+0.0009` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A3** | `val_auc_samples` | 3 | -0.0239 ± 0.0007 | `s0:-0.0242; s1:-0.0244; s2:-0.0231` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0s** | `mae_degC` | 3 | -0.0127 ± 0.0173 | `s0:-0.0327; s1:-0.0015; s2:-0.0040` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0s** | `rmse_degC` | 3 | -0.0074 ± 0.0143 | `s0:-0.0231; s1:+0.0049; s2:-0.0040` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0s** | `best_val_loss` | 3 | +0.0004 ± 0.0011 | `s0:-0.0008; s1:+0.0012; s2:+0.0007` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0s** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0s** | `val_auc_samples` | 3 | -0.0003 ± 0.0014 | `s0:+0.0003; s1:+0.0008; s2:-0.0019` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `mae_degC` | 3 | -0.0117 ± 0.0125 | `s0:-0.0102; s1:-0.0248; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `rmse_degC` | 3 | -0.0100 ± 0.0131 | `s0:-0.0019; s1:-0.0251; s2:-0.0028` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `best_val_loss` | 3 | -0.0002 ± 0.0018 | `s0:-0.0023; s1:+0.0012; s2:+0.0005` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `samples_to_target` | 3 | +48981.0000 ± 48981.0000 | `s0:+97962.0000; s1:+48981.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A0** | `val_auc_samples` | 3 | -0.0257 ± 0.0027 | `s0:-0.0238; s1:-0.0245; s2:-0.0287` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `mae_degC` | 3 | +0.0022 ± 0.0059 | `s0:-0.0018; s1:-0.0005; s2:+0.0089` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `rmse_degC` | 3 | +0.0025 ± 0.0106 | `s0:+0.0025; s1:-0.0081; s2:+0.0131` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `best_val_loss` | 3 | +0.0003 ± 0.0004 | `s0:+0.0005; s1:-0.0002; s2:+0.0005` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `GRU` | **E2** | **A1** | `val_auc_samples` | 3 | -0.0246 ± 0.0009 | `s0:-0.0254; s1:-0.0248; s2:-0.0237` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |

## 4b. Nhân bản trên Backbone TCN (`A0, A0s, A1, A3` — `seeds 0, 1, 2`, Mục 9.2)

> **Ghi chú nhân bản trên TCN (Prompt 9 — Mục 9.2):**
> - Chạy với cùng siêu tham số từ GRU (`alpha=0.5, w_max=2.0` cho `A3`, `alpha=1.0, w_max=3.0` cho `A1`, **không tinh chỉnh lại trên TCN**) và hệ số ngoài mẫu `experiments/analysis/phase3_real/predictive_coef_tcn_seeds34.json` (`fit_scope: "tcn"`, `fit_seeds: [3, 4]`) để kiểm tra tính lặp lại trên kiến trúc tích chập nhân quả giãn nở (TCN). Kết luận chỉ mô tả xu hướng (`3 seeds`).
> - **Quan sát mô tả trên TCN:** TCN hội tụ sớm hơn cả GRU (`best_epoch = 3.0` ở `A0`/`A0s` và `3.7` ở `A1`/`A3`). Ở `2/3` seed (`seed 0, 2`), điểm `val_loss` tốt nhất đạt ngay tại `epoch 3` (cuối giai đoạn `warmup_epochs = 3`), các epoch 4–8 sau khi Policy can thiệp không vượt qua mốc `val_loss` của epoch 3 nên checkpoint tốt nhất của `A0s, A1, A3` trùng nhau (`hiệu số = 0.0000 °C` ở `s0, s2`); ở `seed 1` (`best_epoch = 5`), `A1` (`1.7799 °C`) và `A3` (`1.8061 °C`) có MAE cao hơn nhẹ so với `A0s` (`1.7633 °C`). Nhóm `temp_anomaly` tiếp tục có sai số cao nhất ở mọi cấu hình (`1.9502–1.9798 °C`). Như vậy, hiện tượng hội tụ sớm và không tách biệt được hiệu quả của Policy lặp lại nhất quán trên backbone thứ hai (TCN).

### Bảng tổng hợp TCN (`seeds 0, 1, 2`)

| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `TCN` | **A0** | `baseline` | Baseline uniform (DataLoader chuan, budget 1.0) | 3 (0,1,2) | 1.7739 ± 0.0065 | 2.3364 ± 0.0088 | 0.0819 ± 0.0009 | 3.0 | 130616 (3/3) | trend_only (<5 seeds) |
| `TCN` | **A0s** | `uniform_b1.0` | Baseline uniform qua Scheduler (budget 1.0) | 3 (0,1,2) | 1.7659 ± 0.0089 | 2.3335 ± 0.0124 | 0.0821 ± 0.0004 | 3.0 | 97962 (3/3) | trend_only (<5 seeds) |
| `TCN` | **A1** | `loss_b1.0_a1.0` | Doi chung B2 chi dung loss_ratio (budget 1.0) | 3 (0,1,2) | 1.7714 ± 0.0113 | 2.3441 ± 0.0156 | 0.0820 ± 0.0003 | 3.7 | 97962 (3/3) | trend_only (<5 seeds) |
| `TCN` | **A3** | `predictive_b1.0_a0.5_w2` | predictive (khong dat nguong G3, budget 1.0) | 3 (0,1,2) | 1.7802 ± 0.0241 | 2.3516 ± 0.0267 | 0.0821 ± 0.0003 | 3.7 | 97962 (3/3) | trend_only (<5 seeds) |

### Bảng Test MAE & RMSE (°C) theo từng Pattern trên TCN (`seeds 0, 1, 2`)

| Backbone | Mã | Nhóm | Pattern | Số mẫu test | MAE (°C) (Mean ± Std) | RMSE (°C) (Mean ± Std) | Chênh lệch MAE so với `normal` (°C) |
|---|---|---|---|---:|---:|---:|---:|
| `TCN` | **A0** | `main_pattern` | `pressure_drop` | 929 | 1.7511 ± 0.0177 | 2.3792 ± 0.0013 | +0.0041 |
| `TCN` | **A0** | `main_pattern` | `humidity_net_rise` | 776 | 1.6478 ± 0.0201 | 2.1602 ± 0.0054 | -0.0992 |
| `TCN` | **A0** | `main_pattern` | `temp_anomaly` | 1730 | 1.9502 ± 0.0505 | 2.5572 ± 0.0721 | +0.2031 |
| `TCN` | **A0** | `main_pattern` | `normal` | 6942 | 1.7471 ± 0.0130 | 2.2914 ± 0.0088 | +0.0000 |
| `TCN` | **A0s** | `main_pattern` | `pressure_drop` | 929 | 1.7537 ± 0.0619 | 2.3781 ± 0.0825 | +0.0199 |
| `TCN` | **A0s** | `main_pattern` | `humidity_net_rise` | 776 | 1.6294 ± 0.0112 | 2.1416 ± 0.0018 | -0.1044 |
| `TCN` | **A0s** | `main_pattern` | `temp_anomaly` | 1730 | 1.9620 ± 0.0087 | 2.5656 ± 0.0117 | +0.2281 |
| `TCN` | **A0s** | `main_pattern` | `normal` | 6942 | 1.7339 ± 0.0070 | 2.2868 ± 0.0130 | +0.0000 |
| `TCN` | **A0s** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8893 ± 0.0084 | 2.4613 ± 0.0121 | +0.1555 |
| `TCN` | **A0s** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4634 ± 0.0921 | 3.1927 ± 0.1055 | +0.7295 |
| `TCN` | **A1** | `main_pattern` | `pressure_drop` | 929 | 1.7569 ± 0.0623 | 2.3715 ± 0.0807 | +0.0176 |
| `TCN` | **A1** | `main_pattern` | `humidity_net_rise` | 776 | 1.6360 ± 0.0225 | 2.1503 ± 0.0157 | -0.1032 |
| `TCN` | **A1** | `main_pattern` | `temp_anomaly` | 1730 | 1.9689 ± 0.0093 | 2.5812 ± 0.0210 | +0.2296 |
| `TCN` | **A1** | `main_pattern` | `normal` | 6942 | 1.7393 ± 0.0085 | 2.2986 ± 0.0122 | +0.0000 |
| `TCN` | **A1** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8988 ± 0.0117 | 2.4819 ± 0.0297 | +0.1595 |
| `TCN` | **A1** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.4522 ± 0.0943 | 3.1818 ± 0.1088 | +0.7130 |
| `TCN` | **A3** | `main_pattern` | `pressure_drop` | 929 | 1.7459 ± 0.0632 | 2.3570 ± 0.0823 | -0.0047 |
| `TCN` | **A3** | `main_pattern` | `humidity_net_rise` | 776 | 1.6411 ± 0.0312 | 2.1605 ± 0.0334 | -0.1095 |
| `TCN` | **A3** | `main_pattern` | `temp_anomaly` | 1730 | 1.9798 ± 0.0262 | 2.5967 ± 0.0467 | +0.2292 |
| `TCN` | **A3** | `main_pattern` | `normal` | 6942 | 1.7506 ± 0.0262 | 2.3066 ± 0.0247 | +0.0000 |
| `TCN` | **A3** | `temp_sign_split` | `temp_anomaly_heat` | 1511 | 1.8972 ± 0.0093 | 2.4755 ± 0.0196 | +0.1466 |
| `TCN` | **A3** | `temp_sign_split` | `temp_anomaly_cold` | 219 | 2.5500 ± 0.1752 | 3.3106 ± 0.2214 | +0.7994 |

### Hiệu số ghép cặp theo Seed trên TCN (`A0s - A0`, `A1 - A0s`, `A3 - A0s`, `A3 - A1`)

| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |
|---|---|---|---|---:|---:|---|---|
| `TCN` | **A0s** | **A0** | `mae_degC` | 3 | -0.0080 ± 0.0096 | `s0:-0.0018; s1:-0.0031; s2:-0.0190` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A0s** | **A0** | `rmse_degC` | 3 | -0.0029 ± 0.0155 | `s0:+0.0123; s1:-0.0025; s2:-0.0186` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A0s** | **A0** | `best_val_loss` | 3 | +0.0003 ± 0.0013 | `s0:+0.0009; s1:+0.0011; s2:-0.0012` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A0s** | **A0** | `samples_to_target` | 3 | -32654.0000 ± 28279.1935 | `s0:+0.0000; s1:-48981.0000; s2:-48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A0s** | **A0** | `val_auc_samples` | 3 | -0.0145 ± 0.0019 | `s0:-0.0145; s1:-0.0164; s2:-0.0127` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0s** | `mae_degC` | 3 | +0.0055 ± 0.0096 | `s0:+0.0000; s1:+0.0166; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0s** | `rmse_degC` | 3 | +0.0106 ± 0.0184 | `s0:+0.0000; s1:+0.0318; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0s** | `best_val_loss` | 3 | -0.0001 ± 0.0002 | `s0:+0.0000; s1:-0.0004; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0s** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0s** | `val_auc_samples` | 3 | +0.0008 ± 0.0018 | `s0:-0.0008; s1:+0.0028; s2:+0.0005` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0** | `mae_degC` | 3 | -0.0024 ± 0.0162 | `s0:-0.0018; s1:+0.0135; s2:-0.0190` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0** | `rmse_degC` | 3 | +0.0077 ± 0.0243 | `s0:+0.0123; s1:+0.0294; s2:-0.0186` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0** | `best_val_loss` | 3 | +0.0001 ± 0.0012 | `s0:+0.0009; s1:+0.0007; s2:-0.0012` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0** | `samples_to_target` | 3 | -32654.0000 ± 28279.1935 | `s0:+0.0000; s1:-48981.0000; s2:-48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A1** | **A0** | `val_auc_samples` | 3 | -0.0137 ± 0.0015 | `s0:-0.0152; s1:-0.0136; s2:-0.0122` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0s** | `mae_degC` | 3 | +0.0143 ± 0.0248 | `s0:+0.0000; s1:+0.0429; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0s** | `rmse_degC` | 3 | +0.0181 ± 0.0314 | `s0:+0.0000; s1:+0.0543; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0s** | `best_val_loss` | 3 | -0.0001 ± 0.0001 | `s0:+0.0000; s1:-0.0002; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0s** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0s** | `val_auc_samples` | 3 | +0.0010 ± 0.0010 | `s0:+0.0001; s1:+0.0021; s2:+0.0007` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0** | `mae_degC` | 3 | +0.0063 ± 0.0302 | `s0:-0.0018; s1:+0.0398; s2:-0.0190` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0** | `rmse_degC` | 3 | +0.0152 ± 0.0353 | `s0:+0.0123; s1:+0.0519; s2:-0.0186` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0** | `best_val_loss` | 3 | +0.0002 ± 0.0012 | `s0:+0.0009; s1:+0.0009; s2:-0.0012` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0** | `samples_to_target` | 3 | -32654.0000 ± 28279.1935 | `s0:+0.0000; s1:-48981.0000; s2:-48981.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A0** | `val_auc_samples` | 3 | -0.0136 ± 0.0014 | `s0:-0.0144; s1:-0.0143; s2:-0.0120` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A1** | `mae_degC` | 3 | +0.0088 ± 0.0152 | `s0:+0.0000; s1:+0.0263; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A1** | `rmse_degC` | 3 | +0.0075 ± 0.0130 | `s0:+0.0000; s1:+0.0225; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A1** | `best_val_loss` | 3 | +0.0001 ± 0.0001 | `s0:+0.0000; s1:+0.0002; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A1** | `samples_to_target` | 3 | +0.0000 ± 0.0000 | `s0:+0.0000; s1:+0.0000; s2:+0.0000` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |
| `TCN` | **A3** | **A1** | `val_auc_samples` | 3 | +0.0001 ± 0.0008 | `s0:+0.0008; s1:-0.0007; s2:+0.0002` | Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed) |

## 5. Báo cáo các cấu hình còn thiếu (Missing Configurations)

Tất cả các cấu hình và seed trong danh sách kiểm tra đều đã có đủ trên đĩa.

## Phụ lục: Thời gian chạy (Chỉ tham khảo kỹ thuật — Không dùng để so sánh/kết luận)

> **Lưu ý:** Thời gian chạy mỗi epoch (`Epoch time`), thời gian đạt ngưỡng (`Giây tới val_target`) và `Overhead PALF` phụ thuộc vào tải CPU tại thời điểm chạy cũng như sự khác biệt đường mã giữa `DataLoader(train)` chuẩn và `DataLoader(Subset(train, indices))`. Các số liệu này **không dùng để kết luận hiệu năng thuật toán**.

| Backbone | Mã | Config Tag | Số seed | Giây tới `val_target` | Epoch time (s) | Overhead PALF (s) |
|---|---|---|---:|---:|---:|---:|
| `GRU` | **A0** | `baseline` | 10 (0,1,2,3,4,5,6,7,8,9) | 322.6s | 73.91s | 0.0000s |
| `GRU` | **A0s** | `uniform_b1.0` | 10 (0,1,2,3,4,5,6,7,8,9) | 435.1s | 96.78s | 0.0070s |
| `GRU` | **A0w** | `loss_shuffled_b1.0_a1.0` | 10 (0,1,2,3,4,5,6,7,8,9) | 606.7s | 144.46s | 0.0107s |
| `GRU` | **A0r** | `uniform_b0.7` | 10 (0,1,2,3,4,5,6,7,8,9) | 397.2s | 68.33s | 0.0056s |
| `GRU` | **A1** | `loss_b1.0_a1.0` | 10 (0,1,2,3,4,5,6,7,8,9) | 403.1s | 91.36s | 0.0072s |
| `GRU` | **A3** | `predictive_b1.0_a0.5_w2` | 10 (0,1,2,3,4,5,6,7,8,9) | 379.7s | 93.70s | 0.0073s |
