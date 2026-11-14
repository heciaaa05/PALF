# Báo cáo Phase 3 trên Log Thật (Jena Climate — 4 Backbones × 5 Seeds)

- **Đầu vào:** experiments/results/baseline/jena/{gru,tcn,transformer,patchtst}/seed0..4/ (20 runs, 600 epochs, chế độ FULL có sample_loss_epochs.npy).
- **Đơn vị kiểm định Kruskal-Wallis & Hoán vị khối:** Sự kiện liên tiếp cùng nhãn (event_run, N = 2,251 runs trên tập train Jena).

## 1. Bảng Quyết định Tổng hợp Trung bình qua 4 Backbone (G3 Decision Table)

| variable | rank_spearman_mean | kendall_W_mean | pred_delta_r2_mean | pred_beta_mean | pred_beta_sign_consensus | run_kw_epsilon_sq_mean | null_perm_p95_mean | n_backbones_exceed_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss_norm | 0.9350 | 0.9480 | NaN | NaN | 0 | 0.0178 | 0.0034 | 4 |
| loss_trend | 0.5550 | 0.6440 | 0.0627 | -0.2847 | -1 | 0.0095 | 0.0033 | 3 |
| learning_velocity | 0.2800 | 0.4240 | 0.0282 | 0.2115 | 1 | 0.0122 | 0.0033 | 3 |
| prediction_stability_ps | 0.6000 | 0.6800 | 0.0583 | 0.2630 | 1 | 0.0320 | 0.0033 | 4 |
| within_pattern_dispersion_dis | 0.8900 | 0.9120 | 0.0252 | -0.1485 | -1 | 0.0279 | 0.0032 | 4 |
| grad_norm | 0.7250 | 0.7800 | 0.0037 | -0.0198 | 0 | NaN | NaN | 0 |
| pattern_frequency | 1.0000 | 1.0000 | 0.0002 | 0.0020 | 0 | NaN | NaN | 0 |
| generalization_gap | 0.9000 | 0.9200 | 0.0036 | -0.2695 | -1 | NaN | NaN | 0 |
| forgetting_risk | 0.5671 | 0.0530 | 0.0000 | -13.8250 | -1 | 0.1652 | 0.0029 | 4 |

## 2. Bảng Quyết định Chi tiết theo từng Backbone (Kèm Hệ số Hồi quy pred_beta và Dấu pred_beta_sign cho state_signs)

| model | variable | rank_spearman_mean | kendall_W | pred_delta_r2 | pred_beta | pred_beta_sign | pred_F_stat | pred_p_value | epoch_kw_epsilon_sq | run_kw_epsilon_sq | null_perm_p95 | null_emp_p_value | exceeds_null_p95 | redundant_gt_0p8_with |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gru | loss_norm | 1.0000 | 1.0000 | NaN | NaN | 0 | NaN | NaN | 0.0605 | 0.0274 | 0.0033 | 0.0099 | True | within_pattern_dispersion_dis,generalization_gap |
| gru | loss_trend | 0.8400 | 0.8720 | 0.0951 | -0.3373 | -1 | 280.9961 | 0.0000 | 0.0194 | 0.0076 | 0.0036 | 0.0099 | True | learning_velocity,prediction_stability_ps |
| gru | learning_velocity | 0.0200 | 0.2160 | 0.0366 | 0.2312 | 1 | 81.8580 | 0.0000 | 0.0193 | 0.0068 | 0.0029 | 0.0099 | True | loss_trend |
| gru | prediction_stability_ps | 0.9200 | 0.9360 | 0.0909 | 0.3141 | 1 | 262.3526 | 0.0000 | 0.0291 | 0.0331 | 0.0028 | 0.0099 | True | loss_trend |
| gru | within_pattern_dispersion_dis | 0.9200 | 0.9360 | 0.0379 | -0.2331 | -1 | 85.2126 | 0.0000 | 0.0656 | 0.0228 | 0.0029 | 0.0099 | True | loss_norm,generalization_gap |
| gru | grad_norm | 0.7600 | 0.8080 | 0.0002 | 0.0219 | 1 | 0.2927 | 0.5887 | 0.0818 | NaN | NaN | NaN | False | NaN |
| gru | pattern_frequency | 1.0000 | 1.0000 | 0.0000 | -0.0026 | -1 | 0.0778 | 0.7804 | 1.0000 | NaN | NaN | NaN | False | NaN |
| gru | generalization_gap | 0.9200 | 0.9360 | 0.0115 | -0.4254 | -1 | 23.3355 | 0.0000 | 0.0682 | NaN | NaN | NaN | False | within_pattern_dispersion_dis,loss_norm |
| gru | forgetting_risk | 1.0000 | 0.0000 | NaN | NaN | 0 | NaN | NaN | 0.0000 | 0.1388 | 0.0028 | 0.0099 | True | NaN |
| tcn | loss_norm | 0.7400 | 0.7920 | NaN | NaN | 0 | NaN | NaN | 0.0151 | 0.0099 | 0.0040 | 0.0099 | True | generalization_gap,within_pattern_dispersion_dis |
| tcn | loss_trend | 0.9200 | 0.9360 | 0.0709 | -0.4072 | -1 | 1340.2070 | 0.0000 | 0.0008 | 0.0023 | 0.0036 | 0.2079 | False | learning_velocity,prediction_stability_ps |
| tcn | learning_velocity | 0.6400 | 0.7120 | 0.0387 | 0.3218 | 1 | 342.2283 | 0.0000 | 0.0006 | 0.0005 | 0.0035 | 0.7723 | False | loss_trend |
| tcn | prediction_stability_ps | 1.0000 | 1.0000 | 0.0634 | 0.3722 | 1 | 948.8211 | 0.0000 | 0.0034 | 0.0278 | 0.0038 | 0.0099 | True | loss_trend |
| tcn | within_pattern_dispersion_dis | 1.0000 | 1.0000 | 0.0456 | -0.2627 | -1 | 454.9140 | 0.0000 | 0.1181 | 0.0175 | 0.0033 | 0.0099 | True | generalization_gap,loss_norm |
| tcn | grad_norm | 0.8800 | 0.9040 | 0.0036 | -0.1004 | -1 | 20.0339 | 0.0000 | 0.1420 | NaN | NaN | NaN | False | NaN |
| tcn | pattern_frequency | 1.0000 | 1.0000 | 0.0006 | 0.0096 | 1 | 3.2812 | 0.0706 | 1.0000 | NaN | NaN | NaN | False | NaN |
| tcn | generalization_gap | 0.7600 | 0.8080 | 0.0015 | -0.3783 | -1 | 8.2775 | 0.0042 | 0.0642 | NaN | NaN | NaN | False | within_pattern_dispersion_dis,loss_norm |
| tcn | forgetting_risk | -0.0648 | 0.1000 | 0.0000 | -16.6921 | -1 | 0.0505 | 0.8222 | 0.0051 | 0.1723 | 0.0027 | 0.0099 | True | NaN |
| transformer | loss_norm | 1.0000 | 1.0000 | NaN | NaN | 0 | NaN | NaN | 0.0604 | 0.0127 | 0.0032 | 0.0099 | True | within_pattern_dispersion_dis |
| transformer | loss_trend | 0.6800 | 0.7440 | 0.0731 | -0.2963 | -1 | 349.2082 | 0.0000 | 0.0063 | 0.0034 | 0.0031 | 0.0495 | True | learning_velocity |
| transformer | learning_velocity | 0.6800 | 0.7440 | 0.0325 | 0.2195 | 1 | 114.1839 | 0.0000 | 0.0061 | 0.0047 | 0.0036 | 0.0396 | True | loss_trend |
| transformer | prediction_stability_ps | 0.6800 | 0.7440 | 0.0662 | 0.2707 | 1 | 297.5857 | 0.0000 | 0.0356 | 0.0373 | 0.0033 | 0.0099 | True | NaN |
| transformer | within_pattern_dispersion_dis | 0.8800 | 0.9040 | 0.0000 | 0.0045 | 1 | 0.0263 | 0.8713 | 0.0556 | 0.0328 | 0.0032 | 0.0099 | True | loss_norm |
| transformer | grad_norm | 0.8400 | 0.8720 | 0.0001 | -0.0093 | -1 | 0.1797 | 0.6718 | 0.0288 | NaN | NaN | NaN | False | NaN |
| transformer | pattern_frequency | 1.0000 | 1.0000 | 0.0001 | -0.0032 | -1 | 0.1526 | 0.6962 | 1.0000 | NaN | NaN | NaN | False | NaN |
| transformer | generalization_gap | 1.0000 | 1.0000 | 0.0002 | 0.0498 | 1 | 0.6408 | 0.4238 | 0.2838 | NaN | NaN | NaN | False | NaN |
| transformer | forgetting_risk | 0.3333 | 0.0880 | 0.0000 | -10.9580 | -1 | 0.1039 | 0.7474 | 0.0091 | 0.1747 | 0.0034 | 0.0099 | True | NaN |
| patchtst | loss_norm | 1.0000 | 1.0000 | NaN | NaN | 0 | NaN | NaN | 0.0396 | 0.0210 | 0.0029 | 0.0099 | True | generalization_gap,within_pattern_dispersion_dis,grad_norm |
| patchtst | loss_trend | -0.2200 | 0.0240 | 0.0115 | -0.0980 | -1 | 318.9958 | 0.0000 | 0.0641 | 0.0248 | 0.0030 | 0.0099 | True | learning_velocity |
| patchtst | learning_velocity | -0.2200 | 0.0240 | 0.0050 | 0.0735 | 1 | 104.6238 | 0.0000 | 0.0544 | 0.0369 | 0.0031 | 0.0099 | True | loss_trend |
| patchtst | prediction_stability_ps | -0.2000 | 0.0400 | 0.0125 | 0.0951 | 1 | 365.8159 | 0.0000 | 0.0327 | 0.0300 | 0.0032 | 0.0099 | True | NaN |
| patchtst | within_pattern_dispersion_dis | 0.7600 | 0.8080 | 0.0172 | -0.1027 | -1 | 668.0234 | 0.0000 | 0.0966 | 0.0387 | 0.0035 | 0.0099 | True | generalization_gap,loss_norm |
| patchtst | grad_norm | 0.4200 | 0.5360 | 0.0111 | 0.0084 | 1 | 299.0740 | 0.0000 | 0.0209 | NaN | NaN | NaN | False | loss_norm |
| patchtst | pattern_frequency | 1.0000 | 1.0000 | 0.0000 | 0.0043 | 1 | 0.6842 | 0.4085 | 1.0000 | NaN | NaN | NaN | False | NaN |
| patchtst | generalization_gap | 0.9200 | 0.9360 | 0.0011 | -0.3241 | -1 | 19.1687 | 0.0000 | 0.0370 | NaN | NaN | NaN | False | loss_norm,within_pattern_dispersion_dis |
| patchtst | forgetting_risk | 1.0000 | 0.0240 | NaN | NaN | 0 | NaN | NaN | 0.0050 | 0.1749 | 0.0029 | 0.0099 | True | NaN |

---
## 3.GRU — Chi tiết Backbone GRU (5 Seeds)
# Báo cáo Phân tích Phase 3 (Learning State Investigation)

- **Chế độ chạy (`mode_effective`):** `full`
- **Số thư mục run đầu vào:** `5`
- **Chuẩn hóa (`normalization`):** `epoch1` | **Cửa sổ `W`:** `5` | **Độ trễ dự đoán `k`:** `3`

## Mục 3 — Các cặp biến có tương quan hạng Spearman |ρ| > 0.8

| var_1 | var_2 | spearman_rho | abs_rho | is_redundant_gt_0p8 |
| --- | --- | --- | --- | --- |
| loss_trend | learning_velocity | -0.9520 | 0.9520 | True |
| loss_norm | within_pattern_dispersion_dis | 0.8840 | 0.8840 | True |
| within_pattern_dispersion_dis | generalization_gap | -0.8618 | 0.8618 | True |
| loss_norm | generalization_gap | -0.8353 | 0.8353 | True |
| loss_trend | prediction_stability_ps | -0.8282 | 0.8282 | True |

## Mục 4 — Khác biệt giữa các Pattern (Kruskal-Wallis theo khối Epoch)

| sampling_unit | variable | n_obs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | median_normal | median_pressure_drop | median_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch_block | loss_norm | 600 | 4 | 36.2562 | 0.0000 | 0.0605 | 0.0558 | 0.0616 | 0.2843 | 0.2298 | 0.2077 | 0.2317 | 0.1515 |
| epoch_block | loss_trend | 600 | 4 | 11.6027 | 0.0089 | 0.0194 | 0.0144 | 0.0061 | 0.1560 | -0.0214 | -0.0195 | -0.0230 | -0.0160 |
| epoch_block | learning_velocity | 600 | 4 | 11.5582 | 0.0091 | 0.0193 | 0.0144 | 0.0138 | 0.1686 | 0.0182 | 0.0163 | 0.0187 | 0.0131 |
| epoch_block | prediction_stability_ps | 600 | 4 | 17.4052 | 0.0006 | 0.0291 | 0.0242 | 0.0015 | 0.1660 | 0.0026 | 0.0021 | 0.0030 | 0.0020 |
| epoch_block | within_pattern_dispersion_dis | 600 | 4 | 39.2807 | 0.0000 | 0.0656 | 0.0609 | 0.0403 | 0.3607 | 0.8590 | 0.7890 | 0.8174 | 0.8583 |
| epoch_block | grad_norm | 600 | 4 | 49.0176 | 0.0000 | 0.0818 | 0.0772 | 0.0467 | 0.1487 | 0.1372 | 0.1304 | 0.1520 | 0.1635 |
| epoch_block | pattern_frequency | 600 | 4 | 599.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0894 | 0.6513 | 0.1000 | 0.1593 |
| epoch_block | generalization_gap | 600 | 4 | 40.8320 | 0.0000 | 0.0682 | 0.0635 | 0.0502 | 0.3687 | 0.0910 | 0.0805 | 0.1027 | 0.0647 |
| epoch_block | forgetting_risk | 600 | 4 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mục 4 (Full Mode) — Khác biệt giữa các Pattern trên đơn vị SỰ KIỆN (`event_run`)

| sampling_unit | variable | n_runs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | n_runs_humidity_net_rise | median_normal | n_runs_normal | median_pressure_drop | n_runs_pressure_drop | median_temp_anomaly | n_runs_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_run | run_loss_raw_mean | 2251 | 4 | 10.7734 | 0.0130 | 0.0048 | 0.0035 | 0.0013 | 0.0151 | 0.0333 | 879 | 0.0370 | 900 | 0.0383 | 185 | 0.0386 | 287 |
| event_run | run_loss_norm_mean | 2251 | 4 | 61.5553 | 0.0000 | 0.0274 | 0.0261 | 0.0139 | 0.0461 | 0.2561 | 879 | 0.2482 | 900 | 0.2605 | 185 | 0.1926 | 287 |
| event_run | run_loss_norm_end | 2251 | 4 | 116.7311 | 0.0000 | 0.0519 | 0.0506 | 0.0366 | 0.0680 | 0.0946 | 879 | 0.0871 | 900 | 0.0924 | 185 | 0.0647 | 287 |
| event_run | run_loss_trend | 2251 | 4 | 17.1692 | 0.0007 | 0.0076 | 0.0063 | 0.0033 | 0.0168 | -0.0153 | 879 | -0.0151 | 900 | -0.0165 | 185 | -0.0116 | 287 |
| event_run | run_learning_velocity | 2251 | 4 | 15.2760 | 0.0016 | 0.0068 | 0.0055 | 0.0025 | 0.0162 | 0.0248 | 879 | 0.0240 | 900 | 0.0245 | 185 | 0.0203 | 287 |
| event_run | run_prediction_stability_ps | 2251 | 4 | 74.3807 | 0.0000 | 0.0331 | 0.0318 | 0.0187 | 0.0499 | 0.0755 | 879 | 0.0648 | 900 | 0.0555 | 185 | 0.0545 | 287 |
| event_run | run_within_dispersion_dis | 2251 | 4 | 51.1918 | 0.0000 | 0.0228 | 0.0214 | 0.0135 | 0.0355 | 0.2724 | 879 | 0.3672 | 900 | 0.3851 | 185 | 0.3609 | 287 |
| event_run | run_forgetting_risk | 2251 | 4 | 312.3464 | 0.0000 | 0.1388 | 0.1377 | 0.1188 | 0.1664 | 0.0134 | 879 | 0.0043 | 900 | 0.0031 | 185 | 0.0032 | 287 |

## Mục 5 — Độ ổn định hạng Pattern giữa các Seed

| model | variable | n_seeds | spearman_rank_mean | spearman_rank_std | spearman_rank_min | kendall_W |
| --- | --- | --- | --- | --- | --- | --- |
| gru | loss_norm | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| gru | loss_trend | 5 | 0.8400 | 0.1960 | 0.6000 | 0.8720 |
| gru | learning_velocity | 5 | 0.0200 | 0.6416 | -1.0000 | 0.2160 |
| gru | prediction_stability_ps | 5 | 0.9200 | 0.0980 | 0.8000 | 0.9360 |
| gru | within_pattern_dispersion_dis | 5 | 0.9200 | 0.0980 | 0.8000 | 0.9360 |
| gru | grad_norm | 5 | 0.7600 | 0.1497 | 0.6000 | 0.8080 |
| gru | pattern_frequency | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| gru | generalization_gap | 5 | 0.9200 | 0.0980 | 0.8000 | 0.9360 |
| gru | forgetting_risk | 5 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

## Mục 6 — Phép thử dự đoán Δloss sau k=3 epoch (ngoài thông tin từ loss hiện tại)

| variable | horizon_k | n_obs | r2_baseline_loss_only | r2_with_variable | delta_r2 | beta_variable | F_stat | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss_trend | 3 | 540 | 0.7232 | 0.8183 | 0.0951 | -0.3373 | 280.9961 | 0.0000 |
| prediction_stability_ps | 3 | 540 | 0.7232 | 0.8140 | 0.0909 | 0.3141 | 262.3526 | 0.0000 |
| within_pattern_dispersion_dis | 3 | 540 | 0.7232 | 0.7611 | 0.0379 | -0.2331 | 85.2126 | 0.0000 |
| learning_velocity | 3 | 540 | 0.7232 | 0.7598 | 0.0366 | 0.2312 | 81.8580 | 0.0000 |
| generalization_gap | 3 | 540 | 0.7232 | 0.7347 | 0.0115 | -0.4254 | 23.3355 | 0.0000 |
| grad_norm | 3 | 540 | 0.7232 | 0.7233 | 0.0002 | 0.0219 | 0.2927 | 0.5887 |
| pattern_frequency | 3 | 540 | 0.7232 | 0.7232 | 0.0000 | -0.0026 | 0.0778 | 0.7804 |
| forgetting_risk | 3 | 540 | NaN | NaN | NaN | NaN | NaN | NaN |

## Mục 7 (Full Mode) — Kiểm định đối chứng hoán vị khối (Block Permutation Null Distribution)

| variable | real_kruskal_H | real_epsilon_sq | null_epsilon_sq_mean | null_epsilon_sq_std | null_epsilon_sq_p95 | empirical_p_value | exceeds_null_p95 | n_permutations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run_loss_raw_mean | 10.7734 | 0.0048 | 0.0015 | 0.0012 | 0.0036 | 0.0198 | True | 100 |
| run_loss_norm_mean | 61.5553 | 0.0274 | 0.0015 | 0.0013 | 0.0033 | 0.0099 | True | 100 |
| run_loss_norm_end | 116.7311 | 0.0519 | 0.0014 | 0.0012 | 0.0036 | 0.0099 | True | 100 |
| run_loss_trend | 17.1692 | 0.0076 | 0.0014 | 0.0011 | 0.0036 | 0.0099 | True | 100 |
| run_learning_velocity | 15.2760 | 0.0068 | 0.0012 | 0.0009 | 0.0029 | 0.0099 | True | 100 |
| run_prediction_stability_ps | 74.3807 | 0.0331 | 0.0012 | 0.0008 | 0.0028 | 0.0099 | True | 100 |
| run_within_dispersion_dis | 51.1918 | 0.0228 | 0.0012 | 0.0011 | 0.0029 | 0.0099 | True | 100 |
| run_forgetting_risk | 312.3464 | 0.1388 | 0.0012 | 0.0010 | 0.0028 | 0.0099 | True | 100 |


---
## 3.TCN — Chi tiết Backbone TCN (5 Seeds)
# Báo cáo Phân tích Phase 3 (Learning State Investigation)

- **Chế độ chạy (`mode_effective`):** `full`
- **Số thư mục run đầu vào:** `5`
- **Chuẩn hóa (`normalization`):** `epoch1` | **Cửa sổ `W`:** `5` | **Độ trễ dự đoán `k`:** `3`

## Mục 3 — Các cặp biến có tương quan hạng Spearman |ρ| > 0.8

| var_1 | var_2 | spearman_rho | abs_rho | is_redundant_gt_0p8 |
| --- | --- | --- | --- | --- |
| loss_trend | learning_velocity | -0.9589 | 0.9589 | True |
| within_pattern_dispersion_dis | generalization_gap | -0.8857 | 0.8857 | True |
| loss_norm | generalization_gap | -0.8643 | 0.8643 | True |
| loss_norm | within_pattern_dispersion_dis | 0.8342 | 0.8342 | True |
| loss_trend | prediction_stability_ps | -0.8240 | 0.8240 | True |

## Mục 4 — Khác biệt giữa các Pattern (Kruskal-Wallis theo khối Epoch)

| sampling_unit | variable | n_obs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | median_normal | median_pressure_drop | median_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch_block | loss_norm | 600 | 4 | 9.0456 | 0.0287 | 0.0151 | 0.0101 | 0.0099 | 0.1450 | 0.1834 | 0.1926 | 0.1776 | 0.1578 |
| epoch_block | loss_trend | 600 | 4 | 0.4933 | 0.9204 | 0.0008 | 0.0000 | 0.0004 | 0.0324 | -0.0159 | -0.0165 | -0.0162 | -0.0146 |
| epoch_block | learning_velocity | 600 | 4 | 0.3857 | 0.9432 | 0.0006 | 0.0000 | 0.0004 | 0.0227 | 0.0115 | 0.0117 | 0.0116 | 0.0104 |
| epoch_block | prediction_stability_ps | 600 | 4 | 2.0650 | 0.5590 | 0.0034 | 0.0000 | 0.0011 | 0.0505 | 0.0036 | 0.0030 | 0.0037 | 0.0034 |
| epoch_block | within_pattern_dispersion_dis | 600 | 4 | 70.7668 | 0.0000 | 0.1181 | 0.1137 | 0.0776 | 0.6783 | 0.7765 | 0.7201 | 0.8113 | 0.8179 |
| epoch_block | grad_norm | 600 | 4 | 85.0797 | 0.0000 | 0.1420 | 0.1377 | 0.1114 | 0.2188 | 0.1202 | 0.1411 | 0.1346 | 0.1739 |
| epoch_block | pattern_frequency | 600 | 4 | 599.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0894 | 0.6513 | 0.1000 | 0.1593 |
| epoch_block | generalization_gap | 600 | 4 | 38.4391 | 0.0000 | 0.0642 | 0.0595 | 0.0280 | 0.4050 | 0.0901 | 0.0852 | 0.0886 | 0.0751 |
| epoch_block | forgetting_risk | 600 | 4 | 3.0556 | 0.3831 | 0.0051 | 0.0001 | 0.0000 | 0.0126 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mục 4 (Full Mode) — Khác biệt giữa các Pattern trên đơn vị SỰ KIỆN (`event_run`)

| sampling_unit | variable | n_runs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | n_runs_humidity_net_rise | median_normal | n_runs_normal | median_pressure_drop | n_runs_pressure_drop | median_temp_anomaly | n_runs_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_run | run_loss_raw_mean | 2251 | 4 | 13.6518 | 0.0034 | 0.0061 | 0.0047 | 0.0019 | 0.0167 | 0.0264 | 879 | 0.0285 | 900 | 0.0292 | 185 | 0.0298 | 287 |
| event_run | run_loss_norm_mean | 2251 | 4 | 22.3869 | 0.0001 | 0.0099 | 0.0086 | 0.0036 | 0.0210 | 0.2529 | 879 | 0.2591 | 900 | 0.2422 | 185 | 0.2151 | 287 |
| event_run | run_loss_norm_end | 2251 | 4 | 68.2032 | 0.0000 | 0.0303 | 0.0290 | 0.0201 | 0.0448 | 0.0962 | 879 | 0.0973 | 900 | 0.0905 | 185 | 0.0748 | 287 |
| event_run | run_loss_trend | 2251 | 4 | 5.2071 | 0.1572 | 0.0023 | 0.0010 | 0.0005 | 0.0085 | -0.0168 | 879 | -0.0177 | 900 | -0.0163 | 185 | -0.0151 | 287 |
| event_run | run_learning_velocity | 2251 | 4 | 1.1227 | 0.7716 | 0.0005 | 0.0000 | 0.0001 | 0.0049 | 0.0230 | 879 | 0.0246 | 900 | 0.0234 | 185 | 0.0235 | 287 |
| event_run | run_prediction_stability_ps | 2251 | 4 | 62.4384 | 0.0000 | 0.0278 | 0.0265 | 0.0170 | 0.0470 | 0.0691 | 879 | 0.0591 | 900 | 0.0493 | 185 | 0.0561 | 287 |
| event_run | run_within_dispersion_dis | 2251 | 4 | 39.4499 | 0.0000 | 0.0175 | 0.0162 | 0.0091 | 0.0322 | 0.2808 | 879 | 0.3484 | 900 | 0.3616 | 185 | 0.3453 | 287 |
| event_run | run_forgetting_risk | 2251 | 4 | 387.7106 | 0.0000 | 0.1723 | 0.1712 | 0.1527 | 0.1996 | 0.0132 | 879 | 0.0029 | 900 | 0.0021 | 185 | 0.0025 | 287 |

## Mục 5 — Độ ổn định hạng Pattern giữa các Seed

| model | variable | n_seeds | spearman_rank_mean | spearman_rank_std | spearman_rank_min | kendall_W |
| --- | --- | --- | --- | --- | --- | --- |
| tcn | loss_norm | 5 | 0.7400 | 0.2375 | 0.4000 | 0.7920 |
| tcn | loss_trend | 5 | 0.9200 | 0.0980 | 0.8000 | 0.9360 |
| tcn | learning_velocity | 5 | 0.6400 | 0.2939 | 0.2000 | 0.7120 |
| tcn | prediction_stability_ps | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| tcn | within_pattern_dispersion_dis | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| tcn | grad_norm | 5 | 0.8800 | 0.0980 | 0.8000 | 0.9040 |
| tcn | pattern_frequency | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| tcn | generalization_gap | 5 | 0.7600 | 0.1960 | 0.4000 | 0.8080 |
| tcn | forgetting_risk | 5 | -0.0648 | 0.4869 | -0.5443 | 0.1000 |

## Mục 6 — Phép thử dự đoán Δloss sau k=3 epoch (ngoài thông tin từ loss hiện tại)

| variable | horizon_k | n_obs | r2_baseline_loss_only | r2_with_variable | delta_r2 | beta_variable | F_stat | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss_trend | 3 | 540 | 0.9006 | 0.9716 | 0.0709 | -0.4072 | 1340.2070 | 0.0000 |
| prediction_stability_ps | 3 | 540 | 0.9006 | 0.9641 | 0.0634 | 0.3722 | 948.8211 | 0.0000 |
| within_pattern_dispersion_dis | 3 | 540 | 0.9006 | 0.9462 | 0.0456 | -0.2627 | 454.9140 | 0.0000 |
| learning_velocity | 3 | 540 | 0.9006 | 0.9393 | 0.0387 | 0.3218 | 342.2283 | 0.0000 |
| grad_norm | 3 | 540 | 0.9006 | 0.9042 | 0.0036 | -0.1004 | 20.0339 | 0.0000 |
| generalization_gap | 3 | 540 | 0.9006 | 0.9021 | 0.0015 | -0.3783 | 8.2775 | 0.0042 |
| pattern_frequency | 3 | 540 | 0.9006 | 0.9012 | 0.0006 | 0.0096 | 3.2812 | 0.0706 |
| forgetting_risk | 3 | 540 | 0.9006 | 0.9007 | 0.0000 | -16.6921 | 0.0505 | 0.8222 |

## Mục 7 (Full Mode) — Kiểm định đối chứng hoán vị khối (Block Permutation Null Distribution)

| variable | real_kruskal_H | real_epsilon_sq | null_epsilon_sq_mean | null_epsilon_sq_std | null_epsilon_sq_p95 | empirical_p_value | exceeds_null_p95 | n_permutations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run_loss_raw_mean | 13.6518 | 0.0061 | 0.0015 | 0.0013 | 0.0035 | 0.0198 | True | 100 |
| run_loss_norm_mean | 22.3869 | 0.0099 | 0.0015 | 0.0014 | 0.0040 | 0.0099 | True | 100 |
| run_loss_norm_end | 68.2032 | 0.0303 | 0.0015 | 0.0013 | 0.0035 | 0.0099 | True | 100 |
| run_loss_trend | 5.2071 | 0.0023 | 0.0015 | 0.0012 | 0.0036 | 0.2079 | False | 100 |
| run_learning_velocity | 1.1227 | 0.0005 | 0.0014 | 0.0011 | 0.0035 | 0.7723 | False | 100 |
| run_prediction_stability_ps | 62.4384 | 0.0278 | 0.0013 | 0.0011 | 0.0038 | 0.0099 | True | 100 |
| run_within_dispersion_dis | 39.4499 | 0.0175 | 0.0013 | 0.0011 | 0.0033 | 0.0099 | True | 100 |
| run_forgetting_risk | 387.7106 | 0.1723 | 0.0011 | 0.0009 | 0.0027 | 0.0099 | True | 100 |


---
## 3.TRANSFORMER — Chi tiết Backbone TRANSFORMER (5 Seeds)
# Báo cáo Phân tích Phase 3 (Learning State Investigation)

- **Chế độ chạy (`mode_effective`):** `full`
- **Số thư mục run đầu vào:** `5`
- **Chuẩn hóa (`normalization`):** `epoch1` | **Cửa sổ `W`:** `5` | **Độ trễ dự đoán `k`:** `3`

## Mục 3 — Các cặp biến có tương quan hạng Spearman |ρ| > 0.8

| var_1 | var_2 | spearman_rho | abs_rho | is_redundant_gt_0p8 |
| --- | --- | --- | --- | --- |
| loss_trend | learning_velocity | -0.9559 | 0.9559 | True |
| loss_norm | within_pattern_dispersion_dis | 0.8870 | 0.8870 | True |

## Mục 4 — Khác biệt giữa các Pattern (Kruskal-Wallis theo khối Epoch)

| sampling_unit | variable | n_obs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | median_normal | median_pressure_drop | median_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch_block | loss_norm | 600 | 4 | 36.2016 | 0.0000 | 0.0604 | 0.0557 | 0.0554 | 0.3094 | 0.2622 | 0.2479 | 0.2656 | 0.1979 |
| epoch_block | loss_trend | 600 | 4 | 3.7926 | 0.2847 | 0.0063 | 0.0013 | 0.0029 | 0.0800 | -0.0149 | -0.0136 | -0.0144 | -0.0125 |
| epoch_block | learning_velocity | 600 | 4 | 3.6798 | 0.2982 | 0.0061 | 0.0011 | 0.0057 | 0.0840 | 0.0120 | 0.0114 | 0.0121 | 0.0115 |
| epoch_block | prediction_stability_ps | 600 | 4 | 21.3420 | 0.0001 | 0.0356 | 0.0308 | 0.0083 | 0.1629 | 0.0032 | 0.0023 | 0.0035 | 0.0031 |
| epoch_block | within_pattern_dispersion_dis | 600 | 4 | 33.3267 | 0.0000 | 0.0556 | 0.0509 | 0.0378 | 0.4032 | 0.8605 | 0.8051 | 0.8448 | 0.8784 |
| epoch_block | grad_norm | 600 | 4 | 17.2755 | 0.0006 | 0.0288 | 0.0240 | 0.0172 | 0.0547 | 0.1763 | 0.1865 | 0.1869 | 0.2016 |
| epoch_block | pattern_frequency | 600 | 4 | 599.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0894 | 0.6513 | 0.1000 | 0.1593 |
| epoch_block | generalization_gap | 600 | 4 | 169.9959 | 0.0000 | 0.2838 | 0.2802 | 0.2002 | 0.6782 | 0.1295 | 0.1067 | 0.0901 | 0.0761 |
| epoch_block | forgetting_risk | 600 | 4 | 5.4416 | 0.1422 | 0.0091 | 0.0041 | 0.0000 | 0.0153 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mục 4 (Full Mode) — Khác biệt giữa các Pattern trên đơn vị SỰ KIỆN (`event_run`)

| sampling_unit | variable | n_runs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | n_runs_humidity_net_rise | median_normal | n_runs_normal | median_pressure_drop | n_runs_pressure_drop | median_temp_anomaly | n_runs_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_run | run_loss_raw_mean | 2251 | 4 | 4.9625 | 0.1746 | 0.0022 | 0.0009 | 0.0006 | 0.0105 | 0.0372 | 879 | 0.0401 | 900 | 0.0388 | 185 | 0.0413 | 287 |
| event_run | run_loss_norm_mean | 2251 | 4 | 28.6158 | 0.0000 | 0.0127 | 0.0114 | 0.0050 | 0.0271 | 0.2607 | 879 | 0.2636 | 900 | 0.2797 | 185 | 0.2222 | 287 |
| event_run | run_loss_norm_end | 2251 | 4 | 55.8565 | 0.0000 | 0.0248 | 0.0235 | 0.0149 | 0.0378 | 0.1369 | 879 | 0.1365 | 900 | 0.1494 | 185 | 0.1094 | 287 |
| event_run | run_loss_trend | 2251 | 4 | 7.5432 | 0.0565 | 0.0034 | 0.0020 | 0.0009 | 0.0102 | -0.0121 | 879 | -0.0124 | 900 | -0.0127 | 185 | -0.0105 | 287 |
| event_run | run_learning_velocity | 2251 | 4 | 10.5159 | 0.0147 | 0.0047 | 0.0033 | 0.0014 | 0.0124 | 0.0237 | 879 | 0.0228 | 900 | 0.0208 | 185 | 0.0195 | 287 |
| event_run | run_prediction_stability_ps | 2251 | 4 | 84.0291 | 0.0000 | 0.0373 | 0.0361 | 0.0242 | 0.0555 | 0.0815 | 879 | 0.0645 | 900 | 0.0620 | 185 | 0.0563 | 287 |
| event_run | run_within_dispersion_dis | 2251 | 4 | 73.7368 | 0.0000 | 0.0328 | 0.0315 | 0.0206 | 0.0502 | 0.2903 | 879 | 0.3683 | 900 | 0.3812 | 185 | 0.3659 | 287 |
| event_run | run_forgetting_risk | 2251 | 4 | 393.0586 | 0.0000 | 0.1747 | 0.1736 | 0.1500 | 0.1990 | 0.0168 | 879 | 0.0039 | 900 | 0.0032 | 185 | 0.0042 | 287 |

## Mục 5 — Độ ổn định hạng Pattern giữa các Seed

| model | variable | n_seeds | spearman_rank_mean | spearman_rank_std | spearman_rank_min | kendall_W |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | loss_norm | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| transformer | loss_trend | 5 | 0.6800 | 0.2400 | 0.4000 | 0.7440 |
| transformer | learning_velocity | 5 | 0.6800 | 0.2400 | 0.4000 | 0.7440 |
| transformer | prediction_stability_ps | 5 | 0.6800 | 0.2400 | 0.4000 | 0.7440 |
| transformer | within_pattern_dispersion_dis | 5 | 0.8800 | 0.0980 | 0.8000 | 0.9040 |
| transformer | grad_norm | 5 | 0.8400 | 0.1960 | 0.6000 | 0.8720 |
| transformer | pattern_frequency | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| transformer | generalization_gap | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| transformer | forgetting_risk | 5 | 0.3333 | 0.6667 | -0.3333 | 0.0880 |

## Mục 6 — Phép thử dự đoán Δloss sau k=3 epoch (ngoài thông tin từ loss hiện tại)

| variable | horizon_k | n_obs | r2_baseline_loss_only | r2_with_variable | delta_r2 | beta_variable | F_stat | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss_trend | 3 | 540 | 0.8144 | 0.8876 | 0.0731 | -0.2963 | 349.2082 | 0.0000 |
| prediction_stability_ps | 3 | 540 | 0.8144 | 0.8806 | 0.0662 | 0.2707 | 297.5857 | 0.0000 |
| learning_velocity | 3 | 540 | 0.8144 | 0.8470 | 0.0325 | 0.2195 | 114.1839 | 0.0000 |
| generalization_gap | 3 | 540 | 0.8144 | 0.8147 | 0.0002 | 0.0498 | 0.6408 | 0.4238 |
| grad_norm | 3 | 540 | 0.8144 | 0.8145 | 0.0001 | -0.0093 | 0.1797 | 0.6718 |
| pattern_frequency | 3 | 540 | 0.8144 | 0.8145 | 0.0001 | -0.0032 | 0.1526 | 0.6962 |
| forgetting_risk | 3 | 540 | 0.8144 | 0.8145 | 0.0000 | -10.9580 | 0.1039 | 0.7474 |
| within_pattern_dispersion_dis | 3 | 540 | 0.8144 | 0.8144 | 0.0000 | 0.0045 | 0.0263 | 0.8713 |

## Mục 7 (Full Mode) — Kiểm định đối chứng hoán vị khối (Block Permutation Null Distribution)

| variable | real_kruskal_H | real_epsilon_sq | null_epsilon_sq_mean | null_epsilon_sq_std | null_epsilon_sq_p95 | empirical_p_value | exceeds_null_p95 | n_permutations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run_loss_raw_mean | 4.9625 | 0.0022 | 0.0014 | 0.0012 | 0.0034 | 0.2079 | False | 100 |
| run_loss_norm_mean | 28.6158 | 0.0127 | 0.0014 | 0.0012 | 0.0032 | 0.0099 | True | 100 |
| run_loss_norm_end | 55.8565 | 0.0248 | 0.0015 | 0.0013 | 0.0038 | 0.0099 | True | 100 |
| run_loss_trend | 7.5432 | 0.0034 | 0.0013 | 0.0011 | 0.0031 | 0.0495 | True | 100 |
| run_learning_velocity | 10.5159 | 0.0047 | 0.0014 | 0.0012 | 0.0036 | 0.0396 | True | 100 |
| run_prediction_stability_ps | 84.0291 | 0.0373 | 0.0013 | 0.0009 | 0.0033 | 0.0099 | True | 100 |
| run_within_dispersion_dis | 73.7368 | 0.0328 | 0.0013 | 0.0009 | 0.0032 | 0.0099 | True | 100 |
| run_forgetting_risk | 393.0586 | 0.1747 | 0.0012 | 0.0009 | 0.0034 | 0.0099 | True | 100 |


---
## 3.PATCHTST — Chi tiết Backbone PATCHTST (5 Seeds)
# Báo cáo Phân tích Phase 3 (Learning State Investigation)

- **Chế độ chạy (`mode_effective`):** `full`
- **Số thư mục run đầu vào:** `5`
- **Chuẩn hóa (`normalization`):** `epoch1` | **Cửa sổ `W`:** `5` | **Độ trễ dự đoán `k`:** `3`

## Mục 3 — Các cặp biến có tương quan hạng Spearman |ρ| > 0.8

| var_1 | var_2 | spearman_rho | abs_rho | is_redundant_gt_0p8 |
| --- | --- | --- | --- | --- |
| loss_norm | generalization_gap | -0.9744 | 0.9744 | True |
| within_pattern_dispersion_dis | generalization_gap | -0.8784 | 0.8784 | True |
| loss_norm | within_pattern_dispersion_dis | 0.8727 | 0.8727 | True |
| loss_trend | learning_velocity | -0.8359 | 0.8359 | True |
| loss_norm | grad_norm | 0.8082 | 0.8082 | True |

## Mục 4 — Khác biệt giữa các Pattern (Kruskal-Wallis theo khối Epoch)

| sampling_unit | variable | n_obs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | median_normal | median_pressure_drop | median_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch_block | loss_norm | 600 | 4 | 23.7245 | 0.0000 | 0.0396 | 0.0348 | 0.0447 | 0.2402 | 0.0760 | 0.0899 | 0.0991 | 0.0863 |
| epoch_block | loss_trend | 600 | 4 | 38.4252 | 0.0000 | 0.0641 | 0.0594 | 0.0146 | 0.4355 | -0.0034 | -0.0038 | -0.0051 | -0.0042 |
| epoch_block | learning_velocity | 600 | 4 | 32.6066 | 0.0000 | 0.0544 | 0.0497 | 0.0299 | 0.1950 | 0.0031 | 0.0038 | 0.0050 | 0.0042 |
| epoch_block | prediction_stability_ps | 600 | 4 | 19.5979 | 0.0002 | 0.0327 | 0.0278 | 0.0041 | 0.1475 | 0.0010 | 0.0008 | 0.0014 | 0.0009 |
| epoch_block | within_pattern_dispersion_dis | 600 | 4 | 57.8687 | 0.0000 | 0.0966 | 0.0921 | 0.0797 | 0.4432 | 0.9243 | 0.9785 | 0.9711 | 1.0089 |
| epoch_block | grad_norm | 600 | 4 | 12.5429 | 0.0057 | 0.0209 | 0.0160 | 0.0015 | 0.1128 | 0.2723 | 0.2831 | 0.2952 | 0.4071 |
| epoch_block | pattern_frequency | 600 | 4 | 599.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0894 | 0.6513 | 0.1000 | 0.1593 |
| epoch_block | generalization_gap | 600 | 4 | 22.1492 | 0.0001 | 0.0370 | 0.0321 | 0.0420 | 0.2480 | 0.0572 | 0.0449 | 0.0355 | 0.0426 |
| epoch_block | forgetting_risk | 600 | 4 | 3.0000 | 0.3916 | 0.0050 | 0.0000 | 0.0000 | 0.0050 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mục 4 (Full Mode) — Khác biệt giữa các Pattern trên đơn vị SỰ KIỆN (`event_run`)

| sampling_unit | variable | n_runs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | n_runs_humidity_net_rise | median_normal | n_runs_normal | median_pressure_drop | n_runs_pressure_drop | median_temp_anomaly | n_runs_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_run | run_loss_raw_mean | 2251 | 4 | 18.5346 | 0.0003 | 0.0082 | 0.0069 | 0.0027 | 0.0186 | 0.0618 | 879 | 0.0672 | 900 | 0.0736 | 185 | 0.0699 | 287 |
| event_run | run_loss_norm_mean | 2251 | 4 | 47.2885 | 0.0000 | 0.0210 | 0.0197 | 0.0125 | 0.0372 | 0.0931 | 879 | 0.1085 | 900 | 0.1196 | 185 | 0.1026 | 287 |
| event_run | run_loss_norm_end | 2251 | 4 | 57.6650 | 0.0000 | 0.0256 | 0.0243 | 0.0159 | 0.0414 | 0.0340 | 879 | 0.0391 | 900 | 0.0394 | 185 | 0.0357 | 287 |
| event_run | run_loss_trend | 2251 | 4 | 55.7896 | 0.0000 | 0.0248 | 0.0235 | 0.0134 | 0.0404 | -0.0061 | 879 | -0.0083 | 900 | -0.0092 | 185 | -0.0081 | 287 |
| event_run | run_learning_velocity | 2251 | 4 | 83.0180 | 0.0000 | 0.0369 | 0.0356 | 0.0247 | 0.0547 | 0.0146 | 879 | 0.0250 | 900 | 0.0244 | 185 | 0.0229 | 287 |
| event_run | run_prediction_stability_ps | 2251 | 4 | 67.4916 | 0.0000 | 0.0300 | 0.0287 | 0.0169 | 0.0487 | 0.0530 | 879 | 0.1036 | 900 | 0.0947 | 185 | 0.0951 | 287 |
| event_run | run_within_dispersion_dis | 2251 | 4 | 87.0259 | 0.0000 | 0.0387 | 0.0374 | 0.0257 | 0.0560 | 0.3392 | 879 | 0.4400 | 900 | 0.4530 | 185 | 0.4332 | 287 |
| event_run | run_forgetting_risk | 2251 | 4 | 393.5845 | 0.0000 | 0.1749 | 0.1738 | 0.1511 | 0.2053 | 0.0056 | 879 | 0.0015 | 900 | 0.0014 | 185 | 0.0015 | 287 |

## Mục 5 — Độ ổn định hạng Pattern giữa các Seed

| model | variable | n_seeds | spearman_rank_mean | spearman_rank_std | spearman_rank_min | kendall_W |
| --- | --- | --- | --- | --- | --- | --- |
| patchtst | loss_norm | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| patchtst | loss_trend | 5 | -0.2200 | 0.4512 | -1.0000 | 0.0240 |
| patchtst | learning_velocity | 5 | -0.2200 | 0.4512 | -1.0000 | 0.0240 |
| patchtst | prediction_stability_ps | 5 | -0.2000 | 0.4733 | -0.8000 | 0.0400 |
| patchtst | within_pattern_dispersion_dis | 5 | 0.7600 | 0.1960 | 0.4000 | 0.8080 |
| patchtst | grad_norm | 5 | 0.4200 | 0.3280 | -0.4000 | 0.5360 |
| patchtst | pattern_frequency | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| patchtst | generalization_gap | 5 | 0.9200 | 0.0980 | 0.8000 | 0.9360 |
| patchtst | forgetting_risk | 5 | 1.0000 | 0.0000 | 1.0000 | 0.0240 |

## Mục 6 — Phép thử dự đoán Δloss sau k=3 epoch (ngoài thông tin từ loss hiện tại)

| variable | horizon_k | n_obs | r2_baseline_loss_only | r2_with_variable | delta_r2 | beta_variable | F_stat | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| within_pattern_dispersion_dis | 3 | 540 | 0.9690 | 0.9862 | 0.0172 | -0.1027 | 668.0234 | 0.0000 |
| prediction_stability_ps | 3 | 540 | 0.9690 | 0.9816 | 0.0125 | 0.0951 | 365.8159 | 0.0000 |
| loss_trend | 3 | 540 | 0.9690 | 0.9806 | 0.0115 | -0.0980 | 318.9958 | 0.0000 |
| grad_norm | 3 | 540 | 0.9690 | 0.9801 | 0.0111 | 0.0084 | 299.0740 | 0.0000 |
| learning_velocity | 3 | 540 | 0.9690 | 0.9741 | 0.0050 | 0.0735 | 104.6238 | 0.0000 |
| generalization_gap | 3 | 540 | 0.9690 | 0.9701 | 0.0011 | -0.3241 | 19.1687 | 0.0000 |
| pattern_frequency | 3 | 540 | 0.9690 | 0.9691 | 0.0000 | 0.0043 | 0.6842 | 0.4085 |
| forgetting_risk | 3 | 540 | NaN | NaN | NaN | NaN | NaN | NaN |

## Mục 7 (Full Mode) — Kiểm định đối chứng hoán vị khối (Block Permutation Null Distribution)

| variable | real_kruskal_H | real_epsilon_sq | null_epsilon_sq_mean | null_epsilon_sq_std | null_epsilon_sq_p95 | empirical_p_value | exceeds_null_p95 | n_permutations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run_loss_raw_mean | 18.5346 | 0.0082 | 0.0012 | 0.0010 | 0.0029 | 0.0099 | True | 100 |
| run_loss_norm_mean | 47.2885 | 0.0210 | 0.0012 | 0.0009 | 0.0029 | 0.0099 | True | 100 |
| run_loss_norm_end | 57.6650 | 0.0256 | 0.0014 | 0.0011 | 0.0035 | 0.0099 | True | 100 |
| run_loss_trend | 55.7896 | 0.0248 | 0.0012 | 0.0010 | 0.0030 | 0.0099 | True | 100 |
| run_learning_velocity | 83.0180 | 0.0369 | 0.0012 | 0.0010 | 0.0031 | 0.0099 | True | 100 |
| run_prediction_stability_ps | 67.4916 | 0.0300 | 0.0012 | 0.0010 | 0.0032 | 0.0099 | True | 100 |
| run_within_dispersion_dis | 87.0259 | 0.0387 | 0.0013 | 0.0012 | 0.0035 | 0.0099 | True | 100 |
| run_forgetting_risk | 393.5845 | 0.1749 | 0.0012 | 0.0009 | 0.0029 | 0.0099 | True | 100 |

