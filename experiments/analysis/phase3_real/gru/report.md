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
