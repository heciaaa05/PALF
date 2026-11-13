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
