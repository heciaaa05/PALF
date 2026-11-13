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
