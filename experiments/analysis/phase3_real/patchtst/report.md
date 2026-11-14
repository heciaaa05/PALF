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
