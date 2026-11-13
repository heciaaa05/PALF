# Báo cáo Phân tích Phase 3 (Learning State Investigation)

- **Chế độ chạy (`mode_effective`):** `reduced`
- **Số thư mục run đầu vào:** `1`
- **Chuẩn hóa (`normalization`):** `epoch1` | **Cửa sổ `W`:** `5` | **Độ trễ dự đoán `k`:** `3`

## Mục 3 — Các cặp biến có tương quan hạng Spearman |ρ| > 0.8

| var_1 | var_2 | spearman_rho | abs_rho | is_redundant_gt_0p8 |
| --- | --- | --- | --- | --- |
| loss_norm | generalization_gap | -1.0000 | 1.0000 | True |

## Mục 4 — Khác biệt giữa các Pattern (Kruskal-Wallis theo khối Epoch)

| sampling_unit | variable | n_obs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_humidity_net_rise | median_normal | median_pressure_drop | median_temp_anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch_block | loss_norm | 4 | 4 | 3.0000 | 0.3916 | 1.0000 | NaN | NaN | NaN | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| epoch_block | loss_trend | 4 | 4 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | NaN | NaN | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| epoch_block | learning_velocity | 4 | 4 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | NaN | NaN | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| epoch_block | prediction_stability_ps | 4 | 4 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | NaN | NaN | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| epoch_block | within_pattern_dispersion_dis | 4 | 4 | 3.0000 | 0.3916 | 1.0000 | NaN | NaN | NaN | 1.2080 | 1.3163 | 1.7191 | 1.7576 |
| epoch_block | grad_norm | 4 | 4 | 3.0000 | 0.3916 | 1.0000 | NaN | NaN | NaN | 0.3089 | 0.0737 | 0.1709 | 0.1731 |
| epoch_block | pattern_frequency | 4 | 4 | 3.0000 | 0.3916 | 1.0000 | NaN | NaN | NaN | 0.0894 | 0.6513 | 0.1000 | 0.1593 |
| epoch_block | generalization_gap | 4 | 4 | 3.0000 | 0.3916 | 1.0000 | NaN | NaN | NaN | -0.0668 | -0.1314 | -0.1092 | -0.2202 |
| epoch_block | forgetting_risk | 4 | 4 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | NaN | NaN | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mục 5 — Độ ổn định hạng Pattern giữa các Seed

| model | variable | n_seeds | spearman_rank_mean | spearman_rank_std | spearman_rank_min | kendall_W |
| --- | --- | --- | --- | --- | --- | --- |
| gru | loss_norm | 1 | NaN | NaN | NaN | NaN |
| gru | loss_trend | 1 | NaN | NaN | NaN | NaN |
| gru | learning_velocity | 1 | NaN | NaN | NaN | NaN |
| gru | prediction_stability_ps | 1 | NaN | NaN | NaN | NaN |
| gru | within_pattern_dispersion_dis | 1 | NaN | NaN | NaN | NaN |
| gru | grad_norm | 1 | NaN | NaN | NaN | NaN |
| gru | pattern_frequency | 1 | NaN | NaN | NaN | NaN |
| gru | generalization_gap | 1 | NaN | NaN | NaN | NaN |
| gru | forgetting_risk | 1 | NaN | NaN | NaN | NaN |

## Mục 6 — Phép thử dự đoán Δloss sau k=3 epoch (ngoài thông tin từ loss hiện tại)

| variable | horizon_k | n_obs | r2_baseline_loss_only | r2_with_variable | delta_r2 | beta_variable | F_stat | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss_trend | 3 | 0 | NaN | NaN | NaN | NaN | NaN | NaN |
| learning_velocity | 3 | 0 | NaN | NaN | NaN | NaN | NaN | NaN |
| prediction_stability_ps | 3 | 0 | NaN | NaN | NaN | NaN | NaN | NaN |
| within_pattern_dispersion_dis | 3 | 0 | NaN | NaN | NaN | NaN | NaN | NaN |
| grad_norm | 3 | 0 | NaN | NaN | NaN | NaN | NaN | NaN |
| pattern_frequency | 3 | 0 | NaN | NaN | NaN | NaN | NaN | NaN |
| generalization_gap | 3 | 0 | NaN | NaN | NaN | NaN | NaN | NaN |
| forgetting_risk | 3 | 0 | NaN | NaN | NaN | NaN | NaN | NaN |
