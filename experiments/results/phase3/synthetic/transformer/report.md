# Báo cáo Phân tích Phase 3 (Learning State Investigation)

- **Chế độ chạy (`mode_effective`):** `reduced`
- **Số thư mục run đầu vào:** `1`
- **Chuẩn hóa (`normalization`):** `epoch1` | **Cửa sổ `W`:** `5` | **Độ trễ dự đoán `k`:** `3`

## Mục 3 — Các cặp biến có tương quan hạng Spearman |ρ| > 0.8

| var_1 | var_2 | spearman_rho | abs_rho | is_redundant_gt_0p8 |
| --- | --- | --- | --- | --- |
| pattern_frequency | generalization_gap | 0.8448 | 0.8448 | True |

## Mục 4 — Khác biệt giữa các Pattern (Kruskal-Wallis theo khối Epoch)

| sampling_unit | variable | n_obs_total | k_patterns | kruskal_H | p_value | epsilon_sq | eta_sq_H | epsilon_sq_ci95_low | epsilon_sq_ci95_high | median_falling | median_rising | median_spike | median_stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch_block | loss_norm | 120 | 4 | 74.6592 | 0.0000 | 0.6274 | 0.6178 | 0.5731 | 0.8731 | 0.5649 | 0.5720 | 0.3217 | 0.3298 |
| epoch_block | loss_trend | 120 | 4 | 2.0724 | 0.5575 | 0.0174 | 0.0000 | 0.0082 | 0.1266 | -0.0017 | -0.0013 | -0.0018 | -0.0011 |
| epoch_block | learning_velocity | 120 | 4 | 0.5568 | 0.9062 | 0.0047 | 0.0000 | 0.0031 | 0.0453 | 0.0031 | 0.0026 | 0.0023 | 0.0017 |
| epoch_block | prediction_stability_ps | 120 | 4 | 6.9809 | 0.0725 | 0.0587 | 0.0343 | 0.0127 | 0.2160 | 0.0049 | 0.0052 | 0.0037 | 0.0036 |
| epoch_block | within_pattern_dispersion_dis | 120 | 4 | 85.7553 | 0.0000 | 0.7206 | 0.7134 | 0.7165 | 0.8727 | 3.1895 | 3.4229 | 2.4864 | 3.1703 |
| epoch_block | grad_norm | 120 | 4 | 12.9470 | 0.0048 | 0.1088 | 0.0857 | 0.0847 | 0.2641 | 0.3886 | 0.3654 | 0.3674 | 0.2534 |
| epoch_block | pattern_frequency | 120 | 4 | 119.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2447 | 0.2289 | 0.0478 | 0.4787 |
| epoch_block | generalization_gap | 120 | 4 | 85.7680 | 0.0000 | 0.7207 | 0.7135 | 0.5770 | 0.9229 | -0.0346 | -0.0546 | -0.0687 | -0.0112 |
| epoch_block | forgetting_risk | 120 | 4 | 0.4341 | 0.9331 | 0.0036 | 0.0000 | 0.0006 | 0.0420 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mục 5 — Độ ổn định hạng Pattern giữa các Seed

| model | variable | n_seeds | spearman_rank_mean | spearman_rank_std | spearman_rank_min | kendall_W |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | loss_norm | 1 | NaN | NaN | NaN | NaN |
| transformer | loss_trend | 1 | NaN | NaN | NaN | NaN |
| transformer | learning_velocity | 1 | NaN | NaN | NaN | NaN |
| transformer | prediction_stability_ps | 1 | NaN | NaN | NaN | NaN |
| transformer | within_pattern_dispersion_dis | 1 | NaN | NaN | NaN | NaN |
| transformer | grad_norm | 1 | NaN | NaN | NaN | NaN |
| transformer | pattern_frequency | 1 | NaN | NaN | NaN | NaN |
| transformer | generalization_gap | 1 | NaN | NaN | NaN | NaN |
| transformer | forgetting_risk | 1 | NaN | NaN | NaN | NaN |

## Mục 6 — Phép thử dự đoán Δloss sau k=3 epoch (ngoài thông tin từ loss hiện tại)

| variable | horizon_k | n_obs | r2_baseline_loss_only | r2_with_variable | delta_r2 | beta_variable | F_stat | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| within_pattern_dispersion_dis | 3 | 108 | 0.4156 | 0.8291 | 0.4136 | 0.1322 | 254.1452 | 0.0000 |
| generalization_gap | 3 | 108 | 0.4156 | 0.6696 | 0.2540 | 1.3962 | 80.6987 | 0.0000 |
| forgetting_risk | 3 | 108 | 0.4156 | 0.4280 | 0.0124 | 4.3138 | 2.2847 | 0.1337 |
| grad_norm | 3 | 108 | 0.4156 | 0.4249 | 0.0093 | -0.0376 | 1.7038 | 0.1946 |
| loss_trend | 3 | 108 | 0.4156 | 0.4207 | 0.0051 | 0.0701 | 0.9207 | 0.3395 |
| prediction_stability_ps | 3 | 108 | 0.4156 | 0.4186 | 0.0030 | -0.0516 | 0.5364 | 0.4656 |
| learning_velocity | 3 | 108 | 0.4156 | 0.4178 | 0.0023 | -0.0535 | 0.4070 | 0.5249 |
| pattern_frequency | 3 | 108 | 0.4156 | 0.4157 | 0.0001 | -0.0074 | 0.0244 | 0.8762 |
