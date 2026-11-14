# Báo cáo G3 Follow-up: Kiểm soát Epoch, Lọc Nhiễu & Kiểm định Mean-Reversion (Prompt 7 — Phần 4)

## 1. Kết luận theo ngưỡng đặt trước (Pre-registered Decision Rule)

- **Ngưỡng đặt trước (Prompt 7 Phần 4.3):** *Learning State được giữ nếu $\Delta R^2$ sau kiểm soát epoch $\ge 0.02$ ở ít nhất 3/4 backbone và dấu $\beta$ cùng chiều ở cả 4. Nếu không, báo "lui về phương án dự phòng".*
- **Kết quả của `instability` (`prediction_stability_ps`):** Số backbone có $\Delta R^2 \ge 0.02$ sau kiểm soát epoch = **2/4** | Dấu $\beta$ cùng chiều ở cả 4 backbone = **False**.
- **Quyết định chính thức:** **LUI VỀ PHƯƠNG ÁN DỰ PHÒNG (không đạt ngưỡng ΔR² ≥ 0.02 ở ≥ 3/4 backbone và cùng dấu ở cả 4)**.

## 2. Kiểm tra (a) — Trước và Sau khi Kiểm soát Hiệu ứng Cố định theo Epoch (`controls_epoch`)

| scope | variable | delta_r2_before_fe | beta_before_fe | p_before_fe | delta_r2_after_fe | beta_after_fe | p_after_fe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru | instability | 0.0066 | 0.0634 | 0.0002 | 0.0684 | -0.3657 | 0.0000 |
| gru | loss_trend | 0.0059 | -0.0547 | 0.0004 | 0.0817 | 0.3718 | 0.0000 |
| gru | velocity | 0.0121 | 0.1133 | 0.0000 | 0.1529 | -0.6938 | 0.0000 |
| patchtst | instability | 0.1014 | -0.0465 | 0.0000 | 0.0051 | 0.1028 | 0.0344 |
| patchtst | loss_trend | 0.1019 | 0.0438 | 0.0000 | 0.0042 | -0.0890 | 0.0537 |
| patchtst | velocity | 0.0896 | -0.0417 | 0.0000 | 0.0076 | 0.1488 | 0.0132 |
| tcn | instability | 0.0094 | 0.2422 | 0.0000 | 0.0665 | -0.4324 | 0.0000 |
| tcn | loss_trend | 0.0083 | -0.2055 | 0.0000 | 0.0773 | 0.4218 | 0.0000 |
| tcn | velocity | 0.0037 | 0.1991 | 0.0000 | 0.0881 | -0.5705 | 0.0000 |
| transformer | instability | 0.0005 | 0.0157 | 0.0653 | 0.0172 | -0.1784 | 0.0001 |
| transformer | loss_trend | 0.0004 | -0.0135 | 0.0835 | 0.0225 | 0.1864 | 0.0000 |
| transformer | velocity | 0.0009 | 0.0278 | 0.0137 | 0.0098 | -0.1637 | 0.0048 |
| pooled | instability | 0.0004 | -0.0156 | 0.0356 | 0.0341 | -0.2489 | 0.0000 |
| pooled | loss_trend | 0.0006 | 0.0175 | 0.0112 | 0.0416 | 0.2608 | 0.0000 |
| pooled | velocity | 0.0007 | -0.0275 | 0.0052 | 0.0453 | -0.3639 | 0.0000 |

## 3. Kiểm tra (b) — Lọc nhiễu (`ma3`, `ema`) và Tái lập ngoài mẫu giữa các Seed (`0-2 ↔ 3-4`)

### 3.1. Trên đường cong loss đã làm mượt (có kiểm soát epoch)

| smoothing | scope | variable | delta_r2_after_fe | beta_after_fe | p_after_fe |
| --- | --- | --- | --- | --- | --- |
| raw | gru | instability | 0.0684 | -0.3657 | 0.0000 |
| raw | gru | loss_trend | 0.0817 | 0.3718 | 0.0000 |
| raw | gru | velocity | 0.1529 | -0.6938 | 0.0000 |
| raw | pooled | instability | 0.0341 | -0.2489 | 0.0000 |
| raw | pooled | loss_trend | 0.0416 | 0.2608 | 0.0000 |
| raw | pooled | velocity | 0.0453 | -0.3639 | 0.0000 |
| ma3 | gru | instability | 0.3694 | -1.1063 | 0.0000 |
| ma3 | gru | loss_trend | 0.3764 | 0.9465 | 0.0000 |
| ma3 | gru | velocity | 0.1766 | -0.6802 | 0.0000 |
| ma3 | pooled | instability | 0.2344 | -0.8473 | 0.0000 |
| ma3 | pooled | loss_trend | 0.2419 | 0.7328 | 0.0000 |
| ma3 | pooled | velocity | 0.0422 | -0.3363 | 0.0000 |
| ema | gru | instability | 0.4269 | -1.1745 | 0.0000 |
| ema | gru | loss_trend | 0.4317 | 0.9795 | 0.0000 |
| ema | gru | velocity | 0.3567 | -1.0507 | 0.0000 |
| ema | pooled | instability | 0.2698 | -0.8594 | 0.0000 |
| ema | pooled | loss_trend | 0.2727 | 0.7187 | 0.0000 |
| ema | pooled | velocity | 0.1261 | -0.6036 | 0.0000 |

### 3.2. $R^2$ ngoài mẫu chéo giữa các Seed (`Seed 0-2 -> 3-4` và `Seed 3-4 -> 0-2`)

| scope | variable | r2_oos_012_to_34_loss | r2_oos_012_to_34_full | delta_r2_oos_012_to_34 | r2_oos_34_to_012_loss | r2_oos_34_to_012_full | delta_r2_oos_34_to_012 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru | instability | 0.3334 | 0.4026 | 0.0692 | 0.4673 | 0.5106 | 0.0433 |
| gru | loss_trend | 0.3334 | 0.4165 | 0.0831 | 0.4673 | 0.5162 | 0.0489 |
| gru | velocity | 0.3542 | 0.5278 | 0.1736 | 0.5111 | 0.5759 | 0.0648 |
| patchtst | instability | 0.5967 | 0.5947 | -0.0019 | 0.4094 | 0.4138 | 0.0044 |
| patchtst | loss_trend | 0.5967 | 0.5951 | -0.0016 | 0.4094 | 0.4132 | 0.0038 |
| patchtst | velocity | 0.5674 | 0.5578 | -0.0097 | 0.3899 | 0.3936 | 0.0037 |
| tcn | instability | 0.3387 | 0.3742 | 0.0355 | 0.4152 | 0.4904 | 0.0752 |
| tcn | loss_trend | 0.3387 | 0.3911 | 0.0524 | 0.4152 | 0.4988 | 0.0836 |
| tcn | velocity | 0.3643 | 0.4099 | 0.0456 | 0.4972 | 0.5918 | 0.0946 |
| transformer | instability | 0.5263 | 0.5482 | 0.0219 | 0.3942 | 0.4066 | 0.0124 |
| transformer | loss_trend | 0.5263 | 0.5539 | 0.0276 | 0.3942 | 0.4120 | 0.0178 |
| transformer | velocity | 0.5379 | 0.5445 | 0.0066 | 0.3825 | 0.3940 | 0.0114 |
| pooled | instability | 0.4095 | 0.4465 | 0.0370 | 0.4294 | 0.4613 | 0.0319 |
| pooled | loss_trend | 0.4095 | 0.4557 | 0.0462 | 0.4294 | 0.4672 | 0.0378 |
| pooled | velocity | 0.4276 | 0.4868 | 0.0592 | 0.4626 | 0.5117 | 0.0491 |

## 4. Kiểm tra (c) — Kiểm định trực tiếp hiện tượng "Quay về trung bình" (`dev_ma3 = loss_ratio - ma3`)

| scope | beta_dev_ma3 | delta_r2_dev_ma3 | variable | delta_r2_given_dev_ma3 | beta_given_dev_ma3 |
| --- | --- | --- | --- | --- | --- |
| gru | 0.2600 | 0.0201 | instability | 0.0646 | 1.1833 |
| gru | 0.2600 | 0.0201 | loss_trend | 0.0607 | -1.3559 |
| gru | 0.2600 | 0.0201 | velocity | 0.0477 | -0.4305 |
| patchtst | -0.1648 | 0.0084 | instability | 0.0000 | 0.0161 |
| patchtst | -0.1648 | 0.0084 | loss_trend | 0.0004 | 0.0615 |
| patchtst | -0.1648 | 0.0084 | velocity | 0.0111 | 0.1912 |
| tcn | 0.3881 | 0.0394 | instability | 0.0038 | 0.2770 |
| tcn | 0.3881 | 0.0394 | loss_trend | 0.0009 | -0.1437 |
| tcn | 0.3881 | 0.0394 | velocity | 0.0288 | -0.3942 |
| transformer | 0.1760 | 0.0088 | instability | 0.0255 | 0.7165 |
| transformer | 0.1760 | 0.0088 | loss_trend | 0.0168 | -0.6422 |
| transformer | 0.1760 | 0.0088 | velocity | 0.0032 | 0.1085 |
| pooled | 0.2083 | 0.0128 | instability | 0.1315 | 2.3212 |
| pooled | 0.2083 | 0.0128 | loss_trend | 0.1306 | -2.8110 |
| pooled | 0.2083 | 0.0128 | velocity | 0.0000 | 0.0050 |

## 5. Hệ số xuất ra trong `predictive_coef.json` (Bản GRU học trên Seed 3–4, dùng cho Seed 0–2)

```json
{
  "target": "delta_loss_k3",
  "features": [
    "loss_ratio",
    "instability"
  ],
  "standardize": {
    "loss_ratio": {
      "mean": 0.26393591904045816,
      "std": 0.13999486324882152
    },
    "instability": {
      "mean": 0.022948233657642838,
      "std": 0.040387988647263465
    }
  },
  "intercept": -0.04748577334693152,
  "coef": {
    "loss_ratio": -0.01763983244822315,
    "instability": -0.015401104179535009
  },
  "fit_scope": "gru",
  "fit_seeds": [
    3,
    4
  ],
  "controls_epoch": true
}
```
