# Sổ đăng ký các lần chạy thực nghiệm (RUNS Registry)

Tài liệu này ghi nhận toàn bộ các lần chạy thực nghiệm chính thức (toàn bộ tập Jena Climate: `n_train = 48,981`, `n_val = 10,377`, `n_test = 10,377`, `batch = 128`, `lr = 1e-3`, `device = cpu`) dùng trong báo cáo và giao diện demo.

> **Quy tắc tách biệt & định danh G3:**
> 1. Các lần chạy thử nghiệm thu nhỏ (`--subset` hoặc chạy trực tiếp trên giao diện Streamlit) được lưu riêng trong `experiments/results/smoke/` và **không** được liệt kê hay trộn vào bảng so sánh chính thức dưới đây.
> 2. **Ghi chú bắt buộc về G3:** Tiêu chí G3 đã đặt trước cho kết quả **"lui về phương án dự phòng"** (chỉ 2/4 backbone đạt $\Delta R^2 \ge 0.02$ sau kiểm soát epoch và dấu $\beta$ không cùng chiều ở PatchTST). Do đó, trong mọi bảng báo cáo và demo, cấu hình **A3** và **A3r** phải được gọi đúng tên là **`predictive (không đạt ngưỡng G3)`**.

---

## 1. Cấu hình thực nghiệm chuẩn (Spec Phase 5–6 v1, Mục 7 & Prompt 8)

| Mã cấu hình | Chế độ (`mode`) | Chính sách (`policy`) | Ngân sách (`budget_ratio`) | Tham số `(alpha, w_max)` | Thư mục lưu kết quả |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`A0`** | `baseline` | `uniform` (DataLoader chuẩn) | `1.0` | `-` | `experiments/results/baseline/jena/<model>/seed<s>/` |
| **`A0s`** | `palf` | `uniform` (qua `AdaptiveScheduler`) | `1.0` | `-` | `experiments/results/palf/jena/uniform_b1.0/<model>/seed<s>/` |
| **`A0r`** | `palf` | `uniform` (qua `AdaptiveScheduler`) | `0.7` | `-` | `experiments/results/palf/jena/uniform_b0.7/<model>/seed<s>/` |
| **`A1`** | `palf` | `loss` (chỉ dùng `loss_ratio`, B2) | `1.0` | `(1.0, 3.0)` | `experiments/results/palf/jena/loss_b1.0_a1.0/<model>/seed<s>/` |
| **`A3`** | `palf` | `predictive (không đạt ngưỡng G3)` | `1.0` | `(0.5, 2.0)` (từ `tuning_val.md`) | `experiments/results/palf/jena/predictive_b1.0_a0.5_w2/<model>/seed<s>/` |
| **`A3r`** | `palf` | `predictive (không đạt ngưỡng G3)` | `0.7` | `(0.5, 2.0)` (từ `tuning_val.md`) | `experiments/results/palf/jena/predictive_b0.7_a0.5_w2/<model>/seed<s>/` |
| **`E1`** *(Khám phá)* | `palf` | `predictive (không đạt ngưỡng G3, warmup=1)` | `1.0` | `(0.5, 2.0)` | `experiments/results/palf/jena/e1_warmup1_b1.0_a0.5_w2/<model>/seed<s>/` |
| **`E2`** *(Khám phá)* | `palf` | `loss_abs` (loss tuyệt đối theo Pattern) | `1.0` | `(0.5, 2.0)` | `experiments/results/palf/jena/loss_abs_b1.0_a0.5_w2/<model>/seed<s>/` |

---

## 2. Danh sách các lần chạy Baseline (`A0`, 30 epoch, đầy đủ dữ liệu)

| Mã | Backbone | Seeds | Số mẫu (`train / val / test`) | Epoch | Lệnh chạy tái lập | Đường dẫn kết quả |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `A0` | `gru` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `30` | `python experiments/scripts/run_observation.py --model gru --mode baseline --seeds 0,1,2,3,4 --epochs 30` | `experiments/results/baseline/jena/gru/seed{0..4}/` |
| `A0` | `tcn` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `30` | `python experiments/scripts/run_observation.py --model tcn --mode baseline --seeds 0,1,2,3,4 --epochs 30` | `experiments/results/baseline/jena/tcn/seed{0..4}/` |
| `A0` | `transformer` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `30` | `python experiments/scripts/run_observation.py --model transformer --mode baseline --seeds 0,1,2,3,4 --epochs 30` | `experiments/results/baseline/jena/transformer/seed{0..4}/` |
| `A0` | `patchtst` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `30` | `python experiments/scripts/run_observation.py --model patchtst --mode baseline --seeds 0,1,2,3,4 --epochs 30` | `experiments/results/baseline/jena/patchtst/seed{0..4}/` |

---

## 3. Các lần chạy Tinh chỉnh tham số trên tập Validation (`seed 0`, Phần 5.1)

Hệ số hồi quy tuyến tính sử dụng: `experiments/analysis/phase3_real/predictive_coef_gru_seeds34.json` (được ước lượng chỉ trên `seeds 3, 4` của GRU để tránh rò rỉ thông tin sang `seeds 0, 1, 2`).

| Thẻ cấu hình (`config_tag`) | Backbone | Seed | `(alpha, w_max)` | `budget_ratio` | Epoch | Đường dẫn kết quả |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `pred_a0.5_w2` | `gru` | `0` | `(0.5, 2.0)` | `1.0` | `30` | `experiments/results/tuning/jena/pred_a0.5_w2/gru/seed0/` |
| `pred_a1.0_w3` | `gru` | `0` | `(1.0, 3.0)` | `1.0` | `30` | `experiments/results/tuning/jena/pred_a1.0_w3/gru/seed0/` |
| `pred_a1.0_w5` | `gru` | `0` | `(1.0, 5.0)` | `1.0` | `30` | `experiments/results/tuning/jena/pred_a1.0_w5/gru/seed0/` |
| `pred_a2.0_w3` | `gru` | `0` | `(2.0, 3.0)` | `1.0` | `30` | `experiments/results/tuning/jena/pred_a2.0_w3/gru/seed0/` |

---

## 4. Danh sách các lần chạy PALF trên GRU (`seeds 0..4`, Phần 5.2 & Prompt 8.3–8.4)

- Với **`seeds 0, 1, 2`** của `A3` và `A3r`: dùng tệp hệ số ngoài mẫu `predictive_coef_gru_seeds34.json` (`fit_seeds: [3, 4]`, lưu ý `seed 0` đã dùng để chọn `(alpha=0.5, w_max=2.0)` trên `val`).
- Với **`seeds 3, 4`** của `A3` và `A3r` (Mục 8.3): dùng tệp hệ số ngoài mẫu `predictive_coef_gru_seeds012.json` (`fit_seeds: [0, 1, 2]`) để tránh vòng tròn dữ liệu, và chạy với `--patience 5` (sau khi đã chứng minh tương đương bit-level ở Mục 8.2).

| Mã | Cấu hình (`config_tag`) | Backbone | Seeds | Số mẫu (`train / val / test`) | Epoch / Patience | Đường dẫn kết quả |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `A0s` | `uniform_b1.0` | `gru` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `patience=5` | `experiments/results/palf/jena/uniform_b1.0/gru/seed{0..4}/` |
| `A0r` | `uniform_b0.7` | `gru` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `s0..2: 30 ep; s3..4: patience=5` | `experiments/results/palf/jena/uniform_b0.7/gru/seed{0..4}/` |
| `A1` | `loss_b1.0_a1.0` | `gru` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `s0..2: 30 ep; s3..4: patience=5` | `experiments/results/palf/jena/loss_b1.0_a1.0/gru/seed{0..4}/` |
| `A3` | `predictive_b1.0_a0.5_w2` *(không đạt ngưỡng G3)* | `gru` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `s0..2: 30 ep; s3..4: patience=5` | `experiments/results/palf/jena/predictive_b1.0_a0.5_w2/gru/seed{0..4}/` |
| `A3r` | `predictive_b0.7_a0.5_w2` *(không đạt ngưỡng G3)* | `gru` | `0, 1, 2, 3, 4` | `48981 / 10377 / 10377` | `s0..2: 30 ep; s3..4: patience=5` | `experiments/results/palf/jena/predictive_b0.7_a0.5_w2/gru/seed{0..4}/` |
| `E1` | `e1_warmup1_b1.0_a0.5_w2` *(khám phá)* | `gru` | `0, 1, 2` | `48981 / 10377 / 10377` | `patience=5` | `experiments/results/palf/jena/e1_warmup1_b1.0_a0.5_w2/gru/seed{0..2}/` |
| `E2` | `loss_abs_b1.0_a0.5_w2` *(khám phá)* | `gru` | `0, 1, 2` | `48981 / 10377 / 10377` | `patience=5` | `experiments/results/palf/jena/loss_abs_b1.0_a0.5_w2/gru/seed{0..2}/` |
| `A0s` | `uniform_b1.0` | `tcn` | `0, 1, 2` | `48981 / 10377 / 10377` | `patience=5` | `experiments/results/palf/jena/uniform_b1.0/tcn/seed{0..2}/` |
| `A1` | `loss_b1.0_a1.0` | `tcn` | `0, 1, 2` | `48981 / 10377 / 10377` | `patience=5` | `experiments/results/palf/jena/loss_b1.0_a1.0/tcn/seed{0..2}/` |
| `A3` | `predictive_b1.0_a0.5_w2` *(không đạt ngưỡng G3)* | `tcn` | `0, 1, 2` | `48981 / 10377 / 10377` | `patience=5` | `experiments/results/palf/jena/predictive_b1.0_a0.5_w2/tcn/seed{0..2}/` |

---

## 5. Sao lưu kết quả Baseline `A0` ngoài Repo (Prompt 9 — Mục 9.4)

Vì thư mục `experiments/results/baseline/` nằm trong `.gitignore` để giữ kho mã gọn nhẹ, toàn bộ tệp kết quả gốc của `A0` (`run_config.json`, `history.csv`, `epoch_pattern_log.csv`, `test_metrics.json`, không gồm `*.npy` và `cartography.csv`) cho cả **4 backbone (`gru, tcn, transformer, patchtst` × `seeds 0..4` = 20 runs)** đã được đóng gói và kiểm tra tái lập sạch tại:

- **Đường dẫn bản sao lưu cục bộ / phát hành ngoài lịch sử Git:** `C:\Users\Admin\Downloads\PALF_A0_baseline_backup_all4backbones.zip` (`175,047 bytes`, chứa cấu trúc chuẩn `experiments/results/baseline/jena/<model>/seed{0..4}/`).
- **Cách khôi phục trên bản sao sạch của repo:**
  ```powershell
  Expand-Archive -Path C:\Users\Admin\Downloads\PALF_A0_baseline_backup_all4backbones.zip -DestinationPath . -Force
  .\.venv\Scripts\python.exe experiments\scripts\compare_runs.py --models gru --seeds 0,1,2,3,4
  ```
- **Lưu ý (Sao lưu đa điểm):** Bên cạnh bản cục bộ `C:\Users\Admin\Downloads\PALF_A0_baseline_backup_all4backbones.zip`, bản drive được lưu tại "https://drive.google.com/file/d/1QnN84TE57IYcSBL4QaJ-Mzi6tNwJhYp9/view?usp=sharing"
## 6. Kiểm định cuối 10 Seed trên GRU & Đối chứng Trọng số Xáo trộn `A0w` (Prompt 10)
- **`A0w` (`loss_shuffled_b1.0_a1.0`, `seeds 0..9`):**
  ```powershell
  .\.venv\Scripts\python.exe experiments\scripts\run_observation.py --model gru --mode palf --policy loss --budget-ratio 1.0 --alpha 1.0 --w-max 3.0 --shuffle-weights --config-tag loss_shuffled_b1.0_a1.0 --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 30 --patience 5 --no-save-sample-losses
  ```
- **Mở rộng `A0s, A0r, A1, A3` (`seeds 5..9`):** Chạy cùng cờ như Mục 1/Mục 3 với `--seeds 5,6,7,8,9 --patience 5 --no-save-sample-losses`.
- **Tổng hợp 10 seed + C1–C3 (Bonferroni $\alpha = 0.0167$, t ghép cặp & Wilcoxon):**
  ```powershell
  .\.venv\Scripts\python.exe experiments\scripts\compare_runs.py --models gru --seeds 0,1,2,3,4,5,6,7,8,9
  ```
