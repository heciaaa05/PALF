# Sổ đăng ký các lần chạy thực nghiệm (RUNS Registry)

Tài liệu này ghi nhận toàn bộ các lần chạy thực nghiệm chính thức (30 epoch, toàn bộ tập Jena Climate: `n_train = 48,981`, `n_val = 10,492`, `n_test = 10,493`, `batch = 128`, `lr = 1e-3`, `device = cpu`) dùng trong báo cáo và giao diện demo.

> **Quy tắc tách biệt:** Các lần chạy thử nghiệm thu nhỏ (`--subset` hoặc chạy trực tiếp trên giao diện Streamlit) được lưu riêng trong `experiments/results/smoke/` và **không** được liệt kê hay trộn vào bảng so sánh chính thức dưới đây.

---

## 1. Cấu hình thực nghiệm chuẩn (Spec Phase 5–6 v1, Mục 7)

| Mã cấu hình | Chế độ (`mode`) | Chính sách (`policy`) | Ngân sách (`budget_ratio`) | Tham số `(alpha, w_max)` | Thư mục lưu kết quả |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`A0`** | `baseline` | `uniform` (DataLoader chuẩn) | `1.0` | `-` | `experiments/results/baseline/jena/<model>/seed<s>/` |
| **`A0r`** | `palf` | `uniform` (qua `AdaptiveScheduler`) | `0.7` | `-` | `experiments/results/palf/jena/uniform_b0.7/<model>/seed<s>/` |
| **`A1`** | `palf` | `loss` (chỉ dùng `loss_ratio`, B2) | `1.0` | `(1.0, 3.0)` | `experiments/results/palf/jena/loss_b1.0_a1.0/<model>/seed<s>/` |
| **`A3`** | `palf` | `predictive` (Learning State v1) | `1.0` | `(0.5, 2.0)` (từ `tuning_val.md`) | `experiments/results/palf/jena/predictive_b1.0_a0.5_w2/<model>/seed<s>/` |
| **`A3r`** | `palf` | `predictive` (Learning State v1) | `0.7` | `(0.5, 2.0)` (từ `tuning_val.md`) | `experiments/results/palf/jena/predictive_b0.7_a0.5_w2/<model>/seed<s>/` |

---

## 2. Danh sách các lần chạy Baseline (`A0`, 30 epoch, đầy đủ dữ liệu)

| Mã | Backbone | Seeds | Số mẫu (`train / val / test`) | Epoch | Lệnh chạy tái lập | Đường dẫn kết quả |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `A0` | `gru` | `0, 1, 2, 3, 4` | `48981 / 10492 / 10493` | `30` | `python experiments/scripts/run_observation.py --model gru --mode baseline --seeds 0,1,2,3,4 --epochs 30` | `experiments/results/baseline/jena/gru/seed{0..4}/` |
| `A0` | `tcn` | `0, 1, 2, 3, 4` | `48981 / 10492 / 10493` | `30` | `python experiments/scripts/run_observation.py --model tcn --mode baseline --seeds 0,1,2,3,4 --epochs 30` | `experiments/results/baseline/jena/tcn/seed{0..4}/` |
| `A0` | `transformer` | `0, 1, 2, 3, 4` | `48981 / 10492 / 10493` | `30` | `python experiments/scripts/run_observation.py --model transformer --mode baseline --seeds 0,1,2,3,4 --epochs 30` | `experiments/results/baseline/jena/transformer/seed{0..4}/` |
| `A0` | `patchtst` | `0, 1, 2, 3, 4` | `48981 / 10492 / 10493` | `30` | `python experiments/scripts/run_observation.py --model patchtst --mode baseline --seeds 0,1,2,3,4 --epochs 30` | `experiments/results/baseline/jena/patchtst/seed{0..4}/` |

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

## 4. Danh sách các lần chạy PALF chính thức trên GRU (`seeds 0, 1, 2`, 30 epoch, Phần 5.2)

| Mã | Cấu hình (`config_tag`) | Backbone | Seeds | Số mẫu (`train / val / test`) | Epoch | Lệnh chạy tái lập | Đường dẫn kết quả |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `A0r` | `uniform_b0.7` | `gru` | `0, 1, 2` | `48981 / 10492 / 10493` | `30` | `python experiments/scripts/run_observation.py --model gru --mode palf --policy uniform --budget-ratio 0.7 --seeds 0,1,2 --epochs 30 --no-save-sample-losses` | `experiments/results/palf/jena/uniform_b0.7/gru/seed{0..2}/` |
| `A1` | `loss_b1.0_a1.0` | `gru` | `0, 1, 2` | `48981 / 10492 / 10493` | `30` | `python experiments/scripts/run_observation.py --model gru --mode palf --policy loss --budget-ratio 1.0 --alpha 1.0 --w-max 3.0 --seeds 0,1,2 --epochs 30 --no-save-sample-losses` | `experiments/results/palf/jena/loss_b1.0_a1.0/gru/seed{0..2}/` |
| `A3` | `predictive_b1.0_a0.5_w2` | `gru` | `0, 1, 2` | `48981 / 10492 / 10493` | `30` | `python experiments/scripts/run_observation.py --model gru --mode palf --policy predictive --predictive-coef experiments/analysis/phase3_real/predictive_coef_gru_seeds34.json --budget-ratio 1.0 --alpha 0.5 --w-max 2.0 --seeds 0,1,2 --epochs 30 --no-save-sample-losses` | `experiments/results/palf/jena/predictive_b1.0_a0.5_w2/gru/seed{0..2}/` |
| `A3r` | `predictive_b0.7_a0.5_w2` | `gru` | `0, 1, 2` | `48981 / 10492 / 10493` | `30` | `python experiments/scripts/run_observation.py --model gru --mode palf --policy predictive --predictive-coef experiments/analysis/phase3_real/predictive_coef_gru_seeds34.json --budget-ratio 0.7 --alpha 0.5 --w-max 2.0 --seeds 0,1,2 --epochs 30 --no-save-sample-losses` | `experiments/results/palf/jena/predictive_b0.7_a0.5_w2/gru/seed{0..2}/` |
