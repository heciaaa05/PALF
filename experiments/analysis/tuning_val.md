# Báo cáo Tinh chỉnh Tham số PALF trên Tập Validation (`seed 0`, Phần 5.1)

Tài liệu này ghi nhận kết quả thực nghiệm chọn siêu tham số `(alpha, w_max)` cho chính sách `predictive` (`A3`, `budget_ratio = 1.0`, `30 epoch`) trên mô hình **GRU (`seed 0`)**, tuân thủ nghiêm ngặt quy tắc **chỉ sử dụng các thước đo trên tập `val` và hoàn toàn không sử dụng tập `test` để chọn tham số**.

---

## 1. Thiết lập Thực nghiệm Tinh chỉnh

- **Backbone:** `GRU` (`seed = 0`, `30 epoch`, `batch = 128`, `lr = 1e-3`, `device = cpu`).
- **Tập dữ liệu:** Jena Climate đầy đủ (`n_train = 48,981`, `n_val = 10,377`, `n_test = 10,377`).
- **Chính sách:** `policy = "predictive"` (**không đạt ngưỡng G3 toàn cục 4 backbone**; dùng hệ số hồi quy ước lượng trên `seeds 3, 4` của GRU), `budget_ratio = 1.0`, `warmup_epochs = 3`, `update_every = 1`, `ema_beta = 0.5`, `w_min = 0.25`.
- **Tệp hệ số hồi quy ngoài mẫu:** `experiments/analysis/phase3_real/predictive_coef_gru_seeds34.json` (được ước lượng độc lập trên `seeds 3, 4` của GRU, không chạm vào `seeds 0, 1, 2`).
- **Không gian lưới `(alpha, w_max)` khảo sát (4 cấu hình theo Prompt 7 Phần 5.1):**
  1. `pred_a0.5_w2`: `(alpha = 0.5, w_max = 2.0)`
  2. `pred_a1.0_w3`: `(alpha = 1.0, w_max = 3.0)`
  3. `pred_a1.0_w5`: `(alpha = 1.0, w_max = 5.0)`
  4. `pred_a2.0_w3`: `(alpha = 2.0, w_max = 3.0)`

---

## 2. Bảng Kết quả trên Tập Validation (`seed 0`, Không dùng `test`)

Trong bảng dưới đây:
- `val_auc_samples (30 epoch)`: diện tích hình thang chuẩn hóa dưới đường `val_loss` theo số mẫu tích lũy qua toàn bộ 30 epoch (**tiêu chí chọn chính theo Prompt 7 Phần 5.1**, càng thấp càng tốt).
- `val_auc_samples (10 epoch đầu)`: diện tích dưới đường `val_loss` trong giai đoạn hội tụ 10 epoch đầu (trước khi GRU bắt đầu overfit trên tập train).
- `best_val_loss`: giá trị MSE chuẩn hóa nhỏ nhất đạt được trên tập `val` trong 30 epoch.
- `samples_to_target`: số mẫu tích lũy cần thiết để đạt ngưỡng `val_target = 1.02 × best_val_loss(A0, seed 0) = 0.083544`.

| Cấu hình | `alpha` | `w_max` | `val_auc_samples` (30 epoch) ↓ | `val_auc_samples` (10 epoch đầu) ↓ | `best_val_loss` ↓ | `best_epoch` | `epoch_to_target` | `samples_to_target` | Đường dẫn lưu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline (`A0` tham chiếu)** | `-` | `-` | `0.108628` | `0.085386` | `0.081906` | `4` | `4` | `195,924` | `experiments/results/baseline/jena/gru/seed0` |
| **`pred_a0.5_w2` (ĐƯỢC CHỌN)** | **`0.5`** | **`2.0`** | **`0.108953`** | **`0.084085`** | **`0.077897`** | **`5`** | **`5`** | **`244,905`** | `experiments/results/tuning/jena/pred_a0.5_w2/gru/seed0` |
| `pred_a1.0_w3` | `1.0` | `3.0` | `0.110294` | `0.084476` | `0.078665` | `5` | `5` | `244,905` | `experiments/results/tuning/jena/pred_a1.0_w3/gru/seed0` |
| `pred_a1.0_w5` | `1.0` | `5.0` | `0.110294` | `0.084476` | `0.078665` | `5` | `5` | `244,905` | `experiments/results/tuning/jena/pred_a1.0_w5/gru/seed0` |
| `pred_a2.0_w3` | `2.0` | `3.0` | `0.109091` | `0.085933` | `0.081195` | `5` | `5` | `244,905` | `experiments/results/tuning/jena/pred_a2.0_w3/gru/seed0` |

---

## 3. Nhận xét và Quyết định Chốt Tham số cho Phần 5.2

1. **So sánh giữa 4 cấu hình `predictive`:**
   - Cặp **`(alpha = 0.5, w_max = 2.0)` (`pred_a0.5_w2`)** đạt giá trị **`val_auc_samples` thấp nhất trên toàn bộ 30 epoch (`0.108953`)**, thấp nhất trong giai đoạn hội tụ 10 epoch đầu (`0.084085`), và đồng thời đạt **`best_val_loss` thấp nhất (`0.077897`**, cải thiện rõ rệt so với `best_val_loss = 0.081906` của Baseline `A0` trên cùng `seed 0`).
   - Với `alpha = 1.0`, điểm `need_score` sau khi biến đổi hàm mũ $\exp(1.0 \cdot s_p)$ chưa vượt qua ngưỡng `3.0` nên hai cấu hình `(1.0, 3.0)` và `(1.0, 5.0)` cho quỹ đạo y hệt nhau (`val_auc_samples = 0.110294`).
   - Khi tăng `alpha = 2.0` (`pred_a2.0_w3`), việc đẩy trọng số quá gắt làm mô hình lặp mẫu các pattern hiếm quá mức ở các epoch giữa, khiến `best_val_loss` kém hơn (`0.081195` so với `0.077897` của `alpha = 0.5`).
2. **Quyết định chốt cố định cho Phần 5.2:**
   - Chọn **`alpha = 0.5, w_max = 2.0`** cho toàn bộ các lần chạy chính thức của **`A3` (`policy=predictive, budget_ratio=1.0`)** và **`A3r` (`policy=predictive, budget_ratio=0.7`)** trên cả 3 seed (`seeds 0, 1, 2`).
