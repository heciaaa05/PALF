# Kịch bản Trình diễn (Demo Script) PALF — 5 đến 7 phút

Tài liệu này hướng dẫn trình tự thao tác và lời thuyết minh (5–7 phút) trên giao diện Streamlit (`streamlit run app/app.py`), sử dụng **kết quả thực nghiệm đã lưu sẵn trên đĩa** để không phụ thuộc thời gian chờ huấn luyện trực tiếp.

---

## 1. Mở đầu: Bài toán & Động lực (1 phút)

- **Bài toán thực tế:** Dự báo nhiệt độ chuỗi thời gian trên tập dữ liệu khí tượng **Jena Climate** (`48,981` mẫu train, `10,377` mẫu val, `10,377` mẫu test, có khoảng đệm `purge_gap = 119` bước giữa các tập).
- **Cơ cấu 4 nhóm Pattern theo quy tắc ngưỡng phân vị trên chuỗi đầu vào $X$ (`pattern_labels_test`):**
  - `normal`: **66,9%** (`6,942 / 10,377` mẫu test)
  - `temp_anomaly`: **16,7%** (`1,730 / 10,377` mẫu test; gồm `1,511` mẫu dị thường nóng `heat` và `219` mẫu dị thường lạnh `cold`)
  - `pressure_drop`: **9,0%** (`929 / 10,377` mẫu test)
  - `humidity_net_rise`: **7,5%** (`776 / 10,377` mẫu test)
- **Câu hỏi nghiên cứu của PALF (Pattern-Aware Learning Framework):** Khi huấn luyện đồng đều (*uniform sampling*), nhóm `normal` chiếm hai phần ba số mẫu trong mỗi epoch, trong khi nhóm có sai số tuyệt đối lớn nhất là `temp_anomaly` (`MAE ≈ 1.98 °C`, cao hơn `normal` khoảng `+0.23 °C`). Liệu việc theo dõi trạng thái học theo từng Pattern sau mỗi epoch và điều chỉnh lịch trình lấy mẫu (`AdaptiveScheduler` v1) có thay đổi tốc độ hội tụ theo số mẫu hoặc cải thiện sai số trên các Pattern khó vượt ra ngoài biến động ngẫu nhiên giữa các seed hay không?

---

## 2. Thao tác 1 — So sánh cạnh nhau từ kết quả đã lưu (1.5 phút)

1. Mở giao diện `streamlit run app/app.py`. Mặc định ứng dụng vào **Trang 1: `1. So sánh cạnh nhau (Ngoại tuyến - Kết quả chạy sẵn)`**.
2. Ở bộ lọc bên trái: chọn **Backbone = `GRU`**, **Seed = `0`** (hoặc `Tất cả`).
3. Đặt cạnh nhau các cấu hình tiêu chuẩn:
   - `A0` (`baseline`: lấy mẫu đều, `budget_ratio = 1.0`).
   - `A0r` (`uniform_b0.7`: lấy mẫu đều qua Scheduler, giảm 30% mẫu mỗi epoch, `budget_ratio = 0.7`).
   - `A1` (`loss_b1.0_a1.0`: chỉ dùng `loss_ratio`, `budget_ratio = 1.0`).
   - `A3` (`predictive_b1.0_a0.5_w2`: **chính sách `predictive` (không đạt ngưỡng G3)**, `budget_ratio = 1.0`).
   - `A3r` (`predictive_b0.7_a0.5_w2`: **chính sách `predictive` (không đạt ngưỡng G3)**, `budget_ratio = 0.7`).
4. **Chỉ vào đường `val_loss` theo Epoch và theo Số mẫu tích lũy:**
   - Giải thích rõ: trên GRU, điểm `val_loss` tốt nhất đạt rất sớm ở **epoch 4–6**, sau đó mô hình bắt đầu overfit (`val_loss` tăng từ `~0.080` lên `~0.132` ở epoch 30).
   - Vì tập test được đánh giá tại checkpoint có `val_loss` tốt nhất (`best_epoch`), phép đo hiện tại thực chất phản ánh **1–3 epoch tác động của Policy** ngay sau khi kết thúc giai đoạn khởi động (`warmup_epochs = 3`).
   - *(Lưu ý kỹ thuật: không dùng thời gian chạy `time_sec_to_target` hay `overhead` để so sánh giữa các cấu hình vì thời gian phụ thuộc tải CPU và đường mã lấy mẫu).*

---

## 3. Thao tác 2 — Quan sát Trọng số Pattern kèm Đường `val_loss` và Vạch `best_epoch` (1.5 phút)

1. Chuyển sang **Trang 2: `2. Quan sát PALF (Trọng số, Tỷ lệ mẫu & Learning State)`**.
2. Chọn một lần chạy PALF (ví dụ `A3` hoặc `A1` trên `GRU, seed 0`).
3. **Chỉ vào biểu đồ kết hợp `Trọng số Pattern & Đường val_loss` (có vạch đứng kết thúc warm-up và vạch `best_epoch`):**
   - Trong 3 epoch đầu (`warmup_epochs = 3`, bên trái vạch xanh), mọi Pattern có trọng số `1.0`.
   - Từ epoch 4 (ngay sau vạch warm-up), Policy bắt đầu phân hóa trọng số `weights[p]` và thay đổi tỷ lệ mẫu `quota_planned` (cho phép lặp mẫu có hoàn lại khi $n_p > N_p$). Tuy nhiên, vạch đỏ `best_epoch` xuất hiện ngay ở epoch 5–6, cho thấy khoảng thời gian để Policy phát huy tác dụng trước khi GRU overfit là rất ngắn (1–2 epoch).
   - Chỉ ra hiện tượng quan trọng của biến `loss_ratio` (loss chuẩn hóa về epoch 1): vì `loss_ratio` đo *"đã giảm được bao nhiêu phần so với epoch 1"* chứ không đo *"còn sai số tuyệt đối lớn bao nhiêu"*, nhóm `temp_anomaly` bị hạ trọng số xuống sàn (`0.25`) từ khoảng epoch 10 trở đi dù đây là Pattern có MAE tuyệt đối cao nhất. Phát hiện này dẫn tới hai thí nghiệm khám phá **E1 (`warmup_epochs = 1`)** và **E2 (`policy = loss_abs`)**.

---

## 4. Thao tác 3 — Đọc bảng MAE theo Pattern qua 3 Seed và 5 Seed (1.5 phút)

1. Quay lại **Trang 1: `1. So sánh cạnh nhau`**, cuộn xuống **Bảng tổng hợp Test MAE & RMSE (°C) theo Pattern qua các Seed (`Mean ± Std`)**.
2. Đọc đúng con số thực nghiệm và nêu rõ kết luận trung thực:
   - **Trên 3 seed gốc (`0, 1, 2`):** Mọi cấu hình đều nằm **trong phạm vi biến động giữa các seed**:
     - `A0` (`baseline`): `1.7872 ± 0.0335 °C`
     - `A0r` (`uniform, b=0.7`): `1.7692 ± 0.0111 °C`
     - `A1` (`loss, b=1.0`): `1.7734 ± 0.0246 °C`
     - `A3` (`predictive (không đạt ngưỡng G3), b=1.0`): `1.7704 ± 0.0156 °C` (hiệu số ghép `A3 - A1 = -0.0029 ± 0.0108 °C`)
     - `A3r` (`predictive (không đạt ngưỡng G3), b=0.7`): `1.7701 ± 0.0071 °C`
   - **Trên toàn bộ 5 seed (`0, 1, 2, 3, 4` — Mục 8.3):** Kết luận âm/trung tính được củng cố vững chắc hơn nữa:
     - `A0` (`baseline`): `1.7836 ± 0.0255 °C` (`seeds 1..4`: `1.7872 ± 0.0280 °C`)
     - `A0r` (`uniform, b=0.7`): `1.7678 ± 0.0118 °C` (`seeds 1..4`: `1.7683 ± 0.0135 °C`)
     - `A1` (`loss, b=1.0`): `1.7618 ± 0.0238 °C` (`seeds 1..4`: `1.7621 ± 0.0275 °C`)
     - `A3` (`predictive (không đạt ngưỡng G3), b=1.0`): `1.7682 ± 0.0124 °C` (`seeds 1..4`: `1.7707 ± 0.0128 °C`; hiệu số ghép 5 seed `A3 - A1 = +0.0063 ± 0.0148 °C`)
     - `A3r` (`predictive (không đạt ngưỡng G3), b=0.7`): `1.7699 ± 0.0118 °C` (`seeds 1..4`: `1.7709 ± 0.0134 °C`; hiệu số ghép 5 seed `A3r - A0r = +0.0021 ± 0.0043 °C`)
   - **Hai thí nghiệm khám phá đặt trước (`seeds 0, 1, 2` — Mục 8.4):**
     - `E1` (`warmup_epochs = 1`, `predictive`): `1.7873 ± 0.0247 °C` (hiệu số ghép `E1 - A3 = +0.0169 ± 0.0091 °C`). Can thiệp trọng số ngay từ epoch 2 khi mô hình chưa ổn định làm tăng nhẹ sai số dự báo.
     - `E2` (`policy = loss_abs`, dùng loss tuyệt đối theo Pattern): `1.7756 ± 0.0224 °C` (hiệu số ghép `E2 - A3 = +0.0052 ± 0.0070 °C`, `E2 - A1 = +0.0022 ± 0.0059 °C`).
   - **Điểm then chốt:** Trên cả 3 seed lẫn 5 seed (kể cả khi tách riêng `seeds 1..4` không dùng để chọn siêu tham số), cấu hình đối chứng **`A0r` (chỉ giảm 30% mẫu ngẫu nhiên mỗi epoch, không phân biệt Pattern)** và **`A1` (chỉ dùng `loss_ratio`)** đạt MAE tổng thể tương đương hoặc thấp hơn nhẹ so với `A3` và `A3r` (dù `A3`/`A3r` có MAE thấp hơn trên nhánh phụ `temp_anomaly_cold`: `~2.40 °C` so với `~2.55 °C`). Mọi chênh lệch tổng thể đều **nằm trong phạm vi biến động giữa các seed**.

---

## 5. Điều chúng tôi không khẳng định (Giới hạn nghiên cứu trung thực — 1.5 phút)

1. **Kết quả G3 không đạt ngưỡng đăng ký trước:** Sau khi kiểm soát hiệu ứng cố định theo epoch (`controls_epoch=True`), chỉ có `2/4` backbone (`GRU`, `TCN`) đạt $\Delta R^2 \ge 0.02$ cho `instability`, và dấu $\beta$ đổi chiều ở `PatchTST`. Theo đúng quy tắc đặt trước, kết quả G3 là **"lui về phương án dự phòng"**; cấu hình `A3` chạy thử nghiệm với tên gọi rõ ràng là **`predictive (không đạt ngưỡng G3)`**.
2. **Hiệu ứng làm mượt trong G3 là hiệu ứng cơ học:** Việc làm mượt chuỗi (`ma3`, `ema`) tạo ra $\Delta R^2 = 0.23\text{–}0.43$ là hệ quả cơ học do đặc trưng và mục tiêu cùng tính trên chuỗi trượt chồng lấn, tuyệt đối không được dùng làm bằng chứng ủng hộ giả thuyết.
3. **Hội tụ sớm & hạn chế của `loss_ratio`:** Điểm `val_loss` tốt nhất rơi vào epoch 4–6 ngay sau `warmup_epochs = 3`, khiến Policy chỉ kịp tác động 1–2 epoch trước điểm dừng sớm; đồng thời `loss_ratio` chuẩn hóa theo epoch 1 làm giảm trọng số của `temp_anomaly` dù sai số tuyệt đối của nhóm này còn cao nhất.
4. **Phạm vi dữ liệu:** Kết quả chỉ thực hiện trên bộ dữ liệu Jena Climate với 4 nhãn Pattern phân vị (`normal` 66,9%, `temp_anomaly` 16,7%, `pressure_drop` 9,0%, `humidity_net_rise` 7,5%).
