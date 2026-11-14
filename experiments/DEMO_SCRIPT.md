# Kịch bản Trình diễn (Demo Script) PALF — 5 đến 7 phút

Tài liệu này hướng dẫn trình tự thao tác và lời thuyết minh (5–7 phút) trên giao diện Streamlit (`streamlit run app/app.py`), sử dụng **kết quả thực nghiệm thật 30 epoch** đã lưu sẵn trên đĩa để không phụ thuộc thời gian chờ huấn luyện trực tiếp.

---

## 1. Mở đầu: Bài toán & Động lực (1 phút)

- **Bài toán thực tế:** Dự báo nhiệt độ chuỗi thời gian trên tập dữ liệu khí tượng **Jena Climate** (`48,981` mẫu train, `10,492` mẫu val, `10,493` mẫu test). Dữ liệu không đồng nhất: phần lớn thời gian là trạng thái thời tiết bình thường (`normal`, ~80% dữ liệu), xen kẽ các **sự kiện khí tượng hiếm** nhưng quan trọng như tụt áp suất đột ngột (`pressure_drop`), độ ẩm tăng mạnh (`humidity_net_rise`) hoặc dị thường nhiệt độ (`temp_anomaly`).
- **Hạn chế của huấn luyện tiêu chuẩn (Uniform Sampling):** Mỗi epoch, DataLoader lấy mẫu đồng đều trên toàn bộ tập train. Do mẫu `normal` áp đảo về số lượng, gradient bị chi phối bởi trạng thái bình thường; mô hình hoặc học chậm trên các sự kiện hiếm, hoặc lãng phí ngân sách tính toán lặp đi lặp lại các mẫu dễ đã hội tụ.
- **Câu hỏi của PALF (Pattern-Aware Learning Framework):** Liệu việc theo dõi trạng thái học theo từng nhóm pattern sau mỗi epoch và điều chỉnh lịch trình lấy mẫu (`AdaptiveScheduler` v1 — cho phép lặp mẫu có hoàn lại cho nhóm khó và giảm lấy mẫu nhóm dễ) có giúp mô hình **hội tụ nhanh hơn theo số mẫu xử lý** hoặc **cải thiện sai số trên các pattern sự kiện hiếm** hay không?

---

## 2. Thao tác 1 — So sánh cạnh nhau từ kết quả 30 epoch đã lưu (1.5 phút)

1. Mở giao diện `streamlit run app/app.py`. Mặc định ứng dụng vào **Trang 1: `1. So sánh cạnh nhau (Ngoại tuyến - Kết quả chạy sẵn 30 epoch)`**.
2. Ở bộ lọc bên trái:
   - Chọn **Lọc theo Backbone:** `GRU`.
   - Chọn **Lọc theo Seed:** `0` (hoặc `Tất cả` để chọn nhiều cấu hình trên cùng seed).
3. Ở ô chọn lần chạy, đặt cạnh nhau 3 cấu hình tiêu chuẩn trên cùng `seed 0`:
   - `Baseline | GRU | seed 0` (`A0`: lấy mẫu đều, `budget_ratio = 1.0`).
   - `PALF (predictive_b1.0...) | GRU | seed 0` (`A3`: chính sách hồi quy tuyến tính từ Learning State, `budget_ratio = 1.0`).
   - `PALF (predictive_b0.7...) | GRU | seed 0` (`A3r`: chính sách `predictive` với ngân sách giảm còn `70%` số mẫu mỗi epoch).
4. **Chỉ vào 3 biểu đồ đường `val_loss`:**
   - **Biểu đồ 1 (`val_loss` theo Epoch):** Cho thấy quỹ đạo hội tụ qua 30 epoch và đường ngưỡng mục tiêu màu đỏ `val_target = 1.02 × best_val_loss(A0)`.
   - **Biểu đồ 2 (`val_loss` theo Số mẫu tích lũy):** Nhấn mạnh rằng với `budget_ratio = 0.7` (`A3r` và `A0r`), mỗi epoch chỉ xử lý `34,287` mẫu thay vì `48,981` mẫu, nên đường biểu diễn dịch sang trái — giúp kiểm chứng trực tiếp **tốc độ đạt ngưỡng `val_target` tính theo số mẫu (`samples_to_target`) và diện tích dưới đường cong (`val_auc_samples`)**.
   - **Biểu đồ 3 (`val_loss` theo Thời gian tích lũy):** Cho thấy chi phí thời gian thực tế có tính cả `overhead_sec` của bước cập nhật `Monitor + Tracker + Policy` cuối mỗi epoch (chỉ chiếm `< 0.1%` thời gian mỗi epoch).

---

## 3. Thao tác 2 — Quan sát cơ chế Trọng số & Tỷ lệ mẫu theo Epoch (1.5 phút)

1. Chuyển sang **Trang 2: `2. Quan sát PALF (Trọng số, Tỷ lệ mẫu & Learning State)`**.
2. Chọn lần chạy `PALF (predictive_b1.0...) | GRU | seed 0` (hoặc `loss_b1.0_a1.0`).
3. **Chỉ vào 3 biểu đồ hàng trên:**
   - **Biểu đồ 1 (`weights[p]` theo Epoch):** Trong 3 epoch đầu (`warmup_epochs = 3`), toàn bộ 4 pattern đều giữ trọng số `1.0` để khởi tạo đầy đủ bộ nhớ động học `LearningStateTracker`. Từ **epoch 4 trở đi**, chính sách bắt đầu phân hóa trọng số: các pattern có dự báo tốc độ giảm loss chậm / độ bất ổn cao (như `pressure_drop`, `temp_anomaly`) được đẩy trọng số lên `> 1.0`, trong khi pattern dễ hơn bị giảm trọng số `< 1.0`.
   - **Biểu đồ 2 (`quota_planned` share theo Epoch):** Minh họa bản sửa lỗi cốt lõi của `AdaptiveScheduler` v1 — ở `budget_ratio = 1.0`, tỷ lệ mẫu dự kiến của từng pattern thực sự thay đổi theo `weights[p]` (các pattern có `n_p > N_p` được lấy toàn bộ `N_p` mẫu gốc cộng với `n_p - N_p` mẫu lặp có hoàn lại).
   - **Biểu đồ 3 (`n_seen` share theo Epoch):** Cho thấy tỷ lệ mẫu **duy nhất** được duyệt trong mỗi pattern.
4. **Chỉ vào hàng biểu đồ Learning State phía dưới:** Giải thích ngắn gọn cách `loss_ratio` và `instability` biến thiên theo epoch và tạo ra điểm ưu tiên `need_score`.

---

## 4. Thao tác 3 — Đọc bảng MAE theo Pattern qua các Seed (1.5 phút)

1. Quay lại **Trang 1: `1. So sánh cạnh nhau`**, cuộn xuống phần:
   - **Bảng MAE và RMSE (°C) trên tập Test theo từng Pattern (của các lần chạy đang chọn)**, và
   - **Bảng tổng hợp Test MAE & RMSE (°C) theo Pattern qua các Seed (`Mean ± Std`)**.
2. Đọc trực tiếp con số thật trên màn hình (`Mean ± Std` qua `seeds 0, 1, 2`):
   - So sánh `A0` (`Baseline`), `A1` (`loss_only`, đối chứng chỉ dùng `loss_ratio`), và `A3` (`predictive`, kết hợp `loss_ratio + instability`).
   - So sánh cặp giảm ngân sách `A0r` (`uniform, budget=0.7`) với `A3r` (`predictive, budget=0.7`) để trả lời công bằng câu hỏi: khi cùng cắt 30% mẫu mỗi epoch, việc chọn mẫu có định hướng của PALF có giữ được Test MAE hoặc cải thiện trên các sự kiện hiếm (`pressure_drop`, `humidity_net_rise`, `temp_anomaly`) so với lấy mẫu ngẫu nhiên đồng đều hay không.
   - Chỉ rõ cả những pattern mà PALF thắng lẫn những pattern mà PALF hòa hoặc thua nhẹ (đánh đổi khi rút bớt quota của `normal` để dồn cho các pattern hiếm).

---

## 5. Điều chúng tôi không khẳng định (Giới hạn nghiên cứu trung thực — 1 phút)

Để đảm bảo tính trung thực khoa học, nhóm nêu rõ 4 giới hạn không phóng đại:
1. **Phạm vi dữ liệu và nhãn Pattern:** Toàn bộ kết quả hiện tại chỉ được kiểm chứng trên tập khí tượng **Jena Climate** với bộ nhãn **Pattern Analyzer v1 dựa trên ngưỡng quy tắc (rule-based)** được tính chỉ từ tập `train`. Chúng tôi chưa khẳng định kết quả tự động tổng quát hóa sang miền dữ liệu khác khi chưa có thực nghiệm bổ sung.
2. **Kết quả kiểm định G3 về các biến động học (`g3_followup.md`):**
   - Trước khi kiểm soát hiệu ứng cố định theo epoch (`controls_epoch=False`), các biến đạo hàm (`velocity`, `loss_trend`, `instability`) bị nhiễu bởi nghịch lý Simpson giữa đầu và cuối quá trình huấn luyện ($\Delta R^2 < 0.015$).
   - Sau khi kiểm soát hiệu ứng epoch (`controls_epoch=True`), `instability` đóng góp có ý nghĩa thống kê trên **GRU** ($\Delta R^2 = 0.0684$, $R^2_{\text{oos}}$ chéo seed tăng `+4.3%` đến `+6.9%`) và **TCN** ($\Delta R^2 = 0.0462$), còn `loss_trend` đạt $\Delta R^2 \ge 0.02$ trên `3/4` backbone (`gru`, `tcn`, `transformer`). Tuy nhiên, trên **PatchTST**, các biến động học bậc cao gần như không mang thêm thông tin dự báo ($\Delta R^2 < 0.004$). Do đó, giá trị của biến động học phụ thuộc vào kiến trúc mô hình chứ không phải quy luật phổ quát cho mọi backbone.
3. **Bản chất của chính sách `predictive`:** Trong phiên bản v1, chúng tôi bỏ công thức ghép dấu thủ công (`state` heuristic) vì kiểm định G3 cho thấy dấu giả định ban đầu ngược với dữ liệu thực nghiệm; thay vào đó, chính sách `predictive` dùng hệ số hồi quy tuyến tính đa biến ước lượng ngoài mẫu (`predictive_coef_gru_seeds34.json`, fit trên `seeds 3, 4`, đánh giá độc lập trên `seeds 0, 1, 2`).
4. **Đánh đổi khi tái phân bổ mẫu:** Khi giữ tổng ngân sách cố định (`budget_ratio = 1.0` hoặc `0.7`), việc tăng tần suất lặp mẫu cho các sự kiện hiếm đồng nghĩa với việc giảm số lượt duyệt mẫu `normal`. Mọi cải thiện hoặc suy giảm cục bộ theo từng pattern đều được báo cáo đầy đủ kèm khoảng lệch chuẩn qua các seed.
