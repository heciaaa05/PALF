# PALF — Tờ ôn 1 trang cho buổi Demo & Bảo vệ Tiến độ (30/11)

*Mỗi thành viên nắm chắc 5 câu hỏi cốt lõi dưới đây bằng lời của mình và 8 câu hỏi phản biện nhanh.*

---

## I. Năm câu hỏi cốt lõi (Mục 11.6 — Ai cũng phải trả lời được)

### 1. Pattern là gì, khác patch của PatchTST thế nào?
- **Pattern** trong PALF là **nhãn ngữ nghĩa vật lý** gán cho **toàn bộ cửa sổ đầu vào 96 giờ** ($L=96$) dựa trên luật khí tượng khớp ngưỡng phân vị từ tập `train`: `pressure_drop` (9,0% test), `humidity_net_rise` (7,5%), `temp_anomaly` (16,7%), `normal` (66,9%). Chỉ nhìn đầu vào $X$, tuyệt đối không nhìn phần dự báo $Y$ ($H=24$ giờ).
- **Khác PatchTST:** Trong PatchTST, *patch* là một đoạn con cắt ngắn cố định để làm token đầu vào cho kiến trúc Transformer (đơn vị kiến trúc, không mang ngữ nghĩa). Còn *Pattern* là nhãn ngữ nghĩa của cả cửa sổ 96 giờ, độc lập với kiến trúc mô hình đích.

### 2. Learning State và Learning Dynamics là gì?
- **Learning State** là vector trạng thái học của một Pattern **tại một thời điểm (1 epoch)**, ví dụ tỷ lệ lỗi hiện tại so với epoch đầu `loss_ratio(p, e) = L(p, e) / L(p, 1)`.
- **Learning Dynamics** là **sự biến thiên của trạng thái đó theo thời gian (qua chuỗi epoch)**, gồm tốc độ giảm lỗi `velocity`, xu hướng `loss_trend` và độ dao động `instability` trong cửa sổ trượt $W=3$ epoch. PALF dựa vào Learning Dynamics để biết một Pattern đang tiến bộ hay đang chững lại.

### 3. Vì sao giao thức so sánh của nhóm là công bằng (fair comparison)?
1. **Không rò rỉ dữ liệu:** Chia theo thời gian `70/15/15` (`48.981 / 10.377 / 10.377` cửa sổ) có khoảng đệm `purge_gap = 119` bước; mọi ngưỡng phân vị, đường cơ sở khí hậu (`climatology`) và chuẩn hóa chỉ tính trên `train`.
2. **Cách ly tập đánh giá:** `PolicyEngine` chỉ nhận thống kê từ tập `train` (có unit test chặn mọi khóa `val`/`test`); `val` chỉ dùng để chọn checkpoint tốt nhất; `test` chỉ đo đúng 1 lần ở cuối.
3. **Hệ đối chứng đầy đủ:**
   - `A0` (`DataLoader` chuẩn) và `A0s` (lấy mẫu đều đi qua `AdaptiveScheduler` ở ngân sách `1.0`) để kiểm soát đường mã.
   - `A0r` (lấy mẫu đều ở ngân sách `0.7`) để kiểm soát hiệu ứng giảm số mẫu duy nhất mỗi epoch.
   - `A0w` (trọng số từ `loss` nhưng hoán vị ngẫu nhiên giữa 4 Pattern mỗi epoch) để kiểm soát việc *"đổi cơ cấu mẫu nói chung nhưng mất thông tin Pattern"*.
   - `A1` (chỉ dùng `loss_ratio`) để đối chứng với `A3` (`Learning State` đa biến).
4. **Kiểm định thống kê tiền đăng ký:** 10 seed (`0..9`) trên GRU, kiểm định `t` ghép cặp hai phía + `Wilcoxon`, hiệu chỉnh đa so sánh **Bonferroni** cho 3 phép so sánh chính ($\alpha = 0,05 / 3 = 0,0167$).

### 4. Kết quả nói gì và KHÔNG nói gì?
- **Kết quả NÓI GÌ:**
  - *Ở tầng quan sát (Phase 3 / H1 ủng hộ một phần):* Sau khi kiểm soát hiệu ứng epoch, các biến `Learning Dynamics` dự báo được mức thay đổi loss ở 3 epoch tiếp theo trên **GRU ($\Delta R^2 = 0,07\text{–}0,15$)** và **TCN ($\Delta R^2 = 0,07\text{–}0,09$)**, lặp lại dương khi kiểm chéo giữa các seed; nhưng yếu trên Transformer (`0,01–0,02`) và gần `0` trên PatchTST (`≈ 0,005`).
  - *Ở tầng huấn luyện (Phase 5–6 / 10 seed GRU):* **"Trong thiết lập này (Jena, GRU, 10 seed), chưa có bằng chứng PALF cải thiện so với lấy mẫu đều; Learning State đa biến không hơn loss đơn thuần."** Cụ thể: `C1 (A1 - A0s) = -0,0067 ± 0,0191 °C` (`p_t = 0,2974`); `C2 (A3 - A1) = +0,0075 ± 0,0192 °C` (`p_t = 0,2500`); `C3 (A1 - A0w) = -0,0027 ± 0,0214 °C` (`p_t = 0,6983`) — cả 3 đều không đạt ngưỡng Bonferroni `0,0167`.
- **Kết quả KHÔNG NÓI GÌ (Tuyệt đối không phát biểu):**
  - Không nói *"PALF tốt hơn baseline"* (mọi chênh lệch nhỏ đều nằm trong phạm vi biến động giữa các seed `0,02–0,05 °C`).
  - Không nói *"PALF đã chạy trên cả 4 kiến trúc"* (Baseline `A0` chạy 4 kiến trúc, còn điều phối PALF mới chạy trên GRU 10 seed và TCN 3 seed).
  - Không nói *"Learning State dự đoán được mô hình cần học gì trên mọi kiến trúc"* (vì G3 không đạt ngưỡng $\ge 3/4$ backbone, nên `A3` luôn ghi rõ là `predictive - không đạt ngưỡng G3`).

### 5. Hạn chế lớn nhất và hướng tiếp theo là gì?
- **3 nguyên nhân kỹ thuật / hạn chế chính quan sát được từ dữ liệu:**
  1. *Cửa sổ tác động hẹp (`1–3 epoch`):* Mô hình hội tụ rất sớm (`best_epoch = 5,3–7,6` trên GRU; `epoch 3` trên TCN ngay khi vừa hết `warmup_epochs = 3`) rồi bắt đầu overfit.
  2. *Nhóm `temp_anomaly` có sai số cao nhất (`~1,96–1,97 °C`) và không giảm ở cấu hình nào (kể cả `E2` ưu tiên loss tuyệt đối: `1,991 °C`):* Do gộp chung dị thường nóng (`MAE ~1,88–1,91 °C`) và dị thường lạnh (`MAE ~2,42–2,52 °C`) kèm lệch mùa giữa train và test.
  3. *Một bộ dữ liệu & một hành động duy nhất:* Mới thử trên Jena với 4 nhãn luật và chỉ đổi tỷ trọng lấy mẫu.
- **Hướng tiếp theo có cơ sở thực nghiệm:** (1) Tách riêng `temp_anomaly_heat` và `temp_anomaly_cold` và thử trên dữ liệu có sự kiện cực đoan khác biệt mạnh hơn; (2) Dùng lịch huấn luyện có cửa sổ học dài hơn (giảm `lr`, thêm điều chuẩn) và chạy PALF trên Transformer/PatchTST; (3) Mở rộng không gian hành động (`Replay/Skip` hoặc điều chỉnh hàm mục tiêu).

---

## II. Bảng tra nhanh số liệu chính (GRU, 10 Seed `0..9`)

- **MAE tổng thể (°C):** `A0 = 1,7701 ± 0,0240` · `A0s = 1,7685 ± 0,0190` · `A0r = 1,7646 ± 0,0109` · `A0w = 1,7646 ± 0,0291` · `A1 = 1,7618 ± 0,0164` · `A3 = 1,7693 ± 0,0185`.
- **3 đối chiếu chính (ngưỡng $\alpha = 0,0167$):**
  - `C1 (A1 − A0s)`: `-0,0067 ± 0,0191 °C` (`6/10` seed âm, `p_t = 0,2974`, Wilcoxon `p = 0,4316`) $\rightarrow$ **Không đạt**.
  - `C2 (A3 − A1)`: `+0,0075 ± 0,0192 °C` (`5/10` seed âm, `p_t = 0,2500`, Wilcoxon `p = 0,4316`) $\rightarrow$ **Không đạt**.
  - `C3 (A1 − A0w)`: `-0,0027 ± 0,0214 °C` (`5/10` seed âm, `p_t = 0,6983`, Wilcoxon `p = 1,0000`) $\rightarrow$ **Không đạt**.
- **Vì sao 5 seed đầu khác 10 seed?** Ở 5 seed đầu (`0..4`), `A1, A3, A0r` thấp hơn `A0s` khoảng `0,015–0,021 °C` ở `5/5` seed; khi mở rộng lên 10 seed (`0..9`), hiệu ứng biến mất — minh chứng điển hình cho biến động mẫu nhỏ ($n=5$).
