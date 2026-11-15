# PALF — Kịch bản Trình diễn Thống nhất & Nội dung Slide (Demo 30/11/2026)

> **Tài liệu hợp nhất duy nhất (Prompt 9 — Mục 9.3):** Kết hợp hoàn chỉnh giữa **khung kể chuyện, phân bổ thời gian, nội dung 7 slide & bảng hỏi đáp (`PALF_Kich_ban_Demo_30-11.md`)** với **hướng dẫn thao tác từng màn hình trên ứng dụng Streamlit (`streamlit run app/app.py`)**, đã điền đầy đủ **số liệu thật 5 seed trên GRU (có đối chứng `A0s`)** và **nhân bản 3 seed trên TCN**.
> - **Thời lượng:** `6–7 phút` trình bày + `2–3 phút` hỏi đáp.
> - **Nhánh kết quả chính thức (Nhánh Y):** *"PALF chạy end-to-end, so sánh công bằng; trong điều kiện này chưa tách được khỏi việc chỉ giảm mẫu/đổi đường lấy mẫu"*.

---

## 1. Thông điệp duy nhất cần khán giả mang về

> *Mô hình dự báo chuỗi thời gian thường được huấn luyện bằng cách lấy mẫu ngẫu nhiên, coi mọi đoạn dữ liệu như nhau. PALF thêm một lớp điều phối bên ngoài mô hình: nó theo dõi mô hình đang học từng kiểu đoạn dữ liệu (Pattern) như thế nào và điều chỉnh việc lấy mẫu theo đó, không cần sửa kiến trúc mô hình.*

Đó là điều **chắc chắn** nói được: framework hoạt động, chạy end-to-end trên dữ liệu thật (Jena Climate). *(Lưu ý phạm vi kiến trúc: baseline `A0` đã chạy đủ 4 backbone `GRU, TCN, Transformer, PatchTST`, còn PALF mới chạy thực nghiệm trên **`GRU` và `TCN`** — và trên `TCN`, `best_epoch = 3` trùng đúng điểm kết thúc warm-up ở 2/3 seed nên Policy chưa kịp tác động).* Việc PALF **cải thiện** kết quả dự báo hay không là phần thực nghiệm, được báo cáo trung thực theo đúng quy tắc tiền đăng ký (Prompt 10: kiểm định 10 seed trên GRU kèm đối chứng trọng số xáo trộn `A0w`).

---

## 2. Phân bổ thời gian (6–7 phút trình bày)

| Phút | Nội dung | Ai nói | Màn hình / Slide |
|---|---|---|---|
| `0:00–0:50` | Bài toán và khoảng trống nghiên cứu | Hoàng | **Slide 1–2** |
| `0:50–2:00` | Cách PALF vận hành (một sơ đồ, một vòng lặp) | Hoàng | **Slide 3** |
| `2:00–3:00` | Pattern: 4 nhãn dựa luật trên dữ liệu Jena Climate | Người nói về dữ liệu | **Slide 4** |
| `3:00–5:00` | **Demo ứng dụng Streamlit (`app/app.py`):** chạy PALF, xem trọng số theo Pattern | Người 3 | **Ứng dụng (`Trang 1 -> Trang 2 -> Trang 3`)** *(hoặc Slide dự phòng)* |
| `5:00–6:30` | Kết quả 5 seed GRU (`có A0s`), nhân bản 3 seed TCN & **Điều chúng tôi quan sát được** | Người 4 | **Slide 5–6** |
| `6:30–7:00` | Hạn chế nói thẳng và bước tiếp theo | Hoàng | **Slide 7** |

---

## 3. Lời thoại chi tiết & Thao tác ứng dụng theo từng đoạn

### Đoạn 1 — Bài toán (`0:00–0:50`, 50 giây — Slide 1–2)
- **Lời thoại gợi ý:**
  > *"Khi huấn luyện mô hình dự báo thời tiết, dữ liệu thường được lấy ngẫu nhiên đồng đều: một đoạn thời tiết ổn định (`normal`, chiếm 66,9%) và một đoạn áp suất đang sụt mạnh (`pressure_drop`, 9,0%) hay nhiệt độ dị thường (`temp_anomaly`, 16,7%) được đối xử như nhau. Nhưng mô hình học các kiểu đoạn này không đều: trên tập test Jena Climate, nhóm `temp_anomaly` có sai số MAE cao hơn nhóm `normal` tới hơn `+0,20 °C`. Câu hỏi của đề tài: nếu theo dõi được mô hình đang học từng kiểu đoạn thế nào sau mỗi epoch, ta có điều chỉnh được việc lấy mẫu huấn luyện không? Công trình gần nhất là Dataset Cartography (Swayamdipta et al., 2020), phân tích mẫu khó/dễ sau khi huấn luyện xong. PALF thực hiện theo thời gian thực ngay trong lúc huấn luyện, ở cấp độ nhóm Pattern, và dùng kết quả để điều phối lịch trình lấy mẫu."*

### Đoạn 2 — Vòng lặp PALF (`0:50–2:00`, 70 giây — Slide 3)
- **Chỉ một sơ đồ vòng lặp:**
  `Monitor -> Learning State -> Policy Engine -> Adaptive Scheduler -> Mô hình Backbone -> Loss -> quay lại Monitor`
- **Lời thoại gợi ý:**
  > *"PALF bao quanh mô hình, không nằm bên trong mô hình. Cùng một lớp điều phối chạy được với GRU, TCN, Transformer và PatchTST mà không sửa kiến trúc mạng. Sau giai đoạn khởi động (`warmup_epochs = 3`), ở cuối mỗi epoch, `Monitor` tính 3 biến trạng thái học (`loss_ratio`, `velocity`, `instability`) chỉ từ tập train, `PolicyEngine` tính trọng số ưu tiên bị chặn trong `[w_min, w_max]` và làm mượt EMA, còn `AdaptiveScheduler` phân bổ hạn ngạch mẫu nguyên cho epoch kế tiếp. Hành động duy nhất ở phiên bản v1 này là **đổi tỷ trọng mẫu theo Pattern** (lấy mẫu có hoàn lại khi hạn ngạch vượt số mẫu sẵn có của Pattern, không bỏ mẫu vĩnh viễn)."*

### Đoạn 3 — Bốn nhãn Pattern trên Jena Climate (`2:00–3:00`, 60 giây — Slide 4)
- **Cơ cấu 4 nhãn (`48,981` mẫu train, `10,377` mẫu val, `10,377` mẫu test, khoảng đệm `purge_gap = 119` bước):**
  1. `normal`: **66,9%** (`6,942 / 10,377` mẫu test)
  2. `temp_anomaly`: **16,7%** (`1,730 / 10,377` mẫu test; gồm `1,511` mẫu dị thường nóng `heat` và `219` mẫu dị thường lạnh `cold`)
  3. `pressure_drop`: **9,0%** (`929 / 10,377` mẫu test)
  4. `humidity_net_rise`: **7,5%** (`776 / 10,377` mẫu test)
- **Ba điều bắt buộc nói rõ:**
  - *(i)* Nhãn dựa luật trên **cả cửa sổ đầu vào 96 bước (16 giờ)**, không phải theo từng đoạn patch nhỏ bên trong PatchTST.
  - *(ii)* Mọi ngưỡng phân vị và đường khí hậu (`climatology`) chỉ tính từ tập `train`; bộ gán nhãn chỉ nhìn chuỗi đầu vào $X$, tuyệt đối không nhìn đích dự báo $Y$.
  - *(iii)* Chia dữ liệu theo thời gian (`70/15/15`) nên tập `val/test` lệch mùa so với `train`, làm tỷ lệ Pattern thay đổi theo thời gian (hạn chế thực tế không giấu).

### Đoạn 4 — Demo ứng dụng trực tiếp (`3:00–5:00`, 2 phút — Giao diện `streamlit run app/app.py`)

> **Phương án dự phòng bắt buộc:** Nếu ứng dụng gặp sự cố hiển thị, chuyển ngay sang 3 ảnh chụp màn hình dự phòng đã lưu sẵn tại:
> - [`experiments/analysis/comparison/backup_page1_side_by_side.png`](file:///d:/Github/PALF/experiments/analysis/comparison/backup_page1_side_by_side.png)
> - [`experiments/analysis/comparison/backup_page2_palf_diagnostics.png`](file:///d:/Github/PALF/experiments/analysis/comparison/backup_page2_palf_diagnostics.png)
> - [`experiments/analysis/comparison/backup_page3_live_smoke.png`](file:///d:/Github/PALF/experiments/analysis/comparison/backup_page3_live_smoke.png)

1. **Màn hình 1 (`Trang 1: 1. So sánh cạnh nhau` — 45 giây):**
   - Ở bộ lọc bên trái: chọn **Backbone = `GRU`**, **Seed = `0`** (hoặc `Tất cả`).
   - Đặt cạnh nhau 3 cấu hình cùng seed (`Baseline (A0)`, `PALF (uniform_b1.0 - A0s)` hoặc `PALF (loss_b1.0_a1.0 - A1)`, và `PALF (predictive_b1.0_a0.5_w2 - A3 [không đạt ngưỡng G3])`).
   - **Thứ 1 cần chỉ cho khán giả:** Đường `val_loss` theo **Epoch** và theo **Số mẫu tích lũy**. Chỉ rõ `val_loss` đạt đáy rất sớm ở **epoch 5–6** trên GRU (và **epoch 3** trên TCN), sau đó mô hình bắt đầu overfit.
   - **Thứ 3 cần chỉ cho khán giả:** Cuộn xuống **Bảng tổng hợp toàn cục qua các Seed (`Mean ± Std`)** và **Bảng Test MAE & RMSE (°C) theo Pattern qua các Seed**.

2. **Màn hình 2 (`Trang 2: 2. Quan sát PALF` — 55 giây):**
   - Chọn lần chạy `PALF (predictive_b1.0_a0.5_w2 [predictive - không đạt ngưỡng G3]) | GRU | seed0` (hoặc `loss_abs_b1.0_a0.5_w2`).
   - **Thứ 2 cần chỉ cho khán giả:** Biểu đồ kết hợp **`0. Tương quan giữa Trọng số weights[p], Đường val_loss, mốc hết warmup_epochs và best_epoch`**:
     - Bên trái vạch đứng xanh lá (`Hết warm-up (ep 3)`), cả 4 Pattern đều có trọng số `1.0`.
     - Từ epoch 4, hệ thống bắt đầu phân hóa trọng số `weights[p]` và đổi tỷ lệ mẫu `quota_planned`. Nhưng vạch đứng đỏ (`Best val_loss (ep 5)`) xuất hiện ngay ở epoch 5–6: Policy chỉ có **1–2 epoch** tác động trước khi mô hình đạt checkpoint tốt nhất!
     - Chỉ vào đường trọng số của `temp_anomaly` ở chính sách dùng `loss_ratio`: từ khoảng epoch 10 trở đi, `temp_anomaly` bị hạ trọng số xuống sàn `0.25` vì `loss_ratio` chuẩn hóa về epoch 1 đo *"đã giảm được bao nhiêu phần so với epoch 1"* chứ không đo sai số tuyệt đối còn lại.

3. **Màn hình 3 (`Trang 3: 3. CHẠY THU NHỎ (SMOKE / DEMO TRỰC TIẾP)` — 20 giây nếu còn giờ):**
   - Nhấn mạnh nhãn cảnh báo màu vàng: trang này chỉ chạy thu nhỏ (`500` mẫu, `4` epoch) để minh họa vòng lặp hoạt động trực tiếp trong ~5 giây và lưu tách biệt vào `experiments/results/smoke/`.

### Đoạn 5 — Kết quả thực nghiệm & Điều chúng tôi quan sát được (`5:00–6:30`, 90 giây — Slide 5 & Slide 6)
- Xem nội dung đầy đủ của **Slide 5** và **Slide 6** ở Mục 4 bên dưới.

### Đoạn 6 — Hạn chế nói thẳng (`6:30–7:00`, 30 giây — Slide 7)
- Đọc nguyên văn các ý ở **Slide 7** (Mục 5 bên dưới).

---

## 4. Nội dung Slide 5 & Slide 6 (Bản 1 — Số liệu thật đã đóng băng)

### Slide 5 — Kết quả thực nghiệm: Kiểm định 10 Seed trên GRU (`A0, A0s, A0w, A0r, A1, A3`) & TCN (`3 seed`)

> **Kết luận chính thức sau kiểm định 10 seed trên GRU (chọn đúng theo quy tắc tiền đăng ký Prompt 10 — ngưỡng Bonferroni $\alpha = 0.05/3 = 0.0167$):**
> - **Đối với `C1 (A1 - A0s)` & `C3 (A1 - A0w)` (Phương án 3 — `C1` không đạt ngưỡng $p < 0.0167$):** *"Không có hiệu ứng ổn định của trọng số theo loss ở 10 seed (`C1: -0.0067 ± 0.0191 °C`, `6/10` âm, `p_t = 0.2974`, `p_Bonf = 0.8922`); hiệu ứng ở 5 seed đầu (`-0.0211 °C`) là biến động ngẫu nhiên giữa các seed. Hơn nữa, `A1` gần như trùng khít đối chứng trọng số ngẫu nhiên `A0w` (`C3: -0.0027 ± 0.0214 °C`, `5/10` âm, `p_t = 0.6983`, `p_Bonf = 1.0000`, Wilcoxon `p = 1.0000`)."*
> - **Đối với `C2 (A3 - A1)` (`p = 0.2500 >= 0.0167`, `5/10` âm):** *"Learning State đa biến (`A3`, `+0.0075 ± 0.0192 °C` so với `A1`) không hơn loss đơn thuần."* (Kết quả chính của đề tài, nói thẳng).
>
> **Lời thoại nguyên văn theo Nhánh Y (đã cập nhật 10 seed & phạm vi backbone):**
> *"Trong điều kiện của chúng tôi, lớp điều phối theo Pattern không tạo khác biệt vượt biến động giữa các seed (`10 seed` GRU: `C1: p_t = 0.2974`, `C3: p_t = 0.6983`) và chưa tách được khỏi việc lấy mẫu đều hoặc trọng số ngẫu nhiên. Kết quả này đáng giá vì: (i) framework hoạt động và chạy end-to-end trên các kiến trúc đã thử nghiệm (`GRU` và `TCN`); (ii) chúng tôi đã loại trừ các so sánh không công bằng bằng hệ đối chứng đầy đủ (`A0s` cùng đường mã Scheduler ở ngân sách `1.0`, `A0w` trọng số ngẫu nhiên cùng phân bố, và `A0r` ở ngân sách `0.7`); (iii) với dữ liệu Jena và 4 nhãn Pattern hiện tại, việc dùng Learning State đa biến (`A3`) chưa thêm giá trị dự đoán so với loss đơn thuần (`C2: p_t = 0.2500`). Hướng tiếp theo: áp dụng trên dữ liệu có các sự kiện thời tiết cực đoan khác biệt mạnh hơn và tách riêng dị thường nóng/lạnh."*
>
> **Câu tuyệt đối không nói:** Không nói *"PALF tốt hơn"* khi `C1, C2, C3` đều không đạt ngưỡng Bonferroni; không nói *"PALF hiệu quả trên mọi mô hình"*; không nói *"Learning State dự đoán được mô hình cần học gì"* (vì kiểm định G3 sau kiểm soát epoch chỉ đạt ở 2/4 backbone và lui về phương án dự phòng).

#### Bảng 5A — Kết quả Kiểm định 10 Seed trên Backbone chính GRU (`seeds 0..9`)

| Mã | Cấu hình (`config_tag`) | Cơ chế lấy mẫu | Test MAE (°C) ↓ | Test RMSE (°C) ↓ | Best Val Loss ↓ | Best Epoch | Hiệu MAE vs `A0s` (°C) |
|---|---|---|---:|---:|---:|---:|---:|
| **A0** | `baseline` | `DataLoader(shuffle=True)`, budget `1.0` | `1.7701 ± 0.0240` | `2.3329 ± 0.0265` | `0.0804 ± 0.0014` | `5.6` | — |
| **A0s** | `uniform_b1.0` | `AdaptiveScheduler` đều, budget `1.0` (đối chứng chính) | `1.7685 ± 0.0190` | `2.3340 ± 0.0186` | `0.0807 ± 0.0010` | `5.4` | `0.0000` *(vs `A0`: `-0.0015 ± 0.0143`)* |
| **A0w** | `loss_shuffled_b1.0_a1.0` | Hoán vị ngẫu nhiên trọng số `loss` theo epoch (`b=1.0`) | `1.7646 ± 0.0291` | `2.3309 ± 0.0257` | `0.0808 ± 0.0015` | `5.3` | `-0.0040 ± 0.0243` |
| **A0r** | `uniform_b0.7` | `AdaptiveScheduler` đều, giảm 30% mẫu (`b=0.7`) | `1.7646 ± 0.0109` | `2.3328 ± 0.0109` | `0.0804 ± 0.0007` | `7.6` | `-0.0040 ± 0.0145` |
| **A1** | `loss_b1.0_a1.0` | Chỉ dùng `loss_ratio` (`b=1.0, α=1.0, w_max=3.0`) | `1.7618 ± 0.0164` | `2.3304 ± 0.0221` | `0.0805 ± 0.0015` | `6.0` | `-0.0067 ± 0.0191` |
| **A3** | `predictive_b1.0_a0.5_w2` | `predictive` (không đạt G3, `b=1.0, α=0.5, w_max=2.0`) | `1.7693 ± 0.0185` | `2.3340 ± 0.0182` | `0.0802 ± 0.0015` | `5.4` | `+0.0008 ± 0.0248` |

*(Ghi chú: `A3r` chỉ chạy 5 seed `0..4` ở Prompt 9 với MAE `1.7699 ± 0.0118 °C`, hiệu `A3r - A0r = +0.0021 ± 0.0043 °C`.)*

#### Bảng 5A-bis — 3 Phép so sánh chính (`C1–C3`, 10 seed GRU, Bonferroni $\alpha = 0.05/3 = 0.0167$)

| Mã | Phép so sánh | Mean ± SD (°C) | Dấu (Âm / Dương) | Paired `t` (`p_t`) | `p_Bonf` | Wilcoxon `W` (`p`) | Kết luận |
|---|---|---:|:---:|---:|---:|---:|---|
| **C1** | `A1 - A0s` *(loss vs đều)* | `-0.0067 ± 0.0191` | `6 / 4` | `-1.106` (`0.2974`) | `0.8922` | `19.0` (`0.4316`) | Không đạt |
| **C2** | `A3 - A1` *(Learning State vs loss)* | `+0.0075 ± 0.0192` | `5 / 5` | `+1.230` (`0.2500`) | `0.7501` | `19.0` (`0.4316`) | Không đạt |
| **C3** | `A1 - A0w` *(loss vs trọng số ngẫu nhiên)* | `-0.0027 ± 0.0214` | `5 / 5` | `-0.400` (`0.6983`) | `1.0000` | `27.0` (`1.0000`) | Không đạt |

- **Cách đọc đăng ký trước đối với `A0s` (Mục 9.1):**
  1. **`A0s ≈ A0` (`A0s - A0 = -0.0007 ± 0.0164 °C`, nhỏ hơn nhiều độ lệch chuẩn giữa các seed `0.0161–0.0255 °C`):** Khi lấy đủ 100% mẫu mỗi epoch đúng 1 lần không hoàn lại, đường mã `AdaptiveScheduler` (`A0s`) cho kết quả trùng khít với `DataLoader` chuẩn (`A0`).
  2. **So với `A0s`:** Cả `A1` (`-0.0211 ± 0.0142 °C`), `A3` (`-0.0148 ± 0.0118 °C`) và đối chứng giảm mẫu ngẫu nhiên `A0r` (`-0.0151 ± 0.0079 °C`) đều thấp hơn nhẹ khoảng `0.015–0.021 °C` (ghi nhận là xu hướng với cảnh báo $n=5$ nhỏ và không hiệu chỉnh đa so sánh). Tuy nhiên, **`A3 - A1 = +0.0063 ± 0.0148 °C`**, **`A3 - A0r = +0.0004 °C`** và **`A3r - A0r = +0.0021 ± 0.0043 °C`**: phần chênh lệch này xuất hiện ngay khi giảm số mẫu duy nhất mỗi epoch sau warmup (do giảm ngân sách 30% ở `A0r` hoặc lấy mẫu có hoàn lại `replacement=True` ở `A1/A3`), **chưa tách được khỏi việc chỉ giảm mẫu và không có bằng chứng cho trọng số theo Pattern**.

#### Bảng 5B — Nhân bản trên Backbone thứ hai: TCN (`seeds 0, 1, 2` — dùng hệ số ngoài mẫu `predictive_coef_tcn_seeds34.json`, không tinh chỉnh lại siêu tham số)

| Mã | Cấu hình (`config_tag`) | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Hiệu MAE so với `A0s` (°C) | Hiệu MAE so với `A0` (°C) |
|---|---|---:|---:|---:|---:|---:|---:|
| **A0** | `baseline` | `1.7739 ± 0.0065` | `2.3364 ± 0.0088` | `0.0819 ± 0.0009` | `3.0` | `+0.0080 ± 0.0096` | `0.0000` |
| **A0s** | `uniform_b1.0` | `1.7659 ± 0.0089` | `2.3335 ± 0.0124` | `0.0821 ± 0.0004` | `3.0` | `0.0000` (đối chứng) | `-0.0080 ± 0.0096` |
| **A1** | `loss_b1.0_a1.0` | `1.7714 ± 0.0113` | `2.3441 ± 0.0156` | `0.0820 ± 0.0003` | `3.7` | `+0.0055 ± 0.0096` | `-0.0024 ± 0.0162` |
| **A3** | `predictive_b1.0_a0.5_w2` | `1.7802 ± 0.0241` | `2.3516 ± 0.0267` | `0.0821 ± 0.0003` | `3.7` | `+0.0143 ± 0.0248` | `+0.0063 ± 0.0302` |

- **Nhận xét nhân bản trên TCN:** TCN hội tụ sớm hơn cả GRU (`best_epoch = 3.0–3.7`). Ở `seed 0` và `seed 2`, điểm `val_loss` tốt nhất đạt ngay tại `epoch 3` (cuối `warmup_epochs = 3`), nên `A0s, A1, A3` chọn cùng checkpoint epoch 3 (`hiệu số = 0.0000 °C`); ở `seed 1` (`best_epoch = 5`), `A1` và `A3` có MAE cao hơn nhẹ so với `A0s`. Việc không có cải thiện từ trọng số Pattern lặp lại nhất quán trên kiến trúc thứ hai.

---

### Slide 6 — "Điều chúng tôi quan sát được" (Ba sự thật thực nghiệm có số liệu)

> Slide này trình bày **3 phát hiện thực nghiệm cụ thể từ log huấn luyện** giải thích vì sao điều phối lấy mẫu theo Pattern không tạo ra cải thiện trên bài toán này:

1. **Quan sát 1 — Hội tụ sớm làm cửa sổ tác động của Policy rất ngắn (`1–3 epoch`):**
   - Trên GRU, điểm `val_loss` tốt nhất đạt ở **epoch `5.2–6.4`** (trên TCN là **epoch `3.0–3.7`**), ngay sau khi kết thúc giai đoạn khởi động `warmup_epochs = 3`.
   - Do đó, tại thời điểm lưu checkpoint tốt nhất (`best_epoch`), Policy thực chất mới chỉ can thiệp được **1–3 epoch** trên GRU (và **0–2 epoch** trên TCN). Khi thử giảm `warmup_epochs = 1` ở thí nghiệm khám phá **E1** (`seeds 0, 1, 2`), MAE tăng nhẹ lên **`1.7873 ± 0.0247 °C`** (`E1 - A3 = +0.0169 ± 0.0091 °C`) do trọng số bị biến động khi mô hình chưa ổn định ở epoch 2.

2. **Quan sát 2 — Nhóm `temp_anomaly` có sai số cao nhất và KHÔNG cải thiện ở bất kỳ cấu hình nào (kể cả khi ưu tiên loss tuyệt đối ở E2):**
   - Trên tập test (`1,730` mẫu `temp_anomaly`, gồm `1,511` mẫu nóng và `219` mẫu lạnh), sai số MAE của `temp_anomaly` luôn cao hơn `normal` khoảng `+0.19` đến `+0.25 °C` và **không giảm ở bất kỳ cấu hình nào**:
     - Trên GRU (5 seed): `A0 = 1.9623 °C` · `A0s = 1.9466 °C` · `A0r = 1.9659 °C` · `A1 = 1.9766 °C` · `A3 = 1.9713 °C` · `A3r = 1.9806 °C`.
     - Trên GRU (3 seed khám phá): `E1 (warmup=1) = 1.9923 °C` · **`E2 (loss_abs — ưu tiên trực tiếp loss tuyệt đối) = 1.9911 ± 0.0223 °C`** (`≈ 1,99 °C`).
     - Trên TCN (3 seed): `A0 = 1.9502 °C` · `A0s = 1.9620 °C` · `A1 = 1.9689 °C` · `A3 = 1.9798 °C`.
   - **Ý nghĩa:** Ngay cả khi ép Policy dồn trọng số tối đa (`w_max = 2.0`) vào `temp_anomaly` suốt quá trình học bằng `loss_abs` (**E2**), sai số của `temp_anomaly` vẫn giữ ở `1.99 °C`. Điều này chứng minh **nút cổ chai của `temp_anomaly` không nằm ở tần suất lấy mẫu**, mà do đặc tính bất đối xứng nóng/lạnh (`temp_anomaly_cold` có MAE `2.38–2.56 °C` so với `temp_anomaly_heat` `1.88–1.92 °C`) và lệch phân phối theo mùa giữa tập train và tập test.

3. **Quan sát 3 — Biến `loss_ratio` chuẩn hóa về epoch 1 làm Policy hạ trọng số đúng Pattern khó nhất:**
   - Công thức `loss_ratio(p, e) = L(p, e) / L(p, 1)` đo tốc độ giảm tương đối so với epoch 1 của chính Pattern đó.
   - Vì `temp_anomaly` khởi đầu ở epoch 1 với loss rất lớn (`~0.243`) rồi giảm nhanh về `~0.088` (tỷ lệ `loss_ratio ≈ 0.36`, thấp nhất trong 4 nhóm dù loss tuyệt đối vẫn cao nhất), cả chính sách `loss` (`A1`) lẫn `predictive` (`A3`) đều đánh giá nhầm rằng `temp_anomaly` *"đã học xong"* và hạ trọng số của nó xuống sàn (`w_min = 0.25`) từ khoảng epoch 10 trở đi.

---

## 5. Slide 7 — Hạn chế nói thẳng & Bước tiếp theo (30 giây)

1. **Phạm vi dữ liệu & kiến trúc:** Đánh giá đầy đủ 5 seed trên một bộ dữ liệu (Jena Climate) với backbone chính GRU và nhân bản 3 seed trên backbone thứ hai (TCN).
2. **Cỡ mẫu thống kê:** 5 seed trên GRU và 3 seed trên TCN chỉ đủ để kiểm tra tính nhất quán và mô tả xu hướng, không hiệu chỉnh đa so sánh.
3. **Nhãn Pattern dựa luật cố định:** 4 nhãn dựa luật trên cửa sổ 96 bước; tập `val/test` lệch mùa so với `train` và gộp chung dị thường nóng/lạnh vào một nhãn `temp_anomaly`.
4. **Tín hiệu Learning Dynamics rất nhỏ (G3 lui về phương án dự phòng):** Trước khi kiểm soát epoch, $\Delta R^2 \le 0.06$; sau khi kiểm soát hiệu ứng cố định theo epoch (`controls_epoch=True`), chỉ có `2/4` backbone (`GRU, TCN`) đạt $\Delta R^2 \ge 0.02$ và dấu hệ số $\beta$ đổi chiều ở `PatchTST` (`g3_followup.md`). Hiệu ứng tăng $\Delta R^2$ khi làm mượt chuỗi (`ma3, ema`) là hiệu ứng cơ học của cửa sổ trượt chồng lấn, không dùng làm bằng chứng.
5. **Không gian hành động:** Phiên bản v1 chỉ đổi tỷ trọng lấy mẫu theo nhóm Pattern; chưa tách nhãn dị thường nóng/lạnh, chưa điều chỉnh hàm mục tiêu hoặc trọng số loss theo mẫu.

---

## 6. Bảng Câu hỏi Dự kiến và Câu trả lời Ngắn (Q&A — 2 đến 3 phút)

| Câu hỏi | Trả lời ngắn gọn, chuẩn xác |
|---|---|
| **1. Khác gì Curriculum Learning?** | Curriculum thường sắp xếp thứ tự độ khó cố định đặt trước. PALF quyết định động sau mỗi epoch dựa trên trạng thái học (`Learning State`) đo trực tiếp từ mô hình theo từng nhóm Pattern *(tương đồng hướng tiếp cận learning progress của Graves et al., 2017 nhưng áp dụng ở cấp Pattern chuỗi thời gian)*. |
| **2. Khác gì Dataset Cartography?** | Dataset Cartography (Swayamdipta et al., 2020) phân tích mẫu khó/dễ **sau khi đã huấn luyện xong** toàn bộ các epoch. PALF theo dõi và điều phối lấy mẫu **ngay trong lúc huấn luyện** ở cấp độ nhóm Pattern. |
| **3. Pattern có phải là patch của PatchTST không?** | **Không.** Pattern là nhãn ngữ nghĩa vật lý của **cả cửa sổ đầu vào 96 bước (16 giờ)** theo quy tắc phân vị; còn patch là các đoạn con cắt nhỏ bên trong kiến trúc PatchTST. |
| **4. Vì sao chỉ có 4 nhãn Pattern?** | Được lọc theo tiêu chí tiền đăng ký ở Phase 1–2 trên tập `train` (đủ số sự kiện độc lập $N_{\text{eff}}$, tỷ lệ `normal` hợp lý, ổn định ngưỡng). Nhãn `spike` đã bị loại bỏ từ Phase 1 vì chỉ có 6 sự kiện độc lập, không đủ cỡ mẫu kiểm định. |
| **5. Có rò rỉ dữ liệu (data leakage) không?** | **Không.** Chia theo thời gian `70/15/15` có khoảng đệm `purge_gap = 119` bước; mọi ngưỡng phân vị và `climatology` chỉ tính từ `train`; nhãn Pattern chỉ nhìn đầu vào $X$ (không nhìn $Y$); `PolicyEngine` chỉ nhận thống kê `train` (đã có unit test kiểm chứng tự động). |
| **6. Việc lặp mẫu (`replacement=True`) có làm mô hình học thuộc lòng không?** | Trọng số bị chặn trong `[w_min, w_max]` (`[0.25, 2.0]` ở `A3`), làm mượt EMA ($\gamma=0.3$), và được kiểm chứng công bằng bằng các đối chứng cùng ngân sách mẫu (`A0s` ở ngân sách `1.0`, `A0r` ở ngân sách `0.7`). |
| **7. Vì sao PALF chưa tốt hơn baseline (Nhánh Y)?** | Có 3 lý do thực nghiệm rõ ràng từ Slide 6: (1) Mô hình hội tụ rất sớm (epoch 5–6 trên GRU, epoch 3 trên TCN) nên Policy chỉ kịp tác động 1–3 epoch sau warmup; (2) Nhóm khó nhất `temp_anomaly` (`1.95–1.99 °C`) không giảm sai số ngay cả khi ưu tiên trực tiếp bằng `loss_abs` (`E2`), chứng tỏ lỗi do bất đối xứng nóng/lạnh và lệch mùa chứ không phải do thiếu mẫu; (3) `loss_ratio` chuẩn hóa về epoch 1 làm hạ trọng số của `temp_anomaly`. |
| **8. Chi phí tính toán (Overhead) của PALF là bao nhiêu?** | Phần tính toán `Monitor + PolicyEngine + AdaptiveScheduler` ở cuối mỗi epoch chỉ mất trung bình **`0.004 – 0.006 giây/epoch`** (`4–6 ms/epoch`, nhỏ hơn `0.01%` thời gian một epoch). *(Lưu ý: tổng thời gian mỗi epoch khi đi qua `Subset` phụ thuộc tải CPU và đường đọc chỉ số của PyTorch nên chỉ để ở Phụ lục, không dùng so sánh thuật toán).* |

---

## 7. Danh mục Kiểm tra & Lịch chuẩn bị trước ngày Bảo vệ (30/11)

| Mốc thời gian | Công việc | Trạng thái |
|---|---|---|
| **15–22/11** | Chạy xong `A0s` 5 seed trên GRU (Mục 9.1) và `A0s, A1, A3` 3 seed trên TCN (Mục 9.2) | **Đã hoàn tất** |
| **15–22/11** | Sao lưu `A0` baseline 4 backbone ra ngoài repo & kiểm tra giải nén sạch (Mục 9.4) | **Đã hoàn tất** (`C:\Users\Admin\Downloads\PALF_A0_baseline_backup_all4backbones.zip`) |
| **22–27/11** | Hợp nhất kịch bản duy nhất `experiments/DEMO_SCRIPT.md`, cập nhật bảng 5 seed GRU + 3 seed TCN + Slide 6 ("Điều chúng tôi quan sát được") + chụp ảnh màn hình dự phòng | **Đã hoàn tất** |
| **25/11** | Đóng băng mã nguồn trên nhánh `main` (sau đó chỉ sửa lỗi hiển thị nếu có) | Sẵn sàng |
| **28–29/11** | **Diễn tập có bấm giờ ít nhất 1 lần** (`6–7 phút` nói + `2–3 phút` hỏi đáp); mỗi thành viên nắm vững 3 câu cốt lõi: *(1) Pattern là gì và khác patch ở đâu; (2) Learning State là gì; (3) Vì sao so sánh của chúng tôi công bằng (`A0s`, `A0r`, cùng ngân sách mẫu)* | Lịch diễn tập của nhóm |

---

## 8. Tóm tắt 1 Trang cho Cả Nhóm — Ba Câu Hỏi Bắt Buộc Mọi Thành Viên Phải Trả Lời Được

1. **Câu 1 — Pattern là gì và khác patch của PatchTST ở đâu?**
   - **Pattern** là nhãn ngữ nghĩa vật lý của **toàn bộ một cửa sổ đầu vào 96 bước (16 giờ)** được gán bằng quy tắc ngưỡng phân vị cố định từ tập `train` (`pressure_drop`, `humidity_net_rise`, `temp_anomaly`, `normal`), chỉ nhìn đầu vào $X$ và không nhìn đích $Y$.
   - **Patch** là các đoạn con ngắn (ví dụ 16 bước) được cắt tự động bên trong tầng đầu tiên của riêng mô hình `PatchTST` để tạo token cho Self-Attention. Mọi mô hình (`GRU, TCN, Transformer, PatchTST`) đều dùng chung một bộ nhãn Pattern cấp cửa sổ của PALF.

2. **Câu 2 — Learning State là gì?**
   - Là véc-tơ trạng thái học của từng nhóm Pattern $p$ ở cuối mỗi epoch $e$, được tính **chỉ từ tập `train`** (không rò rỉ `val`/`test`), gồm 3 đại lượng:
     1. `loss_ratio(p, e) = L_train(p, e) / L_train(p, 1)` (mức giảm loss tương đối so với epoch 1),
     2. `velocity(p, e) = L_train(p, e-1) - L_train(p, e)` (tốc độ giảm loss giữa 2 epoch liên tiếp),
     3. `instability(p, e)` (độ lệch chuẩn của `velocity` trong cửa sổ trượt $K=3$ epoch).
   - *(Lưu ý trung thực: kiểm định G3 với kiểm soát hiệu ứng cố định theo epoch cho thấy tín hiệu dự báo của Learning State rất nhỏ, chỉ đạt $\Delta R^2 \ge 0.02$ ở 2/4 backbone nên `A3` chạy theo phương án dự phòng đã đăng ký trước).*

3. **Câu 3 — Vì sao giao thức so sánh của chúng tôi công bằng?**
   - Mọi cấu hình cùng dùng một tập `train/val/test` (`purge_gap = 119`), cùng kiến trúc, cùng `batch = 128`, `lr = 1e-3`, cùng điểm khởi tạo trọng số theo từng `seed`, và đánh giá tập `test` đúng 1 lần tại checkpoint có `val_loss` tốt nhất.
   - Có **3 tầng đối chứng tách biệt từng yếu tố gây nhiễu**:
     - **`A0s` (`uniform_b1.0` qua `AdaptiveScheduler`):** Tách riêng hiệu ứng đường mã Scheduler khi lấy đủ 100% mẫu (`A0s - A0 = -0.0007 ± 0.0164 °C` $\Rightarrow$ `A0s ≈ A0`).
     - **`A0r` (`uniform_b0.7` qua `AdaptiveScheduler`):** Tách riêng hiệu ứng giảm ngân sách mẫu ngẫu nhiên 30% không dùng Pattern.
     - **`A1` (`loss_b1.0_a1.0`):** Tách riêng hiệu ứng chỉ dùng `loss_ratio` so với công thức đa biến `A3` (`predictive`).
