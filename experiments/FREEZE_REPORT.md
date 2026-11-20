# Báo cáo Đóng băng Dự án PALF (FREEZE_REPORT — Mốc 25/11 & Demo 30/11)

- **Thẻ phát hành (Git Tag):** `v1.0-demo` (`"PALF demo 30/11"`)
- **Commit đóng băng:** `23794be` (`23794bea3368b9004903973a98a7b53769149f55`, kết quả thực nghiệm 10 seed tại `2c838ed`)
- **Nhánh chính:** `main`
- **Bộ kiểm thử tự động (`pytest -q`):** **`86 passed in 29.09s`** (`0 failed`)
- **Quy tắc sau đóng băng:** Từ mốc đóng băng `v1.0-demo`, không chạy thêm thực nghiệm, không đổi tham số, không đổi kết luận. Mọi thay đổi (nếu có) chỉ giới hạn ở sửa lỗi hiển thị hoặc lỗi chạy và phải ghi rõ lý do trong commit.

---

## 1. Tình trạng các Cổng Kiểm định Kỹ thuật & Khoa học (`Gates G1–G5`)

| Gate | Hạng mục kiểm định | Trạng thái | Căn cứ & Kết quả thực nghiệm đóng băng |
|:---:|---|:---:|---|
| **G1** | **Pattern Analyzer v1 & Chống rò rỉ dữ liệu** (`Phase 1`) | **ĐẠT (PASS)** | Chia tập theo thời gian `70/15/15` (`48,981 / 10,377 / 10,377` mẫu) với `purge_gap = 119` bước. Bốn nhãn Pattern (`pressure_drop` 9,0%, `humidity_net_rise` 7,5%, `temp_anomaly` 16,7%, `normal` 66,9% trên tập test) chỉ dùng cửa sổ quá khứ $X$ ($L=96$ giờ) và ngưỡng phân vị khớp trên `train`. |
| **G2** | **Learning Dynamics Monitor & Tính tất định** (`Phase 2`) | **ĐẠT (PASS)** | Hoàn tất `20` lần chạy Baseline `A0` (`4 backbone × 5 seed`, 30 epoch) + mở rộng `10 seed` trên GRU. Kiểm tra tái lập bit-level giữa chạy đủ 30 epoch và dừng sớm `patience=5` đạt độ lệch $\le 7,5 \times 10^{-9}$. |
| **G3** | **Phân tích Thống kê & Dự báo Learning Dynamics** (`Phase 3`) | **ỦNG HỘ MỘT PHẦN (Lui về phương án dự phòng)** | Thứ tự độ khó `loss_ratio` giữa các Pattern ổn định cao giữa các seed ($\text{Spearman} \approx 0,94$). Sau khi kiểm soát hiệu ứng epoch (`controls_epoch=True`), các biến động học (`velocity`, `loss_trend`, `instability`) đạt $\Delta R^2 = 0,07\text{–}0,15$ trên **GRU** và $0,07\text{–}0,09$ trên **TCN** (dương khi kiểm chéo giữa các seed), nhưng yếu trên **Transformer** ($0,01\text{–}0,02$) và $\approx 0$ trên **PatchTST** với dấu hệ số đảo chiều. Vì chỉ đạt ở `2/4` backbone (ngưỡng đăng ký trước là $\ge 3/4$), cấu hình `A3/A3r` được định danh bắt buộc là **`predictive (không đạt ngưỡng G3)`**. |
| **G4** | **Vòng lặp Điều phối Đóng & Cách ly Tập Đánh giá** (`Phase 4–6`) | **ĐẠT KỸ THUẬT (PASS)** | `LearningStateTracker` và `PolicyEngine` chặn hoàn toàn mọi thống kê từ `val`/`test`. `AdaptiveScheduler` bảo toàn ngân sách mẫu $\sum_p n_p = n_e$, đảm bảo sàn an toàn $n_{\text{floor}}$ và hỗ trợ hoán vị trọng số đối chứng (`--shuffle-weights` cho `A0w`). |
| **G5** | **Giao thức So sánh Công bằng 10 Seed & Chốt Demo** (`Phase 6 / Kiểm định cuối`) | **ĐÓNG BĂNG (KẾT QUẢ ÂM CÓ KIỂM SOÁT)** | Trên GRU (`10 seed`, ngưỡng Bonferroni $\alpha = 0,05/3 = 0,0167$):<br>• `C1 (A1 − A0s) = -0,0067 ± 0,0191 °C` (`6/10` âm, `p_t = 0,2974`, `W_p = 0,4316`) $\rightarrow$ **Không đạt**.<br>• `C2 (A3 − A1) = +0,0075 ± 0,0192 °C` (`5/10` âm, `p_t = 0,2500`, `W_p = 0,4316`) $\rightarrow$ **Không đạt**.<br>• `C3 (A1 − A0w) = -0,0027 ± 0,0214 °C` (`5/10` âm, `p_t = 0,6983`, `W_p = 1,0000`) $\rightarrow$ **Không đạt**.<br>Kết luận đóng băng: *"Trong thiết lập này (Jena, GRU, 10 seed), chưa có bằng chứng PALF cải thiện so với lấy mẫu đều; Learning State đa biến không hơn loss đơn thuần."* |

---

## 2. Danh sách Hạn chế / Vấn đề Đã Biết Chưa Sửa (Ghi nhận trung thực trong Báo cáo)

1. **Hội tụ sớm làm hẹp cửa sổ tác động của Policy (`1–3 epoch`):** Trên tập Jena với `lr = 1e-3`, GRU đạt `best_val_loss` ở `epoch 5,3–7,6` và TCN đạt ở `epoch 3,0–3,7` (ngay sau `warmup_epochs = 3`), khiến Policy có rất ít epoch can thiệp trước khi mô hình quá khớp.
2. **Nhãn `temp_anomaly` gộp chung hai chiều nóng/lạnh và lệch mùa giữa `train` và `val/test`:** Tỷ lệ dị thường nóng trong `temp_anomaly` tăng từ `48%` (`train`) lên `79%` (`val`) và `87%` (`test`), trong khi dị thường lạnh có sai số cao hơn hẳn (`2,42–2,52 °C` so với `1,88–1,91 °C`).
3. **Phạm vi kiến trúc chạy điều phối PALF:** Điều phối PALF mới chạy trên **GRU (10 seed)** và **TCN (3 seed)**; chưa chạy điều phối PALF trên **Transformer** và **PatchTST**.
4. **Thư mục `experiments/results/baseline/` nằm trong `.gitignore`:** Khi clone mới trên máy khác, cần giải nén `PALF_A0_baseline_backup_all4backbones.zip` theo `experiments/RUNS.md` mục 5 nếu muốn xem đường cong từng epoch của `A0` (các bảng tổng hợp 10 seed và `C1–C3` vẫn hiển thị ngay từ `experiments/analysis/comparison/*.csv`).

---

## 3. Phân công Trách nhiệm Trình bày & Vận hành Demo (`30/11`)

| Thành viên | Vai trò trong buổi Bảo vệ / Demo (`30/11`) | Phần phụ trách chính |
|:---:|---|---|
| **Người 1** | Điều phối chung, Kiến trúc hệ thống & Đóng băng mã nguồn | Trình bày **Đoạn 1 & Đoạn 2** (Bài toán, Kiến trúc 4 khối `Monitor → Learning State → Policy → Scheduler`, tính công bằng & chống rò rỉ dữ liệu). Trả lời câu hỏi về kỹ thuật & pipeline. |
| **Người 2** | Quản lý Máy Demo chính, Máy dự phòng & Hạ tầng thực nghiệm | Chuẩn bị máy demo chính + máy dự phòng, video quay màn hình offline ($\le 3$ phút) và PDF slide trên USB/Drive. Trả lời câu hỏi về quy trình chạy 10 seed, `patience=5` bit-level và các đối chứng `A0s, A0r, A0w`. |
| **Người 3** | Vận hành Giao diện Streamlit Demo & Trình bày Kết quả | Thực hiện **Đoạn 4 (Demo trực tiếp 2 phút trên Trang 1 & Trang 2)** và trình bày **Đoạn 5 (Slide 5 & Slide 6: Bảng 10 seed GRU, `C1–C3`, và 3 sự thật thực nghiệm quan sát được)**. |
| **Người 4** | Kiểm định Thống kê, Hạn chế & Điều phối Q&A | Trình bày **Đoạn 3 (Pattern ≠ Patch, Phase 3 Learning Dynamics)** và **Đoạn 6 (Slide 7: Hạn chế nói thẳng & Hướng phát triển)**. Phụ trách đối chiếu số liệu và hỗ trợ trả lời 8 câu hỏi phản biện Q&A. |
