# Biên bản Kiểm tra Ứng dụng Demo, Clone sạch và Đối chiếu Số liệu (DEMO_CHECK)

- **Ngày kiểm tra:** `20/11/2026` (chuẩn bị cho mốc chốt Demo `24/11` và đóng băng `25/11`)
- **Phiên bản đóng băng mục tiêu:** `v1.0-demo`

---

## 1. Kiểm tra chạy từ Bản Clone Sạch (`Mục 11.1, 11.3 & 12.2`)

1. **Thiết lập kiểm tra:**
   - Xuất cây thư mục sạch (chỉ gồm các tệp được Git theo dõi, không có `.git/`, không có `experiments/results/baseline/`, không có `experiments/results/smoke/`, đã gỡ `pattern_stats_send/` khỏi Git index).
   - Cài đặt qua `requirements.txt` (đã bổ sung `-e .` và cơ chế tự động thêm `src/` vào `sys.path` trong `app/app.py` để chạy trực tiếp `.\.venv\Scripts\python.exe -m streamlit run app\app.py` hoặc `.\scripts\demo.ps1` từ bất kỳ bản clone mới nào mà không gặp lỗi `ModuleNotFoundError: No module named 'palf'`).
2. **Khi chưa giải nén thư mục kết quả lớn `experiments/results/baseline/jena` (đã `.gitignore`):**
   - Hàm `check_results_availability()` và giao diện `app/app.py` **không báo lỗi stack trace**, hiển thị thông báo rõ ràng:
     > *"ℹ️ Thiếu dữ liệu `experiments/results/baseline/jena`, giải nén bản sao lưu theo `experiments/RUNS.md` mục 5 (`Expand-Archive -Path C:\Users\Admin\Downloads\PALF_A0_baseline_backup_all4backbones.zip -DestinationPath . -Force`)."*
   - Các bảng kết quả chính thức 10 seed (`A0, A0s, A0r, A0w, A1, A3`), bảng kiểm định `C1–C3` và bảng MAE theo Pattern vẫn hiển thị đầy đủ từ `experiments/analysis/comparison/*.csv`, đồng thời nhận diện đủ `70` lần chạy PALF có sẵn trong repo.
3. **Sau khi giải nén bản sao lưu `PALF_A0_baseline_backup_all4backbones.zip` (`25 runs`, `100 files`, `197,231 bytes`):**
   - Chạy lệnh tái lập trên bản clone sạch:
     ```powershell
     .\.venv\Scripts\python.exe experiments\scripts\compare_runs.py --models gru --seeds 0,1,2,3,4,5,6,7,8,9 --out-dir comp_verify
     ```
   - Đối chiếu từng ký tự giữa `comp_verify/` và `experiments/analysis/comparison/` trên cả 4 tệp (`comparison_report.md`, `primary_comparisons_prompt10.csv`, `summary_overall.csv`, `summary_by_pattern.csv`): **Sai lệch = `0` (khớp tuyệt đối 100%)**.
4. **Kết quả chạy bộ kiểm thử tự động (`pytest -q`):**
   - Lệnh: `.\.venv\Scripts\pytest.exe -q`
   - Kết quả: **`86 passed in 33.25s`** (`0 failed`, `0 error`).

---

## 2. Biên bản Đối chiếu Thủ công 5 Ô Số Ngẫu nhiên trong Ứng dụng với `comparison_report.md` & CSV (`Mục 11.2`)

| STT | Màn hình / Bảng trên Ứng dụng (`app/app.py`) | Ô số được chọn ngẫu nhiên | Giá trị hiển thị trên Ứng dụng | Giá trị gốc trong `comparison_report.md` & `*.csv` | Kết quả |
|:---:|---|---|---:|---:|:---:|
| **1** | Trang 1 — Bảng 1 (GRU, 10 seed) | `A1 (loss_b1.0_a1.0)` — **Test MAE (°C) (Mean ± Std)** | `1.7618 ± 0.0164` | `1.7618 ± 0.0164` (`summary_overall.csv`: `1.7618483 ± 0.0163759`) | **KHỚP** |
| **2** | Trang 1 — Bảng 1 (GRU, 10 seed) | `A3 (predictive_b1.0_a0.5_w2 - không đạt G3)` — **Best Val Loss** | `0.0802 ± 0.0015` | `0.0802 ± 0.0015` (`summary_overall.csv`: `0.0801763 ± 0.0015162`) | **KHỚP** |
| **3** | Trang 1 — Bảng 2 (Đối chiếu `C1–C3`) | `C1 (A1 − A0s)` — **Hiệu MAE (°C), Số seed âm, `p (t)`, `p (Bonf)`, `Wilcoxon p`** | `-0.0067 ± 0.0191` · `6/10` · `p(t)=0.2974` · `p(Bonf)=0.8922` · `W_p=0.4316` · `Không đạt` | `-0.0067 ± 0.0191` · `6/10` · `p_t=0.2974` · `p_Bonf=0.8922` · `W_p=0.4316` (`primary_comparisons_prompt10.csv`) | **KHỚP** |
| **4** | Trang 1 — Bảng 2 (Đối chiếu `C1–C3`) | `C3 (A1 − A0w)` — **Hiệu MAE (°C), Số seed âm, `p (t)`, `Wilcoxon p`** | `-0.0027 ± 0.0214` · `5/10` · `p(t)=0.6983` · `W_p=1.0000` · `Không đạt` | `-0.0027 ± 0.0214` · `5/10` · `p_t=0.6983` · `W_p=1.0000` (`primary_comparisons_prompt10.csv`) | **KHỚP** |
| **5** | Trang 1 — Bảng 3 (MAE theo Pattern, 10 seed GRU) | Pattern `temp_anomaly` — **MAE (°C) của `A0w` và `A3`** | `A0w: 1.9719 ± 0.0324` · `A3: 1.9715 ± 0.0314` | `A0w: 1.9719 ± 0.0324` · `A3: 1.9715 ± 0.0314` (`summary_by_pattern.csv`) | **KHỚP** |

---

## 3. Kiểm tra chéo Số liệu Slide (`DEMO_SCRIPT.md`) & Quy tắc Hiển thị (`Mục 11.2, 11.4`)

- [x] **Loại bỏ các cột không so sánh được:** Đã xóa hoàn toàn `time_sec_to_target`, `samples_to_target`, `Epoch tới ngưỡng`, `Số mẫu tới ngưỡng`, `AUC Val (theo mẫu)`, và `overhead_sec` khỏi bảng hiển thị của `app/app.py` và `summarize_side_by_side()` (được kiểm chứng tự động bằng unit test trong `tests/test_palf/test_demo_loader.py`).
- [x] **Chế độ chạy trực tiếp (Trang 3):** Mang nhãn rõ ràng **`3. CHẠY THU NHỎ — chỉ minh họa`**, giới hạn mặc định `3 epoch` (tối đa `6 epoch`, không thể vô tình kéo lên `30 epoch`), tập con mặc định `512` mẫu, có nút **`Dừng chạy`** ngay cạnh nút **`Bắt đầu train`**.
- [x] **Định danh trung thực:**
  - Không còn bất kỳ câu nào trong `app/app.py`, `README.md`, `RUNS.md`, `DEMO_SCRIPT.md` nói PALF *"tốt hơn"* hoặc *"PALF đã chạy trên 4 kiến trúc"*.
  - Mọi nhãn `A3` và `A3r` đều ghi rõ **`predictive (không đạt ngưỡng G3)`**.
  - Câu kết luận trên Slide 5 và Trang 1 của ứng dụng khớp nguyên văn: *"Trong thiết lập này (Jena, GRU, 10 seed), chưa có bằng chứng PALF cải thiện so với lấy mẫu đều; Learning State đa biến không hơn loss đơn thuần."*
- [x] **Xác nhận kiểm tra chéo số liệu Slide với CSV:**
  - Người thực hiện & đối chiếu số liệu ứng dụng (Người 3): **Đã ký xác nhận (`20/11/2026`)**
  - Người kiểm tra chéo toàn bộ số liệu Slide với CSV (Người 4): **Đã ký xác nhận (`20/11/2026`)**
  - Người kiểm tra clone sạch & hợp nhất (Người 1): **Đã ký xác nhận (`20/11/2026`)**
