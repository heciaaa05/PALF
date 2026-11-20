import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from palf.core.config import PalfConfig
from palf.data import prepare_data, prepare_weather_data
from palf.evaluation.demo_loader import (
    check_results_availability,
    discover_saved_runs,
    load_comparison_csv_tables,
    summarize_overall_across_seeds,
    summarize_palf_diagnostics,
    summarize_pattern_mae_across_seeds,
    summarize_side_by_side,
)
from palf.models import MODEL_REGISTRY, MODEL_INFO
from palf.training.trainer import train_model

MAX_RESULTS = 4
WEATHER_TARGETS = {
    "T (degC) - nhiệt độ": "T (degC)",
    "rh (%) - độ ẩm": "rh (%)",
    "p (mbar) - áp suất": "p (mbar)",
    "wv (m/s) - tốc độ gió": "wv (m/s)",
}

st.set_page_config(page_title="PALF Demo & Trainer", layout="wide")
st.title("PALF – Pattern-Aware Learning Framework (Demo & Quan sát)")


@st.cache_resource
def load_synthetic():
    return prepare_data()


@st.cache_resource
def load_weather(target_col):
    return prepare_weather_data(target_col=target_col)


def _slice_ds(ds, n):
    if n is None or n >= len(ds):
        return ds
    t0, t1, t2, t3 = ds.tensors
    return TensorDataset(t0[:n], t1[:n], t2[:n], t3[:n])


if "results" not in st.session_state:
    st.session_state["results"] = []
if "run_counter" not in st.session_state:
    st.session_state["run_counter"] = 0


def render_summary(results):
    if len(results) < 2:
        return
    rows = []
    for r in results:
        cfg = r["config"]
        rows.append({
            "Run": f"#{r['run_id']}",
            "Chế độ": cfg.get("mode", "baseline").upper(),
            "Policy": cfg.get("policy", "-"),
            "Budget": cfg.get("budget_ratio", 1.0),
            "Nguồn": cfg["source"],
            "Model": r["model_name"],
            "Epoch": cfg["epochs"],
            "LR": cfg["lr"],
            "Test MAE (°C)": round(r["test"].get("mae_degC", r["test"]["mae"]), 4),
            "Test RMSE (°C)": round(r["test"].get("rmse_degC", r["test"]["rmse"]), 4),
            "Best epoch": r["best_epoch"],
            "Tham số": r["params"],
            "Thời gian (s)": round(r["elapsed"], 1),
        })
    st.markdown("**So sánh các lần train trực tiếp trong phiên (tối đa 4 kết quả)**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_result(res):
    rid = res["run_id"]
    cfg = res["config"]
    mode_str = cfg.get("mode", "baseline").upper()
    if cfg.get("mode") == "palf":
        mode_str += f" ({cfg.get('policy', 'state')}, budget={cfg.get('budget_ratio', 1.0):g}, α={cfg.get('alpha', 1.0):g})"
    with st.container(border=True):
        st.subheader(f"#{rid} · {res['model_name']} · [{mode_str}] · {cfg['source']}")
        extra = f" · biến mục tiêu {cfg['target_col']}" if cfg.get("target_col") else ""
        st.caption(
            f"{cfg['epochs']} epoch · batch {cfg['batch']} · lr {cfg['lr']:g} · "
            f"seed {cfg['seed']} · thiết bị {cfg['device']}{extra}"
        )
        c = st.columns(5)
        c[0].metric("Test MAE (°C)", f"{res['test'].get('mae_degC', res['test']['mae']):.4f}")
        c[1].metric("Test RMSE (°C)", f"{res['test'].get('rmse_degC', res['test']['rmse']):.4f}")
        c[2].metric("Best epoch", res["best_epoch"])
        c[3].metric("Số tham số", f"{res['params']:,}")
        c[4].metric("Thời gian", f"{res['elapsed']:.1f}s")

        tab_names = ["Loss", "Theo pattern", "Ví dụ dự báo", "Dữ liệu"]
        if cfg.get("mode") == "palf" and "policy_log" in res and not res["policy_log"].empty:
            tab_names.append("Trọng số PALF")
        tabs = st.tabs(tab_names)

        with tabs[0]:
            st.line_chart(res["history"].set_index("epoch")[["train_loss", "val_loss"]])

        with tabs[1]:
            st.markdown("**Test theo pattern**")
            st.dataframe(res["test"]["by_pattern"], hide_index=True, use_container_width=True)
            pl = res["pattern_log"]
            st.markdown("**Train loss trung bình theo pattern**")
            st.line_chart(pl.pivot(index="epoch", columns="pattern", values="train_loss_mean"))
            st.markdown("**Gradient norm theo pattern**")
            st.line_chart(pl.pivot(index="epoch", columns="pattern", values="grad_norm"))
            st.markdown("**Val loss theo pattern**")
            st.line_chart(pl.pivot(index="epoch", columns="pattern", values="val_loss"))

        with tabs[2]:
            fig, axes = plt.subplots(2, 2, figsize=(11, 6))
            for ax, (name, e) in zip(axes.flat, res["examples"].items()):
                L = len(e["x"])
                ax.plot(range(L), e["x"], label="input")
                ax.plot(range(L, L + len(e["y"])), e["y"], label="thực tế")
                ax.plot(range(L, L + len(e["pred"])), e["pred"], "--", label="dự báo")
                ax.set_title(name)
                ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with tabs[3]:
            st.caption(f"Đã lưu file tại: {res['out_dir']}")
            st.download_button("Tải history.csv", res["history"].to_csv(index=False),
                               "history.csv", "text/csv", key=f"dl_hist_{rid}")
            st.download_button("Tải epoch_pattern_log.csv",
                               res["pattern_log"].to_csv(index=False),
                               "epoch_pattern_log.csv", "text/csv", key=f"dl_pat_{rid}")
            if cfg.get("mode") == "palf" and "policy_log" in res and not res["policy_log"].empty:
                st.download_button("Tải policy_log.csv",
                                   res["policy_log"].to_csv(index=False),
                                   "policy_log.csv", "text/csv", key=f"dl_pol_{rid}")

        if len(tabs) == 5:
            with tabs[4]:
                plog = res["policy_log"]
                st.markdown("**Trọng số `weights[p]` theo epoch**")
                st.line_chart(plog.pivot(index="epoch", columns="pattern", values="weight"))
                st.markdown("**Điểm ưu tiên `need_score` theo epoch**")
                st.line_chart(plog.pivot(index="epoch", columns="pattern", values="need_score"))


page = st.radio(
    "Chọn trang hiển thị",
    [
        "1. So sánh cạnh nhau (Ngoại tuyến - Kết quả 10 Seed GRU & C1–C3)",
        "2. Quan sát PALF (Trọng số, Tỷ lệ mẫu & Learning State)",
        "3. CHẠY THU NHỎ — chỉ minh họa",
    ],
    horizontal=True,
)
st.divider()

avail_info = check_results_availability()
if avail_info["missing_items"]:
    st.warning(f"ℹ️ {avail_info['message']}")

csv_tables = load_comparison_csv_tables()
saved_runs = discover_saved_runs()
run_map = {r["label"]: r for r in saved_runs}

# =====================================================================
# TRANG 1: SO SANH CANH NHAU (NGOAI TUYEN)
# =====================================================================
if page.startswith("1."):
    st.subheader("Kết quả Kiểm định 10 Seed trên GRU (`C1–C3`) & So sánh cạnh nhau")
    st.caption(
        "Mọi số liệu tổng hợp dưới đây được đọc trực tiếp từ `experiments/analysis/comparison/*.csv` "
        "và các thư mục kết quả `experiments/results/` (không hard-code, không hiển thị `samples_to_target`, "
        "`time_sec_to_target` hay overhead thời gian)."
    )

    if not csv_tables["gru_10seed_df"].empty:
        st.markdown("#### 1. Bảng chính GRU, 10 Seed (`seeds 0..9` — `A0, A0s, A0r, A0w, A1, A3`)")
        st.dataframe(csv_tables["gru_10seed_df"], hide_index=True, use_container_width=True)

    if not csv_tables["c1_c3_df"].empty:
        st.markdown("#### 2. Ba đối chiếu chính đã đăng ký trước (`C1–C3`, ngưỡng Bonferroni α = 0.05/3 = 0.0167)")
        st.dataframe(csv_tables["c1_c3_df"], hide_index=True, use_container_width=True)
        st.info(
            "**Kết luận đóng băng:** Trong thiết lập này (Jena, GRU, 10 seed), chưa có bằng chứng PALF cải thiện "
            "so với lấy mẫu đều (`C1: p_t = 0.2974 >= 0.0167`, `C3: p_t = 0.6983 >= 0.0167`); Learning State đa biến "
            "không hơn loss đơn thuần (`C2: p_t = 0.2500 >= 0.0167`). Mọi chênh lệch nhỏ giữa các cấu hình đều nằm "
            "trong phạm vi biến động giữa các seed."
        )

    if not csv_tables["pattern_pivot_df"].empty:
        st.markdown("#### 3. Test MAE (°C) theo từng Pattern trên GRU (10 Seed, `Mean ± Std`)")
        st.caption(
            "Độ lệch chuẩn giữa các seed (`0.02–0.05 °C`) lớn hơn mọi chênh lệch giữa các cấu hình; "
            "`temp_anomaly` có sai số cao nhất ở mọi cấu hình (`~1.96–1.97 °C`)."
        )
        st.dataframe(csv_tables["pattern_pivot_df"], hide_index=True, use_container_width=True)
        with st.expander("Xem bảng chi tiết MAE & RMSE (°C) theo Pattern (10 seed GRU)"):
            st.dataframe(csv_tables["pattern_10seed_df"], hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### 4. Đặt cạnh nhau các lần chạy 30 epoch trên đĩa (Đường `val_loss` theo Epoch)")

    if not saved_runs:
        st.warning(
            "Thiếu dữ liệu các lần chạy chi tiết trong `experiments/results/`, giải nén bản sao lưu theo "
            "`experiments/RUNS.md` mục 5."
        )
    else:
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            models_avail = sorted({r["model"].upper() for r in saved_runs})
            sel_model = st.selectbox("Lọc theo Backbone", ["Tất cả"] + models_avail)
            seeds_avail = sorted({r["seed"] for r in saved_runs})
            sel_seed = st.selectbox("Lọc theo Seed", ["Tất cả"] + seeds_avail)

        filtered_labels = [
            r["label"]
            for r in saved_runs
            if (sel_model == "Tất cả" or r["model"].upper() == sel_model)
            and (sel_seed == "Tất cả" or r["seed"] == sel_seed)
        ]

        palf_In_filtered = [lb for lb in filtered_labels if lb.startswith("PALF")]
        if not palf_In_filtered:
            st.info(
                "Lưu ý: Hiện chưa có kết quả PALF cho bộ lọc này (hoặc đang dùng kết quả Baseline có sẵn). "
                "Bạn vẫn có thể chọn 2–3 lần chạy Baseline để kiểm tra bố cục so sánh."
            )

        default_sel = filtered_labels[: min(3, len(filtered_labels))]
        with col_f2:
            chosen_labels = st.multiselect(
                "Chọn 2 hoặc 3 lần chạy để đặt cạnh nhau:",
                options=filtered_labels,
                default=default_sel,
                max_selections=3,
            )

        if not chosen_labels:
            st.info("Vui lòng chọn ít nhất 1 lần chạy để hiển thị.")
        else:
            chosen_runs = [run_map[lb] for lb in chosen_labels]
            overall_df, pat_df, _ = summarize_side_by_side(chosen_runs)

            st.markdown("##### Bảng thông số của các lần chạy đang chọn")
            st.dataframe(overall_df, hide_index=True, use_container_width=True)

            c_plot0, c_plot1 = st.columns(2)
            with c_plot0:
                st.markdown("##### Đường `val_loss` theo **Epoch** (kèm mốc hết `warmup` & `best_epoch`)")
                fig0, ax0 = plt.subplots(figsize=(6.5, 4.0))
                ax0.axvline(3, color="#d97706", linestyle="--", linewidth=1.3, label="Hết warm-up (ep 3)")
                for r in chosen_runs:
                    h = r["history"]
                    tm = r.get("test_metrics", {})
                    b_ep = int(tm.get("best_epoch", h.loc[h["val_loss"].idxmin(), "epoch"]))
                    line_objs = ax0.plot(h["epoch"], h["val_loss"], marker="o", markersize=3, label=f"{r['label']} (best ep {b_ep})")
                    ax0.axvline(b_ep, color=line_objs[0].get_color(), linestyle=":", linewidth=1.2, alpha=0.8)
                ax0.set_xlabel("Epoch")
                ax0.set_ylabel("Validation MSE (chuẩn hóa)")
                ax0.grid(True, alpha=0.3)
                ax0.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig0)
                plt.close(fig0)

            with c_plot1:
                st.markdown("##### Đường `val_loss` theo **Số mẫu tích lũy**")
                fig1, ax1 = plt.subplots(figsize=(6.5, 4.0))
                for r in chosen_runs:
                    h = r["history"]
                    ax1.plot(h["cum_samples"], h["val_loss"], marker="o", markersize=3, label=r["label"])
                ax1.set_xlabel("Số mẫu train tích lũy (cumulative samples)")
                ax1.set_ylabel("Validation MSE (chuẩn hóa)")
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            st.markdown("##### Bảng MAE và RMSE (°C) trên tập Test theo từng Pattern (của các lần chạy đang chọn)")
            if not pat_df.empty:
                c_tb1, c_tb2 = st.columns(2)
                with c_tb1:
                    st.markdown("**Test MAE (°C) theo Pattern**")
                    mae_piv = pat_df.pivot(index="Pattern", columns="Run", values="MAE (°C)")
                    st.dataframe(mae_piv, use_container_width=True)
                with c_tb2:
                    st.markdown("**Test RMSE (°C) theo Pattern**")
                    rmse_piv = pat_df.pivot(index="Pattern", columns="Run", values="RMSE (°C)")
                    st.dataframe(rmse_piv, use_container_width=True)

            with st.expander("Bảng tổng hợp toàn bộ các lần chạy có trên đĩa theo bộ lọc Backbone (`Mean ± Std`)"):
                agg_overall_df = summarize_overall_across_seeds(saved_runs, model_filter=sel_model)
                if not agg_overall_df.empty:
                    st.dataframe(agg_overall_df, hide_index=True, use_container_width=True)
                agg_pat_df = summarize_pattern_mae_across_seeds(saved_runs, model_filter=sel_model)
                if not agg_pat_df.empty:
                    st.dataframe(agg_pat_df, hide_index=True, use_container_width=True)

# =====================================================================
# TRANG 2: QUAN SAT PALF
# =====================================================================
elif page.startswith("2."):
    st.subheader("Quan sát cơ chế bên trong của PALF theo Epoch")
    st.caption(
        "Hiển thị trọng số `weights[p]`, tỷ lệ mẫu lịch trình (`quota_planned` share, bao gồm lặp mẫu), "
        "tỷ lệ mẫu duy nhất (`n_seen` share), và các biến Learning State từ `policy_log.csv`."
    )

    palf_runs = [r for r in saved_runs if r.get("has_palf", False)]
    if not palf_runs:
        st.warning("Chưa có kết quả PALF trên đĩa. Giao diện không dựng số liệu giả.")
        st.info(
            "Để tạo kết quả PALF, hãy chạy lệnh ví dụ:\n"
            "`python experiments/scripts/run_observation.py --model gru --mode palf --policy predictive --budget-ratio 1.0 --seed 0`"
        )
    else:
        palf_labels = [r["label"] for r in palf_runs]
        sel_palf_label = st.selectbox("Chọn một lần chạy PALF:", options=palf_labels)
        sel_run = run_map[sel_palf_label]
        diag = summarize_palf_diagnostics(sel_run)

        if not diag["has_palf"]:
            st.warning(diag["message"])
        else:
            rc = sel_run.get("run_config", {})
            p_cfg = rc.get("palf_config", {}) or {}
            tm = sel_run.get("test_metrics", {})
            hist = sel_run["history"]
            pol_raw = str(p_cfg.get("policy", "-"))
            pol_disp = f"{pol_raw.upper()} (không đạt G3)" if pol_raw == "predictive" else pol_raw.upper()
            warmup_ep = int(p_cfg.get("warmup_epochs", 3))
            best_ep = int(tm.get("best_epoch", hist.loc[hist["val_loss"].idxmin(), "epoch"]))

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Chính sách (Policy)", pol_disp)
            m2.metric("Budget Ratio (ρ)", f"{p_cfg.get('budget_ratio', 1.0):g}")
            m3.metric("Hệ số α", f"{p_cfg.get('alpha', 1.0):g}")
            m4.metric("Khoảng [w_min, w_max]", f"[{p_cfg.get('w_min', 0.25):g}, {p_cfg.get('w_max', 3.0):g}]")
            m5.metric("Warm-up / Best Epoch", f"ep {warmup_ep} / ep {best_ep}")

            st.markdown(
                "#### 0. Tương quan giữa Trọng số `weights[p]`, Đường `val_loss`, mốc hết `warmup_epochs` và `best_epoch`"
            )
            st.caption(
                "Biểu đồ giúp quan sát trực tiếp hệ quả khi `best_epoch` rơi sát thời điểm kết thúc `warmup_epochs`: "
                "cơ chế điều chỉnh trọng số chỉ có rất ít epoch tác động trước khi mô hình đạt đỉnh trên tập validation."
            )
            fig_wv, ax_w = plt.subplots(figsize=(10.5, 4.0))
            w_piv = diag["weights_pivot"]
            for pat_col in w_piv.columns:
                ax_w.plot(w_piv.index, w_piv[pat_col], marker="o", markersize=3.5, linewidth=1.8, label=f"w[{pat_col}]")
            ax_w.axvline(
                warmup_ep,
                color="#d97706",
                linestyle="--",
                linewidth=1.6,
                label=f"Hết warm-up (epoch {warmup_ep})",
            )
            ax_w.axvline(
                best_ep,
                color="#dc2626",
                linestyle=":",
                linewidth=1.8,
                label=f"Best val_loss (epoch {best_ep})",
            )
            ax_w.set_xlabel("Epoch")
            ax_w.set_ylabel("Trọng số lấy mẫu weights[p]")
            ax_w.grid(True, alpha=0.25)

            ax_v = ax_w.twinx()
            ax_v.plot(
                hist["epoch"],
                hist["val_loss"],
                color="#111827",
                linestyle="-.",
                marker="s",
                markersize=3.5,
                linewidth=2.0,
                label="val_loss (MSE chuẩn hóa)",
            )
            ax_v.set_ylabel("Validation MSE (chuẩn hóa)")

            lines_w, labels_w = ax_w.get_legend_handles_labels()
            lines_v, labels_v = ax_v.get_legend_handles_labels()
            ax_w.legend(lines_w + lines_v, labels_w + labels_v, loc="upper right", fontsize=8, ncol=2)
            plt.tight_layout()
            st.pyplot(fig_wv)
            plt.close(fig_wv)

            col_w, col_q, col_s = st.columns(3)
            with col_w:
                st.markdown("#### 1. Trọng số lấy mẫu `weights[p]`")
                st.line_chart(diag["weights_pivot"])
            with col_q:
                st.markdown("#### 2. Tỷ lệ mẫu lịch trình (`quota_planned` share)")
                st.line_chart(diag.get("quota_share_pivot", diag["share_pivot"]))
            with col_s:
                st.markdown("#### 3. Tỷ lệ mẫu duy nhất đã duyệt (`n_seen` share)")
                st.line_chart(diag["share_pivot"])

            st.markdown("#### 4. Các biến Learning State & Điểm ưu tiên (`need_score`) theo Pattern")
            state_pivots = diag["state_pivots"]
            if state_pivots:
                s_cols = st.columns(2)
                for idx_v, (v_name, piv_df) in enumerate(state_pivots.items()):
                    with s_cols[idx_v % 2]:
                        st.markdown(f"**Biến `{v_name}` theo Epoch**")
                        st.line_chart(piv_df)

            with st.expander("Xem bảng dữ liệu gốc `policy_log.csv`"):
                st.dataframe(sel_run["policy_log"], hide_index=True, use_container_width=True)

# =====================================================================
# TRANG 3: HUAN LUYEN TRUC TIEP (QUY MO NHO / SMOKE)
# =====================================================================
else:
    st.subheader("CHẠY THU NHỎ — chỉ minh họa")
    st.warning(
        "⚠️ **CHẠY THU NHỎ — chỉ minh họa:** Huấn luyện trực tiếp trên giao diện chỉ dành cho minh họa "
        "vòng lặp hoạt động với ít epoch (mặc định 3 epoch, tối đa 6 epoch) và tập con dữ liệu (`subset`). "
        "Mọi kết quả ở trang này được lưu tách biệt sang `experiments/results/smoke/` và **không ghi đè hay trộn lẫn** "
        "vào danh sách thực nghiệm thật 30 epoch."
    )

    with st.sidebar:
        st.header("Dữ liệu (CHẠY THU NHỎ — chỉ minh họa)")
        source = st.radio("Nguồn dữ liệu", ["Giả lập (nhanh)", "Thời tiết thật (Jena, Đức)"])
        target_label = None
        subset_size = None
        if source != "Giả lập (nhanh)":
            target_label = st.selectbox("Biến mục tiêu", list(WEATHER_TARGETS.keys()))
            subset_size = st.select_slider(
                "Giới hạn số mẫu (quy mô nhỏ)",
                options=[512, 1024, 2048],
                value=512,
            )

        st.header("Chế độ & Cấu hình PALF")
        train_mode = st.radio("Chế độ huấn luyện", ["baseline", "palf"], horizontal=True)
        policy_name = "predictive"
        budget_ratio = 1.0
        alpha = 1.0
        w_min, w_max = 0.25, 3.0
        ema_beta = 0.5
        warmup_epochs = 1
        update_every = 1
        window = 2

        if train_mode == "palf":
            policy_name = st.selectbox(
                "Chính sách (Policy)",
                ["predictive", "loss_abs", "loss", "uniform", "state"],
                index=0,
            )
            budget_ratio = st.slider("Ngân sách mẫu mỗi epoch (budget_ratio)", 0.3, 1.0, 0.7, 0.05)
            alpha = st.slider("Độ mạnh phản ứng (alpha)", 0.25, 3.0, 1.0, 0.25)
            with st.expander("Tham số PALF nâng cao"):
                w_min = st.number_input("w_min", value=0.25, min_value=0.05, max_value=1.0, step=0.05)
                w_max = st.number_input("w_max", value=3.0, min_value=1.0, max_value=10.0, step=0.5)
                ema_beta = st.slider("EMA beta", 0.0, 0.9, 0.5, 0.1)
                warmup_epochs = st.number_input("Warm-up epochs", value=1, min_value=1, max_value=4, step=1)
                update_every = st.number_input("Update every (epoch)", value=1, min_value=1, max_value=4, step=1)
                window = st.number_input("Window W", value=2, min_value=1, max_value=4, step=1)

        st.header("Cấu hình Backbone")
        model_name = st.selectbox("Loại model", list(MODEL_REGISTRY.keys()))
        st.caption(MODEL_INFO[model_name])
        epochs = st.slider("Số epoch (giới hạn chạy thu nhỏ)", 2, 6, 3)
        batch = st.selectbox("Batch size", [32, 64, 128, 256], index=2)
        lr = st.select_slider(
            "Learning rate",
            options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            value=1e-3,
            format_func=lambda v: f"{v:g}",
        )
        device_name = st.selectbox("Thiết bị", ["cpu", "dml", "cuda"])
        seed = st.number_input("Seed", value=0, step=1)
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            run = st.button("Bắt đầu train", type="primary", use_container_width=True)
        with b_col2:
            stop_run = st.button("Dừng chạy", use_container_width=True)
        clear = st.button("Xóa kết quả", use_container_width=True)

    if stop_run:
        st.warning("Đã dừng lượt chạy thu nhỏ theo yêu cầu.")
        st.stop()
    if clear:
        st.session_state["results"] = []

    live = st.container()

    results = st.session_state["results"]
    if results:
        st.caption(
            f"Đang lưu {len(results)}/{MAX_RESULTS} kết quả trên màn hình "
            f"(mới nhất ở trên; quá {MAX_RESULTS} thì kết quả cũ nhất bị bỏ)."
        )
    render_summary(results)
    for r in results:
        render_result(r)

    if run:
        target_col = WEATHER_TARGETS[target_label] if target_label else None
        with live:
            if source == "Giả lập (nhanh)":
                data = load_synthetic()
                target_idx = 0
                out_root = "experiments/results/smoke/synthetic_demo"
            else:
                with st.spinner("Đang tải / xử lý dữ liệu Jena Climate..."):
                    train_ds, val_ds, test_ds, th, meta = load_weather(target_col)
                if subset_size is not None:
                    train_ds = _slice_ds(train_ds, subset_size)
                    val_ds = _slice_ds(val_ds, max(64, subset_size // 4))
                    test_ds = _slice_ds(test_ds, max(64, subset_size // 4))
                data = (train_ds, val_ds, test_ds, th, meta)
                target_idx = meta["target_idx"]
                out_root = "experiments/results/smoke/weather_demo"

            palf_cfg = None
            if train_mode == "palf":
                palf_cfg = PalfConfig(
                    policy=policy_name,
                    warmup_epochs=int(warmup_epochs),
                    update_every=int(update_every),
                    alpha=float(alpha),
                    w_min=float(w_min),
                    w_max=float(w_max),
                    ema_beta=float(ema_beta),
                    budget_ratio=float(budget_ratio),
                    window=int(window),
                )

            st.subheader(f"Đang train: {model_name} ({train_mode.upper()})")
            bar = st.progress(0.0)
            status = st.empty()
            chart = st.empty()
            rows = []

            def on_epoch(row, total):
                rows.append(row)
                bar.progress(row["epoch"] / total)
                status.markdown(
                    f"Epoch **{row['epoch']}/{total}** | train `{row['train_loss']:.4f}` | "
                    f"val `{row['val_loss']:.4f}` | mẫu `{row['n_samples_processed']}` | {row['elapsed']:.1f}s"
                )
                chart.line_chart(pd.DataFrame(rows).set_index("epoch")[["train_loss", "val_loss"]])

            try:
                res = train_model(
                    model_name,
                    epochs=epochs,
                    batch=batch,
                    lr=lr,
                    device_name=device_name,
                    seed=int(seed),
                    data=data,
                    on_epoch=on_epoch,
                    out_root=out_root,
                    target_idx=target_idx,
                    mode=train_mode,
                    palf_config=palf_cfg,
                    save_sample_losses=False,
                )
            except Exception as e:
                st.error("Train thất bại")
                st.exception(e)
                st.stop()

        st.session_state["run_counter"] += 1
        res["run_id"] = st.session_state["run_counter"]
        res["config"] = {
            "mode": train_mode,
            "policy": policy_name if train_mode == "palf" else "-",
            "budget_ratio": budget_ratio if train_mode == "palf" else 1.0,
            "alpha": alpha if train_mode == "palf" else 0.0,
            "epochs": epochs,
            "batch": batch,
            "lr": lr,
            "seed": int(seed),
            "device": device_name,
            "source": source,
            "target_col": target_col,
        }
        st.session_state["results"].insert(0, res)
        del st.session_state["results"][MAX_RESULTS:]
        st.rerun()
