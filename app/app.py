import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from palf.core.config import PalfConfig
from palf.data import prepare_data, prepare_weather_data
from palf.evaluation.demo_loader import (
    discover_saved_runs,
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
        "1. So sánh cạnh nhau (Ngoại tuyến - Kết quả chạy sẵn 30 epoch)",
        "2. Quan sát PALF (Trọng số, Tỷ lệ mẫu & Learning State)",
        "3. CHẠY THU NHỎ (SMOKE / DEMO TRỰC TIẾP)",
    ],
    horizontal=True,
)
st.divider()

saved_runs = discover_saved_runs()
run_map = {r["label"]: r for r in saved_runs}

# =====================================================================
# TRANG 1: SO SANH CANH NHAU (NGOAI TUYEN)
# =====================================================================
if page.startswith("1."):
    st.subheader("So sánh cạnh nhau giữa các lần chạy sẵn 30 epoch trên đĩa")
    st.caption(
        "Chế độ mặc định (ngoại tuyến): đọc trực tiếp các lần chạy thật 30 epoch từ "
        "`experiments/results/baseline/jena` và `experiments/results/palf/jena` "
        "(tách biệt hoàn toàn khỏi các lần chạy thu nhỏ trong `experiments/results/smoke/`)."
    )

    if not saved_runs:
        st.warning("Chưa có kết quả nào trong `experiments/results/`. Hãy chạy `run_observation.py` trước.")
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
            overall_df, pat_df, default_target = summarize_side_by_side(chosen_runs)

            custom_target = st.number_input(
                "Ngưỡng `val_target` (mặc định = 1.02 × best val loss của lần chạy đối chứng đầu tiên):",
                value=float(round(default_target, 5)),
                step=0.001,
                format="%.5f",
            )
            overall_df, pat_df, _ = summarize_side_by_side(chosen_runs, val_target=custom_target)

            st.markdown("#### Bảng tổng hợp: Test MAE/RMSE (°C) & Tốc độ đạt ngưỡng Val")
            st.dataframe(overall_df, hide_index=True, use_container_width=True)

            c_plot0, c_plot1, c_plot2 = st.columns(3)
            with c_plot0:
                st.markdown("#### 1. Đường `val_loss` theo **Epoch**")
                fig0, ax0 = plt.subplots(figsize=(5.5, 3.8))
                for r in chosen_runs:
                    h = r["history"]
                    ax0.plot(h["epoch"], h["val_loss"], marker="o", markersize=3, label=r["label"])
                ax0.axhline(custom_target, color="red", linestyle="--", linewidth=1, label=f"val_target={custom_target:.4f}")
                ax0.set_xlabel("Epoch")
                ax0.set_ylabel("Validation MSE (chuẩn hóa)")
                ax0.grid(True, alpha=0.3)
                ax0.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig0)
                plt.close(fig0)

            with c_plot1:
                st.markdown("#### 2. Đường `val_loss` theo **Số mẫu tích lũy**")
                fig1, ax1 = plt.subplots(figsize=(5.5, 3.8))
                for r in chosen_runs:
                    h = r["history"]
                    ax1.plot(h["cum_samples"], h["val_loss"], marker="o", markersize=3, label=r["label"])
                ax1.axhline(custom_target, color="red", linestyle="--", linewidth=1, label=f"val_target={custom_target:.4f}")
                ax1.set_xlabel("Số mẫu train tích lũy (cumulative samples)")
                ax1.set_ylabel("Validation MSE (chuẩn hóa)")
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            with c_plot2:
                st.markdown("#### 3. Đường `val_loss` theo **Thời gian tích lũy (s)**")
                fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
                for r in chosen_runs:
                    h = r["history"]
                    ax2.plot(h["cum_time_sec"], h["val_loss"], marker="s", markersize=3, label=r["label"])
                ax2.axhline(custom_target, color="red", linestyle="--", linewidth=1, label=f"val_target={custom_target:.4f}")
                ax2.set_xlabel("Thời gian huấn luyện tích lũy (giây)")
                ax2.set_ylabel("Validation MSE (chuẩn hóa)")
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            st.markdown("#### Bảng MAE và RMSE (°C) trên tập Test theo từng Pattern (của các lần chạy đang chọn)")
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

            st.markdown("#### Bảng tổng hợp Test MAE & RMSE (°C) theo Pattern qua các Seed (`Mean ± Std`)")
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
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Chính sách (Policy)", str(p_cfg.get("policy", "-")).upper())
            m2.metric("Budget Ratio (ρ)", f"{p_cfg.get('budget_ratio', 1.0):g}")
            m3.metric("Hệ số α", f"{p_cfg.get('alpha', 1.0):g}")
            m4.metric("Khoảng [w_min, w_max]", f"[{p_cfg.get('w_min', 0.25):g}, {p_cfg.get('w_max', 3.0):g}]")
            m5.metric("Warm-up / Window", f"{p_cfg.get('warmup_epochs', 3)} / {p_cfg.get('window', 3)}")

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
    st.subheader("CHẠY THU NHỎ (SMOKE / DEMO TRỰC TIẾP)")
    st.warning(
        "⚠️ **CHẠY THU NHỎ (SMOKE / DEMO TRỰC TIẾP):** Huấn luyện trực tiếp trên giao diện chỉ dành cho thử nghiệm "
        "nhanh với ít epoch và tập con dữ liệu (`subset`). Mọi kết quả ở trang này được lưu tách biệt sang "
        "`experiments/results/smoke/` và **không ghi đè hay trộn lẫn** vào danh sách thực nghiệm thật 30 epoch."
    )

    with st.sidebar:
        st.header("Dữ liệu (CHẠY THU NHỎ / SMOKE)")
        source = st.radio("Nguồn dữ liệu", ["Giả lập (nhanh)", "Thời tiết thật (Jena, Đức)"])
        target_label = None
        subset_size = None
        if source != "Giả lập (nhanh)":
            target_label = st.selectbox("Biến mục tiêu", list(WEATHER_TARGETS.keys()))
            subset_size = st.select_slider(
                "Giới hạn số mẫu (quy mô nhỏ)",
                options=[512, 1024, 2048, 4096],
                value=1024,
            )

        st.header("Chế độ & Cấu hình PALF")
        train_mode = st.radio("Chế độ huấn luyện", ["baseline", "palf"], horizontal=True)
        policy_name = "predictive"
        budget_ratio = 1.0
        alpha = 1.0
        w_min, w_max = 0.25, 3.0
        ema_beta = 0.5
        warmup_epochs = 3
        update_every = 1
        window = 3

        if train_mode == "palf":
            policy_name = st.selectbox("Chính sách (Policy)", ["predictive", "loss", "uniform", "state"], index=0)
            budget_ratio = st.slider("Ngân sách mẫu mỗi epoch (budget_ratio)", 0.3, 1.0, 0.7, 0.05)
            alpha = st.slider("Độ mạnh phản ứng (alpha)", 0.25, 3.0, 1.0, 0.25)
            with st.expander("Tham số PALF nâng cao"):
                w_min = st.number_input("w_min", value=0.25, min_value=0.05, max_value=1.0, step=0.05)
                w_max = st.number_input("w_max", value=3.0, min_value=1.0, max_value=10.0, step=0.5)
                ema_beta = st.slider("EMA beta", 0.0, 0.9, 0.5, 0.1)
                warmup_epochs = st.number_input("Warm-up epochs", value=3, min_value=1, max_value=10, step=1)
                update_every = st.number_input("Update every (epoch)", value=1, min_value=1, max_value=10, step=1)
                window = st.number_input("Window W", value=3, min_value=1, max_value=10, step=1)

        st.header("Cấu hình Backbone")
        model_name = st.selectbox("Loại model", list(MODEL_REGISTRY.keys()))
        st.caption(MODEL_INFO[model_name])
        epochs = st.slider("Số epoch", 2, 30, 8)
        batch = st.selectbox("Batch size", [32, 64, 128, 256], index=2)
        lr = st.select_slider(
            "Learning rate",
            options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            value=1e-3,
            format_func=lambda v: f"{v:g}",
        )
        device_name = st.selectbox("Thiết bị", ["cpu", "dml", "cuda"])
        seed = st.number_input("Seed", value=0, step=1)
        run = st.button("Bắt đầu train", type="primary")
        clear = st.button("Xóa kết quả")

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
