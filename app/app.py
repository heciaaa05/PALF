import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from palf.data import prepare_data
from palf.models import MODEL_REGISTRY, MODEL_INFO
from palf.training.trainer import train_model

MAX_RESULTS = 4

st.set_page_config(page_title="PALF Trainer", layout="wide")
st.title("PALF – Huấn luyện mô hình dự báo chuỗi thời gian")


@st.cache_resource
def load_data():
    return prepare_data()


if "results" not in st.session_state:
    st.session_state["results"] = []
if "run_counter" not in st.session_state:
    st.session_state["run_counter"] = 0


def render_summary(results):
    if len(results) < 2:
        return
    rows = [{
        "Run": f"#{r['run_id']}",
        "Model": r["model_name"],
        "Epoch": r["config"]["epochs"],
        "LR": r["config"]["lr"],
        "Test MSE": round(r["test"]["mse"], 4),
        "Test MAE": round(r["test"]["mae"], 4),
        "Best epoch": r["best_epoch"],
        "Tham số": r["params"],
        "Thời gian (s)": round(r["elapsed"], 1),
    } for r in results]
    st.markdown("**So sánh các lần train**")
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_result(res):
    rid = res["run_id"]
    cfg = res["config"]
    with st.container(border=True):
        st.subheader(f"#{rid} · {res['model_name']}")
        st.caption(
            f"{cfg['epochs']} epoch · batch {cfg['batch']} · lr {cfg['lr']:g} · "
            f"seed {cfg['seed']} · thiết bị {cfg['device']}"
        )
        c = st.columns(5)
        c[0].metric("Test MSE", f"{res['test']['mse']:.4f}")
        c[1].metric("Test MAE", f"{res['test']['mae']:.4f}")
        c[2].metric("Best epoch", res["best_epoch"])
        c[3].metric("Số tham số", f"{res['params']:,}")
        c[4].metric("Thời gian", f"{res['elapsed']:.1f}s")

        t1, t2, t3, t4 = st.tabs(["Loss", "Theo pattern", "Ví dụ dự báo", "Dữ liệu"])

        with t1:
            st.line_chart(res["history"].set_index("epoch")[["train_loss", "val_loss"]])

        with t2:
            st.markdown("**Test theo pattern**")
            st.dataframe(res["test"]["by_pattern"])
            pl = res["pattern_log"]
            st.markdown("**Train loss trung bình theo pattern**")
            st.line_chart(pl.pivot(index="epoch", columns="pattern", values="train_loss_mean"))
            st.markdown("**Gradient norm theo pattern**")
            st.line_chart(pl.pivot(index="epoch", columns="pattern", values="grad_norm"))
            st.markdown("**Val loss theo pattern**")
            st.line_chart(pl.pivot(index="epoch", columns="pattern", values="val_loss"))

        with t3:
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

        with t4:
            st.caption(f"Đã lưu file tại: {res['out_dir']}")
            st.download_button("Tải history.csv", res["history"].to_csv(index=False),
                               "history.csv", "text/csv", key=f"dl_hist_{rid}")
            st.download_button("Tải epoch_pattern_log.csv",
                               res["pattern_log"].to_csv(index=False),
                               "epoch_pattern_log.csv", "text/csv", key=f"dl_pat_{rid}")


with st.sidebar:
    st.header("Cấu hình")
    model_name = st.selectbox("Loại model", list(MODEL_REGISTRY.keys()))
    st.caption(MODEL_INFO[model_name])
    epochs = st.slider("Số epoch", 5, 100, 30)
    batch = st.selectbox("Batch size", [32, 64, 128, 256], index=2)
    lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                          value=1e-3, format_func=lambda v: f"{v:g}")
    device_name = st.selectbox("Thiết bị", ["cpu", "dml", "cuda"])
    seed = st.number_input("Seed", value=42, step=1)
    run = st.button("Bắt đầu train", type="primary")
    clear = st.button("Xóa kết quả")

if clear:
    st.session_state["results"] = []

live = st.container()  # khu vực tiến trình, luôn nằm trên cùng

results = st.session_state["results"]
if results:
    st.caption(f"Đang lưu {len(results)}/{MAX_RESULTS} kết quả (mới nhất ở trên; "
               f"quá {MAX_RESULTS} thì kết quả cũ nhất bị bỏ).")
render_summary(results)
for r in results:
    render_result(r)

if run:
    data = load_data()
    with live:
        st.subheader(f"Đang train: {model_name}")
        bar = st.progress(0.0)
        status = st.empty()
        chart = st.empty()
        rows = []

        def on_epoch(row, total):
            rows.append(row)
            bar.progress(row["epoch"] / total)
            status.markdown(
                f"Epoch **{row['epoch']}/{total}** | train `{row['train_loss']:.4f}` | "
                f"val `{row['val_loss']:.4f}` | {row['elapsed']:.1f}s"
            )
            chart.line_chart(pd.DataFrame(rows).set_index("epoch")[["train_loss", "val_loss"]])

        try:
            res = train_model(model_name, epochs=epochs, batch=batch, lr=lr,
                              device_name=device_name, seed=int(seed),
                              data=data, on_epoch=on_epoch)
        except Exception as e:
            st.error("Train thất bại")
            st.exception(e)
            st.stop()

    st.session_state["run_counter"] += 1
    res["run_id"] = st.session_state["run_counter"]
    res["config"] = {"epochs": epochs, "batch": batch, "lr": lr,
                     "seed": int(seed), "device": device_name}
    st.session_state["results"].insert(0, res)
    del st.session_state["results"][MAX_RESULTS:]
    st.rerun()