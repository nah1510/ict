"""Streamlit Live Demo for Session-based Recommendation (Chủ đề 9 - IUH)."""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.gru4rec import GRU4RecRecommender
from src.metrics import hit_rate_at_k, mrr_at_k

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


@st.cache_resource
def load_artifacts():
    with open(DATA_DIR / "test_sessions.pkl", "rb") as f:
        test_sessions = pickle.load(f)
    with open(DATA_DIR / "item_map.json", "r", encoding="utf-8") as f:
        item_map = json.load(f)

    idx_to_name: Dict[int, str] = {}
    meta_path = DATA_DIR / "item_meta.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        for _, row in meta_df.iterrows():
            idx_to_name[int(row["item_idx"])] = f"{row['item_name']} [{row['category']}]"
    else:
        for raw, idx in item_map.items():
            idx_to_name[int(idx)] = f"Item #{raw}"

    with open(RESULTS_DIR / "pop.pkl", "rb") as f:
        pop_model = pickle.load(f)
    with open(RESULTS_DIR / "vsknn.pkl", "rb") as f:
        vsknn_model = pickle.load(f)
    gru_model = GRU4RecRecommender.load(str(RESULTS_DIR / "gru4rec_ce.pt"), device="cpu")

    return test_sessions, idx_to_name, pop_model, vsknn_model, gru_model


def format_recs(recs: List[int], idx_to_name: Dict[int, str], target: int | None = None) -> pd.DataFrame:
    rows = []
    for rank, item_idx in enumerate(recs, start=1):
        label = idx_to_name.get(item_idx, f"Item {item_idx}")
        if target is not None and item_idx == target:
            label = f"🎯 {label} (ĐÚNG)"
        rows.append({"Hạng": rank, "ID": item_idx, "Sản phẩm": label})
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Session-based RS Demo", layout="wide")
    st.title("🛒 Demo: Khuyến Nghị Dựa Trên Hành Vi Duyệt Web (Session-based RS)")
    st.caption("Môn: Xu hướng mới trong ICT (CHKHMT15B1) — Chủ đề 9: POP vs. V-SKNN vs. GRU4Rec")

    if not (RESULTS_DIR / "gru4rec_ce.pt").exists():
        st.error(
            "Chưa tìm thấy checkpoint! Hãy chạy:\n"
            "`python -m src.preprocess --sample --output_dir data/processed` và "
            "`python -m src.train_eval --data_dir data/processed --results_dir results`"
        )
        return

    test_sessions, idx_to_name, pop_model, vsknn_model, gru_model = load_artifacts()

    tab_live, tab_bench = st.tabs(["⚡ Live Session Simulator", "📊 Kết quả Benchmark & Phân tích"])

    with tab_live:
        mode = st.radio(
            "Chế độ mô phỏng:",
            ["Tự chọn sản phẩm (Interactive Click)", "Phát lại phiên từ tập Test (Test Session Replay)"],
            horizontal=True,
        )

        if "custom_session" not in st.session_state:
            st.session_state.custom_session = []

        target_item = None
        current_prefix: List[int] = []

        if mode == "Tự chọn sản phẩm (Interactive Click)":
            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                all_items = sorted(idx_to_name.keys())
                selected = st.selectbox(
                    "Chọn sản phẩm khách vãng lai vừa click:",
                    options=all_items,
                    format_func=lambda x: f"[{x}] {idx_to_name[x]}",
                )
            with col_btn:
                st.write("")
                st.write("")
                if st.button("➕ Click sản phẩm này"):
                    st.session_state.custom_session.append(selected)
                if st.button("🔄 Xóa phiên (Khách mới)"):
                    st.session_state.custom_session = []
            current_prefix = st.session_state.custom_session
        else:
            valid_test = [s for s in test_sessions if len(s) >= 3]
            sess_idx = st.slider("Chọn chỉ số phiên Test:", 0, max(0, len(valid_test) - 1), 0)
            chosen_sess = valid_test[sess_idx]
            step = st.slider("Bước thời gian t (số cú click đã thực hiện):", 1, len(chosen_sess) - 1, 1)
            current_prefix = chosen_sess[:step]
            target_item = chosen_sess[step]

        st.subheader("Chuỗi hành vi trong phiên hiện tại (Khách ẩn danh):")
        if not current_prefix:
            st.info("Phiên trống — hãy click sản phẩm đầu tiên hoặc chọn chế độ Test Session Replay.")
        else:
            chain_str = "  ➜  ".join(f"`{idx_to_name.get(i, i)}`" for i in current_prefix)
            st.markdown(chain_str)
            if target_item is not None:
                st.success(f"🎯 Sản phẩm thực tế kế tiếp (Ground Truth): **{idx_to_name.get(target_item, target_item)}**")

            top_k = 10
            eval_k = 20
            c1, c2, c3 = st.columns(3)
            for col, model, title in [
                (c1, pop_model, "1. POP (Baseline phổ biến)"),
                (c2, vsknn_model, "2. V-SKNN (Hàng xóm theo phiên)"),
                (c3, gru_model, "3. GRU4Rec (Mạng nơ-ron GRU)"),
            ]:
                recs_20 = model.predict_next(current_prefix, top_k=eval_k)
                with col:
                    st.markdown(f"### {title}")
                    if target_item is not None:
                        hr = hit_rate_at_k(recs_20, target_item, k=eval_k)
                        mrr = mrr_at_k(recs_20, target_item, k=eval_k)
                        st.metric(f"HR@{eval_k} | MRR@{eval_k}", f"{hr:.0f} | {mrr:.3f}")
                    st.dataframe(
                        format_recs(recs_20[:top_k], idx_to_name, target=target_item),
                        use_container_width=True,
                        hide_index=True,
                    )

    with tab_bench:
        bench_path = RESULTS_DIR / "benchmark.csv"
        len_path = RESULTS_DIR / "session_length_analysis.csv"
        if bench_path.exists():
            st.subheader("Bảng so sánh tổng thể (Mean ± Std qua nhiều random seeds)")
            st.dataframe(pd.read_csv(bench_path), use_container_width=True, hide_index=True)
        if (RESULTS_DIR / "benchmark_chart.png").exists():
            st.image(str(RESULTS_DIR / "benchmark_chart.png"), caption="So sánh HR@20 và MRR@20")
        if len_path.exists():
            st.subheader("Phân tích theo độ dài tiền tố phiên")
            st.dataframe(pd.read_csv(len_path), use_container_width=True, hide_index=True)
        if (RESULTS_DIR / "session_length_chart.png").exists():
            st.image(str(RESULTS_DIR / "session_length_chart.png"), caption="Hiệu năng theo độ dài phiên")


if __name__ == "__main__":
    main()
