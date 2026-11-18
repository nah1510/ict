# Khuyến Nghị Dựa Trên Hành Vi Duyệt Web (Session-based Recommendation)

Repository phục vụ bài thuyết trình và thực nghiệm **Chủ đề 9: Khuyến nghị theo phiên duyệt web (Session-based Recommendation)** — Môn **Xu hướng mới trong ICT (Lớp CHKHMT15B1 - HK1 2026-2027)**, Trường Đại học Công nghiệp TP. Hồ Chí Minh (IUH).

---

## 1. Cấu trúc thư mục

```text
ict/
├── README.md                        # Hướng dẫn cài đặt, chạy thực nghiệm & demo
├── requirements.txt                 # Thư viện Python cần thiết
├── docs/
│   ├── kich_ban_thuyet_trinh.md     # Kịch bản thuyết trình chi tiết (13 slides, phân vai 3 người, Q&A)
│   └── kich_ban_demo.md             # Kịch bản thao tác Live Demo khi thuyết trình
├── src/
│   ├── __init__.py
│   ├── preprocess.py                # Tiền xử lý, lọc phiên/item, chia theo thời gian, sinh tiền tố & dữ liệu mẫu
│   ├── metrics.py                   # Độ đo HR@K (Recall@K) và MRR@K
│   ├── baselines.py                 # POP, S-POP, Item-KNN, SKNN, V-SKNN
│   ├── gru4rec.py                   # GRU4Rec (PyTorch) với session-parallel minibatch & CE/BPR/TOP1 loss
│   └── train_eval.py                # Pipeline chạy thực nghiệm đa seed, phân tích theo độ dài phiên & vẽ biểu đồ
├── app_demo.py                      # Ứng dụng Streamlit cho Live Demo tương tác thời gian thực
├── data/                            # Thư mục chứa dữ liệu thô & đã xử lý (tự tạo khi chạy)
└── results/                         # Bảng kết quả (.csv), checkpoint (.pt) và biểu đồ (.png)
```

---

## 2. Cài đặt môi trường

Yêu cầu Python 3.9+ (khuyến nghị tạo môi trường ảo `venv` hoặc `conda`):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Trên Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Chuẩn bị dữ liệu

Hệ thống hỗ trợ 2 chế độ:

### Cách A: Chạy nhanh với dữ liệu mẫu mô phỏng (khuyên dùng để test pipeline & demo ngay)
Không cần tải file lớn, script sẽ tự sinh bộ dữ liệu hành vi duyệt web giả lập có cấu trúc ý định theo phiên:
```bash
python -m src.preprocess --sample --output_dir data/processed
```

### Cách B: Chạy trên dữ liệu thực tế công khai
Chuẩn bị file CSV có 3 cột `session_id,item_id,timestamp` (hỗ trợ **RetailRocket**, **YOOCHOOSE 1/64**, **DIGINETICA** sau khi chuẩn hóa tên cột):
```bash
python -m src.preprocess --input_csv data/raw/events.csv --output_dir data/processed
```

**Quy trình tiền xử lý áp dụng (chuẩn Chủ đề 9):**
1. Loại bỏ sản phẩm xuất hiện dưới 5 lần (`min_item_freq = 5`).
2. Loại bỏ phiên có độ dài 1 (`min_session_len = 2`).
3. Chia tập train / validation / test **theo thời gian** (không chia ngẫu nhiên để tránh rò rỉ dữ liệu tương lai).
4. Tăng cường dữ liệu (Data Augmentation) bằng cách sinh mọi tiền tố của phiên: $[i_1, i_2, i_3] \rightarrow ([i_1] \to i_2),\ ([i_1, i_2] \to i_3)$.

---

## 4. Chạy thực nghiệm & So sánh mô hình

Chạy toàn bộ benchmark (`POP`, `S-POP`, `Item-KNN`, `SKNN`, `V-SKNN`, `GRU4Rec (CE)`, `GRU4Rec (BPR)`) qua **3 random seeds** (`42, 43, 44`) để báo cáo trung bình $\pm$ độ lệch chuẩn:

```bash
python -m src.train_eval \
  --data_dir data/processed \
  --results_dir results \
  --seeds 42 43 44 \
  --epochs 10 \
  --k 20
```

Kết quả đầu ra trong thư mục `results/`:
- `benchmark.csv`: Bảng `HR@20` và `MRR@20` (mean $\pm$ std) của tất cả mô hình.
- `session_length_analysis.csv`: Phân tích hiệu năng theo độ dài phiên (Ngắn: 2–3, Trung bình: 4–7, Dài: $\ge 8$).
- `gru4rec_ce.pt` & `vsknn.pkl`: Checkpoint đã huấn luyện dùng cho Live Demo.
- `benchmark_chart.png` & `session_length_chart.png`: Biểu đồ trực quan đưa vào slide.

---

## 5. Chạy Live Demo tương tác (Streamlit)

Sau khi đã chạy `preprocess` và `train_eval`, khởi chạy giao diện web demo:

```bash
streamlit run app_demo.py
```

Xem chi tiết kịch bản thao tác demo trong [docs/kich_ban_demo.md](docs/kich_ban_demo.md) và kịch bản thuyết trình trong [docs/kich_ban_thuyet_trinh.md](docs/kich_ban_thuyet_trinh.md).

---

## 6. Tài liệu tham khảo chính

1. **Hidasi B., Karatzoglou A., Baltrunas L., Tikk D. (2016).** *Session-based recommendations with recurrent neural networks.* ICLR 2016. *(Bài đọc cốt lõi)*
2. **Ludewig M., Jannach D. (2018).** *Evaluation of session-based recommendation algorithms.* User Modeling and User-Adapted Interaction, 28(4-5).
3. **Li J., Ren P., Chen Z., Ren Z., Lian T., Ma J. (2017).** *Neural attentive session-based recommendation (NARM).* CIKM 2017.
4. **Wu S., Tang Y., Zhu Y., Wang L., Xie X., Tan T. (2019).** *Session-based recommendation with graph neural networks (SR-GNN).* AAAI 2019.
