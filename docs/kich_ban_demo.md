# Kịch Bản Thao Tác Live Demo (Thời lượng: ~3 phút)

## Chuẩn bị trước giờ thuyết trình
Chạy sẵn trước ở nhà/trước buổi học để lưu checkpoint và mở sẵn tab trình duyệt:
```bash
# 1. Tạo dữ liệu (hoặc dùng dữ liệu thật đã tải)
python -m src.preprocess --sample --output_dir data/processed

# 2. Huấn luyện & xuất bảng kết quả + checkpoint
python -m src.train_eval --data_dir data/processed --results_dir results --seeds 42 43 44 --epochs 10 --k 20

# 3. Khởi động ứng dụng demo
streamlit run app_demo.py
```

---

## Luồng thao tác Live Demo trên màn hình (TV3 thực hiện)

### Cảnh 1: Giới thiệu giao diện Khách vãng lai (0:30)
- Mở tab trình duyệt đang chạy `app_demo.py`.
- **Lời nói:** *"Đây là giao diện mô phỏng một khách vãng lai chưa đăng nhập vào trang thương mại điện tử. Hệ thống chưa biết gì về lịch sử trước đây của người dùng này."*

### Cảnh 2: Mô phỏng từng cú click trong phiên (1:30)
- Chọn tab **"Tự chọn sản phẩm (Interactive Click)"** (hoặc chọn một phiên mẫu từ tập Test ở sidebar).
- **Bước 1 — Click sản phẩm đầu tiên (ví dụ thuộc nhóm Điện tử / Laptop):**
  - Chỉ vào 3 cột kết quả hiển thị song song: `POP` (baseline phổ biến), `V-SKNN` (hàng xóm theo phiên), `GRU4Rec` (mạng hồi quy).
  - **Nhận xét:** *"Ở cú click đầu tiên, POP chỉ gợi ý đồ bán chạy chung chung. Trong khi đó, V-SKNN tìm các phiên lịch sử chứa sản phẩm này, còn GRU4Rec cập nhật vector trạng thái ẩn $h_1$ để dự đoán."*
- **Bước 2 — Click tiếp sản phẩm thứ 2 và thứ 3:**
  - Quan sát Top-10/Top-20 tự động cập nhật ngay lập tức.
  - Nếu chọn chế độ **Test Session Replay**, chỉ vào ô đánh dấu **Ground Truth (Sản phẩm thực tế tiếp theo)** đang nhảy vào Top-20 và giá trị `HR@20 = 1`, `MRR@20` tăng lên.
  - **Nhận xét:** *"Khi chuỗi dài hơn, mô hình nắm bắt rõ hơn ý định tức thời của phiên. Đặc biệt V-SKNN ưu tiên cú click gần nhất, còn GRU4Rec học được thứ tự chuyển tiếp."*

### Cảnh 3: Chuyển sang Tab "Kết quả Benchmark & Phân tích" (1:00)
- Bấm sang tab thứ 2 trên giao diện Streamlit (hiển thị bảng `results/benchmark.csv` và biểu đồ `results/session_length_chart.png`).
- **Lời nói:** *"Trên toàn bộ tập test qua 3 lần chạy với random seed cố định, nhóm thu được bảng so sánh HR@20 và MRR@20. Có thể thấy V-SKNN đạt hiệu năng rất cao so với các baseline đơn giản và bám sát GRU4Rec — đúng với nhận xét của Ludewig & Jannach (2018)."*
- Quay lại Slide 11 để phân tích sâu hơn về độ dài phiên và hàm mất mát.

---

## Phương án dự phòng (Fallback)
- Nếu máy chiếu/mạng gặp sự cố không chạy được Streamlit trực tiếp: mở sẵn ảnh chụp màn hình hoặc chạy trực tiếp lệnh CLI:
  ```bash
  python -m src.train_eval --data_dir data/processed --results_dir results --eval_only
  ```
  để in bảng kết quả trực tiếp trên terminal.
