# Kịch Bản Thuyết Trình: Khuyến Nghị Theo Phiên Duyệt Web (Session-based Recommendation)

- **Môn học:** Xu hướng mới trong ICT — Lớp CHKHMT15B1 — HK1 (2026 - 2027)
- **Giảng viên hướng dẫn:** TS. Lê Thị Vĩnh Thanh
- **Chủ đề:** Chủ đề 9 — Khuyến nghị theo phiên duyệt web (*Session-based Recommendation*)
- **Tổng thời gian dự kiến:** 16–18 phút (Trình bày + Demo) + 5 phút Hỏi đáp (Q&A)

---

## Phân công nhóm 3 học viên

| Thành viên | Vai trò & Nội dung phụ trách | Thời lượng |
|---|---|---|
| **Thành viên 1 (TV1)** | Slide 1–5: Đặt vấn đề, phân biệt Session-based vs Sequential, phát biểu bài toán & tiền xử lý | ~4.5 phút |
| **Thành viên 2 (TV2)** | Slide 6–8: Baselines (POP, S-POP, SKNN/V-SKNN), GRU4Rec (session-parallel minibatch, loss), ý tưởng NARM & SR-GNN | ~5.5 phút |
| **Thành viên 3 (TV3)** | Slide 9–13: Thiết lập thực nghiệm, **Live Demo**, bảng kết quả, phân tích ablation, bài học baseline mạnh & kết luận | ~7 phút |

---

## Kịch bản chi tiết từng Slide

### Slide 1: Trang bìa *(0:30 — TV1)*
- **Nội dung trên slide:** Tên đề tài, tên môn học, tên giảng viên, danh sách 3 thành viên.
- **Lời nói gợi ý:** *"Kính chào cô và các bạn. Nhóm chúng em xin trình bày Chủ đề 9: Khuyến nghị dựa trên hành vi duyệt web (Session-based Recommendation). Nội dung gồm 4 phần: Đặt vấn đề, Phương pháp, Demo & Thực nghiệm, và Thảo luận."*

### Slide 2: Bối cảnh & Động lực *(1:00 — TV1)*
- **Hình minh họa:** Một người dùng ẩn danh (chưa đăng nhập) truy cập trang TMĐT, click xem *Điện thoại → Ốp lưng → ...*.
- **Nội dung trên slide:**
  - Phần lớn lưu lượng TMĐT/tin tức là khách vãng lai (không có `user_id`, cookie bị xóa hoặc hết hạn).
  - Ý định trong phiên mang tính tức thời (hôm nay mua quà tặng khác với sở thích cá nhân mọi ngày).
  - *Câu hỏi:* Làm thế nào dự đoán sản phẩm tiếp theo chỉ từ chuỗi click ngắn hiện tại?

### Slide 3: Phân biệt Session-based RS vs. Sequential RS (Chủ đề 15) *(1:00 — TV1)*
- **Bảng so sánh trực quan:**

| Tiêu chí | Session-based RS (Chủ đề 9) | Sequential RS (Chủ đề 15) |
|---|---|---|
| Danh tính người dùng | **Không có** (ẩn danh theo phiên) | **Có** `user_id` dài hạn |
| Độ dài ngữ cảnh | Ngắn, nhất thời (vài đến vài chục click) | Dài, tích lũy nhiều phiên theo thời gian |
| Mô hình tiêu biểu | SKNN, V-SKNN, GRU4Rec, NARM, SR-GNN | FPMC, Caser, SASRec, BERT4Rec |

### Slide 4: Phát biểu bài toán *(1:00 — TV1)*
- **Công thức trên slide:**
  - Một phiên duyệt web: $s = [i_1, i_2, \dots, i_t]$ với $i_k \in I$.
  - Đầu vào: Chuỗi tiền tố hiện tại $[i_1, \dots, i_t]$.
  - Đầu ra: Phân phối xác suất cho sản phẩm kế tiếp $i_{t+1}$:
    $$P(i_{t+1} = j \mid i_1, \dots, i_t) = \text{softmax}_j(h_t^\top e_j)$$
  - Trong đó $h_t$ là vector biểu diễn phiên ở bước $t$, $e_j$ là embedding của sản phẩm $j$.

### Slide 5: Tiền xử lý & Tăng cường dữ liệu *(1:00 — TV1)*
- **Sơ đồ pipeline:** Raw logs $\to$ Lọc item ($< 5$ lần) $\to$ Lọc phiên (độ dài $= 1$) $\to$ Chia theo thời gian (ngày/tuần cuối làm test) $\to$ **Sequence Splitting**.
- **Ví dụ minh họa trên slide:**
  - Phiên $[A, B, C, D]$ $\longrightarrow$ 3 mẫu huấn luyện:
    1. $[A] \to B$
    2. $[A, B] \to C$
    3. $[A, B, C] \to D$
- *Chuyển giao:* *"Để giải quyết bài toán trên, mời [TV2] trình bày các phương pháp từ cơ sở đến học sâu."*

### Slide 6: Nhóm Baseline & Hàng xóm theo phiên (SKNN / V-SKNN) *(1:45 — TV2)*
- **Nội dung trên slide:**
  - `POP` (phổ biến toàn cục), `S-POP` (phổ biến trong chính phiên hiện tại), `Item-KNN` (đồng xuất hiện cặp item).
  - **SKNN (Session-KNN):** Tìm $K$ phiên lịch sử giống phiên hiện tại nhất (Cosine trên tập item), chấm điểm item ứng viên theo tổng độ tương tự của các phiên chứa nó.
  - **V-SKNN:** Thêm trọng số suy giảm theo thời gian click (click gần nhất quan trọng hơn).
- **Lời nói gợi ý:** Nhấn mạnh ưu điểm của SKNN là cực kỳ đơn giản, không cần train mạng nơ-ron, cập nhật phiên mới ngay lập tức.

### Slide 7: GRU4Rec (Hidasi et al., ICLR 2016) *(2:15 — TV2)*
- **Nội dung trên slide:**
  - Kiến trúc: Item ID $\to$ Embedding $\to$ Lớp **GRU** ($z_t, r_t, \tilde{h}_t, h_t$) $\to$ Linear/Dot-product $\to$ Điểm dự đoán.
  - **Sơ đồ Session-parallel Minibatch** *(điểm ăn tiền lý thuyết)*: Thay vì pad mọi phiên về cùng độ dài, các phiên chạy song song theo bước thời gian; phiên nào kết thúc thì nạp phiên mới vào ngay slot đó.
  - **Negative sampling & Hàm mất mát:** Dùng các item khác trong cùng minibatch làm mẫu âm. So sánh **Cross-Entropy**, **BPR** ($-\ln \sigma(\hat{r}_i - \hat{r}_j)$) và **TOP1**.

### Slide 8: Mở rộng ý tưởng: NARM & SR-GNN *(1:30 — TV2)*
- **NARM (Li et al., CIKM 2017):** Dùng cơ chế **Attention** kết hợp *trạng thái cuối* (hành vi tuần tự cục bộ) và *tổng có trọng số mọi trạng thái* (mục đích chính của cả phiên).
- **SR-GNN (Wu et al., AAAI 2019):** Biểu diễn phiên thành **đồ thị có hướng** để bắt các chuyển tiếp lặp lại (ví dụ xem A $\to$ B $\to$ quay lại A).
- *Chuyển giao:* *"Mời [TV3] trình bày phần thực nghiệm, live demo và phân tích kết quả."*

### Slide 9: Thiết lập thực nghiệm & Độ đo *(1:00 — TV3)*
- **Dữ liệu:** Mô tả thống kê tập dữ liệu nhóm chạy (số phiên train/test, số item, độ dài phiên trung bình).
- **Độ đo (tại $K=20$):**
  - **HR@20 (Hit Rate / Recall@20):** Tỉ lệ phiên có sản phẩm kế tiếp nằm trong Top-20.
  - **MRR@20 (Mean Reciprocal Rank):** Trung bình của $1/\text{rank}$ nếu nằm trong Top-20 (đo mức độ xếp hạng cao).
- *Ví dụ tính tay nhanh:* Nếu item đúng đứng hạng 4 $\Rightarrow \text{HR@20}=1,\ \text{MRR@20}=1/4=0.25$.

### Slide 10: LIVE DEMO *(3:00 — TV3)*
- Chuyển sang ứng dụng Streamlit (`app_demo.py`) — thực hiện đúng theo [kich_ban_demo.md](kich_ban_demo.md).

### Slide 11: Kết quả thực nghiệm & Phân tích Ablation *(1:30 — TV3)*
- **Bảng kết quả** (trung bình $\pm$ std qua 3 seed từ `results/benchmark.csv`): so sánh `POP`, `S-POP`, `Item-KNN`, `SKNN`, `V-SKNN`, `GRU4Rec (CE)`, `GRU4Rec (BPR)`.
- **Biểu đồ phân tích** (`results/session_length_chart.png`):
  - Phiên ngắn (2–3 click) vs. trung bình (4–7) vs. dài ($\ge 8$).
  - So sánh Cross-Entropy vs. BPR trong GRU4Rec.

### Slide 12: Thảo luận — Bài học về Baseline mạnh *(1:00 — TV3)*
- Dẫn kết luận nổi tiếng của **Ludewig & Jannach (2018)**: Khi tinh chỉnh kỹ, `V-SKNN` thường cạnh tranh ngang ngửa hoặc vượt `GRU4Rec` trên nhiều bộ dữ liệu TMĐT.
- **Khi nào dùng gì?**
  - Cần triển khai nhanh, ít tài nguyên, cập nhật real-time: `V-SKNN`.
  - Chuỗi có thứ tự phức tạp, cần học biểu diễn ẩn hoặc kết hợp thuộc tính: `GRU4Rec` / `NARM` / `SR-GNN`.

### Slide 13: Kết luận, Hạn chế & Tài liệu tham khảo *(0:30 — TV3)*
- Tóm tắt 3 đóng góp chính của bài trình bày.
- Hạn chế: Chưa xử lý item mới hoàn toàn (item cold-start), chưa tận dụng metadata sản phẩm (giá, danh mục, hình ảnh).
- Tài liệu tham khảo: Hidasi et al. (2016), Ludewig & Jannach (2018), Li et al. (2017), Wu et al. (2019).

---

## Bộ câu hỏi Q&A dự kiến (Chuẩn bị cho 10% điểm trả lời câu hỏi)

1. **Hỏi:** *Tại sao không dùng Collaborative Filtering (UserKNN/ItemKNN) hay Matrix Factorization cho bài toán này?*
   - **Đáp:** Khách vãng lai không có `user_id` ổn định nên không xây được vector người dùng $p_u$. Ngoài ra, ý định trong một phiên rất ngắn hạn và thay đổi liên tục theo từng cú click.
2. **Hỏi:** *Tại sao khi chia tập train/test phải chia theo thời gian mà không chia ngẫu nhiên?*
   - **Đáp:** Chia ngẫu nhiên làm rò rỉ thông tin tương lai (data leakage) — mô hình học từ các xu hướng/phiên ở ngày sau để dự đoán ngày trước, khiến kết quả offline bị ảo và không phản ánh thực tế triển khai.
3. **Hỏi:** *Cơ chế Session-parallel minibatch của GRU4Rec khác gì với cách padding thông thường?*
   - **Đáp:** Độ dài phiên rất chênh lệch; padding nhiều gây lãng phí tính toán trên các token `<PAD>`. Session-parallel minibatch ghép các phiên liên tiếp vào từng cột batch để bước nào cũng tính trên dữ liệu thật, đồng thời tận dụng các item khác trong batch làm mẫu âm (negative samples).
4. **Hỏi:** *Nếu sản phẩm mới vừa được thêm lên sàn (chưa ai click bao giờ) thì GRU4Rec và SKNN có gợi ý được không?*
   - **Đáp:** Cả hai đều gặp vấn đề **item cold-start** vì chỉ dựa trên ID và lịch sử click. Để khắc phục cần kết hợp đặc trưng nội dung (Content-based / metadata embedding) hoặc mô hình lai.
