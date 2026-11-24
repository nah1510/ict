# Outline Bài Thuyết Trình & Báo Cáo Theo 8 Bài Báo Liên Quan (Chủ đề 9)

Tài liệu này hệ thống hóa **8 bài báo gốc** trong thư mục [papers/](../papers) theo đúng cấu trúc **Mục 2: Công trình liên quan (Related Work)** trong *Sườn báo cáo khoa học CNTT*, đồng thời ánh xạ trực tiếp vào từng phần của bài thuyết trình.

---

## Phần A. Outline tổng thể bài thuyết trình gắn với các bài báo

1. **Nền tảng kiến trúc hồi quy (RNN / GRU):**
   - `08-Chung-2014-Gated-RNNs.pdf` → Cơ sở lý thuyết của cổng cập nhật ($z_t$) và cổng xóa ($r_t$) trong GRU.
2. **Khởi đầu kỷ nguyên Deep Learning cho Session-based RS:**
   - `01-Hidasi-2016-GRU4Rec.pdf` *(Đọc trước tiên)* → Đề xuất `GRU4Rec`, cơ chế *session-parallel minibatch*, hàm mất mát `BPR` & `TOP1`.
   - `02-Hidasi-Karatzoglou-2018-GRU4Rec-Top-k.pdf` → Cải tiến hàm mất mát (`BPR-max`, `TOP1-max`) và chiến lược lấy mẫu âm cho dữ liệu quy mô lớn.
3. **Trào lưu phản biện & sức mạnh của Baseline hàng xóm (Neighborhood):**
   - `06-Jannach-Ludewig-2017-RNN-and-Session-kNN.pdf` → Kết hợp `GRU4Rec` và `SKNN`, phát hiện hàng xóm theo phiên rất mạnh ở ngữ cảnh ngắn hạn.
   - `07-Ludewig-Jannach-2018-Evaluation.pdf` → Đánh giá toàn diện: khi tinh chỉnh đúng, `SKNN` / `V-SKNN` thường cạnh tranh ngang ngửa hoặc vượt `GRU4Rec`.
4. **Mở rộng sang Cơ chế Chú ý (Attention) & Bộ nhớ ngắn hạn:**
   - `03-Li-2017-NARM.pdf` → `NARM`: kết hợp hành vi tuần tự (trạng thái cuối) và mục đích chính của phiên (global attention).
   - `04-Liu-2018-STAMP.pdf` → `STAMP`: ưu tiên thao tác click cuối cùng (short-term memory) kết hợp ngữ cảnh phiên, không cần RNN.
5. **Mở rộng sang Đồ thị (Graph Neural Networks):**
   - `05-Wu-2019-SR-GNN.pdf` → `SR-GNN`: biểu diễn phiên thành đồ thị có hướng để bắt các chuyển tiếp lặp lại ($A \to B \to A$).

---

## Phần B. Phân tích chi tiết từng nhóm bài báo (Phục vụ viết Mục 2 Báo cáo)

### Nhóm 1: Nền tảng & Mô hình RNN theo phiên (GRU4Rec)
| Bài báo | Mục tiêu | Ý tưởng chính | Kết quả & Đóng góp | Hạn chế |
|---|---|---|---|---|
| **Chung et al. (2014)** (`08`) | So sánh LSTM và GRU trên dữ liệu chuỗi | Đề xuất đánh giá thực nghiệm GRU với 2 cổng ($z_t, r_t$), ít tham số hơn LSTM | GRU đạt hiệu năng tương đương/nhỉnh hơn LSTM với tốc độ hội tụ nhanh | Chưa thiết kế riêng cho bài toán khuyến nghị & tập nhãn lớn |
| **Hidasi et al. (2016)** (`01`) | Dự đoán click kế tiếp cho phiên ẩn danh | Áp dụng GRU + *session-parallel minibatch* + negative sampling trong batch (`BPR`, `TOP1`) | Vượt xa `POP`, `S-POP`, `Item-KNN` tại thời điểm công bố (+20–30% Recall@20) | Chỉ dựa vào trạng thái cuối $h_t$, dễ quên ý định đầu phiên nếu phiên dài |
| **Hidasi & Karatzoglou (2018)** (`02`) | Khắc phục suy giảm gradient ở bài gốc khi tập item lớn | Đề xuất hàm mất mát `TOP1-max`, `BPR-max` và thêm mẫu âm theo độ phổ biến | Cải thiện đáng kể `Recall@20` & `MRR@20` so với `GRU4Rec (2016)` | Vẫn giữ giả định chuyển tiếp tuần tự một chiều |

### Nhóm 2: Phương pháp Hàng xóm & Đánh giá công bằng (SKNN / V-SKNN)
| Bài báo | Mục tiêu | Ý tưởng chính | Kết quả & Đóng góp | Hạn chế |
|---|---|---|---|---|
| **Jannach & Ludewig (2017)** (`06`) | Kiểm chứng `GRU4Rec` trước phương pháp hàng xóm | Đề xuất `SKNN` (cosine trên tập item của phiên) và mô hình lai `RNN + SKNN` | `SKNN` đạt kết quả sát `GRU4Rec`; kết hợp cả hai cho hiệu năng cao nhất | Chi phí tra cứu láng giềng tăng khi lịch sử phiên quá lớn |
| **Ludewig & Jannach (2018)** (`07`) | Chuẩn hóa quy trình đánh giá Session-based RS | So sánh có hệ thống `SKNN`, `V-SKNN` (trọng số thời gian), `GRU4Rec`, `NARM` trên nhiều bộ dữ liệu | Chứng minh `V-SKNN` thường ngang hoặc vượt mô hình sâu trên `YOOCHOOSE`, `DIGINETICA` | Không học được biểu diễn ẩn phi tuyến hay tích hợp metadata phức tạp |

### Nhóm 3: Cơ chế Chú ý (Attention) & Mạng Đồ thị (GNN)
| Bài báo | Mục tiêu | Ý tưởng chính | Kết quả & Đóng góp | Hạn chế |
|---|---|---|---|---|
| **Li et al. (2017) — NARM** (`03`) | Nắm đồng thời mục đích chính và hành vi tuần tự | Encoder toàn cục (attention trên mọi $h_t$) + Encoder cục bộ ($h_t$ cuối) | Cải thiện rõ trên cả phiên ngắn và dài so với `GRU4Rec` | Vẫn phụ thuộc mạng RNN tuần tự khi mã hóa |
| **Liu et al. (2018) — STAMP** (`04`) | Nhấn mạnh ý định tức thời ở cú click cuối | Thay RNN bằng mạng MLP + Attention ưu tiên item cuối $x_t$ (short-term priority) | Nhanh hơn RNN, xử lý tốt hiện tượng đổi ý định đột ngột (intent drift) | Bỏ qua một phần thông tin thứ tự chặt chẽ ở giữa phiên |
| **Wu et al. (2019) — SR-GNN** (`05`) | Bắt các chuyển tiếp phức tạp & lặp lại trong phiên | Chuyển phiên thành đồ thị có hướng, dùng Gated GNN học embedding nút + attention | Vượt `GRU4Rec`, `NARM`, `STAMP` trên `YOOCHOOSE` và `DIGINETICA` | Độ phức tạp tính toán và bộ nhớ cao hơn đáng kể |

---

## Phần C. Kết nối vào Khoảng trống Nghiên cứu (Research Gap) của Nhóm

Từ tổng hợp 8 bài báo trên, nhóm xác định trọng tâm thực nghiệm cho bài thuyết trình & báo cáo:
1. **Khoảng trống kiểm chứng:** So sánh trực tiếp, công bằng (cùng tập dữ liệu, cùng cách chia theo thời gian, đa seed) giữa nhóm **Baseline hàng xóm** (`POP`, `S-POP`, `Item-KNN`, `SKNN`, `V-SKNN`) và **Học sâu** (`GRU4Rec` với `CE` và `BPR`).
2. **Phân tích điều kiện biên:** Kiểm chứng độ nhạy của từng mô hình theo **độ dài phiên** (ngắn / trung bình / dài) và **hàm mất mát**.
