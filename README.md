# Latent Timeline Graph (LTG) — Temporal Relation Extraction

Mô hình trích xuất **quan hệ thời gian giữa các sự kiện** (Event Temporal Relation Extraction) ở mức tài liệu, kết hợp **Allen's Interval Algebra** với **Graph Propagation**. Mỗi sự kiện được ánh xạ thành một *khoảng thời gian tiềm ẩn* `(s, t)` trên trục thời gian, từ đó suy ra quan hệ giữa các cặp sự kiện bằng cả tín hiệu hình học (geometry) lẫn ngữ nghĩa (relation).

Đây là phần code của khoá luận tốt nghiệp (KLTN). Repo cung cấp 2 notebook end-to-end với 2 backbone encoder khác nhau và 4 bộ dữ liệu chuẩn.

---

## 1. Nội dung repo

```
.
├── data/                        # 4 dataset, mỗi dataset có train/dev/test.csv
│   ├── MATRES/MATRES/
│   ├── TBD/TBD/
│   ├── TDDAutoProcessed/TDDAutoProcessed/
│   └── TDDManProcessed/TDDManProcessed/
├── ltg-final-roberta.ipynb      # pipeline đầy đủ, backbone = roberta-base
├── ltg-final-bert-base.ipynb    # pipeline đầy đủ, backbone = bert-base-uncased
└── README.md
```

> Hai notebook **giống hệt nhau** ngoài một tham số duy nhất: `model_name` (`roberta-base` vs `bert-base-uncased`). Mọi mô tả kiến trúc dưới đây áp dụng cho cả hai.

---

## 2. Dữ liệu

Mỗi dataset là một bài toán phân loại quan hệ thời gian giữa cặp sự kiện trong cùng một văn bản. Tất cả file CSV dùng **chung một schema 11 cột**:

| Cột | Ý nghĩa |
|---|---|
| `entity1_id`, `entity2_id` | ID của sự kiện 1 và 2 trong document |
| `entity1_start`, `entity1_end` | char-span (vị trí ký tự) của sự kiện 1 trong `text` |
| `entity2_start`, `entity2_end` | char-span của sự kiện 2 |
| `entity1_text`, `entity2_text` | chuỗi từ kích hoạt (trigger) của mỗi sự kiện |
| `document_id` | định danh tài liệu (mọi cặp cùng document chia sẻ cùng `text`) |
| `text` | toàn văn tài liệu |
| `label` | nhãn quan hệ thời gian (xem bảng dưới) |

### Các dataset và tập nhãn

| Dataset | Thư mục | Train / Dev / Test (số cặp) | Tập nhãn chuẩn |
|---|---|---|---|
| **MATRES** | `data/MATRES/MATRES/` | ~282k / ~30k / ~16k | BEFORE, AFTER, EQUAL, VAGUE |
| **TBD** (TimeBank-Dense) | `data/TBD/TBD/` | ~4.0k / 630 / 1.4k | BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS, VAGUE |
| **TDDMan** (TDDiscourse, manual) | `data/TDDManProcessed/TDDManProcessed/` | ~4.0k / 651 / 1.5k | BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS |
| **TDDAuto** (TDDiscourse, auto) | `data/TDDAutoProcessed/TDDAutoProcessed/` | ~32k / 1.4k / 4.3k | BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS |

> *Số dòng ở trên tính cả header. Đếm chính xác bằng `wc -l data/<DS>/<DS>/*.csv`.*

**Ghi chú về nhãn:**
- TDDMan/TDDAuto lưu nhãn ở dạng viết tắt thô (`b`, `a`, `i`, `ii`, `s`) → notebook tự ánh xạ về nhãn chuẩn (`BEFORE`, `AFTER`, `INCLUDES`, `IS_INCLUDED`, `SIMULTANEOUS`) qua bảng `aliases` trong `DATASET_REGISTRY`.
- Nhãn **VAGUE** (chỉ có ở MATRES, TBD) được **loại khỏi tính F1** (macro/micro) nhưng vẫn dùng trong huấn luyện.
- MATRES, TBD đã có sẵn `dev.csv`; TDDMan/TDDAuto được tách dev từ train theo `document` (`dev_ratio=0.1`) khi `auto_split_dev=True`.

---

## 3. Kiến trúc mô hình

Luồng xử lý: **encode document → pool sự kiện → toạ độ thời gian tiềm ẩn → graph propagation → fusion 2 nhánh decoder**.

```
            ┌─────────────────────────── document text ───────────────────────────┐
            │  Sliding-window encoder (RoBERTa / BERT, phủ trọn doc kể cả >512 tok) │
            └──────────────────────────────────┬──────────────────────────────────┘
                                  attention span-pooling mỗi sự kiện
                                                ▼
                            z_i  (vector sự kiện, m sự kiện)
                                                ▼
                      EventTimeHead:  s_i,  t_i = s_i + softplus(d_i) + eps
                                                ▼
        ┌──────────────── Latent Timeline Graph Propagation (×graph_layers) ───────────────┐
        │  node = [z; s; t; d] · edge = EdgeUpdate(h, geometry φ, sim_sem, sim_time)         │
        │  DropEdge → Top-K theo attention → softmax theo node đích → GRU NodeUpdate          │
        │  Mặc định **dense graph**: mọi cặp có hướng i≠j  (m·(m-1) cạnh)                     │
        └──────────────────────────────────────┬───────────────────────────────────────────┘
                                                ▼
          ┌─────────── nhánh Allen (geometry) ───────────┐   ┌──── nhánh Relation (semantic) ────┐
          │  AllenDecoder: công thức hình học mỗi nhãn    │   │  RelationEncoder h_ij=FFN([z_i;z_j])│
          │  từ interval (s_i,t_i,s_j,t_j) → z_geo         │   │  RelationDecoder → z_rel            │
          └───────────────────────┬──────────────────────┘   └──────────────┬─────────────────────┘
                                  └──────── logits = z_rel + α·z_geo ────────┘
```

### Thành phần chính

- **`EventTimeHead`** — biến vector sự kiện `z` thành khoảng thời gian `(s, t)` với `t = s + softplus(d) + eps` (đảm bảo độ dài interval dương).
- **`GeometryComputer`** — từ cặp interval sinh ra **φ_ij** gồm 10 đặc trưng hình học (delta_s, delta_t, gap, overlap, ov_ratio, containment, dur_ratio, eq_score, …).
- **`RelationEncoder`** — `h_ij = FFN([z_i; z_j])`, biểu diễn quan hệ theo ngữ nghĩa.
- **`GraphPropagationLayer`** — một vòng lan truyền trên Latent Timeline Graph: EdgeUpdate → geometry head → message passing có attention + Top-K + DropEdge → GRU cập nhật node. Hỗ trợ 2 chế độ:
  - **dense** (`graph_dense=True`, mặc định): propagation trên **mọi** cạnh có hướng `i≠j`.
  - **sparse**: chỉ trên cặp đã gán nhãn (đồ thị hai chiều với φ và φ_rev).
- **`AllenDecoder`** — sinh logit hình học theo **đúng tập nhãn** của từng dataset; mỗi nhãn Allen có một công thức trên interval (vd. `BEFORE: s_j − t_i`, `IS_INCLUDED: min(s_i−s_j, t_j−t_i)`), riêng `VAGUE` dùng head học được.
- **`RelationDecoder`** — `[h; g; φ; cos_sim] → z_rel`.
- **Fusion** — `logits = z_rel + α·z_geo`, với `z_geo` được LayerNorm theo lớp và `α` là tham số học được theo từng lớp.

### Hàm mất mát

```
L = L_cls + λ_reg · L_reg + λ_align · L_align
```
- **`L_cls`** — cross-entropy có class weights.
- **`L_reg`** — `d² + 1/(d+ε)`, phạt interval quá dài lẫn quá ngắn.
- **`L_align`** — `‖P_g(g) − P_φ(φ)‖²`, căn chỉnh geometry dự đoán (`g`) với geometry tính từ interval (`φ`); chỉ bật khi graph đang hoạt động.

### Huấn luyện 2 giai đoạn
1. **No-graph** — học encoder + toạ độ thời gian trước (`epoch < graph_start_epoch`).
2. **Graph** — bật graph propagation + `L_align` (`epoch ≥ graph_start_epoch`).
3. **Early-stop** theo dev macro-F1 (`f1_patience`).

---

## 4. Quy trình notebook

Mỗi notebook gồm 9 phần đánh số:

1. **Imports, Device & Cấu hình** — `DATASET_REGISTRY`, `BASE_CFG`, `build_config()`.
2. **Data Pipeline** — tokenize trọn document, map char-span → token-span, gom cặp đã gán nhãn; `prepare_dataset()` trả về *bundle* dùng lại cho mọi seed.
3. **Kiến trúc mô hình** — các module ở mục 3 trên.
4. **Loss, Đánh giá & Huấn luyện** — `compute_loss`, `evaluate` (macro/micro-F1 loại VAGUE), vòng huấn luyện 2 giai đoạn.
5. **Thí nghiệm chính** — 4 dataset × 3 seed (`SEEDS = [42, 123, 2024]`), lưu mọi kết quả và giữ run micro-F1 cao nhất mỗi dataset.
6. **Tổng hợp kết quả** — bảng `mean ± std`, classification report + confusion matrix (theo seed tốt nhất), biểu đồ so sánh.
7. **Ablation Study** — 1 seed (42), các case `−Graph` / `−Relation` / `−Event` / `−Align loss` so với Full, bật/tắt qua cờ `abl_*`.
8. **Trực quan hoá Latent Timeline** — vẽ timeline tiềm ẩn cho document có micro-F1 cao nhất mỗi dataset.
9. **Phân tích phân bố toạ độ tiềm ẩn** — histogram/violin/scatter của `s`, `t`, `d = t − s` trên test set.

### Các công tắc điều khiển

| Biến | Giá trị | Tác dụng |
|---|---|---|
| `FULL_MODE` | `'all'` \| `'single'` | chạy cả 4 dataset hay 1 dataset đại diện (`FULL_SINGLE='tddman'`) |
| `SEEDS` | `[42, 123, 2024]` | các seed của thí nghiệm chính |
| `ABLATION_MODE` | `'all'` \| `'single'` \| `'off'` | phạm vi ablation |
| `graph_dense` | `True` / `False` | dense (mọi cặp) vs sparse (chỉ cặp đã gán nhãn) |
| `abl_use_graph`, `abl_relation_branch`, `abl_event_branch`, `abl_align_loss` | `True`/`False` | bật/tắt từng thành phần cho ablation (mặc định = mô hình đầy đủ) |

---

## 5. Cách chạy

Notebook được thiết kế để chạy trên **Kaggle / Colab có GPU** (cần `transformers`, `torch`; không chạy được trên CPU thuần do chi phí encode toàn document).

1. **Chuẩn bị dữ liệu.** Trỏ `data_root` của từng dataset tới đúng vị trí:
   - Trên Kaggle: tự nhận `/kaggle/input/...` (xem `DATASET_REGISTRY`).
   - Chạy local: sửa nhánh `else` trong `DATASET_REGISTRY` để trỏ về thư mục `data/` của repo này, ví dụ:
     ```python
     'matres': dict(data_root='data/MATRES/MATRES', ...)
     ```
     (lưu ý cấu trúc thư mục lồng `data/<DS>/<DS>/`).
2. **Chọn backbone.** Mở `ltg-final-roberta.ipynb` (RoBERTa) hoặc `ltg-final-bert-base.ipynb` (BERT).
3. **Chạy lần lượt từ trên xuống.** Phần 5 sẽ train `4 dataset × 3 seed`; phần 6–9 tự sinh bảng số liệu và biểu đồ.
4. **Model tốt nhất** mỗi dataset được lưu thành `.pt` trong `save_dir` (gồm `state_dict` + `cfg` + metrics; nạp lại bằng `LatentTimelineGraphModel(ckpt['cfg'])`).

### Siêu tham số mặc định (trích)

| Tham số | Giá trị | Tham số | Giá trị |
|---|---|---|---|
| `max_length` / `window` | 512 | `graph_layers` | 2 |
| `stride` | 128 | `graph_topk` | 15 |
| `hidden_size` | 768 | `drop_edge` | 0.15 |
| `rel_dim` | 256 | `lr` | 2e-5 |
| `lam_align` | 1 | `lam_reg` | 0.005 |
| `total_epochs` | 100 | `graph_start_epoch` | 10 |
| `accum_steps` | 4 | `f1_patience` | 10 |

---

## 6. Đánh giá

- Chỉ số chính: **macro-F1** và **micro-F1**, **loại bỏ nhãn VAGUE** khỏi phép tính.
- Bảng tổng hợp báo cáo `mean ± std` trên 3 seed; classification report + confusion matrix lấy từ seed có micro-F1 cao nhất mỗi dataset.

---

## 7. Datasets gốc

- **MATRES** — Ning et al., *A Multi-Axis Annotation Scheme for Event Temporal Relations* (ACL 2018).
- **TimeBank-Dense (TBD)** — Cassidy et al., *An Annotation Framework for Dense Event Ordering* (ACL 2014).
- **TDDiscourse (TDDMan / TDDAuto)** — Naik et al., *TDDiscourse: A Dataset for Discourse-Level Temporal Ordering of Events* (SIGDIAL 2019).

Vui lòng trích dẫn các công trình gốc khi sử dụng dữ liệu.
