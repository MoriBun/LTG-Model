# Latent Timeline Graph (LTG) — Temporal Relation Extraction

A model for **Event Temporal Relation Extraction** at the document level, combining **Allen's Interval Algebra** with **Graph Propagation**. Each event is mapped to a *latent time interval* `(s, t)` on a timeline, from which the relation between event pairs is inferred using both geometric (geometry) and semantic (relation) signals.

The repo provides 2 end-to-end notebooks with 2 different backbone encoders and 4 standard datasets.

---

## 1. Repo contents

```
.
├── data/                        # 4 datasets, each with train/dev/test.csv
│   ├── MATRES/MATRES/
│   ├── TBD/TBD/
│   ├── TDDAutoProcessed/TDDAutoProcessed/
│   └── TDDManProcessed/TDDManProcessed/
├── ltg-final-roberta.ipynb      # full pipeline, backbone = roberta-base
├── ltg-final-bert-base.ipynb    # full pipeline, backbone = bert-base-uncased
└── README.md
```

> The two notebooks are **identical** except for a single parameter: `model_name` (`roberta-base` vs `bert-base-uncased`). Every architectural description below applies to both.

---

## 2. Data

Each dataset is a temporal-relation classification problem over event pairs within the same document. All CSV files share **one common 11-column schema**:

| Column | Meaning |
|---|---|
| `entity1_id`, `entity2_id` | IDs of event 1 and event 2 in the document |
| `entity1_start`, `entity1_end` | char-span (character positions) of event 1 in `text` |
| `entity2_start`, `entity2_end` | char-span of event 2 |
| `entity1_text`, `entity2_text` | trigger string of each event |
| `document_id` | document identifier (all pairs in the same document share the same `text`) |
| `text` | full document text |
| `label` | temporal relation label (see table below) |

### Datasets and label sets

| Dataset | Directory | Train / Dev / Test (number of pairs) | Standard label set |
|---|---|---|---|
| **MATRES** | `data/MATRES/MATRES/` | ~282k / ~30k / ~16k | BEFORE, AFTER, EQUAL, VAGUE |
| **TBD** (TimeBank-Dense) | `data/TBD/TBD/` | ~4.0k / 630 / 1.4k | BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS, VAGUE |
| **TDDMan** (TDDiscourse, manual) | `data/TDDManProcessed/TDDManProcessed/` | ~4.0k / 651 / 1.5k | BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS |
| **TDDAuto** (TDDiscourse, auto) | `data/TDDAutoProcessed/TDDAutoProcessed/` | ~32k / 1.4k / 4.3k | BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS |

> *The row counts above include the header. Count exactly with `wc -l data/<DS>/<DS>/*.csv`.*

**Notes on labels:**
- TDDMan/TDDAuto store labels in raw abbreviated form (`b`, `a`, `i`, `ii`, `s`) → the notebook maps them automatically to the standard labels (`BEFORE`, `AFTER`, `INCLUDES`, `IS_INCLUDED`, `SIMULTANEOUS`) via the `aliases` table in `DATASET_REGISTRY`.
- The **VAGUE** label (present only in MATRES, TBD) is **excluded from the F1 computation** (macro/micro) but is still used during training.
- MATRES, TBD already provide a `dev.csv`; for TDDMan/TDDAuto, dev is split from train by `document` (`dev_ratio=0.1`) when `auto_split_dev=True`.

---

## 3. Model architecture

Processing flow: **encode document → pool events → latent time coordinates → graph propagation → two-branch decoder fusion**.

```
            ┌─────────────────────────── document text ───────────────────────────┐
            │  Sliding-window encoder (RoBERTa/BERT, covers full doc even > 512 tok)│
            └──────────────────────────────────┬──────────────────────────────────┘
                                  attention span-pooling per event
                                               ▼
                            z_i  (event vector, m events)
                                               ▼
                      EventTimeHead:  s_i,  t_i = s_i + softplus(d_i) + eps
                                               ▼
        ┌──────────────── Latent Timeline Graph Propagation (×graph_layers) ───────────────┐
        │  node = [z; s; t; d] · edge = EdgeUpdate(h, geometry φ, sim_sem, sim_time)       │
        │  DropEdge → Top-K by attention → softmax over destination node → GRU NodeUpdate  │
        │  Default **dense graph**: every directed pair i≠j  (m·(m-1) edges)              │
        └─────────────────────┬──────────────────────────────────────────────┬─────────────┘
                              ▼                                              ▼
        ┌─────────── Allen branch (geometry) ───────────┐   ┌──── Relation branch (semantic) ──────┐
        │  AllenDecoder: per-label geometric formula    │   │  RelationEncoder h_ij=FFN([z_i;z_j]) │
        │  from interval (s_i,t_i,s_j,t_j) → z_geo       │   │  RelationDecoder → z_rel            │
        └───────────────────────┬──────────────────────┘   └───────────────┬─────────────────────┘
                                └──────── logits = z_rel + α·z_geo ────────┘
```

### Main components

- **`EventTimeHead`** — turns an event vector `z` into a time interval `(s, t)` with `t = s + softplus(d) + eps` (guaranteeing a positive interval length).
- **`GeometryComputer`** — from a pair of intervals produces **φ_ij**, consisting of 10 geometric features (delta_s, delta_t, gap, overlap, ov_ratio, containment, dur_ratio, eq_score, …).
- **`RelationEncoder`** — `h_ij = FFN([z_i; z_j])`, a semantic representation of the relation.
- **`GraphPropagationLayer`** — one round of propagation over the Latent Timeline Graph: EdgeUpdate → geometry head → message passing with attention + Top-K + DropEdge → GRU node update. Supports 2 modes:
  - **dense** (`graph_dense=True`, default): propagation over **all** directed edges `i≠j`.
  - **sparse**: only over labeled pairs (a bidirectional graph with φ and φ_rev).
- **`AllenDecoder`** — produces geometric logits according to the **exact label set** of each dataset; each Allen label has a formula over intervals (e.g. `BEFORE: s_j − t_i`, `IS_INCLUDED: min(s_i−s_j, t_j−t_i)`), while `VAGUE` uses a learned head.
- **`RelationDecoder`** — `[h; g; φ; cos_sim] → z_rel`.
- **Fusion** — `logits = z_rel + α·z_geo`, where `z_geo` is LayerNorm-ed per layer and `α` is a learned parameter per layer.

### Loss function

```
L = L_cls + λ_reg · L_reg + λ_align · L_align
```
- **`L_cls`** — cross-entropy with class weights.
- **`L_reg`** — `d² + 1/(d+ε)`, penalizing intervals that are both too long and too short.
- **`L_align`** — `‖P_g(g) − P_φ(φ)‖²`, aligning the predicted geometry (`g`) with the geometry computed from intervals (`φ`); enabled only when the graph is active.

### Two-stage training
1. **No-graph** — first learn the encoder + time coordinates (`epoch < graph_start_epoch`).
2. **Graph** — enable graph propagation + `L_align` (`epoch ≥ graph_start_epoch`).
3. **Early-stop** by dev macro-F1 (`f1_patience`).

---

## 4. Notebook workflow

Each notebook consists of 9 numbered sections:

1. **Imports, Device & Configuration** — `DATASET_REGISTRY`, `BASE_CFG`, `build_config()`.
2. **Data Pipeline** — tokenize the whole document, map char-span → token-span, gather labeled pairs; `prepare_dataset()` returns a *bundle* reused across every seed.
3. **Model architecture** — the modules in section 3 above.
4. **Loss, Evaluation & Training** — `compute_loss`, `evaluate` (macro/micro-F1 excluding VAGUE), the two-stage training loop.
5. **Main experiments** — 4 datasets × 3 seeds (`SEEDS = [42, 123, 2024]`), saving all results and keeping the highest-micro-F1 run per dataset.
6. **Results aggregation** — `mean ± std` table, classification report + confusion matrix (for the best seed), comparison charts.
7. **Ablation Study** — 1 seed (42), the cases `−Graph` / `−Relation` / `−Event` / `−Align loss` compared against Full, toggled via the `abl_*` flags.
8. **Latent Timeline visualization** — plot the latent timeline for the highest-micro-F1 document per dataset.
9. **Latent coordinate distribution analysis** — histogram/violin/scatter of `s`, `t`, `d = t − s` on the test set.

### Control switches

| Variable | Value | Effect |
|---|---|---|
| `FULL_MODE` | `'all'` \| `'single'` | run all 4 datasets or 1 representative dataset (`FULL_SINGLE='tddman'`) |
| `SEEDS` | `[42, 123, 2024]` | seeds for the main experiments |
| `ABLATION_MODE` | `'all'` \| `'single'` \| `'off'` | scope of the ablation |
| `graph_dense` | `True` / `False` | dense (all pairs) vs sparse (labeled pairs only) |
| `abl_use_graph`, `abl_relation_branch`, `abl_event_branch`, `abl_align_loss` | `True`/`False` | enable/disable each component for ablation (default = full model) |

---

## 5. How to run

The notebooks are designed to run on **Kaggle / Colab with a GPU** (require `transformers`, `torch`; cannot run on pure CPU due to the cost of encoding the full document).

1. **Prepare the data.** Point each dataset's `data_root` to the correct location:
   - On Kaggle: auto-detects `/kaggle/input/...` (see `DATASET_REGISTRY`).
   - Running locally: edit the `else` branch in `DATASET_REGISTRY` to point to this repo's `data/` directory, for example:
     ```python
     'matres': dict(data_root='data/MATRES/MATRES', ...)
     ```
     (note the nested directory structure `data/<DS>/<DS>/`).
2. **Choose a backbone.** Open `ltg-final-roberta.ipynb` (RoBERTa) or `ltg-final-bert-base.ipynb` (BERT).
3. **Run top to bottom.** Section 5 will train `4 datasets × 3 seeds`; sections 6–9 automatically generate the metric tables and charts.
4. **The best model** per dataset is saved as a `.pt` file in `save_dir` (containing `state_dict` + `cfg` + metrics; reload with `LatentTimelineGraphModel(ckpt['cfg'])`).

### Default hyperparameters (excerpt)

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| `max_length` / `window` | 512 | `graph_layers` | 2 |
| `stride` | 128 | `graph_topk` | 15 |
| `hidden_size` | 768 | `drop_edge` | 0.15 |
| `rel_dim` | 256 | `lr` | 2e-5 |
| `lam_align` | 1 | `lam_reg` | 0.005 |
| `total_epochs` | 100 | `graph_start_epoch` | 10 |
| `accum_steps` | 4 | `f1_patience` | 10 |

---

## 6. Evaluation

- Main metrics: **macro-F1** and **micro-F1**, **excluding the VAGUE label** from the computation.
- The summary table reports `mean ± std` over 3 seeds; the classification report + confusion matrix are taken from the seed with the highest micro-F1 per dataset.

---

## 7. Original datasets

- **MATRES** — Ning et al., *A Multi-Axis Annotation Scheme for Event Temporal Relations* (ACL 2018).
- **TimeBank-Dense (TBD)** — Cassidy et al., *An Annotation Framework for Dense Event Ordering* (ACL 2014).
- **TDDiscourse (TDDMan / TDDAuto)** — Naik et al., *TDDiscourse: A Dataset for Discourse-Level Temporal Ordering of Events* (SIGDIAL 2019).

Please cite the original works when using the data.
