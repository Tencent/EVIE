---
license: apache-2.0
library_name: colpali-engine
pipeline_tag: visual-document-retrieval
base_model: tencent/EVIE-Preview-4.5B
tags:
- colpali-engine
- qwen3_5
- vision-language
- colbert
- late-interaction
- multi-vector
- matryoshka
- vidore
- token-compression
datasets:
- vidore/vidore_benchmark
- vidore/vidore_benchmark_v2
- jinaai/jina-vdr
---

<div align="center">

# 🏆 EVIE: The Most Accurate and Lightweight Visual Document Retriever
### Evidence-Vector-Informed Embedding (EVIE)

<p align="center">
  <a href="#-comprehensive-vidore-leaderboard-comparison"><img src="https://img.shields.io/badge/🥇_ViDoRe_V3-66.75_·_Rank_%231-FFD700?style=for-the-badge&labelColor=1a1a2e" alt="ViDoRe V3 Rank 1"></a>
  <a href="#-comprehensive-vidore-leaderboard-comparison"><img src="https://img.shields.io/badge/🥇_ViDoRe_V1+V2-92.18_·_Rank_%231-FFD700?style=for-the-badge&labelColor=1a1a2e" alt="ViDoRe V1+V2 Rank 1"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square&logo=apache" alt="License"></a>
  <a href="https://github.com/Tencent/EVIE"><img src="https://img.shields.io/badge/GitHub-Tencent%2FEVIE-black?style=flat-square&logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/tencent/EVIE-4.5B"><img src="https://img.shields.io/badge/🤗_Hugging_Face-EVIE--4.5B-yellow?style=flat-square" alt="Hugging Face 4.5B"></a>
  <a href="https://huggingface.co/tencent/EVIE-8B"><img src="https://img.shields.io/badge/🤗_Hugging_Face-EVIE--8B-purple?style=flat-square" alt="Hugging Face 8B"></a>
  <a href="https://github.com/QwenLM/Qwen2.5-VL"><img src="https://img.shields.io/badge/Backbone-Qwen3.5-orange?style=flat-square&logo=deepnote" alt="Backbone"></a>
  <a href="#-token-compression-hac"><img src="https://img.shields.io/badge/Index_Storage-3.81_GiB_%2F_1M_pages-brightgreen?style=flat-square&logo=databricks" alt="Storage"></a>
</p>

<p align="center">
  <b>High-Precision Late-Interaction Retrieval</b> • 
  <b>Dynamic Prefix-MRL (64D–2048D)</b> • 
  <b>Training-Free HAC Token Compression</b>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/tencent/EVIE-4.5B"><b>EVIE-4.5B (Prefix-MRL & HAC)</b></a> &nbsp;•&nbsp;
  🤗 <a href="https://huggingface.co/tencent/EVIE-8B"><b>EVIE-8B (Flagship Teacher)</b></a> &nbsp;•&nbsp;
  🐙 <a href="https://github.com/Tencent/EVIE"><b>GitHub: Tencent/EVIE</b></a>
</p>

</div>

---

> 📢 **Release Announcement**: All model weights, training pipelines, token compression algorithms (HAC), and evaluation suites have been fully open-sourced in this repository. Full technical details, architectural ablations, and the formal research paper will be updated in an upcoming release.

---

## 🌟 Highlights

- **Top-Tier Benchmark Performance**: **66.75** on ViDoRe V3 for **EVIE-8B** and **66.02** for **EVIE-4.5B** with single-projection Prefix-MRL.
- **⚡ Prefix-MRL Elasticity**: Single 2048D linear projection. Freely truncate at runtime into $\{64, 128, 256, 512, 1024, 2048\}$ dimensions without separate models.
- **📦 Ultra-Compact Index (HAC)**: Training-free Hierarchical Agglomerative Clustering compresses token counts from ~750 down to **32 vectors/page**, slashing index storage to **3.81 GiB per million pages**.
- **🌐 138 Multilingual Tasks Evaluated**: Thoroughly evaluated across ViDoRe V1, V2, V3, and JinaVDR across 4 metric families (nDCG, Recall, MAP, MRR @1/5/10).
- **🔬 EVIE-ARD Distillation Recipe**: Anchor-preserving, capacity-aware relation distillation reproducing full student training from the 8B teacher.

---

## 🧠 Architecture & Technical Highlights

```text
 Query Text  ────────► ColQwen3.5 (BiDir Attention) ────► Elastic Multi-Vectors (64D–2048D)
                                                                    │
                                                           MaxSim Matching
                                                                    │
 Doc Image   ────────► ColQwen3.5 (Vision Encoder)  ────► HAC Compression ──► 32 Vectors / Page
```

- **Late-Interaction Multi-Vector Paradigm**: Unlike dense single-vector retrieval that collapses high-resolution document pages into a single point, EVIE preserves fine-grained visual details (complex tables, layout structures, charts, and small typography) through token-level representations, scoring relevance via late-interaction MaxSim:
  $$S(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} (q_i \cdot d_j)$$
- **Prefix-MRL (Single-Head Elastic Representation)**: EVIE-4.5B introduces single-projection Prefix-MRL. A single 2048D linear projection natively supports runtime truncation down to $\{64, 128, 256, 512, 1024, 2048\}$ dimensions without maintaining multiple heads or separate checkpoints.
- **EVIE-ARD (Anchor-preserving Relation Distillation)**: The 4.5B student is distilled from the 8B teacher using token-relation topological geometry, hard-negative margin calibration, and anchor-preserving alignment, maintaining peak retrieval accuracy even under low-dimensional prefixes.
- **HAC Token Compression (Hierarchical Agglomerative Clustering)**: A plug-and-play, training-free token reduction algorithm that aggregates visual patch tokens into 32 or 64 semantic centroids in joint feature-spatial space, reducing 1M-page index footprints to as little as **3.81 GiB**.

---

## 📊 Comprehensive ViDoRe Leaderboard Comparison

Performance comparison across modern multi-vector late-interaction visual document retrievers on ViDoRe:

| Rank | Model | Base Model | Param | Embed Dim | ViDoRe V1 (nDCG@5) | ViDoRe V2 (nDCG@5) | ViDoRe V3 (nDCG@10) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[EVIE-8B](https://huggingface.co/tencent/EVIE-8B)** | Qwen3.5-9B | 8.41B | 4096D | **92.18** | **74.23** | **66.75** |
| 🥈 | **[EVIE-4.5B](https://huggingface.co/tencent/EVIE-4.5B)** | Qwen3.5-4B | 4.61B | 64–2048D Prefix-MRL | **92.07** | **73.38** | **66.02** |
| 🥉 | **EVIE-Preview-4.5B** | Qwen3.5-4B | 4.54B | 128D | 91.73 | 70.87 | 65.36 |
| 4 | webAI-ColVec1.1-8b | Qwen2.5-VL | 8.40B | 640D | 91.30 | 65.82 | 65.32 |
| 5 | VultronRetrieverPrime-8B | Qwen3.5-9B | 8.40B | 320D | 92.08 | 68.18 | 64.26 |
| 6 | webAI-ColVec1.1-4b | Qwen2.5-VL | 4.54B | 640D | 90.49 | 63.60 | 63.90 |
| 7 | VultronRetrieverCore-4.5B | Qwen3.5-4B | 4.50B | 320D | 92.21 | 66.12 | 63.57 |
| 8 | nemotron-colembed-vl-8b-v2 | Nemotron-8B | 8.80B | 4096D | 92.65 | 65.16 | 63.54 |
| 9 | tomoro-colqwen3-embed-8b | Qwen2.5-VL | 8.00B | 320D | 90.76 | 65.40 | 61.60 |
| 10 | nemotron-colembed-vl-4b-v2 | Nemotron-4B | 4.80B | 2560D | 91.62 | 64.49 | 61.42 |
| 11 | athrael-soju/colqwen3.5-4.5B-v3 | Qwen3.5-4B | 4.60B | 128D | 91.54 | 64.25 | 61.46 |
| 12 | tomoro-colqwen3-embed-4b | Qwen2.5-VL | 4.00B | 320D | 90.57 | 64.69 | 60.16 |
| 13 | VultronRetrieverFlash-0.8B | Qwen3.5-0.8B | 0.85B | 320D | 88.15 | 60.36 | 56.16 |

---

### 🔍 ViDoRe V3 Per-Domain Breakdown (nDCG@10)

| Model | Avg | CompSci | Energy | Finance EN | Finance FR | HR | Industrial | Pharma | Physics |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EVIE-8B** | **66.75** | **81.86** | **72.51** | **71.23** | **56.40** | **69.29** | **59.77** | **70.81** | **52.11** |
| **EVIE-4.5B** | **66.02** | 81.72 | 72.32 | 70.00 | 54.90 | 67.82 | 59.40 | 70.27 | 51.69 |
| webAI-ColVec1.1-8b | 65.32 | 80.08 | 70.12 | 71.90 | 54.87 | 68.55 | 57.65 | 67.88 | 51.50 |
| nemotron-colembed-vl-8b-v2 | 63.54 | 79.30 | 69.82 | 67.29 | 51.54 | 66.32 | 56.03 | 67.19 | 50.84 |
| VultronRetrieverPrime-8B | 64.26 | 79.80 | 70.30 | 69.00 | 54.50 | 66.80 | 57.40 | 68.20 | 51.70 |
| VultronRetrieverCore-4.5B | 63.57 | 79.80 | 69.20 | 68.90 | 52.00 | 66.10 | 56.10 | 67.50 | 50.20 |
| tomoro-colqwen3-embed-8b | 61.60 | 75.35 | 68.41 | 65.08 | 49.10 | 63.98 | 54.41 | 66.36 | 50.13 |

---

## 🎯 Prefix-MRL Elastic Multi-Vector Head

EVIE-4.5B embeds document and query tokens with a single 2048D linear projection head trained via **EVIE-ARD**. You can truncate the channel dimension on-the-fly without maintaining different models:

```text
Full Projection (2048D)  [========================================================] 66.02
Prefix 1024D             [============================]                             65.94
Prefix 512D              [==============]                                           65.90
Prefix 256D              [=======]                                                  65.68
Prefix 128D              [===]                                                      65.27
Prefix 64D               [=]                                                        64.51
```

| Dimension | Bytes / Vector | ViDoRe V1 | ViDoRe V2 | ViDoRe V3 | JinaVDR | 138-Task Avg4 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 128 B | 92.16 | 73.18 | 64.51 | 81.00 | 77.71 |
| **128** | 256 B | 92.28 | 73.37 | 65.27 | 81.83 | 78.18 |
| **256** | 512 B | 92.20 | 73.83 | 65.68 | 82.09 | 78.45 |
| **512** | 1 KiB | 92.38 | 74.53 | 65.90 | 82.27 | 78.77 |
| **1024** | 2 KiB | 92.39 | 74.53 | 65.94 | 82.43 | 78.82 |
| **2048** | 4 KiB | **92.53** | **74.91** | **66.02** | **82.48** | **78.98** |

---

## 🗜️ Token Compression (HAC)

Raw late-interaction representations keep all visual patch vectors (~750 vectors/page), requiring substantial storage. EVIE integrates **Hierarchical Agglomerative Clustering (HAC)** in a joint semantic-position space to cluster page tokens at indexing time:

$$
z = \text{L2}\left[(1-w)\text{L2}(v) + w p\right], \quad \mu_c = \text{L2}\left(\text{mean}_{i \in c} v_i\right)
$$

### Production Ready SKUs

| SKU | Payload / Page | Index Size (1M Pages) | Vectors / Page | ViDoRe V1 | ViDoRe V2 | ViDoRe V3 | Avg4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **d64 K32** | **4 KiB** | **3.81 GiB** | **32** | 89.27 | 67.83 | 59.58 | 73.47 |
| **d64 K64** | 8 KiB | 7.63 GiB | 64 | **90.63** | **70.71** | **62.06** | **75.57** |
| **d128 K32** | 8 KiB | 7.63 GiB | 32 | 90.00 | 69.45 | 61.40 | 74.87 |

---

## 📋 Complete 138-Task Evaluation Matrix

Protocol `paired-all-pages-dedup+process_queries+ndcg2r-20260827` ($\text{MVT} = 1024$, bidirectional attention):

| Metric | ViDoRe V1 | ViDoRe V2 | ViDoRe V3 | JinaVDR | 4-Board Macro Avg |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **nDCG@1** | 88.35 | 72.76 | 61.16 | 74.26 | 74.13 |
| **nDCG@5** | 92.07 | 73.38 | 63.39 | 81.57 | 77.60 |
| **nDCG@10** | **92.53** | **74.91** | **66.02** | **82.48** | **78.98** |
| **Recall@1** | 88.35 | 36.73 | 30.14 | 74.26 | 57.37 |
| **Recall@5** | 94.95 | 65.59 | 58.06 | 87.39 | 76.50 |
| **Recall@10** | 96.36 | 76.56 | 69.67 | 90.17 | 83.19 |
| **MAP@1** | 88.35 | 73.18 | 65.03 | 74.26 | 75.20 |
| **MAP@5** | 91.09 | 66.52 | 55.02 | 79.61 | 73.06 |
| **MAP@10** | 91.29 | 66.25 | 55.50 | 79.99 | 73.25 |
| **MRR@1** | 88.35 | 73.18 | 65.03 | 74.26 | 75.20 |
| **MRR@5** | 91.09 | 81.66 | 74.76 | 79.61 | 81.78 |
| **MRR@10** | 91.29 | 82.11 | 75.43 | 79.99 | 82.20 |

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/Tencent/EVIE.git
cd EVIE
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/colpali${PYTHONPATH:+:$PYTHONPATH}"
```

### Self-Contained Python Inference

```python
import torch
from PIL import Image
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import set_active_head

model_id = "tencent/EVIE-4.5B"

# 1. Load model with FlashAttention and bidirectional attention
model = ColQwen3_5.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
).eval()
model.enable_bidirectional_attention()

# 2. Select any Matryoshka head: 64, 128, 256, 512, 1024, or 2048
set_active_head(model, 128)

# 3. Process query and document image
processor = ColQwen3_5Processor.from_pretrained(model_id)
images = [Image.open("examples/demo/pages/q3_revenue.png").convert("RGB")]
queries = ["What is the total quarterly revenue?"]

image_batch = processor.process_images(images).to(model.device)
query_batch = processor.process_queries(queries).to(model.device)

# 4. Generate multi-vector representations and late-interaction score
with torch.inference_mode():
    image_embeddings = model(**image_batch)
    model.rope_deltas = None  # Reset RoPE deltas before text query forward
    query_embeddings = model(**query_batch)

scores = processor.score(query_embeddings, image_embeddings)
print("Late-interaction MaxSim Relevance Score:", scores)
```

---

## 📂 Repository Layout

```text
EVIE/
├── model.safetensors         # EVIE-4.5B weights (safetensors)
├── config.json               # Model configuration (Prefix-MRL, max 2048)
├── processor_config.json     # Multimodal processor config
├── infer.py                  # Standalone inference & scoring CLI
├── colpali/                  # ColQwen3.5 + Prefix-MRL + EVIE-ARD
├── code/
│   ├── teacher/              # EVIE-8B training arms and soup merging
│   ├── student/              # EVIE-4.5B Prefix-MRL / EVIE-ARD distillation
│   ├── shared/               # Data loaders, adapter merges, and 138-task eval harness
│   └── compress/             # Training-free HAC token compression pipeline
├── examples/demo/            # 8-page retrieval demo (run.sh)
└── env.sh.example            # Environment variables template
```

---

## 🔬 Training & Distillation Reproduction

Student training executes Prefix-MRL plus EVIE-ARD: capacity-aware relation distillation and margin distillation against frozen **EVIE-8B**, with a Preview-anchor term on the 128D prefix:

```bash
cp env.sh.example env.sh
source env.sh

# Step 1: Train EVIE-8B teacher arms and merge
bash code/teacher/run.sh

# Step 2: Distill EVIE-4.5B student with ARD loss
export TEACHER_DIR=../Evie-8B
bash code/student/run.sh

# Step 3: Run comprehensive 138-task evaluation
MODEL_DIR=. RUN_NAME=evie-4.5b bash code/shared/eval_run.sh
```

---

## 📚 Citation

```bibtex
@misc{tencent2026evie,
  title        = {EVIE: High-Performance Multilingual Visual Document Retrieval with Matryoshka Embeddings and Token Compression},
  author       = {{Tencent}},
  year         = {2026},
  howpublished = {\url{https://github.com/Tencent/EVIE}}
}
```

---

## 📄 License

This repository is licensed under the [Apache-2.0 License](LICENSE).
