# TESS: Temporal Evolution Semantic Space

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.12664-b31b1b.svg)](https://arxiv.org/abs/2603.12664)
[![ICML 2026](https://img.shields.io/badge/ICML-2026%20Oral-blue.svg)](https://icml.cc/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space**

*Lehui Li, Yuyao Wang, Jisheng Yan, Wei Zhang, Jinliang Deng, Haoliang Sun, Zhongyi Han, Yongshun Gong*

*ICML 2026 Oral*

</div>

## Overview

TESS bridges the modality gap between text and time-series forecasting by introducing a **Temporal Evolution Semantic Space** — an interpretable intermediate bottleneck. An LLM (Qwen3-8B) extracts four temporal primitives (distribution shift, volatility, shape, temporal influence) from text via structured prompting. These primitives feed into a forecasting model through confidence-aware gating and late additive fusion, achieving up to **29% error reduction** over SOTA baselines on four real-world datasets.

## Key Designs

:star2: **Temporal Primitives** — LLM extracts four interpretable, numerically-grounded primitive labels from text, forming a discrete semantic bottleneck between modalities.

:star2: **Confidence-Aware Gating** — A learned gate filters unreliable LLM predictions using self-consistency margins.

:star2: **Late Additive Fusion** — Numeric and primitive branches are decoded independently, then combined: `y_hat = y_numeric + text_delta_scale * y_primitive_delta`.

:star2: **Teacher-Student Distillation** — A teacher trained on oracle (GT) primitives transfers primitive-level knowledge to a student using only LLM-extracted text primitives.

## Installation

```bash
conda create -n tess python=3.10 -y && conda activate tess
pip install torch --index-url https://download.pytorch.org/whl/cu118  # adjust for your CUDA
pip install -r requirements.txt
```

## Primitive Inference

Primitive labels must be generated before training. **Inference results for all four datasets are already provided** in `data_cache/` — you can skip straight to training.

To reproduce the inference pipeline from scratch, TESS uses **Qwen3-8B** via an OpenAI-compatible API. Fill in your own API key and endpoint, then run:

```bash
BASE_URL=<your-api-endpoint> \
API_KEY=<your-api-key> \
MODEL=qwen3-8b \
bash scripts/run_other_datasets_sampling4.sh
```

## Training & Main Results

The main paper results use teacher-student distillation:

**Step 1** — Train the teacher (GT primitives, oracle upper bound):

```bash
ROOT=. MODE=gt_scale1 bash scripts/run_legacy_multimodal_primitive_additive_fnspid.sh
```

**Step 2** — Train students via distillation:

```bash
# Hard distillation
ROOT=. bash scripts/run_additive_hard_distill_fnspid.sh

# Soft distillation (probability-weighted embeddings)
ROOT=. bash scripts/run_additive_soft_distill_fnspid_v2.sh
```

| Variant | Student Model | Difference |
|---------|--------------|------------|
| Hard | `legacy_multimodal_primitive_additive` | Hard ID embedding lookup |
| Soft | `legacy_multimodal_primitive_additive_soft` | Probability-weighted embedding (captures LLM uncertainty) |

Both initialize the student's numeric backbone from the teacher and distill only the primitive delta (`λ_delta=0.1`, `λ_pred=0.0`).

**View results:**

```bash
python -m experiments.summarize_additive_results --out-root outputs/tess_basic/fnspid_legacy_standard
```

## Citation

```bibtex
@inproceedings{li2026tess,
  title     = {From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space},
  author    = {Lehui Li and Yuyao Wang and Jisheng Yan and Wei Zhang and Jinliang Deng and Haoliang Sun and Zhongyi Han and Yongshun Gong},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Oral Presentation}
}
```

