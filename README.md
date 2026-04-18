# KTAC 🔬

**Re-implementation of FOCUS for Few-shot Ovarian Cancer WSI Classification**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

KTAC is a personal re-implementation of the core ideas from [FOCUS (CVPR 2025)](https://github.com/dddavid4real/FOCUS), applied to the **UBC-OCEAN** ovarian cancer dataset under few-shot settings (4-shot, 8-shot, 16-shot).

The original FOCUS framework proposes a three-stage visual token compression strategy for few-shot Whole Slide Image classification. This project reimplements that pipeline with some architectural differences:

- **Text encoder**: DistilBERT (frozen) instead of CONCH text encoder
- **Aggregator**: Custom cross-modal multi-head attention module
- **Feature backbone**: CONCH visual features (512-dim, pre-extracted)
- **Scope**: Single dataset (UBC-OCEAN), single scale (low-resolution thumbnail features)

> This is a learning project. Results may differ from the original paper.

---

## Task

Classify ovarian cancer WSIs into 5 subtypes:

| Label | Subtype |
|---|---|
| CC | Clear Cell Carcinoma |
| EC | Endometrioid Carcinoma |
| HGSC | High-Grade Serous Carcinoma |
| LGSC | Low-Grade Serous Carcinoma |
| MC | Mucinous Carcinoma |

**Dataset:** [UBC-OCEAN](https://www.kaggle.com/competitions/UBC-OCEAN) (Kaggle)  
**Features:** Pre-extracted with CONCH (512-dim per patch)  
**Evaluation:** 10-fold cross-validation, reported as mean ± std

---

## Method

### Pipeline

```
WSI patch bag [N × 512]
    → Feature Encoder (Linear 512→256, LayerNorm, ReLU, Dropout)

    → Stage 1: Global Redundancy Removal
               Sliding-window cosine similarity (window_size=32)
               Removes patches with mean similarity > μ + σ

    → Stage 2: Language-guided Token Selection
               Cosine similarity between each token and DistilBERT text anchor
               Keep top-k tokens (k = γ × N', γ=0.8, max=1024)

    → Stage 3: Sequential Visual Token Compression (SVTC)
               Remove consecutive tokens with cosine similarity > 0.7

    → Cross-Modal Aggregator
               Multi-head cross-attention (text as query, visual as key/value)

    → Linear Classifier → [5 logits]
```

### Key Differences from FOCUS

| Component | FOCUS (original) | KTAC (this repo) |
|---|---|---|
| Text encoder | CONCH text encoder | DistilBERT (frozen) |
| Visual backbone | CONCH / UNI / GPFM | CONCH (fixed) |
| Dataset | TCGA-NSCLC, CAMELYON, UBC-OCEAN | UBC-OCEAN only |
| Input scale | Multi-scale | Single scale (low-res) |

---

## Project Structure

```
KTAC/
├── configs/
│   └── model_config.yaml         # All paths and hyperparameters
├── src/
│   └── model_trainer/
│       ├── utils.py              # Config loader and seed setter
│       ├── dataset.py            # WSIFocusDataset and custom_collate
│       ├── modules.py            # FocusModules (Stage 1 & 3) + CrossModalAggregator
│       ├── architecture.py       # FocusOnSpark — full model
│       ├── engine.py             # train_one_epoch, evaluate
│       └── main_train.py         # Training entry point
├── data/
│   ├── pt/                       # CONCH .pt feature files (one per slide)
│   ├── splits/                   # Fold split CSVs (4shot/, 8shot/, 16shot/)
│   └── prompts/                  # LLM-generated text prompt CSV
├── checkpoints/                  # Saved model weights (auto-generated)
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/TranQuocTroq/KTAC.git
cd KTAC
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+

---

## Usage

**Step 1 — Edit `configs/model_config.yaml`**

Set the paths to your feature files, split CSVs, and prompt file.

**Step 2 — Run training**

```bash
python -m src.model_trainer.main_train --config configs/model_config.yaml
```

Results are printed as a summary table at the end:


| Setting | AUC              | F1               | BACC             |
|---------|------------------|------------------|------------------|
| 4shot   | 0.9117 ± 0.0600  | 0.7357 ± 0.1103  | 0.7192 ± 0.1202  |
| 8shot   | 0.9894 ± 0.0159  | 0.9288 ± 0.0865  | 0.9283 ± 0.0879  |


---

## References

- Guo et al. [FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification](https://github.com/dddavid4real/FOCUS). CVPR 2025.
- Lu et al. [A Visual-Language Foundation Model for Computational Pathology (CONCH)](https://github.com/mahmoodlab/CONCH). Nature Medicine, 2024.
- Ilse et al. [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712). ICML 2018.

---

## Contact

**Tran Quoc Trong** · tranquoct157@gmail.com  
Water Resources University · Faculty of Information Technology · Class of 2026
