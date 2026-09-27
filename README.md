# 🖼️ Transformer-Based Image Super-Resolution & Knowledge Distillation

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset: DIV2K](https://img.shields.io/badge/Dataset-DIV2K%20x4-00A4E4?style=for-the-badge)](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework for **4× Single-Image Super-Resolution (SISR)** featuring a **Swin Transformer (SwinIR-Lite)** teacher model, a lightweight **Residual CNN** student model, and a **multi-objective Knowledge Distillation** pipeline trained on the high-resolution **DIV2K** dataset.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architectures](#-architectures)
  - [1. SwinIR-Lite (Teacher Network)](#1-swinir-lite-teacher-network)
  - [2. Lightweight Residual CNN (Student Network)](#2-lightweight-residual-cnn-student-network)
  - [3. ESPCN-Style CNN (Baseline Network)](#3-espcn-style-cnn-baseline-network)
- [Knowledge Distillation Framework](#-knowledge-distillation-framework)
- [Experimental Results & Evaluation Metrics](#-experimental-results--evaluation-metrics)
  - [Quantitative Benchmarks](#quantitative-benchmarks)
  - [Visual & Perceptual Comparison](#visual--perceptual-comparison)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Dataset Preparation](#dataset-preparation)
  - [Running the Notebook](#running-the-notebook)
- [Hyperparameters & Training Configuration](#-hyperparameters--training-configuration)
- [Authors & Acknowledgements](#-authors--acknowledgements)

---

## 🔍 Overview

Single-Image Super-Resolution (SISR) aims to reconstruct high-fidelity High-Resolution (HR) images from Low-Resolution (LR) inputs. While Vision Transformer architectures like SwinIR achieve state-of-the-art reconstruction quality, their computational demands (FLOPs, latency, parameter count) make them impractical for edge deployment and real-time inference.

This project implements:
1. A **Teacher SwinIR-Lite** network leveraging Shifted Window Self-Attention (W-MSA & SW-MSA) combined with VGG-19 perceptual loss.
2. A **Student CNN** utilizing residual feature extraction, pixel shuffle upsampling, and Exponential Moving Average (EMA) weight updates.
3. A **Unified Knowledge Distillation (KD) Engine** transferring both pixel-level details, intermediate feature representations, and Laplacian edge boundaries from teacher to student.

```
LR Input (64x64) ─────────────────────────────────────────────────────────────┐
   │                                                                           │
   ▼                                                                           ▼
┌──────────────────────────────┐                             ┌──────────────────────────────┐
│  Teacher: SwinIR-Lite        │                             │     Student: Residual CNN    │
│  (3.70M Parameters)          │                             │     (815K Parameters)        │
│  - Window Attention (W-MSA)  │                             │  - 3x Conv Residual Blocks   │
│  - Shifted Window (SW-MSA)   │                             │  - Sub-pixel PixelShuffle x4 │
└──────────────┬───────────────┘                             └──────────────┬───────────────┘
               │                                                            │
               │ [Teacher Output & Features]                                │ [Student Output & Features]
               └──────────────────────┬─────────────────────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │  Knowledge Distillation   │
                        │  - Distillation MSE Loss  │
                        │  - Ground Truth L1 Loss   │
                        │  - VGG19 Perceptual Loss  │
                        │  - Laplacian Edge Loss    │
                        │  - Intermediate Feat Loss │
                        └───────────────────────────┘
```

---

## ✨ Key Features

- **Shifted Window Self-Attention**: Custom Keras implementation of Window Multi-Head Self-Attention with cyclic shift masks and relative position bias.
- **Multi-Component Distillation Objective**:
  $$\mathcal{L}_{total} = \alpha \mathcal{L}_{student} + (1 - \alpha)\mathcal{L}_{distill} + \lambda_{edge}\mathcal{L}_{edge} + \lambda_{feat}\mathcal{L}_{feature}$$
- **VGG-19 Feature-Level Perceptual Loss**: Captures semantic high-frequency structures beyond naive pixel-wise distance metrics.
- **Laplacian Edge Regularization**: Preserves sharp edge boundaries and suppresses blur artifacts.
- **Exponential Moving Average (EMA)**: Enhances student model generalization during optimization.
- **Cosine Annealing with Warmup**: Dynamic learning rate schedule ensuring stable convergence.

---

## 🧠 Architectures

### 1. SwinIR-Lite (Teacher Network)
- **Parameters**: `3,705,603` (~3.71M)
- **Embedding Dimension**: `128`
- **Attention Heads**: `8`
- **Window Size**: `8 × 8`
- **Depth**: `6` Swin Transformer Blocks (alternating regular & shifted windows)
- **Upsampling**: Sub-pixel convolution (`tf.nn.depth_to_space`) with 4× upscale factor.

### 2. Lightweight Residual CNN (Student Network)
- **Parameters**: `815,939` (~816K — **78% reduction** compared to Teacher)
- **Structure**:
  - Head: Conv2D ($3 \to 128$)
  - Body: 3 Residual Blocks with ReLU activations and identity skips
  - Upsampler: Conv2D ($128 \to 128 \times 4^2 = 2048$) + Sub-pixel PixelShuffle 4×
  - Tail: Conv2D ($128 \to 3$) with Sigmoid activation

### 3. ESPCN-Style CNN (Baseline Network)
- **Parameters**: `37,200` (~37.2K)
- **Structure**: 3-layer feed-forward CNN ($5\times 5$, $3\times 3$, $3\times 3$) with direct sub-pixel upscaling.

---

## 🧪 Knowledge Distillation Framework

The custom `Distiller` model coordinates feature transfer and gradient updates:

| Loss Component | Objective | Target | Weight |
|---|---|---|---|
| **Student Task Loss ($\mathcal{L}_{student}$)** | Content + Perceptual | HR Ground Truth | $\alpha = 0.7$ |
| **Distillation Loss ($\mathcal{L}_{distill}$)** | Output alignment | Teacher Output | $(1 - \alpha) = 0.3$ |
| **Edge Loss ($\mathcal{L}_{edge}$)** | Edge fidelity via Laplacian filter | HR Ground Truth | $\lambda_{edge} = 0.02$ |
| **Feature Loss ($\mathcal{L}_{feat}$)** | Latent representation alignment | Teacher Conv Embeddings | $\lambda_{feat} = 0.1$ |

---

## 📊 Experimental Results & Evaluation Metrics

Evaluated on the **DIV2K Validation Set** (4× bicubic downsampled pairs):

### Quantitative Benchmarks

| Model | Parameters | Val Loss | PSNR (dB) ↑ | SSIM ↑ | Relative Size |
|---|---|---|---|---|---|
| **Baseline ESPCN** | 37,200 | 0.0085 | 21.9740 | 0.5550 | 1.0× |
| **Student CNN (Distilled)** | **815,939** | **0.4157** | **22.5911** | **0.6110** | **21.9×** |
| **Teacher SwinIR-Lite** | 3,705,603 | 64.2689* | 20.9048 | 0.5038 | 99.6× |

> \* *Note: Teacher loss is numerically high because its objective function incorporates deep feature-space VGG-19 L2 distances (`block5_conv2`), which operate on unnormalized activation scales.*

### Key Insights
1. **Distillation Advantage**: The Student CNN achieves the **highest PSNR (22.59 dB)** and **highest SSIM (0.6110)**, outperforming the baseline by **+0.62 dB PSNR** and **+0.056 SSIM**.
2. **Artifact Reduction**: While the Teacher SwinIR produces high-frequency details, the distilled student eliminates high-frequency ringing and color distortion artifacts.
3. **Efficiency vs Quality Trade-off**: The Student CNN maintains a compact footprint (815K params) suitable for low-latency inference while retaining superior structural similarity.

### Visual & Perceptual Comparison

| Aspect | Baseline CNN | Teacher SwinIR-Lite | Student CNN (Distilled) | Ground Truth |
|---|---|---|---|---|
| **Color Fidelity** | Accurate | Slight color shifts | Highly Accurate | Reference |
| **Sharpness** | Low (over-smoothed) | High (aggressive detail) | Balanced & Natural | Reference |
| **Edge Artifacts** | None | Visible near sharp edges | Minimal / Suppressed | None |
| **Inference Cost** | Ultra Low | High | Low | N/A |

---

## 📁 Project Directory Structure

```plaintext
Image-Upscaling-Project/
├── project2025.ipynb             # Complete Jupyter Notebook (data, models, KD training, evals)
├── baseline_cnn_weights.h5       # Checkpoint weights for Baseline CNN (optional/saved)
├── swinir_teacher_weights.h5     # Checkpoint weights for SwinIR Teacher (optional/saved)
├── student_best.h5               # Best distilled checkpoint weights for Student CNN
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and CUDA-enabled GPU acceleration (recommended):

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib h5py
```

### Dataset Preparation

The pipeline automatically downloads and streams the **DIV2K 4× bicubic dataset** via TensorFlow Datasets:
- **Train Set**: 800 HR images paired with bicubic downsampled LR versions
- **Validation Set**: 100 benchmark image pairs
- **Preprocessing**: Paired random crops ($64\times 64$ LR $\to$ $256\times 256$ HR), random flips, 90-degree rotations, and light color jitter.

### Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/RathanPai/Image-Upscaling-Project.git
   cd Image-Upscaling-Project
   ```
2. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook project2025.ipynb
   ```
3. Run all cells sequentially to reproduce the data pipeline, train all three models, perform knowledge distillation, and plot metric comparisons.

---

## ⚙️ Hyperparameters & Training Configuration

```python
UPSCALE_FACTOR   = 4
LR_PATCH_SIZE    = 64
HR_PATCH_SIZE    = 256
BATCH_SIZE       = 8
EPOCHS           = 25
BASE_LR          = 2e-4
WARMUP_STEPS     = 500
EMA_DECAY        = 0.999
ALPHA_KD         = 0.7
EDGE_WEIGHT      = 0.02
FEATURE_WEIGHT   = 0.1
```

---

## 📚 References & Acknowledgements

- **DIV2K Dataset**: [Agustsson & Timofte (CVPRW 2017)](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **SwinIR**: [Liang et al., "SwinIR: Image Restoration Using Swin Transformer" (ICCVW 2021)](https://arxiv.org/abs/2108.10257)
- **ESPCN**: [Shi et al., "Real-Time Single Image Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" (CVPR 2016)](https://arxiv.org/abs/1609.05158)
- **Knowledge Distillation**: [Hinton et al., "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531)
