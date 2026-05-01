# 🛰️ Satellite Land Classification: CNN–ViT Hybrid Deep Learning

A 4-module deep learning capstone project for **agricultural vs. non-agricultural land classification** from satellite imagery, built with both **Keras/TensorFlow** and **PyTorch**, culminating in a CNN–Vision Transformer (ViT) hybrid architecture.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Results & Evaluation](#results--evaluation)
- [Key Concepts](#key-concepts)

---

## Overview

This project tackles a real-world geospatial analysis problem — classifying satellite images into **agricultural** and **non-agricultural** land — using state-of-the-art deep learning techniques.

The project is structured across four progressive modules:

1. **Data pipeline engineering** — efficient loading and augmentation strategies
2. **CNN classifiers** — building, training, and evaluating models in both frameworks
3. **Vision Transformers** — implementing CNN–ViT hybrid architectures
4. **Final evaluation** — end-to-end inference and cross-framework benchmarking

---

## Project Structure

```
├── Module 1 — Data Loading & Augmentation
│   ├── AI-capstone-M1L1-v1.ipynb       # Memory-based vs. generator-based loading
│   ├── AI-capstone-M1L2-v1.ipynb       # Data loading & augmentation in Keras
│   └── AI-capstone-M1L3-v1.ipynb       # Data loading & augmentation in PyTorch
│
├── Module 2 — CNN Classifiers
│   ├── Lab-M2L1-...Keras-Classifier    # Train & evaluate a Keras CNN
│   ├── Lab-M2L2-...PyTorch-Classifier  # Train & evaluate a PyTorch CNN
│   └── Lab-M2L3-...Comparative-Analysis# Cross-framework performance comparison
│
├── Module 3 — Vision Transformers
│   ├── Lab-M3L1-...ViT-Keras           # CNN–ViT hybrid in Keras
│   └── Lab-M3L2-...ViT-PyTorch         # CNN–ViT hybrid in PyTorch
│
└── Module 4 — Integration & Final Evaluation
    └── lab-M4L1-...CNN-ViT-Evaluation  # End-to-end evaluation of both ViT models
```

---

## Modules

### Module 1 — Data Pipeline Engineering

Focuses on building efficient, scalable data pipelines for satellite imagery.

- **Lab 1:** Compares sequential (memory-based) vs. generator-based image loading — trade-offs in memory usage, I/O performance, and code complexity
- **Lab 2 (Keras):** Implements a custom data generator and `image_dataset_from_directory` with `tf.data` optimizations (`.map()`, `.cache()`, `.prefetch()`)
- **Lab 3 (PyTorch):** Builds a custom `Dataset` class, uses `torchvision.datasets.ImageFolder`, and wraps both with `DataLoader` for batching and shuffling

### Module 2 — CNN Classifiers

Builds and benchmarks CNN image classifiers in both frameworks.

- Trains a **Keras CNN** on the satellite image dataset and evaluates with accuracy, loss curves, and confusion matrix
- Trains an equivalent **PyTorch CNN**, with manual training loops, seed control for reproducibility, and checkpoint saving
- **Comparative analysis** across both models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves

### Module 3 — CNN–ViT Hybrid Architecture

Extends the trained CNN backbones with a **Vision Transformer** encoder.

- **Keras hybrid:** Extracts feature maps from the pre-trained CNN, tokenizes them with positional embeddings, and feeds them through a Transformer encoder block
- **PyTorch hybrid:** Implements Patch Embedding, Multi-Head Self-Attention (MHSA), and a full Transformer encoder on top of the CNN backbone
- Both models trained with early stopping via model checkpointing

### Module 4 — Final Evaluation & Benchmarking

End-to-end evaluation of both CNN–ViT hybrid models.

- Loads pre-trained Keras and PyTorch ViT models
- Runs inference on held-out test data
- Computes full evaluation metrics for both and plots comparative ROC curves
- Provides a structured discussion of results, trade-offs, and architectural insights

---

## Tech Stack

| Category | Tools |
|---|---|
| Frameworks | TensorFlow / Keras, PyTorch, torchvision |
| Data | Satellite imagery dataset (IBM Cloud Object Storage) |
| Evaluation | Scikit-learn (metrics), matplotlib (visualization) |
| Architecture | CNN, Vision Transformer (ViT), CNN–ViT Hybrid |
| Environment | Jupyter Notebooks, IBM Skills Network Labs |

---

## Getting Started

### Prerequisites

```bash
pip install numpy==1.26 matplotlib==3.9.2 tensorflow==2.19 scikit-learn
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Run the notebooks in order

```
Module 1 → Module 2 → Module 3 → Module 4
```

The dataset is downloaded automatically inside each notebook from IBM's cloud storage. No manual data setup is required.

---

## Results & Evaluation

Models are evaluated on the following metrics:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Minimizing false positives |
| **Recall** | Minimizing false negatives |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Ability to distinguish between classes across thresholds |

The final module (M4) provides a side-by-side comparison of the Keras and PyTorch CNN–ViT hybrid models with ROC curves and a structured performance summary.

---

## Key Concepts

- **CNN–ViT Hybrid:** CNNs excel at capturing local spatial features (edges, textures); Vision Transformers capture long-range global dependencies via self-attention. The hybrid architecture leverages both strengths.
- **Data Augmentation:** Applied in both Keras and PyTorch pipelines to improve generalization on satellite imagery.
- **Framework Parity:** Every component — data loading, model architecture, training, and evaluation — is implemented in both Keras and PyTorch, making this a practical side-by-side reference.
- **Reproducibility:** Random seeds are explicitly controlled in all PyTorch experiments.
