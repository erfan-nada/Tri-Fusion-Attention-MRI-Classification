# Tri-fusion CNN-based Models with Attention for MRI Brain Tumor Classification

This repository contains the implementation accompanying the paper **"Tri-fusion CNN-based Models with Attention for MRI Brain Tumor Classification"**, accepted for publication at the **IEEE IMSA 2026 Conference**.

> **Citation / DOI:** The paper has been accepted but is not yet published. The full citation, DOI, and IEEE Xplore link will be added here once available. The manuscript itself is not distributed in this repository — please refer to the official publication once it is live.

**Academic Research and Implementation**
>This framework and the accompanying research paper were developed and published during my **3rd year of undergraduate studies**, representing a contribution to the fields of Computer Vision, Deep Learning, and Automated Clinical Decision Support Systems.

## Overview

Manual interpretation of brain MRI scans is time-consuming and subject to significant inter- and intra-observer variability. While transfer learning with pre-trained CNNs (VGG16, DenseNet, EfficientNet) has shown strong results for automated brain tumor classification, single-stream architectures are limited by architecture-specific inductive biases, and conventional fusion approaches (static concatenation) treat all extracted features equally — introducing redundancy and noise.

This project proposes a **Tri-Fusion CNN architecture with a learned attention gating mechanism** that adaptively weighs features from three complementary pre-trained backbones before fusion, rather than combining them uniformly.

### Architecture

1. **Multi-Backbone Feature Extraction** — Three pre-trained CNNs run in parallel as feature extractors (classification heads removed):
   - **VGG16** — fine-grained spatial textures and local tumor margins
   - **EfficientNetV2-S** — efficient, high-level global contextual representation
   - **DenseNet121** — dense connectivity for detecting subtle tissue-density gradients
2. **Dimensional Projection Layer** — Each backbone's features (4096 / 1280 / 1024-dim) are projected into a shared 512-dim space.
3. **Learned Attention Gating** — A two-layer MLP computes a softmax-normalized weight for each of the three projected streams, dynamically prioritizing the most discriminative backbone per scan.
4. **Weighted Fusion + Classification** — The gated, weighted features are concatenated and passed through a feed-forward classification head to predict one of four tumor classes.

## Dataset

The models are trained and evaluated on a balanced brain MRI dataset of **7,200 images** across **four classes**: glioma, meningioma, pituitary tumor, and no tumor (1,800 images per class; 5,600 training / 1,600 testing, evenly split across classes).

> Dataset source: Nickparvar, "Brain Tumor MRI Dataset" (2026). The dataset is not included in this repository — see [Setup](#setup) below for how to point the code at your local copy.

## Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| EfficientNetV2-S (baseline) | 77.50% | 0.7810 | 0.7750 | 0.7668 |
| DenseNet121 (baseline) | 82.94% | 0.8497 | 0.8294 | 0.8258 |
| VGG16 (baseline) | 92.00% | 0.9234 | 0.9200 | 0.9179 |
| Tri-Fusion (concatenation) | 94.44% | 0.9486 | 0.9444 | 0.9431 |
| **Tri-Fusion + Attention (proposed)** | **95.06%** | **0.9540** | **0.9506** | **0.9498** |

The proposed Tri-Fusion + Attention model improves accuracy by **3.06%** over the strongest single-model baseline (VGG16), with the attention gate contributing a further **0.62%** gain over plain feature concatenation.

## Repository Structure

```
.
├── Research_Project_2026.ipynb   # End-to-end training/evaluation notebook
└── README.md
```

## Notebook Contents

The notebook (`Research_Project_2026.ipynb`) implements the full pipeline in PyTorch:

1. **Preprocessing** — A custom `CropBrainContour` transform (OpenCV-based) crops each MRI scan to its brain contour before resizing to 224×224 and applying ImageNet normalization. Training data is additionally augmented with random horizontal flips, small rotations, and color jitter.
2. **Baseline models** — VGG16, DenseNet121, and EfficientNetV2-S are each fine-tuned individually (frozen backbone, replaced classification head) to establish single-stream baselines.
3. **Tri-Fusion (concatenation)** — `TriFusionEfficient`: the three backbones' raw feature vectors (4096 + 1280 + 1024 = 6400-dim) are concatenated and passed to a fusion classifier head.
4. **Tri-Fusion + Attention (proposed)** — `AttentionTriFusion`: features from each backbone are projected to a common 512-dim space, a learned attention gate produces per-stream softmax weights, and the weighted features are concatenated before classification.
5. **Evaluation** — Accuracy, macro-averaged precision/recall/F1, AUC-ROC (one-vs-rest), and a full classification report are computed on the held-out test set for every model.

A custom 5-layer CNN trained from scratch (`CustomCNN`) is also included in the notebook as an additional from-scratch comparison point; it is not part of the formal baseline/fusion comparison reported in the paper.

## Setup

### Requirements

- Python 3.8+
- PyTorch & torchvision
- OpenCV (`opencv-python-headless`)
- `imutils`
- `scikit-learn`
- `numpy`, `Pillow`

```bash
pip install torch torchvision opencv-python-headless imutils scikit-learn numpy pillow
```

### Dataset Setup

1. Download the Brain Tumor MRI Dataset (Nickparvar, 2026) and organize it into `Training/` and `Testing/` folders, each containing one subfolder per class (`glioma`, `meningioma`, `notumor`, `pituitary`) — i.e. the standard `torchvision.datasets.ImageFolder` layout.
2. Update the dataset paths at the top of the notebook:

```python
TRAINING_DATA_PATH = "path/to/dataset/Training"
TESTING_DATA_PATH  = "path/to/dataset/Testing"
```

### Training

Run the notebook cells in order. Each model section (VGG16, DenseNet121, EfficientNetV2-S, Custom CNN, Tri-Fusion, Tri-Fusion + Attention) trains and evaluates independently, using early stopping (patience = 5, min delta = 0.001) over a maximum of 30 epochs. The Tri-Fusion models use differential learning rates: `1e-5` for the pre-trained backbones and `1e-4` for the fusion/attention/classification layers, trained with the Adam optimizer.

Trained model weights (`.pth` files) are saved locally during training.


## Acknowledgments

Faculty of Computer Science, Modern Science and Arts University, Cairo, Egypt.
