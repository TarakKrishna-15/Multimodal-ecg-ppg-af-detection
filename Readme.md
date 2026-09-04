# Multimodal ECG-PPG AF Detection

## Attention-Guided Residual–Gated Multimodal Learning for Real-Time Atrial Fibrillation Detection Using ECG and PPG

A multimodal deep learning framework for detecting Atrial Fibrillation (AF) using synchronized Electrocardiography (ECG) and Photoplethysmography (PPG) signals.

## Research Overview

Atrial Fibrillation is a common cardiac arrhythmia associated with increased risk of stroke and other cardiovascular complications.

ECG provides electrical information about cardiac activity, while PPG provides information related to peripheral blood-volume changes.

This project combines these complementary physiological modalities using a multimodal deep learning framework.

### Overall Pipeline

ECG + PPG  
↓  
Modality-Specific Preprocessing  
↓  
MultiScale CNN  
↓  
BiGRU  
↓  
Attention  
↓  
256-D ECG + 256-D PPG Representations  
↓  
512-D Feature Fusion  
↓  
Residual-Gated Classification  
↓  
AF / Non-AF

---

## Problem Statement

ECG-based approaches can be affected by noise, motion artifacts, sensor characteristics, and inter-patient variability.

PPG provides complementary cardiovascular information but is indirectly related to cardiac electrical activity and can also be affected by motion artifacts.

The objective of this research is to combine ECG and PPG to develop a robust multimodal framework for automated AF detection while exploring real-time edge deployment.

---

## Proposed Methodology

The proposed framework processes ECG and PPG independently before multimodal fusion.

The main stages are:

1. Modality-specific preprocessing
2. Multiscale temporal feature extraction
3. Bidirectional temporal modelling using BiGRU
4. Attention-guided representation learning
5. ECG-PPG feature-level fusion
6. Residual learning
7. Gated feature modulation
8. AF / Non-AF classification
9. Edge deployment evaluation

---

## System Architecture

### ECG Branch

ECG Segment (1500 × 1)

→ MultiScale 1D CNN  
→ Kernel Sizes: 3, 7, 11  
→ Batch Normalization  
→ Max Pooling  
→ BiGRU  
→ Attention  
→ Global Average Pooling  
→ 256-D ECG Feature

### PPG Branch

PPG Segment (1500 × 1)

→ MultiScale 1D CNN  
→ Kernel Sizes: 3, 5, 9  
→ Batch Normalization  
→ Max Pooling  
→ BiGRU  
→ Attention  
→ Global Average Pooling  
→ 256-D PPG Feature

### Multimodal Fusion

256-D ECG Feature + 256-D PPG Feature

→ 512-D Fused Representation

→ Residual-Gated Classification Network

→ AF / Non-AF

---

## Signal Preprocessing

### ECG

The ECG preprocessing pipeline consists of:

- Band-pass filtering: 0.5–40 Hz
- Adaptive Median Filtering (AMF)
- Z-score normalization
- Segmentation into 1500-sample windows

### PPG

The PPG preprocessing pipeline consists of:

- Band-pass filtering: 0.5–8 Hz
- Savitzky-Golay smoothing
- Z-score normalization
- Segmentation into 1500-sample windows

Both modalities are processed independently while maintaining synchronization.

---

## Multimodal Feature Learning

The ECG and PPG branches generate compact modality-specific representations.

Each branch produces a 256-dimensional feature vector.

These representations are concatenated:

256 ECG + 256 PPG = 512-dimensional fused representation.

The fused representation is then processed using residual learning and gated feature modulation before final classification.

---

## Dataset

The experiments use the MIMIC PERform AF dataset.

Dataset configuration used in this work:

| Parameter | Value |
|---|---|
| Patients used | 35 |
| Sampling Frequency | 125 Hz |
| Segment Length | 1500 samples |
| Segment Duration | 12 seconds |
| Generated Segments | 3500 |
| Modalities | ECG + PPG |
| Classes | AF / Non-AF |

Dataset files are not included in this repository.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1 × 10⁻⁴ |
| Weight Decay | 1 × 10⁻⁵ |
| Epochs | 150 |
| Batch Size | 32 |
| Cross Validation | Stratified 5-Fold |

---

## Results

The reported experimental evaluation achieved:

| Metric | Result |
|---|---:|
| Accuracy | 99.89% |
| Precision | 99.84% |
| Recall | 99.95% |
| F1-Score | 99.89% |
| Specificity | 99.81% |
| ROC-AUC | 1.00 |

### Confusion Matrix

The reported confusion matrix contains:

- True Negatives: 1597
- False Positives: 3
- False Negatives: 1
- True Positives: 1899

---

## Edge Deployment

The trained multimodal framework was evaluated on an NVIDIA Jetson Nano.

Reported deployment performance:

| Parameter | Result |
|---|---:|
| Samples Evaluated | 3500 |
| Accuracy | 99.89% |
| Total Inference Time | 1.255 s |
| Average Latency / Sample | 0.3587 ms |

This evaluation explores the feasibility of low-latency multimodal physiological signal analysis on an edge computing platform.

---

## Project Structure

```text
Multimodal-ecg-ppg-af-detection/
│
├── ECG(AMF).ipynb
├── ppg(sg)2.ipynb
├── ecg feature extractor.ipynb
├── ecg ppg feature extractor.ipynb
├── kfold.ipynb
│
├── tarak_jetson/
│   ├── classification_report.txt
│   ├── evaluation_af.py
│   ├── multimodal_label_encoder.pkl
│   └── multimodal_scaler.pkl
│
├── .gitignore
└── README.md
