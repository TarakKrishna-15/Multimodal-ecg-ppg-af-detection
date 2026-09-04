# Multimodal ECG-PPG AF Detection

## Attention-Guided Residual–Gated Multimodal Learning for Real-Time Atrial Fibrillation Detection Using ECG and PPG

A multimodal deep learning framework for detecting Atrial Fibrillation (AF) using synchronized Electrocardiography (ECG) and Photoplethysmography (PPG) signals.

## Research Overview

Atrial Fibrillation is a common cardiac arrhythmia associated with increased risk of stroke and other cardiovascular complications.

ECG provides electrical information about cardiac activity, while PPG provides information related to peripheral blood-volume changes.

This project combines these complementary physiological modalities using a multimodal deep learning framework.

### Overall Pipeline

```text
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
