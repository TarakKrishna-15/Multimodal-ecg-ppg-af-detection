# Multimodal ECG-PPG AF Detection

## Attention-Guided Residual–Gated Multimodal Learning for Real-Time Atrial Fibrillation Detection Using ECG and PPG

A multimodal deep learning framework for detecting **Atrial Fibrillation (AF)** using synchronized **Electrocardiography (ECG)** and **Photoplethysmography (PPG)** signals.

---

## Research Overview

Atrial Fibrillation (AF) is a common cardiac arrhythmia characterized by irregular cardiac activity and is associated with an increased risk of cardiovascular complications.

ECG provides information about the electrical activity of the heart, while PPG provides information related to peripheral blood-volume changes.

This research combines these complementary physiological modalities using an **attention-guided residual-gated multimodal deep learning architecture** for AF detection.

The framework is designed with real-time and edge deployment considerations.

---

## Overall Pipeline

```text
                    ECG + PPG
                       │
                       ▼
          Modality-Specific Preprocessing
                 │             │
                 ▼             ▼
             ECG Branch     PPG Branch
                 │             │
                 ▼             ▼
          MultiScale CNN   MultiScale CNN
                 │             │
                 ▼             ▼
               BiGRU         BiGRU
                 │             │
                 ▼             ▼
             Attention      Attention
                 │             │
                 ▼             ▼
              256-D ECG     256-D PPG
                 │             │
                 └──────┬──────┘
                        ▼
                 512-D Feature
                     Fusion
                        │
                        ▼
              Residual-Gated
                 Classifier
                        │
                        ▼
                  AF / Non-AF
