# AN2DL Challenges - The Big Batch Theory

This repository contains the projects developed for the **Artificial Neural Networks and Deep Learning (AN2DL)** course at Politecnico di Milano. Our team, **The Big Batch Theory**, successfully completed two competitive deep learning challenges, earning a final evaluation of **9.5/10**.

---

## Performance Overview

| **Competition**                       | **Final Rank** | **Final Score (F1)** |
| ------------------------------------- | -------------- | -------------------- |
| **Challenge 1: Time Series**          | **15 / 193**   | 0.9567               |
| **Challenge 2: Image Classification** | **106 / 193**  | 0.3795               |

---

## Challenge 1: Multi-class Time-Series Classification

**Objective:** Classify subject pain status into three categories (_no_pain_, _low_pain_, _high_pain_) using multi-sensor temporal data.

- **Approach:** Implementation of a flexible `RecurrentClassifier` supporting RNN, LSTM, and GRU architectures.
    
- **Strategy:** Data preprocessing involved sliding-window generation, PCA for dimensionality reduction, and class-weighted loss functions.
    
- **Validation:** Robust inference via a 5-fold cross-validation ensemble.
    

For a comprehensive breakdown of the hyperparameters, grid search results, and regularization techniques, please refer to the full technical report. 

---

## Challenge 2: Histological Slide Classification

**Objective:** Fine-grained classification of histological slides into four distinct subtypes: _Luminal A, Luminal B, HER2(+), and Triple Negative_.

- **Approach:** Evolution from a baseline ResNet-18 to an advanced Multiple Instance Learning (MIL) pipeline.
    
- **Architecture:** Attention-based MIL utilizing a Vision Transformer (`ViT`) as the feature extraction backbone.
    
- **Data Pipeline:** Implementation of an automated cleaning script using HSV color-space filtering to eliminate artifacts and "label noise".
    

Detailed information regarding ROI-centric preprocessing, multi-scale tiling, and fine-tuning strategies can be found in the technical report.


---

## Team Members

   
- **Andrea Rossi**
    
- **Fabio Rossi**
    
- **Francesco Sarra**

- **Benedetta Mussini**
