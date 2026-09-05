# Imbalance-Aware Dynamic FedAvg (IA-D FedAvg) for IoMT Skin Cancer Classification

This repository contains the official PyTorch implementation of **IA-D FedAvg**, a novel federated learning framework designed to solve severe **Class Imbalance** and **Non-IID Data Heterogeneity** in Internet of Medical Things (IoMT) environments using the HAM10000 dataset.

## Key Features
- **Dynamic Loss Regularization ($\mu$-Adaptation):** Automatically adjusts client-side loss penalty based on class distribution and local gradient variance without needing heavy RL models.
- **Automated Data Pipeline:** Built-in Dataset downloader with real-time download/extraction progress tracking.
- **Resource Efficient:** Significantly faster training convergence compared to `AdaFedProx` and `FedCAD`.

## Project Structure
```text
IA_D_FedAvg_SkinCancer/
│
├── data/                    # Automated dataset storage
├── src/
│   ├── dataset.py           # Auto downloader, extract & loader
│   ├── models.py            # CNN/ResNet models
│   ├── utils.py             # Imbalance metrics & dynamic weights
│   └── algorithms/          # Baseline & Proposed algorithms
│
├── main.py                  # Experiment Runner
├── requirements.txt         # Dependencies
└── README.md                # Project documentation