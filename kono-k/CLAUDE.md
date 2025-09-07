# CMI 2025 Gesture Recognition Competition

## Project Overview
This is a machine learning competition project for gesture recognition using IMU and TOF/THM sensor data. The model uses a two-branch neural network architecture to classify gestures from sensor sequences.

## Project Structure
```
├── data/                   # Training and test data
├── input/                  # Configuration files
├── result/                 # Model outputs and artifacts
├── run/                    # Main training script
└── src/                    # Source code
    ├── dataset/            # Data processing and augmentation
    ├── metric/             # Competition metrics
    ├── model/              # Model architectures
    └── utils/              # Utilities and configuration
```

## Key Commands
- **Train model**: `python run/main.py`
- **Python environment**: Use `python3` instead of `python`
- **Dependencies**: torch, pandas, numpy, sklearn, wandb, timm, joblib

## Configuration
- Main config: `input/config.yaml`
- Experiment tracking: WandB (entity: CMI2025, project: CMI2025)
- Cross-validation: 5-fold stratified

## Data
- Training data: `data/train.csv`, `data/train_demographics.csv`
- Test data: `data/test.csv`, `data/test_demographics.csv`
- Features: IMU sensors + TOF/THM sensors

## Model Details
- Architecture: Two-branch model for different sensor types
- Sequence length: 100 (configurable via PAD_PERCENTILE)
- Augmentation: Jitter, dropout, modality dropout
- Metric: Hierarchical F1 score