# Predictive Safety and Dynamic Risk Estimation in Human–Robot Collaboration

**Authors:** Alessandro Tommasi (s353532), Alessio Sorrentino (s353528), Francesco Sicilia (s354909)

## 1. Project Overview

This project implements a two-phase risk estimation system for human-robot collaboration scenarios using the CHICO dataset. The system predicts collision risk based on 3D skeleton data of humans and robots working together.

## 2. Dataset

Based on the public [CHICO-PoseForecasting](https://github.com/AlessioSam/CHICO-PoseForecasting) dataset.

### Structure
- **20 subjects** (S00–S19) performing realistic industrial actions
- **7 action types**: hammer, lift, place-hp, place-lp, polish, span_heavy, span_light
- **24 joints per frame** (15 human + 9 robot), each with 3D coordinates (x, y, z) in millimeters
- **Normal and CRASH versions** for each action (with and without physical collision)

### Dataset Split
Following the protocol from [arXiv:2208.07308](https://arxiv.org/abs/2208.07308) Section 6.1:

| Split | Subjects | Count |
|-------|----------|-------|
| **Train** | S01, S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, S15, S16, S17 | 14 |
| **Validation** | S00, S04 | 2 |
| **Test** | S02, S03, S18, S19 | 4 |

### Risk Labels
Risk levels are computed based on minimum human-robot joint distance:

| Risk Level | Class | Distance Threshold |
|------------|-------|-------------------|
| Safe | 0 | > 630 mm |
| Near-collision | 1 | 130–630 mm |
| Collision | 2 | ≤ 130 mm |

## 3. Architecture

### Phase 1: MLP for Static Risk Classification
A Multilayer Perceptron that classifies the risk level of a single frame.

- **Input**: Single frame with 24 joints × 3 coordinates = **72 features**
- **Architecture**: Fully connected layers (72 → 128 → 64 → 3)
- **Output**: Risk class (0, 1, or 2)
- **Loss**: Weighted Cross-Entropy

### Phase 2: LSTM for Future Risk Prediction
An LSTM-based encoder-decoder that predicts future risk levels from past observations.

- **Input**: Sequence of **T=10 past frames** → (10, 72)
- **Encoder**: 2-layer LSTM (hidden_size=128)
- **Decoder**: Linear layers mapping final hidden state to predictions
- **Output**: **P=25 future risk labels** (1 second at 25fps)
- **Loss**: Weighted Cross-Entropy

## 4. Project Structure

```
human-robot-collaboration/
├── datasets/
│   └── 3d_skeletons_risk/     # Dataset with risk labels
│       ├── S00/
│       ├── S01/
│       └── ...
├── models/
│   ├── mlp.py                 # MLP architecture
│   └── lstm.py                # LSTM architecture
├── utils/
│   └── pkl_data_loader.py     # Dataset loading and preprocessing
├── train_mlp.py               # MLP training script
├── train_lstm.py              # LSTM training script
└── add_risk_label.py          # Script to add risk labels to dataset
```

## 5. Data Preprocessing

1. **Root-Relative Normalization**: All joint positions are normalized relative to the human root joint (first joint), making the model invariant to absolute position in space while preserving human-robot relative distances.

2. **Z-Score Normalization**: Statistics (mean, std) are computed **only on the training set** and applied to all splits to prevent data leakage.

## 6. Training Setup

### Weighted Loss
Due to class imbalance (most frames are "Safe"), we use weighted Cross-Entropy:
- Safe (0): weight = 1.0
- Near-collision (1): weight = 5.0
- Collision (2): weight = 20.0

### Hyperparameters
| Parameter | MLP | LSTM |
|-----------|-----|------|
| Batch Size | 32 | 32 |
| Learning Rate | 1e-3 | 1e-3 |
| Optimizer | Adam | Adam |
| Epochs | 20 | 20 |

## 7. Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class metrics
- **Confusion Matrix**: To analyze error patterns
- **Collision Recall**: Critical metric - how many actual collisions are correctly predicted

## 8. Usage

### Prerequisites
```bash
pip install torch numpy scikit-learn tqdm
```

### Add Risk Labels to Dataset
```bash
python add_risk_label.py
```

### Train MLP
```bash
python train_mlp.py --data_path datasets/3d_skeletons_risk --epochs 20
```

### Train LSTM
```bash
python train_lstm.py --data_path datasets/3d_skeletons_risk --epochs 20 --input_frames 10 --output_frames 25
```

## 9. References

- CHICO Dataset: [GitHub](https://github.com/AlessioSam/CHICO-PoseForecasting)
- Reference Paper: [arXiv:2208.07308](https://arxiv.org/abs/2208.07308)
