# Human Robot Collaboration Task

Implementation of **Human Pose Forecasting** models for collaborative robots on the CHICO dataset (Cobots and Humans in Industrial COllaboration).

## 🎯 Pipeline

### 1. Real-time Data Acquisition
- **Input**: Video Stream (from one or more Cameras)
- **Processing**: 
  - **Pose Estimation AI**: Extracts 3D Skeleton (Human + Robot) from video frames.
  - **Model**: ResNet18-based Pose Estimator (trained on CHICO).
  - **Procedure**: Detects both Human and Robot arms, aligned with CHICO ground truth.
  - **Note**: The system is trained on multiple views (Top, Side, Front), so it can work with different camera angles.

### 2. Pose Forecasting (Action Prediction)
- **Input**: Sequence of 3D Skeletons (10 frames)
- **Model**: MLP (Multi-Layer Perceptron), LSTM, or CNN (1D Temporal)
- **Output**: Future 3D Skeleton (25 frames)
- **Goal**: Anticipate human and robot movement.
- **Note**: LSTM and CNNs can also be tested here (CNNs can process coordinate sequences as 1D temporal data, not just images).

### 3. Safety Analysis (Geometric Calculation)
- **Input**: Predicted Future Skeleton
- **Algorithm**: Geometric Distance Calculation
- **Logic**: Calculate Euclidean distance between Human joints and Robot joints.
- **Output**: Safety Score, Collision Warning.

## 🏗️ Project Structure (TO-MODIFY)

```
human-robot-collaboration/
├── datasets/                   # Dataset
│   └── 3d_skeletons/
│       └── S00/ ... S19/       # 20 subjects
├── models/                     # Neural architectures
│   ├── mlp.py                  # MLP
│   └── __init__.py
├── utils/                      # Utilities and metrics
│   ├── metrics.py              # MPJPE and other metrics
│   ├── cnn_data_loader.py                      # CNN Dataset loading
│   ├── pkl_data_loader.py                      # .pkl Dataset loading
│   ├── pose_estimation_data_loader.py          # Pose Dataset loading
|   └── __init__.py
├── inspect_pkl.py              # Visualization script
├── coppelia_sim/               # CoppeliaSim integration
│   ├── pick_and_place_UR5.py
│   └── coppeliasim_zmqremoteapi_client/
├── checkpoints/                # Model checkpoints
├── runs/                       # TensorBoard logs
├── train.py                    # Training script
└── README_POSE_FORECASTING.md
```

## 📈 Implemented Models

### MLP 
- **Input**: 10 frames (400ms @ 25fps)
- **Output**: 25 frames (1 second @ 25fps)
- **Architecture**: FC(1024) → FC(512) → FC(256) → Output
- **Parameters**: ~1.6M

## ⚙️ Configuration

### Temporal Parameters
```python
INPUT_FRAMES = 10       # 400ms history (observes movement)
OUTPUT_FRAMES = 25      # 480ms prediction @ 25fps
```

### Dataset Split
```python
Training:   S00-S14  (16 subjects, ~75%)
Validation: S15-S17  (2 subjects, ~15%)
Test:       S18-S19  (2 subjects, ~10%)
```

# Appendix: Dataset & References

## 📊 CHICO Dataset

Human-robot interaction dataset in industrial environments:
- **20 subjects** (S00-S19) performing collaborative tasks
- **7 industrial actions** (realistic)
- **24 3D joints** per frame:
  - 15 human skeleton joints (x,y,z coordinates in mm)
  - 9 robotic arm joints (x,y,z coordinates in mm)
- **Normal and CRASH versions** (with collision) for each action

### Available Actions:
| Action | Description |
|--------|-------------|
| `hammer` | Hammering |
| `lift` | Lifting an object |
| `place-hp` | High placement |
| `place-lp` | Low placement |
| `polish` | Polishing a surface |
| `span_heavy` | Spanning heavy object |
| `span_light` | Spanning light object |

### 📊 CHICO Dataset Insights (Structure Discovery)

The CHICO dataset has a unique structure that we leverage:
- **3 Parallel Video Views**: For every action (e.g., `S00_hammer`), there are 3 video files (e.g., `00_03.mp4`, `00_06.mp4`, `00_12.mp4`).
- **Synchronization**: These 3 videos are **synchronized views** (Top, Side, Front) of the exact same event.
- **Alignment**: Frame $N$ in *any* of the 3 videos corresponds exactly to row $N$ in the `.pkl` skeleton file.

**Implication for Training**:
- We treat each video view as an independent training sample.
- This **triples** our effective training data size.
- The model learns to estimate 3D pose from multiple viewing angles, making it robust to camera placement.

## 📚 Reference Paper

**Title**: "Pushing back the frontiers of collaborative robots in industrial environments"

**Key Contributions**:
1. SeS-GCN: Separable-Sparse GCN for pose forecasting
2. CHICO Dataset: 20 operators, 7 actions, 226 genuine collisions
3. Results: 85.3mm MPJPE @ 1sec, 2.3ms runtime, 1.72% parameters

### Metrics Explained

#### MPJPE (Mean Per Joint Position Error)
Mean error between predicted poses and ground truth in millimeters.

```python
MPJPE = mean(||predicted_joint - target_joint||₂)
```

#### Paper Target:
- **MPJPE @ 1 sec**: 85.3 mm
- **Runtime**: 2.3 msec
- **Parameters**: ~1.72% of state-of-the-art

#### Loss vs MPJPE
- **Loss (MSE)**: Abstract value (squared error). Hard to visualize.
- **MPJPE**: **Physical Error**. Very easy to interpret because it tells you exactly **how many millimeters** the prediction is away from the real position.
