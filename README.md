# Predictive Safety and Dynamic Risk Estimation in Human–Robot Collaboration

## Overview
This project proposes a two‑stage pipeline for risk estimation in Human–Robot Collaboration (HRC) using 3D skeletons (lighter and more privacy‑preserving than RGB):

1) **Instantaneous risk (MLP)**
- An **MLP** classifies each frame as **Safe / Near‑Collision / Collision** from human–robot joint distances.

2) **Predictive risk (LSTM)**
- An **LSTM encoder‑decoder** uses **10 past frames** to predict the risk evolution over **25 future frames (~1s at 25 fps)**.
- No explicit TTC is used: temporal dynamics are learned directly from data.

The dataset is an annotated version of **CHICO** with 20 subjects and realistic industrial actions; the split follows the subject‑independent protocol from arXiv:2208.07308. Coordinates are normalized in a **root‑relative** manner to improve spatial invariance.

---

## Dataset
The dataset is not included in the repo. Download it from Hugging Face:
- https://huggingface.co/datasets/frasic3/3d_skeletons_risk/tree/main

Expected structure:
```
 datasets/
   3d_skeletons_risk/
     S00/
     S01/
     ...
     S19/
```

Each sequence is a .pkl file with frames structured as:
```
[human_joints, robot_joints, risk]
```
- **Human joints:** 15 joints, 3D coordinates (x, y, z) in mm
- **Robot joints:** 9 joints, 3D coordinates (x, y, z) in mm
- **Risk class:**
  - Safe (0): $d_{min} > 630$ mm
  - Near‑Collision (1): $130 < d_{min} \le 630$ mm
  - Collision (2): $d_{min} \le 130$ mm

> If you have the original CHICO dataset without labels, you can generate them with [add_risk_label.py](add_risk_label.py).

---

## How to run (training/eval)
### 1) Main dependencies
- Python 3.10+
- PyTorch
- NumPy, tqdm
- Matplotlib, seaborn
- scikit‑learn

### 2) MLP training (instantaneous risk)
Example:
```
python train_mlp.py --data_path path/to/the/dataset/3d_skeletons_risk --epochs 10 --threshold 0.10
```
Outputs are saved in `runs/mlp_<timestamp>/` with metrics and plots.

### 3) LSTM training (predictive risk)
Example:
```
python train_lstm.py --data_path path/to/the/dataset/3d_skeletons_risk --epochs 5 --threshold 0.10
```
Outputs are saved in `runs/lstm_<timestamp>/` with metrics and plots.

### 4) Checkpoint evaluation (no retraining)
Examples:
```
# MLP
python eval_checkpoint.py --model mlp --data_path path/to/the/dataset/3d_skeletons_risk \
  --checkpoint path/to/checkpoints/runs/mlp_YYYYMMDD_HHMMSS/mlp_best.pth --threshold 0.10

# LSTM
python eval_checkpoint.py --model lstm --data_path path/to/the/datasets/3d_skeletons_risk \
  --checkpoint path/to/checkpointsruns/lstm_YYYYMMDD_HHMMSS/lstm_best.pth --threshold 0.10
```

---

## Notes
- Results and visualizations are saved in the `runs/` folder.
- Risk thresholds and subject‑independent splits are consistent with the paper.

---

## References
- Alessio Sampieri et al. (2022). *Pose Forecasting in Industrial Human‑Robot Collaboration*. https://arxiv.org/abs/2208.07308
- Amir Shahroudy et al. (2016). *NTU RGB+D: A large scale dataset for 3D human activity analysis*. https://arxiv.org/abs/1604.02808
