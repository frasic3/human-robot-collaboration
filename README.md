# Predictive Safety and Dynamic Risk Estimation in Human–Robot Collaboration

## Panoramica (dal paper)
Questo progetto propone una pipeline in due fasi per la stima del rischio in scenari di Human–Robot Collaboration (HRC) usando scheletri 3D (più leggeri e privacy‑preserving rispetto a RGB):

1) **Rischio istantaneo (MLP)**
- Un **MLP** classifica ogni frame in **Safe / Near‑Collision / Collision** a partire dalle distanze tra giunti umani e robot.

2) **Rischio predittivo (LSTM)**
- Un **LSTM encoder‑decoder** usa **10 frame passati** per predire l’evoluzione del rischio su **25 frame futuri (~1s a 25 fps)**.
- Non si usa TTC esplicito: il modello apprende dinamiche temporali direttamente dai dati.

Il dataset è una versione annotata di **CHICO** con 20 soggetti e azioni industriali realistiche; lo split segue il protocollo subject‑independent di arXiv:2208.07308. Le coordinate sono normalizzate in modo **root‑relative** per rendere il modello più robusto alla posizione globale.

---

## Dataset
Il dataset non è incluso nel repo. Puoi scaricarlo da Hugging Face:
- https://huggingface.co/datasets/frasic3/3d_skeletons_risk/tree/main

Struttura attesa:
```
 datasets/
   3d_skeletons_risk/
     S00/
     S01/
     ...
     S19/
```

Ogni sequenza è un file .pkl con frame strutturati come:
```
[human_joints, robot_joints, risk]
```
- **Human joints:** 15 giunti, coordinate 3D (x, y, z) in mm
- **Robot joints:** 9 giunti, coordinate 3D (x, y, z) in mm
- **Risk class:**
  - Safe (0): $d_{min} > 630$ mm
  - Near‑Collision (1): $130 < d_{min} \le 630$ mm
  - Collision (2): $d_{min} \le 130$ mm

> Se disponi del CHICO originale senza etichette, puoi generarle con [add_risk_label.py](add_risk_label.py).

---

## Come avviare le simulazioni (training/eval)
### 1) Dipendenze principali
- Python 3.10+
- PyTorch
- NumPy, tqdm
- Matplotlib, seaborn
- scikit‑learn

### 2) Training MLP (rischio istantaneo)
Esempio:
```
python train_mlp.py --data_path datasets/3d_skeletons_risk --epochs 10 --threshold 0.10
```
Output in `runs/mlp_<timestamp>/` con metriche e grafici.

### 3) Training LSTM (rischio predittivo)
Esempio:
```
python train_lstm.py --data_path datasets/3d_skeletons_risk --epochs 5 --threshold 0.10
```
Output in `runs/lstm_<timestamp>/` con metriche e grafici.

### 4) Valutazione di un checkpoint (senza retraining)
Esempi:
```
# MLP
python eval_checkpoint.py --model mlp --data_path datasets/3d_skeletons_risk \
  --checkpoint runs/mlp_YYYYMMDD_HHMMSS/mlp_best.pth --threshold 0.10

# LSTM
python eval_checkpoint.py --model lstm --data_path datasets/3d_skeletons_risk \
  --checkpoint runs/lstm_YYYYMMDD_HHMMSS/lstm_best.pth --threshold 0.10
```

---

## Note
- I risultati e le visualizzazioni vengono salvati nella cartella `runs/`.
- Le soglie di rischio e gli split soggetto‑indipendenti sono coerenti con il paper.

---

## Riferimenti
- Alessio Sampieri et al. (2022). *Pose Forecasting in Industrial Human‑Robot Collaboration*. https://arxiv.org/abs/2208.07308
- Amir Shahroudy et al. (2016). *NTU RGB+D: A large scale dataset for 3D human activity analysis*. https://arxiv.org/abs/1604.02808
