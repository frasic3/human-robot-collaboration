import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import random
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.lstm import RiskLSTM
from utils.pkl_data_loader import create_pkl_dataloaders

TRAIN_SUBJECTS = [
    'S01', 'S05', 'S06', 'S07', 'S08', 'S09',
    'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17'
]

VAL_SUBJECTS = ['S00', 'S04']

TEST_SUBJECTS = ['S02', 'S03', 'S18', 'S19']

# Risk Configuration
RISK_WEIGHTS = [1.0, 2.0, 12.0]  # Safe, Near, Collision
CLASS_NAMES = ['Safe', 'Near-Collision', 'Collision']
BATCH_SIZE = 64
INPUT_FRAMES = 10
OUTPUT_FRAMES = 25
EVAL_FRAMES = 15
SAMPLER_STRATEGY = 'balanced_batch'
SAMPLER_COLLISION_SCALE = 0.0
SAMPLER_COLLISION_POWER = 1.0
SAMPLER_MIN_WEIGHT = 1.0
URGENT_FRAMES = 15
DECAY_FACTOR = 0.3
TRAIN_STRIDE = 10
VAL_STRIDE = 1
TEST_STRIDE = 1


class TemporalWeightedCrossEntropy(nn.Module):

    def __init__(
        self,
        class_weights,
        output_frames=25,
        urgent_frames=15,
        decay_factor=0.3,
        device='cpu',
    ):

        super().__init__()
        self.class_weights = torch.tensor(class_weights).float().to(device)
        self.output_frames = int(output_frames)
        temporal_weights = torch.ones(self.output_frames, device=device)
        urgent_frames = int(max(0, min(urgent_frames, self.output_frames)))
        if urgent_frames < self.output_frames:
            temporal_weights[urgent_frames:] = float(decay_factor)
        self.temporal_weights = temporal_weights

    def forward(self, inputs, targets):

        # inputs: (B, P, C)
        # targets: (B, P)
        ce_loss = F.cross_entropy(
            inputs.view(-1, inputs.shape[-1]),
            targets.view(-1),
            weight=self.class_weights,
            reduction='none',
        )
        ce_loss = ce_loss.view(inputs.shape[0], self.output_frames)
        weighted_loss = ce_loss * self.temporal_weights
        return weighted_loss.mean()


def compute_forecasting_loss(criterion, outputs, targets):

    """Compute loss for forecasting outputs.
    - TemporalWeightedCrossEntropy expects outputs (B,P,C) and targets (B,P)
    - nn.CrossEntropyLoss expects flattened (B*P,C) and (B*P)
    """
    if isinstance(criterion, TemporalWeightedCrossEntropy):

        return criterion(outputs, targets)
    return criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))


def resolve_risk_weights(args):

    """Fixed loss weights (Safe, Near, Collision)."""
    w = getattr(args, 'risk_weights', None)
    if w is None:
        return RISK_WEIGHTS
    return list(w)


def mean_class_recall_first_k(targets: np.ndarray, preds: np.ndarray, k: int) -> float:
    """Compute mean-class recall over the first k steps."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    k = int(max(1, k))
    t = targets[:, :k].reshape(-1)
    p = preds[:, :k].reshape(-1)
    recalls = []
    for c in range(3):

        mask = (t == c)
        if mask.sum() > 0:
            recall = np.sum((p == c) & mask) / mask.sum()
            recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0


def min_class_recall_first_k(targets: np.ndarray, preds: np.ndarray, k: int) -> tuple[float, dict]:
    """Compute min-class recall over the first k steps.
    Returns:
        (min_recall, per_class_recalls_dict)
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    k = int(max(1, k))
    t = targets[:, :k].reshape(-1)
    p = preds[:, :k].reshape(-1)
    recalls = {}
    values = []
    for c, name in enumerate(CLASS_NAMES):

        mask = (t == c)
        denom = int(mask.sum())
        if denom == 0:
            r = 0.0
        else:
            r = float((p[mask] == c).sum()) / float(denom)
        recalls[name] = r
        values.append(r)
    return float(min(values)), recalls


def _write_tuning_metrics(run_dir: str, payload: dict) -> None:
    import json
    path = os.path.join(run_dir, 'tuning_metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _safe_bincount(arr: np.ndarray, minlength: int = 3) -> np.ndarray:
    arr = np.asarray(arr).astype(np.int64).reshape(-1)
    return np.bincount(arr, minlength=minlength)


def print_sequence_dataset_stats(dataset, name: str, limit: int = 2000):

    """Print quick stats for sequence datasets without scanning everything."""
    try:
        sequences = dataset.sequences
    except Exception:
        print(f"[{name}] Could not access dataset.sequences")
        return
    n = len(sequences)
    if n == 0:
        print(f"[{name}] Empty dataset")
        return
    k = min(n, max(1, int(limit)))
    idxs = np.random.choice(n, k, replace=False) if k < n else np.arange(n)
    # Per-sequence label = max future risk
    seq_labels = []
    frame_counts = np.zeros(3, dtype=np.int64)
    collision_frames_per_seq = []
    for i in idxs:
        s = sequences[int(i)]
        y = s.get('output_risk', None)
        if y is None:
            continue
        y = np.asarray(y)
        seq_labels.append(int(np.max(y)))
        frame_counts += _safe_bincount(y, minlength=3)
        collision_frames_per_seq.append(int(np.sum(y == 2)))
    seq_labels = np.asarray(seq_labels, dtype=np.int64)
    seq_counts = _safe_bincount(seq_labels, minlength=3)
    total_frames = int(frame_counts.sum())
    frame_pct = (frame_counts / max(1, total_frames)) * 100.0
    seq_pct = (seq_counts / max(1, seq_counts.sum())) * 100.0
    print(f"\n[{name}] Stats on {len(seq_labels)}/{n} sequences (sampled)")
    print(f"  Per-sequence max(y) distribution: {dict(zip(CLASS_NAMES, seq_counts.tolist()))}  (%: {seq_pct.round(2).tolist()})")
    print(f"  Per-frame distribution over future horizon: {dict(zip(CLASS_NAMES, frame_counts.tolist()))}  (%: {frame_pct.round(2).tolist()})")
    if len(collision_frames_per_seq) > 0:
        cf = np.asarray(collision_frames_per_seq, dtype=np.int64)
        print(f"  Collision frames per collision-seq: mean={cf[seq_labels==2].mean() if np.any(seq_labels==2) else 0:.2f}, max={cf.max()}")
# ============================================================================
# LEVEL 1: PER-TIMESTEP METRICS
# ============================================================================


def compute_per_timestep_metrics(all_targets, all_preds, output_frames, num_classes=3):

    """
    Calcola metriche per ogni timestep futuro.
    Args:
        all_targets: (N, P) array di ground truth per ogni sample
        all_preds: (N, P) array di predizioni per ogni sample
        output_frames: P - numero di frame futuri
        num_classes: numero di classi
    Returns:
        dict con:
        - accuracy_per_step: (P,) accuracy per ogni timestep
        - recall_per_class_per_step: (num_classes, P) recall per classe/timestep
        - precision_per_class_per_step: (num_classes, P) precision per classe/timestep
    """
    N = all_targets.shape[0]
    P = output_frames
    accuracy_per_step = np.zeros(P)
    recall_per_class_per_step = np.zeros((num_classes, P))
    precision_per_class_per_step = np.zeros((num_classes, P))
    for t in range(P):

        targets_t = all_targets[:, t]
        preds_t = all_preds[:, t]
        # Accuracy
        accuracy_per_step[t] = np.mean(targets_t == preds_t)
        # Per-class metrics
        for c in range(num_classes):

            true_positives = np.sum((targets_t == c) & (preds_t == c))
            actual_positives = np.sum(targets_t == c)
            predicted_positives = np.sum(preds_t == c)
            recall_per_class_per_step[c, t] = true_positives / (actual_positives + 1e-8)
            precision_per_class_per_step[c, t] = true_positives / (predicted_positives + 1e-8)
    return {
        'accuracy_per_step': accuracy_per_step,
        'recall_per_class_per_step': recall_per_class_per_step,
        'precision_per_class_per_step': precision_per_class_per_step
    }


def save_temporal_recall_curve(metrics, save_dir, filename='temporal_recall_curve.png'):

    """
    Curva di Recall Temporale per ogni classe.
    Asse X: Orizzonte temporale (+1 a +P frame)
    Asse Y: Recall
    """
    P = metrics['recall_per_class_per_step'].shape[1]
    timesteps = np.arange(1, P + 1)
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Verde, Arancione, Rosso
    for c, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):

        recall = metrics['recall_per_class_per_step'][c]
        plt.plot(timesteps, recall, '-o', color=color, label=f'{class_name}',
                 linewidth=2, markersize=4)
    plt.xlabel('Orizzonte Temporale (Frame Futuro)', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Curva di Recall Temporale - Degradazione della Predizione', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xlim(1, P)
    # Aggiungi annotazione per Collision al primo e ultimo frame
    collision_recall = metrics['recall_per_class_per_step'][2]
    plt.annotate(f'{collision_recall[0]:.2f}', (1, collision_recall[0]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.annotate(f'{collision_recall[-1]:.2f}', (P, collision_recall[-1]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()


def save_temporal_heatmap(all_targets, all_preds, output_frames, save_dir, filename='temporal_heatmap.png'):

    """
    Heatmap Temporale delle predizioni.
    Mostra RECALL per classe e timestep (i.e. accuracy condizionata sulla classe vera).
    """
    P = output_frames
    num_classes = 3
    # Crea matrice (num_classes, P) con accuracy per ogni combinazione
    heatmap_data = np.zeros((num_classes, P))
    for t in range(P):

        for c in range(num_classes):

            mask = all_targets[:, t] == c
            if np.sum(mask) > 0:
                heatmap_data[c, t] = np.mean(all_preds[:, t][mask] == c)
    plt.figure(figsize=(14, 5))
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                     xticklabels=[f'+{i+1}' for i in range(P)],
                     yticklabels=CLASS_NAMES,
                     vmin=0, vmax=1,
                     annot_kws={"size": 8})
    plt.xlabel('Frame Futuro', fontsize=12)
    plt.ylabel('Classe Vera', fontsize=12)
    plt.title('Heatmap Temporale: Recall per Classe e Orizzonte', fontsize=14)
    # Evidenzia la riga Collision
    ax.axhline(y=2, color='red', linewidth=3)
    ax.axhline(y=3, color='red', linewidth=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()


def compute_per_timestep_class_rates(all_targets, all_preds, output_frames, num_classes=3):

    """Per timestep: true-rate, predicted-rate, and true support per class."""
    N = int(all_targets.shape[0])
    P = int(output_frames)
    true_rate = np.zeros((num_classes, P), dtype=np.float64)
    pred_rate = np.zeros((num_classes, P), dtype=np.float64)
    true_support = np.zeros((num_classes, P), dtype=np.int64)
    for t in range(P):

        targets_t = np.asarray(all_targets[:, t], dtype=np.int64)
        preds_t = np.asarray(all_preds[:, t], dtype=np.int64)
        tc = _safe_bincount(targets_t, minlength=num_classes)
        pc = _safe_bincount(preds_t, minlength=num_classes)
        true_support[:, t] = tc
        true_rate[:, t] = tc / max(1, N)
        pred_rate[:, t] = pc / max(1, N)
    return {
        'true_rate': true_rate,
        'pred_rate': pred_rate,
        'true_support': true_support,
    }
# ============================================================================
# LEVEL 2: EVENT-BASED METRICS
# ============================================================================


def compute_mean_anticipation_time(all_targets, all_preds, fps=25):

    """
    Mean Anticipation Time (MAT) - Tempo di Preavviso Medio.
    Per ogni sequenza dove avviene una collisione reale al frame T,
    calcola quanti frame PRIMA il modello inizia a predire "Collision" stabilmente.
    Args:
        all_targets: (N, P) ground truth
        all_preds: (N, P) predizioni
        fps: frame rate per conversione in secondi
    Returns:
        dict con statistiche MAT
    """
    N, P = all_targets.shape
    anticipation_times = []
    for i in range(N):

        target_seq = all_targets[i]
        pred_seq = all_preds[i]
        # Trova il primo frame con Collision nel ground truth
        collision_indices = np.where(target_seq == 2)[0]
        if len(collision_indices) == 0:
            continue  # Nessuna collisione in questa sequenza
        first_collision = collision_indices[0]
        # Trova il primo frame dove il modello predice Collision PRIMA della collisione reale
        pred_collision_before = np.where((pred_seq == 2) & (np.arange(P) < first_collision))[0]
        if len(pred_collision_before) > 0:
            first_pred = pred_collision_before[0]
            anticipation = first_collision - first_pred
            anticipation_times.append(anticipation)
        else:
            # Il modello non ha anticipato, verifica se predice al momento della collisione
            if pred_seq[first_collision] == 2:
                anticipation_times.append(0)  # Nessun anticipo ma corretta
            else:
                anticipation_times.append(-1)  # Mancata completamente
    if len(anticipation_times) == 0:
        return {'mat_frames': 0, 'mat_seconds': 0, 'anticipation_rate': 0}
    # Filtra i casi mancati per il calcolo della media
    valid_anticipations = [a for a in anticipation_times if a >= 0]
    if len(valid_anticipations) == 0:
        mat_frames = 0
    else:
        mat_frames = np.mean(valid_anticipations)
    # Calcola quante collisioni sono state anticipate (>0 frame prima)
    anticipated = sum(1 for a in anticipation_times if a > 0)
    detected_at_moment = sum(1 for a in anticipation_times if a == 0)
    missed = sum(1 for a in anticipation_times if a == -1)
    return {
        'mat_frames': mat_frames,
        'mat_seconds': mat_frames / fps,
        'anticipated_count': anticipated,
        'detected_at_moment': detected_at_moment,
        'missed_count': missed,
        'total_collisions': len(anticipation_times),
        'anticipation_rate': anticipated / len(anticipation_times) if len(anticipation_times) > 0 else 0,
        'detection_rate': (anticipated + detected_at_moment) / len(anticipation_times) if len(anticipation_times) > 0 else 0
    }


def compute_event_level_recall(all_targets, all_preds, min_consecutive=1):

    """
    Event-Level Recall (Recall per Episodio).
    Una sequenza contenente una Collision è considerata "rilevata" se
    il modello predice Collision almeno una volta (o per min_consecutive frame).
    Args:
        all_targets: (N, P) ground truth
        all_preds: (N, P) predizioni
        min_consecutive: minimo frame consecutivi per considerare una detection
    Returns:
        dict con event recall statistics
    """
    N = all_targets.shape[0]
    sequences_with_collision = 0
    detected_sequences = 0
    for i in range(N):

        target_seq = all_targets[i]
        pred_seq = all_preds[i]
        has_collision = np.any(target_seq == 2)
        if not has_collision:
            continue
        sequences_with_collision += 1
        # Verifica se il modello ha predetto Collision
        pred_collisions = (pred_seq == 2)
        if min_consecutive == 1:
            if np.any(pred_collisions):

                detected_sequences += 1
        else:
            # Conta frame consecutivi di Collision
            max_consecutive = 0
            current_consecutive = 0
            for pc in pred_collisions:
                if pc:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            if max_consecutive >= min_consecutive:
                detected_sequences += 1
    event_recall = detected_sequences / sequences_with_collision if sequences_with_collision > 0 else 0
    return {
        'event_recall': event_recall,
        'detected_sequences': detected_sequences,
        'total_collision_sequences': sequences_with_collision,
        'min_consecutive_required': min_consecutive
    }
# ============================================================================
# LEVEL 3: PREDICTION STABILITY (FLICKERING)
# ============================================================================


def compute_flicker_rate(all_preds):

    """
    Flicker Rate (Tasso di Sfarfallio).
    Conta quante volte la predizione cambia classe all'interno di ogni sequenza.
    Args:
        all_preds: (N, P) predizioni
    Returns:
        dict con flicker statistics
    """
    N, P = all_preds.shape
    flicker_counts = []
    for i in range(N):

        pred_seq = all_preds[i]
        changes = np.sum(pred_seq[1:] != pred_seq[:-1])
        flicker_counts.append(changes)
    flicker_counts = np.array(flicker_counts)
    return {
        'mean_flicker': np.mean(flicker_counts),
        'std_flicker': np.std(flicker_counts),
        'max_flicker': np.max(flicker_counts),
        'min_flicker': np.min(flicker_counts),
        'flicker_histogram': flicker_counts
    }


def save_stability_analysis(all_preds, save_dir, filename='stability_analysis.png'):

    """
    Visualizza l'analisi della stabilità delle predizioni.
    """
    flicker_stats = compute_flicker_rate(all_preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Istogramma del flicker rate
    axes[0].hist(flicker_stats['flicker_histogram'], bins=20, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0].axvline(flicker_stats['mean_flicker'], color='red', linestyle='--', linewidth=2,
                     label=f"Media: {flicker_stats['mean_flicker']:.2f}")
    axes[0].set_xlabel('Numero di Cambi di Classe', fontsize=12)
    axes[0].set_ylabel('Frequenza', fontsize=12)
    axes[0].set_title('Distribuzione del Flicker Rate', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Box plot
    axes[1].boxplot(flicker_stats['flicker_histogram'], vert=True)
    axes[1].set_ylabel('Numero di Cambi di Classe', fontsize=12)
    axes[1].set_title('Box Plot Flicker Rate', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    # Aggiungi statistiche testuali
    stats_text = f"Media: {flicker_stats['mean_flicker']:.2f}\n"
    stats_text += f"Std: {flicker_stats['std_flicker']:.2f}\n"
    stats_text += f"Min: {flicker_stats['min_flicker']}\n"
    stats_text += f"Max: {flicker_stats['max_flicker']}"
    axes[1].text(1.15, 0.5, stats_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()
    return flicker_stats
# ============================================================================
# LEVEL 4: VISUAL METRICS (FORECAST PLOTS)
# ============================================================================


def save_forecast_plots(all_targets, all_preds, save_dir, num_samples=10, filename_prefix='forecast'):

    """
    Crea grafici "a raggiera" che confrontano ground truth e predizioni.
    Per ogni sample:
    - Linea nera: classe reale nei 25 frame futuri
    - Linea colorata: classe predetta
    """
    N, P = all_targets.shape
    # Seleziona sample con collision per visualizzazione interessante
    collision_indices = np.where(np.any(all_targets == 2, axis=1))[0]
    if len(collision_indices) < num_samples:
        sample_indices = collision_indices
    else:
        sample_indices = np.random.choice(collision_indices, num_samples, replace=False)
    # Aggiungi anche alcuni sample senza collision
    non_collision_indices = np.where(~np.any(all_targets == 2, axis=1))[0]
    if len(non_collision_indices) > 0:
        extra = min(3, len(non_collision_indices))
        sample_indices = np.concatenate([sample_indices,
                                          np.random.choice(non_collision_indices, extra, replace=False)])
    num_plots = len(sample_indices)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    timesteps = np.arange(1, P + 1)
    class_mapping = {0: 0, 1: 1, 2: 2}  # Safe=0, Near=1, Collision=2
    for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):

        gt = all_targets[sample_idx]
        pred = all_preds[sample_idx]
        # Plot ground truth
        ax.plot(timesteps, gt, 'k-', linewidth=3, label='Ground Truth', alpha=0.7)
        ax.scatter(timesteps, gt, c='black', s=30, zorder=5)
        # Plot prediction con colore basato sulla correttezza
        colors = ['green' if g == p else 'red' for g, p in zip(gt, pred)]
        ax.plot(timesteps, pred, '--', color='blue', linewidth=2, label='Prediction', alpha=0.7)
        ax.scatter(timesteps, pred, c=colors, s=30, zorder=5, edgecolors='blue')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Safe', 'Near', 'Coll'])
        ax.set_xlabel('Frame Futuro')
        ax.set_ylabel('Classe')
        ax.set_title(f'Sample {sample_idx}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, P + 1)
        ax.set_ylim(-0.2, 2.5)
    # Nascondi assi vuoti
    for ax in axes[len(sample_indices):]:
        ax.set_visible(False)
    plt.suptitle('Forecast Plots: Ground Truth vs Prediction', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename_prefix}_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_aggregated_forecast_plot(all_targets, all_preds, save_dir, filename='forecast_aggregated.png'):

    """
    Grafico aggregato che mostra la distribuzione delle predizioni nel tempo.
    """
    N, P = all_targets.shape
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    timesteps = np.arange(1, P + 1)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    # Plot 1: Distribuzione Ground Truth nel tempo
    gt_dist = np.zeros((3, P))
    for t in range(P):

        for c in range(3):

            gt_dist[c, t] = np.mean(all_targets[:, t] == c)
    axes[0].stackplot(timesteps, gt_dist[0], gt_dist[1], gt_dist[2],
                      labels=CLASS_NAMES, colors=colors, alpha=0.8)
    axes[0].set_xlabel('Frame Futuro', fontsize=12)
    axes[0].set_ylabel('Proporzione', fontsize=12)
    axes[0].set_title('Distribuzione Ground Truth nel Tempo', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].set_xlim(1, P)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    # Plot 2: Distribuzione Predizioni nel tempo
    pred_dist = np.zeros((3, P))
    for t in range(P):

        for c in range(3):

            pred_dist[c, t] = np.mean(all_preds[:, t] == c)
    axes[1].stackplot(timesteps, pred_dist[0], pred_dist[1], pred_dist[2],
                      labels=CLASS_NAMES, colors=colors, alpha=0.8)
    axes[1].set_xlabel('Frame Futuro', fontsize=12)
    axes[1].set_ylabel('Proporzione', fontsize=12)
    axes[1].set_title('Distribuzione Predizioni nel Tempo', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(1, P)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()
# ============================================================================
# STANDARD VISUALIZATION FUNCTIONS
# ============================================================================


def save_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):

    """Salva le curve di loss e accuracy per training e validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, 'b-o', label='Train Loss', markersize=3)
    axes[0].plot(val_losses, 'r-o', label='Val Loss', markersize=3)
    axes[0].set_title('Loss Curves', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(train_accs, 'b-o', label='Train Acc', markersize=3)
    axes[1].plot(val_accs, 'r-o', label='Val Acc', markersize=3)
    axes[1].set_title('Accuracy Curves', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, save_dir, filename='confusion_matrix.png', info_text=''):

    """Salva la matrice di confusione (aggregata su tutti i timestep)."""
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.title(f'Confusion Matrix (Aggregated)\n{info_text}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()


def save_metrics_summary(metrics_dict, mat_stats, event_stats, flicker_stats, save_dir, filename='metrics_summary.png'):

    """
    Salva un riepilogo visivo di tutte le metriche.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1. Collision Recall nel tempo
    ax1 = axes[0, 0]
    P = metrics_dict['recall_per_class_per_step'].shape[1]
    timesteps = np.arange(1, P + 1)
    collision_recall = metrics_dict['recall_per_class_per_step'][2]
    ax1.fill_between(timesteps, collision_recall, alpha=0.3, color='red')
    ax1.plot(timesteps, collision_recall, 'r-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Frame Futuro')
    ax1.set_ylabel('Recall Collision')
    ax1.set_title('Degradazione Recall Collision')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target 80%')
    ax1.legend()
    # 2. MAT Statistics
    ax2 = axes[0, 1]
    categories = ['Anticipate', 'On Time', 'Missed']
    values = [mat_stats['anticipated_count'], mat_stats['detected_at_moment'], mat_stats['missed_count']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax2.bar(categories, values, color=colors, edgecolor='black')
    ax2.set_ylabel('Numero di Collisioni')
    ax2.set_title(f"Anticipazione Collisioni (MAT: {mat_stats['mat_frames']:.1f} frame = {mat_stats['mat_seconds']:.2f}s)")
    for bar, val in zip(bars, values):

        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
                 ha='center', va='bottom', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    # 3. Event-Level Stats
    ax3 = axes[1, 0]
    event_data = [event_stats['event_recall'], 1 - event_stats['event_recall']]
    labels = [f"Detected\n({event_stats['detected_sequences']})",
              f"Missed\n({event_stats['total_collision_sequences'] - event_stats['detected_sequences']})"]
    colors = ['#2ecc71', '#e74c3c']
    ax3.pie(event_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
            explode=(0.05, 0))
    ax3.set_title(f"Event-Level Recall: {event_stats['event_recall']:.1%}")
    # 4. Summary Text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = "=" * 40 + "\n"
    summary_text += "RIEPILOGO METRICHE FORECASTING\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += "PER-TIMESTEP:\n"
    summary_text += f"  • Recall Collision @+1:  {collision_recall[0]:.2%}\n"
    summary_text += f"  • Recall Collision @+25: {collision_recall[-1]:.2%}\n"
    summary_text += f"  • Recall Medio:          {np.mean(collision_recall):.2%}\n\n"
    summary_text += "ANTICIPAZIONE:\n"
    summary_text += f"  • MAT: {mat_stats['mat_frames']:.1f} frame ({mat_stats['mat_seconds']:.2f}s)\n"
    summary_text += f"  • Anticipation Rate: {mat_stats['anticipation_rate']:.1%}\n"
    summary_text += f"  • Detection Rate: {mat_stats['detection_rate']:.1%}\n\n"
    summary_text += "EVENT-LEVEL:\n"
    summary_text += f"  • Event Recall: {event_stats['event_recall']:.1%}\n"
    summary_text += f"  • Sequenze Rilevate: {event_stats['detected_sequences']}/{event_stats['total_collision_sequences']}\n\n"
    summary_text += "STABILITA':\n"
    summary_text += f"  • Flicker Medio: {flicker_stats['mean_flicker']:.2f} cambi/seq\n"
    summary_text += f"  • Flicker Max: {flicker_stats['max_flicker']} cambi/seq\n"
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()
# ============================================================================
# COMPREHENSIVE EVALUATION FUNCTION
# ============================================================================


def comprehensive_evaluation(all_targets, all_preds, all_probs, output_frames, save_dir, prefix=''):

    """
    Esegue la valutazione completa a 4 livelli e salva tutti i grafici.
    Args:
        all_targets: (N, P) ground truth sequences
        all_preds: (N, P) predicted sequences
        all_probs: (N, P, 3) probability sequences
        output_frames: P
        save_dir: directory per salvare i risultati
        prefix: prefisso per i filename (es. 'val_' o 'test_')
    """
    # LEVEL 1: Per-Timestep Metrics
    timestep_metrics = compute_per_timestep_metrics(all_targets, all_preds, output_frames)
    save_temporal_recall_curve(timestep_metrics, save_dir, f'{prefix}temporal_recall_curve.png')
    save_temporal_heatmap(all_targets, all_preds, output_frames, save_dir, f'{prefix}temporal_heatmap.png')
    collision_recall_start = timestep_metrics['recall_per_class_per_step'][2, 0]
    collision_recall_end = timestep_metrics['recall_per_class_per_step'][2, -1]
    # Diagnostics: interpret heatmap + threshold trade-offs
    rates = compute_per_timestep_class_rates(all_targets, all_preds, output_frames, num_classes=3)
    first_half = max(1, int(output_frames) // 2)
    recall = timestep_metrics['recall_per_class_per_step']
    recall_first = recall[:, :first_half].mean(axis=1)
    recall_second = recall[:, first_half:].mean(axis=1) if first_half < output_frames else recall_first
    pred_rate_mean_first = rates['pred_rate'][:, :first_half].mean(axis=1)
    true_rate_mean_first = rates['true_rate'][:, :first_half].mean(axis=1)
    # LEVEL 2: Event-Based Metrics
    mat_stats = compute_mean_anticipation_time(all_targets, all_preds)
    event_stats = compute_event_level_recall(all_targets, all_preds, min_consecutive=1)
    event_stats_3 = compute_event_level_recall(all_targets, all_preds, min_consecutive=3)
    # LEVEL 3: Stability Analysis
    flicker_stats = save_stability_analysis(all_preds, save_dir, f'{prefix}stability_analysis.png')
    # LEVEL 4: Visual Metrics
    save_forecast_plots(all_targets, all_preds, save_dir, num_samples=12, filename_prefix=f'{prefix}forecast')
    save_aggregated_forecast_plot(all_targets, all_preds, save_dir, f'{prefix}forecast_aggregated.png')
    # Summary
    save_metrics_summary(timestep_metrics, mat_stats, event_stats, flicker_stats, save_dir, f'{prefix}metrics_summary.png')
    # Save standard confusion matrix (aggregated)
    save_confusion_matrix(all_targets, all_preds, CLASS_NAMES, save_dir, f'{prefix}confusion_matrix.png')
    return {
        'timestep_metrics': timestep_metrics,
        'mat_stats': mat_stats,
        'event_stats': event_stats,
        'event_stats_3': event_stats_3,
        'flicker_stats': flicker_stats
    }


def _collect_targets_and_probs(args, model, loader, device, criterion=None, desc='', show_progress=True):

    """Collect (N,P) targets and (N,P,3) probabilities for a loader."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    with torch.no_grad():

        for inputs, output_risks in tqdm(loader, desc=desc, disable=not show_progress):

            inputs = inputs.to(device)
            targets = output_risks.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=2)
            if criterion is not None:
                loss = compute_forecasting_loss(criterion, outputs, targets)
                total_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    avg_loss = (total_loss / len(loader)) if (criterion is not None and len(loader) > 0) else None
    return avg_loss, all_targets, all_probs


def save_val_artifacts(args, model, val_loader, run_dir, device):

    """Save the same validation artifacts we used to produce (plots + summaries)."""
    best_model_path = os.path.join(run_dir, 'lstm_best.pth')
    if not os.path.exists(best_model_path):

        print(f"Warning: Best model not found at {best_model_path}; skipping val artifacts")
        return None
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    risk_weights = resolve_risk_weights(args)
    criterion = TemporalWeightedCrossEntropy(
        class_weights=risk_weights,
        output_frames=args.output_frames,
        urgent_frames=getattr(args, 'urgent_frames', URGENT_FRAMES),
        decay_factor=getattr(args, 'decay_factor', DECAY_FACTOR),
        device=device,
    )
    # Silent: this pass is only to save plots/metrics
    avg_val_loss, all_targets, all_probs = _collect_targets_and_probs(
        args, model, val_loader, device, criterion=criterion, desc='[Val]', show_progress=False
    )
    all_preds = apply_collision_threshold(all_probs, args.threshold)
    # Save all plots and summaries (silent by default)
    eval_pack = comprehensive_evaluation(
        all_targets,
        all_preds,
        all_probs,
        args.output_frames,
        run_dir,
        prefix='val_',
    )
    return {
        'val_loss': float(avg_val_loss) if avg_val_loss is not None else None,
        'val_accuracy': float(np.mean(all_preds == all_targets)),
        **eval_pack,
    }
# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def apply_collision_threshold(probs, threshold):

    """
    Applica threshold sulla probabilità di Collision.
    Args:
        probs: (N, P, 3) probabilità per ogni classe
        threshold: soglia per classe Collision
    Returns:
        (N, P) predizioni con threshold applicato
    """
    N, P, _ = probs.shape
    # Predizione di default: argmax su Safe e Near
    preds = np.argmax(probs[:, :, :2], axis=2)  # (N, P)
    # Override con Collision se prob >= threshold
    collision_probs = probs[:, :, 2]  # (N, P)
    preds[collision_probs >= threshold] = 2
    return preds


def train_lstm(args, model, train_loader, val_loader, run_dir, device):

    """Main training loop con weighted loss e threshold."""
    risk_weights = resolve_risk_weights(args)
    print(f"Using device: {device}")
    print(f"Collision threshold: {args.threshold}")
    print(f"Risk Weights: {risk_weights}")
    # Loss (always temporal-weighted)
    criterion = TemporalWeightedCrossEntropy(
        class_weights=risk_weights,
        output_frames=args.output_frames,
        urgent_frames=getattr(args, 'urgent_frames', URGENT_FRAMES),
        decay_factor=getattr(args, 'decay_factor', DECAY_FACTOR),
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_min_recall_first = -1.0
    best_mean_recall_first = -1.0
    best_epoch = 0
    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    # Best model data
    best_val_preds = None
    best_val_targets = None
    best_val_probs = None
    epochs_no_improve = 0
    try:
        for epoch in range(args.epochs):

            # --- TRAIN ---
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
            for inputs, output_risks in pbar:
                inputs = inputs.to(device)  # (B, T, 72)
                targets = output_risks.to(device)  # (B, P)
                optimizer.zero_grad()
                outputs = model(inputs)  # (B, P, 3)
                loss = compute_forecasting_loss(criterion, outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # Apply threshold per training accuracy
                probs = F.softmax(outputs, dim=2).detach().cpu().numpy()
                preds = apply_collision_threshold(probs, args.threshold)
                targets_np = targets.cpu().numpy()
                total += targets.numel()
                correct += np.sum(preds == targets_np)
                pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            avg_train_loss = train_loss / len(train_loader)
            train_acc = correct / total
            # --- VALIDATION ---
            model.eval()
            val_loss = 0.0
            val_targets_list = []
            val_probs_list = []
            with torch.no_grad():

                for inputs, output_risks in val_loader:
                    inputs = inputs.to(device)
                    targets = output_risks.to(device)
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=2)
                    loss = compute_forecasting_loss(criterion, outputs, targets)
                    val_loss += loss.item()
                    val_targets_list.append(targets.cpu().numpy())
                    val_probs_list.append(probs.cpu().numpy())
            # Concatenate all batches
            val_targets_array = np.concatenate(val_targets_list, axis=0)  # (N, P)
            val_probs_array = np.concatenate(val_probs_list, axis=0)  # (N, P, 3)
            # Apply threshold (or just argmax)
            val_preds_array = apply_collision_threshold(val_probs_array, args.threshold)
            # Calculate metrics
            avg_val_loss = val_loss / len(val_loader)
            val_acc = np.mean(val_preds_array == val_targets_array)
            k = int(getattr(args, 'eval_frames', EVAL_FRAMES))
            min_recall_first, per_class = min_class_recall_first_k(val_targets_array, val_preds_array, k)
            mean_recall_first = mean_class_recall_first_k(val_targets_array, val_preds_array, k)
            print(
                f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f} | "
                f"min-recall@{k}={min_recall_first:.2%} | mean-recall@{k}={mean_recall_first:.2%} | "
                f"(Safe={per_class['Safe']:.1%}, Near={per_class['Near-Collision']:.1%}, Coll={per_class['Collision']:.1%})"
            )
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            # Hierarchical selection: min-recall@k > mean-recall@k > loss
            save_model = False
            if min_recall_first > best_min_recall_first:
                save_model = True
            elif min_recall_first == best_min_recall_first:
                if mean_recall_first > best_mean_recall_first:
                    save_model = True
                elif mean_recall_first == best_mean_recall_first:
                    if avg_val_loss < best_val_loss:
                        save_model = True

            if save_model:
                best_val_loss = avg_val_loss
                best_min_recall_first = float(min_recall_first)
                best_mean_recall_first = float(mean_recall_first)
                best_val_preds = val_preds_array.copy()
                best_val_targets = val_targets_array.copy()
                best_val_probs = val_probs_array.copy()
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(run_dir, 'lstm_best.pth'))
                print("Saved best model")

                _write_tuning_metrics(
                    run_dir,
                    {
                        'best_epoch': int(best_epoch),
                        'best_val_loss': float(best_val_loss),
                        'best_min_recall': float(best_min_recall_first),
                        'best_mean_recall': float(best_mean_recall_first),
                        'eval_frames': int(k),
                        'threshold': float(args.threshold),
                        'risk_weights': list(resolve_risk_weights(args)),
                        'augment_factor': int(getattr(args, 'augment_factor', 0)),
                    }
                )
            else:
                epochs_no_improve += 1
                # Early stopping
                if epochs_no_improve >= args.patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs (no improvement for {args.patience} epochs)")
                    break

            # Optional success criterion for autotuning
            if getattr(args, 'stop_on_target', False):
                if float(min_recall_first) >= 0.80 and float(mean_recall_first) >= 0.80:
                    print(f"\nTarget reached on validation (@{k}): min>=0.80 and mean>=0.80. Stopping early.")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Proceeding with best checkpoint collected so far...")
    # --- SAVE TRAINING VISUALIZATIONS ---
    if not getattr(args, 'skip_artifacts', False):
        save_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, run_dir)
    print(f"\nTraining complete. Best model at epoch {best_epoch}")
    return best_epoch


def test_lstm(args, model, test_loader, run_dir, device):

    """Test phase."""
    # Load best model
    best_model_path = os.path.join(run_dir, 'lstm_best.pth')
    if not os.path.exists(best_model_path):

        print(f"Error: Best model not found at {best_model_path}")
        return
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    # Loss (match training): always temporal-weighted
    risk_weights = resolve_risk_weights(args)
    criterion = TemporalWeightedCrossEntropy(
        class_weights=risk_weights,
        output_frames=args.output_frames,
        urgent_frames=getattr(args, 'urgent_frames', URGENT_FRAMES),
        decay_factor=getattr(args, 'decay_factor', DECAY_FACTOR),
        device=device,
    )
    avg_test_loss, all_targets, all_probs = _collect_targets_and_probs(
        args, model, test_loader, device, criterion=criterion, desc='[Test]', show_progress=True
    )
    # Apply threshold
    all_preds = apply_collision_threshold(all_probs, args.threshold)
    # Calculate basic metrics
    test_acc = np.mean(all_preds == all_targets)
    k = int(getattr(args, 'eval_frames', EVAL_FRAMES))
    min_first, per_class = min_class_recall_first_k(all_targets, all_preds, k)
    mean_first = mean_class_recall_first_k(all_targets, all_preds, k)
    print(
        f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.4f} | "
        f"min-recall@{k}: {min_first:.2%} | mean-recall@{k}: {mean_first:.2%}"
    )
    # Save all plots and summaries (silent by default)
    eval_pack = comprehensive_evaluation(
        all_targets,
        all_preds,
        all_probs,
        args.output_frames,
        run_dir,
        prefix='test_',
    )
    results = {
        'test_loss': float(avg_test_loss) if avg_test_loss is not None else None,
        'test_accuracy': float(test_acc),
        'min_recall_first_k': float(min_first),
        'mean_recall_first_k': float(mean_first),
        'per_class_first_k': per_class,
        **eval_pack,
    }
    save_final_report(args, results, run_dir)
    return results


def save_final_report(args, results, save_dir):

    """Salva un report testuale finale."""
    report_path = os.path.join(save_dir, 'final_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LSTM FORECASTING - FINAL REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"  Input Frames: {args.input_frames}\n")
        f.write(f"  Output Frames: {args.output_frames}\n")
        f.write(f"  Collision Threshold: {args.threshold}\n")
        f.write(f"  Risk Weights: {resolve_risk_weights(args)}\n")
        f.write(
            f"  Temporal Loss: ENABLED (urgent_frames={getattr(args, 'urgent_frames', URGENT_FRAMES)}, "
            f"decay_factor={getattr(args, 'decay_factor', DECAY_FACTOR)})\n"
        )
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write("  Weighted Sampler: True\n")
        f.write(f"  Augmentation: {args.augment_factor}\n")
        f.write(f"  Strides: train={TRAIN_STRIDE}, val={VAL_STRIDE}, test={TEST_STRIDE}\n\n")
        f.write("RESULTS:\n")
        if results.get('test_loss') is not None:
            f.write(f"  Test Loss: {results['test_loss']:.4f}\n")
        f.write(f"  Test Accuracy: {results['test_accuracy']:.4f}\n")
        k = int(getattr(args, 'eval_frames', EVAL_FRAMES))
        if results.get('min_recall_first_k') is not None:
            f.write(f"  Min Recall (first {k}): {results['min_recall_first_k']:.2%}\n")
        if results.get('mean_recall_first_k') is not None:
            f.write(f"  Mean Recall (first {k}): {results['mean_recall_first_k']:.2%}\n")
        pc = results.get('per_class_first_k', {})
        if pc:
            f.write(
                f"  Recall (first {k}): "
                f"Safe={pc.get('Safe', 0.0):.2%}, "
                f"Near-Collision={pc.get('Near-Collision', 0.0):.2%}, "
                f"Collision={pc.get('Collision', 0.0):.2%}\n"
            )
        mat_stats = results.get('mat_stats')
        if isinstance(mat_stats, dict):

            f.write("\nEVENT/ANTICIPATION (from saved evaluation):\n")
            if 'mat_frames' in mat_stats:
                f.write(f"  MAT Frames: {mat_stats['mat_frames']:.2f}\n")
            if 'mat_seconds' in mat_stats:
                f.write(f"  MAT Seconds: {mat_stats['mat_seconds']:.3f}\n")
            if 'anticipation_rate' in mat_stats:
                f.write(f"  Anticipation Rate: {mat_stats['anticipation_rate']:.2%}\n")
            if 'detection_rate' in mat_stats:
                f.write(f"  Detection Rate: {mat_stats['detection_rate']:.2%}\n")
        event_stats = results.get('event_stats')
        if isinstance(event_stats, dict):

            f.write("\nEVENT RECALL (>=1 frame):\n")
            if 'event_recall' in event_stats:
                f.write(f"  Event Recall: {event_stats['event_recall']:.2%}\n")
            if 'detected_sequences' in event_stats and 'total_collision_sequences' in event_stats:
                f.write(
                    f"  Detected/Total Collision Seqs: {event_stats['detected_sequences']}/{event_stats['total_collision_sequences']}\n"
                )
        event_stats_3 = results.get('event_stats_3')
        if isinstance(event_stats_3, dict):

            f.write("\nEVENT RECALL (>=3 consecutive frames):\n")
            if 'event_recall' in event_stats_3:
                f.write(f"  Event Recall: {event_stats_3['event_recall']:.2%}\n")
        flicker_stats = results.get('flicker_stats')
        if isinstance(flicker_stats, dict):

            f.write("\nSTABILITY (from saved evaluation):\n")
            if 'mean_flicker' in flicker_stats:
                f.write(f"  Mean Flicker: {flicker_stats['mean_flicker']:.3f}\n")
            if 'max_flicker' in flicker_stats:
                f.write(f"  Max Flicker: {flicker_stats['max_flicker']}\n")
    print(f"\nFinal report saved to: {report_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'c:\Users\Proprietario\Desktop\human-robot-collaboration\datasets\3d_skeletons_risk')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--threshold', type=float, default=0.12, help='Collision probability threshold (lower = more sensitive)')
    parser.add_argument('--risk_weights', type=float, nargs=3, default=RISK_WEIGHTS, help='Loss weights for (Safe Near Collision), e.g. --risk_weights 1 4 10')
    parser.add_argument('--augment_factor', type=int, default=10, help='0 disables augmentation; otherwise number of augmented copies per collision sequence')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run folder name under runs/')
    parser.add_argument('--skip_test', action='store_true', help='Skip test evaluation (useful for autotuning)')
    parser.add_argument('--skip_artifacts', action='store_true', help='Skip saving plots/artifacts (useful for autotuning)')
    parser.add_argument('--stop_on_target', action='store_true', help='Stop early if val min+mean recall@15 reach 0.80/0.80')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Fixed settings (kept out of CLI)
    args.batch_size = BATCH_SIZE
    args.input_frames = INPUT_FRAMES
    args.output_frames = OUTPUT_FRAMES
    args.eval_frames = EVAL_FRAMES
    args.sampler_strategy = SAMPLER_STRATEGY
    args.sampler_collision_scale = SAMPLER_COLLISION_SCALE
    args.sampler_collision_power = SAMPLER_COLLISION_POWER
    args.sampler_min_weight = SAMPLER_MIN_WEIGHT
    args.urgent_frames = URGENT_FRAMES
    args.decay_factor = DECAY_FACTOR
    
    # Reproducibility
    seed = int(getattr(args, 'seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older torch without warn_only
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    
    # Setup run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = args.run_name.strip() if isinstance(args.run_name, str) and args.run_name.strip() else f'lstm_{timestamp}'
    run_dir = os.path.join('runs', run_id)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    train_loader, val_loader, test_loader, stats = create_pkl_dataloaders(
        dataset_path=args.data_path,
        train_subjects=TRAIN_SUBJECTS,
        val_subjects=VAL_SUBJECTS,
        test_subjects=TEST_SUBJECTS,
        input_frames=INPUT_FRAMES,
        output_frames=OUTPUT_FRAMES,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        mode='sequence',
        use_weighted_sampler=True,
        augment_collision=bool(int(args.augment_factor) > 0),
        augment_factor=int(args.augment_factor),
        sampler_strategy=args.sampler_strategy,
        sampler_collision_scale=args.sampler_collision_scale,
        sampler_collision_power=args.sampler_collision_power,
        sampler_min_weight=args.sampler_min_weight,
        seed=seed,
        train_stride=TRAIN_STRIDE,
        val_stride=VAL_STRIDE,
        test_stride=TEST_STRIDE
    )
    
    # Create model
    model = RiskLSTM(
        input_size=72,
        hidden_size=512,
        num_layers=2,
        num_classes=3,
        output_frames=args.output_frames
    ).to(device)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train and Test
    train_lstm(args, model, train_loader, val_loader, run_dir, device)
    if not getattr(args, 'skip_artifacts', False):
        save_val_artifacts(args, model, val_loader, run_dir, device)
    if not getattr(args, 'skip_test', False):
        test_lstm(args, model, test_loader, run_dir, device)
    print(f"Results saved to: {run_dir}")