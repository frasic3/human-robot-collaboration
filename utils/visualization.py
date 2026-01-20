"""
Visualization utilities for training metrics and model evaluation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_curve, 
    average_precision_score, 
    classification_report,
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.preprocessing import label_binarize


# -----------------------------------------------------------------------------
# LSTM / Forecasting utilities
# -----------------------------------------------------------------------------


DEFAULT_CLASS_NAMES = ['Safe', 'Near-Collision', 'Collision']


def _safe_bincount(arr: np.ndarray, minlength: int = 3) -> np.ndarray:
    arr = np.asarray(arr).astype(np.int64).reshape(-1)
    return np.bincount(arr, minlength=minlength)


def compute_per_timestep_metrics(all_targets, all_preds, output_frames, num_classes=3):

    """Compute metrics for each future timestep.

    Args:
        all_targets: (N, P) array of ground truth per sample
        all_preds: (N, P) array of predictions per sample
        output_frames: P - number of future frames
        num_classes: number of classes
    Returns:
        dict with:
        - accuracy_per_step: (P,) accuracy per timestep
        - recall_per_class_per_step: (num_classes, P) recall per class/timestep
        - precision_per_class_per_step: (num_classes, P) precision per class/timestep
    """
    N = all_targets.shape[0]
    P = output_frames
    accuracy_per_step = np.zeros(P)
    recall_per_class_per_step = np.zeros((num_classes, P))
    precision_per_class_per_step = np.zeros((num_classes, P))
    for t in range(P):

        targets_t = all_targets[:, t]
        preds_t = all_preds[:, t]
        accuracy_per_step[t] = np.mean(targets_t == preds_t)
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


def save_temporal_recall_curve(metrics, save_dir, filename='temporal_recall_curve.png', class_names=None):

    """Temporal recall curve for each class."""
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    P = metrics['recall_per_class_per_step'].shape[1]
    timesteps = np.arange(1, P + 1)
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for c, (class_name, color) in enumerate(zip(class_names, colors)):

        recall = metrics['recall_per_class_per_step'][c]
        plt.plot(timesteps, recall, '-o', color=color, label=f'{class_name}',
                 linewidth=2, markersize=4)
    plt.xlabel('Temporal Horizon (Future Frame)', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Temporal Recall Curve - Prediction Degradation', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xlim(1, P)
    collision_recall = metrics['recall_per_class_per_step'][2]
    plt.annotate(f'{collision_recall[0]:.2f}', (1, collision_recall[0]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.annotate(f'{collision_recall[-1]:.2f}', (P, collision_recall[-1]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()


def save_temporal_heatmap(all_targets, all_preds, output_frames, save_dir, filename='temporal_heatmap.png', class_names=None):

    """Temporal heatmap of predictions (recall per class and timestep)."""
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    P = output_frames
    num_classes = 3
    heatmap_data = np.zeros((num_classes, P))
    for t in range(P):

        for c in range(num_classes):

            mask = all_targets[:, t] == c
            if np.sum(mask) > 0:
                heatmap_data[c, t] = np.mean(all_preds[:, t][mask] == c)
    plt.figure(figsize=(14, 5))
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                     xticklabels=[f'+{i+1}' for i in range(P)],
                     yticklabels=class_names,
                     vmin=0, vmax=1,
                     annot_kws={"size": 8})
    plt.xlabel('Future Frame', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('Temporal Heatmap: Recall by Class and Horizon', fontsize=14)
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


def compute_mean_anticipation_time(all_targets, all_preds, fps=25):

    """Mean Anticipation Time (MAT) - Tempo di Preavviso Medio."""
    N, P = all_targets.shape
    anticipation_times = []
    for i in range(N):

        target_seq = all_targets[i]
        pred_seq = all_preds[i]
        collision_indices = np.where(target_seq == 2)[0]
        if len(collision_indices) == 0:
            continue
        first_collision = collision_indices[0]
        pred_collision_before = np.where((pred_seq == 2) & (np.arange(P) < first_collision))[0]
        if len(pred_collision_before) > 0:
            first_pred = pred_collision_before[0]
            anticipation = first_collision - first_pred
            anticipation_times.append(anticipation)
        else:
            if pred_seq[first_collision] == 2:
                anticipation_times.append(0)
            else:
                anticipation_times.append(-1)
    if len(anticipation_times) == 0:
        return {'mat_frames': 0, 'mat_seconds': 0, 'anticipation_rate': 0}
    valid_anticipations = [a for a in anticipation_times if a >= 0]
    if len(valid_anticipations) == 0:
        mat_frames = 0
    else:
        mat_frames = np.mean(valid_anticipations)
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

    """Event-Level Recall (Recall per Episodio)."""
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
        pred_collisions = (pred_seq == 2)
        if min_consecutive == 1:
            if np.any(pred_collisions):

                detected_sequences += 1
        else:
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


def compute_flicker_rate(all_preds):

    """Flicker rate."""
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

    """Show prediction stability analysis."""
    flicker_stats = compute_flicker_rate(all_preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(flicker_stats['flicker_histogram'], bins=20, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0].axvline(flicker_stats['mean_flicker'], color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {flicker_stats['mean_flicker']:.2f}")
    axes[0].set_xlabel('Number of Class Changes', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Flicker Rate Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].boxplot(flicker_stats['flicker_histogram'], vert=True)
    axes[1].set_ylabel('Number of Class Changes', fontsize=12)
    axes[1].set_title('Flicker Rate Box Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    stats_text = f"Mean: {flicker_stats['mean_flicker']:.2f}\n"
    stats_text += f"Std: {flicker_stats['std_flicker']:.2f}\n"
    stats_text += f"Min: {flicker_stats['min_flicker']}\n"
    stats_text += f"Max: {flicker_stats['max_flicker']}"
    axes[1].text(1.15, 0.5, stats_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()
    return flicker_stats


def save_forecast_plots(all_targets, all_preds, save_dir, num_samples=10, filename_prefix='forecast'):

    """Create plots comparing ground truth and predictions."""
    N, P = all_targets.shape
    collision_indices = np.where(np.any(all_targets == 2, axis=1))[0]
    if len(collision_indices) < num_samples:
        sample_indices = collision_indices
    else:
        sample_indices = np.random.choice(collision_indices, num_samples, replace=False)
    non_collision_indices = np.where(~np.any(all_targets == 2, axis=1))[0]
    if len(non_collision_indices) > 0:
        extra = min(3, len(non_collision_indices))
        sample_indices = np.concatenate([sample_indices,
                                         np.random.choice(non_collision_indices, extra, replace=False)])
    num_plots = len(sample_indices)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    timesteps = np.arange(1, P + 1)
    for ax, sample_idx in zip(axes, sample_indices):

        gt = all_targets[sample_idx]
        pred = all_preds[sample_idx]
        ax.plot(timesteps, gt, 'k-', linewidth=3, label='Ground Truth', alpha=0.7)
        ax.scatter(timesteps, gt, c='black', s=30, zorder=5)
        colors = ['green' if g == p else 'red' for g, p in zip(gt, pred)]
        ax.plot(timesteps, pred, '--', color='blue', linewidth=2, label='Prediction', alpha=0.7)
        ax.scatter(timesteps, pred, c=colors, s=30, zorder=5, edgecolors='blue')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Safe', 'Near', 'Coll'])
        ax.set_xlabel('Future Frame')
        ax.set_ylabel('Class')
        ax.set_title(f'Sample {sample_idx}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, P + 1)
        ax.set_ylim(-0.2, 2.5)
    for ax in axes[len(sample_indices):]:
        ax.set_visible(False)
    plt.suptitle('Forecast Plots: Ground Truth vs Prediction', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename_prefix}_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_aggregated_forecast_plot(all_targets, all_preds, save_dir, filename='forecast_aggregated.png', class_names=None):

    """Aggregated plot showing prediction distribution over time."""
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    N, P = all_targets.shape
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    timesteps = np.arange(1, P + 1)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    gt_dist = np.zeros((3, P))
    for t in range(P):

        for c in range(3):

            gt_dist[c, t] = np.mean(all_targets[:, t] == c)
    axes[0].stackplot(timesteps, gt_dist[0], gt_dist[1], gt_dist[2],
                      labels=class_names, colors=colors, alpha=0.8)
    axes[0].set_xlabel('Future Frame', fontsize=12)
    axes[0].set_ylabel('Proportion', fontsize=12)
    axes[0].set_title('Ground Truth Distribution Over Time', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].set_xlim(1, P)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    pred_dist = np.zeros((3, P))
    for t in range(P):

        for c in range(3):

            pred_dist[c, t] = np.mean(all_preds[:, t] == c)
    axes[1].stackplot(timesteps, pred_dist[0], pred_dist[1], pred_dist[2],
                      labels=class_names, colors=colors, alpha=0.8)
    axes[1].set_xlabel('Future Frame', fontsize=12)
    axes[1].set_ylabel('Proportion', fontsize=12)
    axes[1].set_title('Prediction Distribution Over Time', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(1, P)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()


def save_confusion_matrix_forecasting(y_true, y_pred, class_names, save_dir, filename='confusion_matrix.png'):

    """Save the confusion matrix aggregated over all timesteps (forecasting)."""
    cm = confusion_matrix(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.title('Confusion Matrix (Aggregated)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()


def save_metrics_summary(
    metrics_dict,
    mat_stats,
    event_stats,
    flicker_stats,
    save_dir,
    filename='metrics_summary.png',
    title: str | None = None,
    header: str | None = None,
):

    """Save a visual summary of all metrics (forecasting)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if title:
        fig.suptitle(str(title), fontsize=16, fontweight='bold')
    ax1 = axes[0, 0]
    P = metrics_dict['recall_per_class_per_step'].shape[1]
    timesteps = np.arange(1, P + 1)
    collision_recall = metrics_dict['recall_per_class_per_step'][2]
    ax1.fill_between(timesteps, collision_recall, alpha=0.3, color='red')
    ax1.plot(timesteps, collision_recall, 'r-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Future Frame')
    ax1.set_ylabel('Collision Recall')
    ax1.set_title('Collision Recall Degradation')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target 80%')
    ax1.legend()
    ax2 = axes[0, 1]
    categories = ['Anticipate', 'On Time', 'Missed']
    values = [mat_stats['anticipated_count'], mat_stats['detected_at_moment'], mat_stats['missed_count']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax2.bar(categories, values, color=colors, edgecolor='black')
    ax2.set_ylabel('Number of Collisions')
    ax2.set_title(
        f"Collision Anticipation (MAT: {mat_stats['mat_frames']:.1f} frames = {mat_stats['mat_seconds']:.2f}s)"
    )
    for bar, val in zip(bars, values):

        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val),
                 ha='center', va='bottom', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax3 = axes[1, 0]
    event_data = [event_stats['event_recall'], 1 - event_stats['event_recall']]
    labels = [
        f"Detected\n({event_stats['detected_sequences']})",
        f"Missed\n({event_stats['total_collision_sequences'] - event_stats['detected_sequences']})",
    ]
    colors = ['#2ecc71', '#e74c3c']
    ax3.pie(event_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
            explode=(0.05, 0))
    ax3.set_title(f"Event-Level Recall: {event_stats['event_recall']:.1%}")
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = "=" * 40 + "\n"
    summary_text += (str(header) if header else "FORECASTING METRICS SUMMARY") + "\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += "PER-TIMESTEP:\n"
    summary_text += f"  • Recall Collision @+1:  {collision_recall[0]:.2%}\n"
    summary_text += f"  • Recall Collision @+{P}: {collision_recall[-1]:.2%}\n"
    summary_text += f"  • Mean Recall:           {np.mean(collision_recall):.2%}\n\n"
    summary_text += "ANTICIPATION:\n"
    summary_text += f"  • MAT: {mat_stats['mat_frames']:.1f} frames ({mat_stats['mat_seconds']:.2f}s)\n"
    summary_text += f"  • Anticipation Rate: {mat_stats['anticipation_rate']:.1%}\n"
    summary_text += f"  • Detection Rate: {mat_stats['detection_rate']:.1%}\n\n"
    summary_text += "EVENT-LEVEL:\n"
    summary_text += f"  • Event Recall: {event_stats['event_recall']:.1%}\n"
    summary_text += f"  • Detected Sequences: {event_stats['detected_sequences']}/{event_stats['total_collision_sequences']}\n\n"
    summary_text += "STABILITY:\n"
    summary_text += f"  • Mean Flicker: {flicker_stats['mean_flicker']:.2f} changes/seq\n"
    summary_text += f"  • Max Flicker: {flicker_stats['max_flicker']} changes/seq\n"
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Leave room for suptitle if present
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()


def comprehensive_evaluation(all_targets, all_preds, all_probs, output_frames, save_dir, prefix='', class_names=None):

    """Run a full forecasting evaluation and save all plots."""
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    timestep_metrics = compute_per_timestep_metrics(all_targets, all_preds, output_frames)
    save_temporal_recall_curve(timestep_metrics, save_dir, f'{prefix}temporal_recall_curve.png', class_names=class_names)
    save_temporal_heatmap(all_targets, all_preds, output_frames, save_dir, f'{prefix}temporal_heatmap.png', class_names=class_names)

    rates = compute_per_timestep_class_rates(all_targets, all_preds, output_frames, num_classes=3)
    first_half = max(1, int(output_frames) // 2)
    recall = timestep_metrics['recall_per_class_per_step']
    recall_first = recall[:, :first_half].mean(axis=1)
    recall_second = recall[:, first_half:].mean(axis=1) if first_half < output_frames else recall_first
    pred_rate_mean_first = rates['pred_rate'][:, :first_half].mean(axis=1)
    true_rate_mean_first = rates['true_rate'][:, :first_half].mean(axis=1)

    mat_stats = compute_mean_anticipation_time(all_targets, all_preds)
    event_stats = compute_event_level_recall(all_targets, all_preds, min_consecutive=1)
    event_stats_3 = compute_event_level_recall(all_targets, all_preds, min_consecutive=3)
    flicker_stats = save_stability_analysis(all_preds, save_dir, f'{prefix}stability_analysis.png')
    save_forecast_plots(all_targets, all_preds, save_dir, num_samples=12, filename_prefix=f'{prefix}forecast')
    save_aggregated_forecast_plot(all_targets, all_preds, save_dir, f'{prefix}forecast_aggregated.png', class_names=class_names)
    save_metrics_summary(timestep_metrics, mat_stats, event_stats, flicker_stats, save_dir, f'{prefix}metrics_summary.png')
    save_confusion_matrix_forecasting(all_targets, all_preds, class_names, save_dir, f'{prefix}confusion_matrix.png')

    # Keep the same diagnostic fields returned by train_lstm.py
    collision_recall_start = float(timestep_metrics['recall_per_class_per_step'][2, 0])
    collision_recall_end = float(timestep_metrics['recall_per_class_per_step'][2, -1])

    return {
        'timestep_metrics': timestep_metrics,
        'mat_stats': mat_stats,
        'event_stats': event_stats,
        'event_stats_3': event_stats_3,
        'flicker_stats': flicker_stats,
        'diagnostics': {
            'collision_recall_start': collision_recall_start,
            'collision_recall_end': collision_recall_end,
            'recall_first_half_mean': recall_first.tolist(),
            'recall_second_half_mean': recall_second.tolist() if isinstance(recall_second, np.ndarray) else recall_first.tolist(),
            'pred_rate_first_half_mean': pred_rate_mean_first.tolist(),
            'true_rate_first_half_mean': true_rate_mean_first.tolist(),
        }
    }


def save_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Save loss and accuracy curves for training and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(train_losses, 'b-o', label='Train Loss', markersize=3)
    axes[0].plot(val_losses, 'r-o', label='Val Loss', markersize=3)
    axes[0].set_title('Loss Curves', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
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


def save_confusion_matrix(y_true, y_pred, class_names, save_dir, filename='confusion_matrix.png', info_text=None):
    """Save the confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    
    title = 'Confusion Matrix'
    if info_text:
        title += f'\n{info_text}'
    plt.title(title, fontsize=14)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def save_pr_curves(y_true, y_probs, class_names, save_dir, filename='pr_curves.png', info_text=None):
    """Save Precision-Recall curves for each class."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, orange, red
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, lw=2, color=colors[i],
                 label=f'{class_names[i]} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    
    title = 'Precision-Recall Curves'
    if info_text:
        title += f'\n{info_text}'
    plt.title(title, fontsize=14)
    
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_report(y_true, y_pred, class_names, save_dir, filename='metrics_report.png', info_text=None):
    """Save the metrics report as a PNG image."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    
    # Prepare table data
    metrics = ['precision', 'recall', 'f1-score', 'support']
    rows = class_names + ['accuracy', 'macro avg', 'weighted avg']
    
    cell_text = []
    for row in rows:
        if row == 'accuracy':
            cell_text.append(['', '', f"{report['accuracy']:.4f}", f"{int(report['weighted avg']['support'])}"])
        else:
            cell_text.append([f"{report[row]['precision']:.4f}", 
                              f"{report[row]['recall']:.4f}", 
                              f"{report[row]['f1-score']:.4f}", 
                              f"{int(report[row]['support'])}"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    table = ax.table(cellText=cell_text,
                     rowLabels=rows,
                     colLabels=metrics,
                     cellLoc='center',
                     rowLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color the header
    for j in range(len(metrics)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color class rows
    colors = ['#E2EFDA', '#FFF2CC', '#FCE4D6']
    for i, row in enumerate(rows):
        if i < len(class_names):
            for j in range(-1, len(metrics)):
                table[(i+1, j)].set_facecolor(colors[i % len(colors)])
    
    title = 'Classification Report'
    if info_text:
        title += f'\n{info_text}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def save_per_class_metrics(y_true, y_pred, class_names, save_dir, filename='per_class_metrics.png'):
    """Save a bar chart with precision, recall, and f1 per class."""
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values above bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()
