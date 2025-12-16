"""
Threshold Tuning Script - Analizza come diverse soglie influenzano le metriche
"""
import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

from models.mlp import RiskMLP
from utils.pkl_data_loader import create_pkl_dataloaders

# Dataset Split Configuration
TEST_SUBJECTS = ['S02', 'S03', 'S18', 'S19']
CLASS_NAMES = ['Safe', 'Near-Collision', 'Collision']


def load_model(model_path, device):
    """Carica il modello salvato"""
    model = RiskMLP(input_size=72, hidden_sizes=[128, 64], num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_predictions(model, test_loader, device):
    """Ottiene le probabilità dal modello"""
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Computing predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_targets), np.array(all_probs)


def apply_threshold(probs, threshold):
    """Applica la soglia per la classe Collision"""
    collision_probs = probs[:, 2]
    preds = np.argmax(probs[:, :2], axis=1)  # argmax tra Safe e Near-Collision
    preds[collision_probs >= threshold] = 2  # Se P(Collision) >= threshold -> Collision
    return preds


def compute_metrics(targets, preds):
    """Calcola precision, recall, f1 per ogni classe"""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=[0, 1, 2], zero_division=0
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }


def plot_metrics_vs_threshold(thresholds, all_metrics, output_dir):
    """Grafico che mostra come cambiano le metriche al variare della soglia"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_names = ['precision', 'recall', 'f1']
    titles = ['Precision vs Threshold', 'Recall vs Threshold', 'F1-Score vs Threshold']
    
    for ax, metric_name, title in zip(axes, metrics_names, titles):
        for i, class_name in enumerate(CLASS_NAMES):
            values = [m[metric_name][i] for m in all_metrics]
            ax.plot(thresholds, values, marker='o', label=class_name, linewidth=2)
        
        ax.set_xlabel('Collision Threshold', fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([thresholds[0], thresholds[-1]])
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_threshold.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_collision_metrics_detail(thresholds, all_metrics, output_dir):
    """Grafico dettagliato per la classe Collision (la più importante)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    precision_collision = [m['precision'][2] for m in all_metrics]
    recall_collision = [m['recall'][2] for m in all_metrics]
    f1_collision = [m['f1'][2] for m in all_metrics]
    
    ax.plot(thresholds, precision_collision, 'b-o', label='Precision', linewidth=2, markersize=8)
    ax.plot(thresholds, recall_collision, 'r-s', label='Recall', linewidth=2, markersize=8)
    ax.plot(thresholds, f1_collision, 'g-^', label='F1-Score', linewidth=2, markersize=8)
    
    ax.set_xlabel('Collision Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Collision Class Metrics vs Threshold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([thresholds[0], thresholds[-1]])
    ax.set_ylim([0, 1.05])
    
    # Trova il threshold ottimale (max F1 per Collision)
    best_idx = np.argmax(f1_collision)
    best_threshold = thresholds[best_idx]
    ax.axvline(x=best_threshold, color='purple', linestyle='--', alpha=0.7, 
               label=f'Best F1 @ {best_threshold:.2f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collision_metrics_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_threshold


def plot_confusion_matrices_grid(targets, probs, thresholds_subset, output_dir):
    """Griglia di confusion matrices per diverse soglie"""
    n_thresholds = len(thresholds_subset)
    n_cols = 5
    n_rows = (n_thresholds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, threshold in enumerate(thresholds_subset):
        preds = apply_threshold(probs, threshold)
        cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        axes[idx].set_title(f'Threshold = {threshold:.2f}', fontsize=10)
        axes[idx].set_xlabel('Predicted', fontsize=8)
        axes[idx].set_ylabel('True', fontsize=8)
    
    # Nascondi assi extra se dispari
    for idx in range(len(thresholds_subset), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_tradeoff(thresholds, all_metrics, output_dir):
    """Grafico precision-recall tradeoff per Collision"""
    precision_collision = [m['precision'][2] for m in all_metrics]
    recall_collision = [m['recall'][2] for m in all_metrics]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(recall_collision, precision_collision, c=thresholds, 
                         cmap='viridis', s=100, edgecolors='black')
    
    # Connetti i punti con una linea
    ax.plot(recall_collision, precision_collision, 'k--', alpha=0.3)
    
    # Annota alcuni punti chiave
    for i, t in enumerate(thresholds):
        if t in [0.95, 0.96, 0.97, 0.98, 0.99]:
            ax.annotate(f'{t:.1f}', (recall_collision[i], precision_collision[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold', fontsize=11)
    
    ax.set_xlabel('Recall (Collision)', fontsize=12)
    ax.set_ylabel('Precision (Collision)', fontsize=12)
    ax.set_title('Precision-Recall Tradeoff for Collision Class', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_tradeoff.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_summary_report(thresholds, all_metrics, best_threshold, output_dir):
    """Salva un report testuale con i risultati"""
    report_path = os.path.join(output_dir, 'threshold_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("THRESHOLD TUNING ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Best threshold for Collision F1: {best_threshold:.2f}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("COLLISION CLASS METRICS BY THRESHOLD\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 60 + "\n")
        
        for t, m in zip(thresholds, all_metrics):
            f.write(f"{t:<12.2f} {m['precision'][2]:<12.4f} {m['recall'][2]:<12.4f} {m['f1'][2]:<12.4f}\n")
        
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n")
        f.write("- Lower threshold: More Collision predictions (higher recall, lower precision)\n")
        f.write("- Higher threshold: Fewer Collision predictions (lower recall, higher precision)\n")
        f.write("- For safety-critical applications: prefer lower threshold (catch more collisions)\n")


def main():
    parser = argparse.ArgumentParser(description="Threshold Tuning Analysis")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--output_dir', type=str, default='threshold_analysis', help="Output directory")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, device)
    
    # Load test data (use TEST_SUBJECTS as train to compute normalization stats)
    print("Loading test data...")
    _, _, test_loader, _ = create_pkl_dataloaders(
        dataset_path=args.data_path,
        train_subjects=TEST_SUBJECTS,
        val_subjects=[],
        test_subjects=TEST_SUBJECTS,
        input_frames=1,
        output_frames=0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode='single_frame',
        use_weighted_sampler=False,
        augment_collision=False
    )
    
    # Get predictions
    targets, probs = get_predictions(model, test_loader, device)
    print(f"Total samples: {len(targets)}")
    
    # Test different thresholds
    thresholds = np.arange(0.80, 1.0, 0.01)
    print(f"\nTesting {len(thresholds)} thresholds from {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
    
    all_metrics = []
    for t in tqdm(thresholds, desc="Evaluating thresholds"):
        preds = apply_threshold(probs, t)
        metrics = compute_metrics(targets, preds)
        all_metrics.append(metrics)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_metrics_vs_threshold(thresholds, all_metrics, args.output_dir)
    print("  - metrics_vs_threshold.png")
    
    best_threshold = plot_collision_metrics_detail(thresholds, all_metrics, args.output_dir)
    print("  - collision_metrics_detail.png")
    
    # Confusion matrices for all thresholds from 0.80 to 0.99
    thresholds_subset = list(np.arange(0.80, 1.0, 0.01))
    plot_confusion_matrices_grid(targets, probs, thresholds_subset, args.output_dir)
    print("  - confusion_matrices_grid.png")
    
    plot_precision_recall_tradeoff(thresholds, all_metrics, args.output_dir)
    print("  - precision_recall_tradeoff.png")
    
    save_summary_report(thresholds, all_metrics, best_threshold, args.output_dir)
    print("  - threshold_analysis_report.txt")
    
    print(f"\n{'='*50}")
    print(f"Best threshold for Collision F1: {best_threshold:.2f}")
    best_idx = list(thresholds).index(best_threshold) if best_threshold in thresholds else np.argmin(np.abs(thresholds - best_threshold))
    print(f"  Precision: {all_metrics[best_idx]['precision'][2]:.4f}")
    print(f"  Recall: {all_metrics[best_idx]['recall'][2]:.4f}")
    print(f"  F1-Score: {all_metrics[best_idx]['f1'][2]:.4f}")
    print(f"{'='*50}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
