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


def save_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Salva le curve di loss e accuracy per training e validation."""
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
    """Salva la matrice di confusione come heatmap."""
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
    """Salva le curve Precision-Recall per ogni classe."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Verde, Arancione, Rosso
    
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
    """Salva il report delle metriche come immagine PNG."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    
    # Prepara i dati per la tabella
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
    
    # Colora l'header
    for j in range(len(metrics)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Colora le righe delle classi
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
    """Salva un grafico a barre con precision, recall, f1 per ogni classe."""
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
    
    # Aggiungi valori sopra le barre
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
