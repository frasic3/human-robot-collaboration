import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from models.lstm import RiskLSTM
from utils.pkl_data_loader import create_pkl_dataloaders

# Dataset Split Configuration
TRAIN_SUBJECTS = [
    'S01', 'S05', 'S06', 'S07', 'S08', 'S09', 
    'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17'
]
VAL_SUBJECTS = ['S00', 'S04']
TEST_SUBJECTS = ['S02', 'S03', 'S18', 'S19']

# Risk Configuration
RISK_WEIGHTS = [1.0, 1.5, 75.0]
CLASS_NAMES = ['Safe', 'Near-Collision', 'Collision']

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

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

def save_confusion_matrix(y_true, y_pred, class_names, save_dir, filename='confusion_matrix.png'):
    """Salva la matrice di confusione come heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()

def save_pr_curves(y_true, y_probs, class_names, save_dir, filename='pr_curves.png'):
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
    plt.title('Precision-Recall Curves', fontsize=16)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()

def save_metrics_report(y_true, y_pred, class_names, save_dir, filename='metrics_report.png'):
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
    colors = ['#E2EFDA', '#FFF2CC', '#FCE4D6']  # Verde chiaro, Giallo chiaro, Arancione chiaro
    for i, row in enumerate(rows):
        if i < len(class_names):
            for j in range(-1, len(metrics)):
                table[(i+1, j)].set_facecolor(colors[i % len(colors)])
    
    plt.title('Classification Report', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def save_per_class_metrics(y_true, y_pred, class_names, save_dir, filename='per_class_metrics.png'):
    """Salva un grafico a barre con precision, recall, f1 per ogni classe."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
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

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_lstm(args, model, train_loader, val_loader, run_dir, device):

    # Weighted Loss
    weights = torch.tensor(RISK_WEIGHTS).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)    
    best_val_loss = float('inf')
    
    # Lists for plotting
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # Per salvare le predizioni della migliore epoca di validation
    best_val_preds = None
    best_val_targets = None
    best_val_probs = None
    
    checkpoint_interval = max(1, args.epochs // 5)
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for inputs, output_risks in pbar:
            inputs = inputs.to(device) # (B, 10, 72)
            targets = output_risks.to(device) # (B, 25)
            
            optimizer.zero_grad()
            outputs = model(inputs) # (B, 25, 3)
            
            loss = criterion(outputs.view(-1, 3), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 2) # (B, 25)
            total += targets.numel()
            correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets_list = []
        val_probs = []
        
        with torch.no_grad():
            for inputs, output_risks in val_loader:
                inputs = inputs.to(device)
                targets = output_risks.to(device)
                
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=2) # (B, 25, 3)
                loss = criterion(outputs.view(-1, 3), targets.view(-1))
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 2)
                val_total += targets.numel()
                val_correct += (predicted == targets).sum().item()
                
                val_preds.extend(predicted.cpu().numpy().flatten())
                val_targets_list.extend(targets.cpu().numpy().flatten())
                val_probs.extend(probs.cpu().numpy().reshape(-1, 3))
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_preds = np.array(val_preds)
            best_val_targets = np.array(val_targets_list)
            best_val_probs = np.array(val_probs)
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, 'lstm_best.pth'))
            print("Saved best model.")
        else:
            epochs_no_improve += 1
            print(f"Validation Loss not improved for {epochs_no_improve}/{args.patience} epochs.")

        # Periodic Checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Controlla la condizione di stop
        if epochs_no_improve == args.patience:
            print(f"Early Stopping triggered after {epoch + 1} epochs.")
            break
    
    # --- SAVE TRAINING VISUALIZATIONS ---
    save_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, run_dir)
    
    # Salva confusion matrix e metriche per la migliore epoca di validation
    if best_val_preds is not None:
        save_confusion_matrix(best_val_targets, best_val_preds, CLASS_NAMES, run_dir, 'confusion_matrix_val.png')
        save_pr_curves(best_val_targets, best_val_probs, CLASS_NAMES, run_dir, 'pr_curves_val.png')
        save_per_class_metrics(best_val_targets, best_val_preds, CLASS_NAMES, run_dir, 'per_class_metrics_val.png')
        save_metrics_report(best_val_targets, best_val_preds, CLASS_NAMES, run_dir, 'metrics_report_val.png')
    
    print("Training complete.")

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_lstm(args, model, test_loader, run_dir, device):
    print("\n" + "="*50)
    print("Starting Final Test Phase...")
    
    # Weighted Loss
    weights = torch.tensor(RISK_WEIGHTS).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Load the best model
    best_model_path = os.path.join(run_dir, 'lstm_best.pth')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model checkpoint not found at {best_model_path}. Cannot perform test.")
        return
        
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    test_loss = 0.0
    test_total_samples = 0
    test_correct = 0
    all_targets = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, output_risks in tqdm(test_loader, desc="[Test]"):
            inputs = inputs.to(device)
            targets = output_risks.to(device) # (B, 25)
            
            outputs = model(inputs) # (B, 25, 3)
            
            loss = criterion(outputs.view(-1, 3), targets.view(-1))
            test_loss += loss.item()
            
            probs = F.softmax(outputs, dim=2) # (B, 25, 3)
            _, predicted = torch.max(outputs.data, 2)
            
            num_samples_in_batch = targets.numel()
            test_total_samples += num_samples_in_batch
            test_correct += (predicted == targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy().flatten())
            all_predictions.extend(predicted.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().reshape(-1, 3))

    avg_test_batch_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total_samples
    
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # --- SAVE TEST VISUALIZATIONS ---
    save_confusion_matrix(all_targets, all_predictions, CLASS_NAMES, run_dir, 'confusion_matrix_test.png')
    save_pr_curves(all_targets, all_probs, CLASS_NAMES, run_dir, 'pr_curves_test.png')
    save_per_class_metrics(all_targets, all_predictions, CLASS_NAMES, run_dir, 'per_class_metrics_test.png')
    save_metrics_report(all_targets, all_predictions, CLASS_NAMES, run_dir, 'metrics_report_test.png')

    print("-" * 50)
    print(f"Final Test Results (on {len(TEST_SUBJECTS)} subjects):")
    print(f"Average Batch Loss: {avg_test_batch_loss:.4f} | Total Accuracy: {test_acc:.4f}")
    print("="*50)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'c:\Users\Proprietario\Desktop\human-robot-collaboration\datasets\3d_skeletons_risk')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--output_frames', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()
    
    # Setup Runs Directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', f'lstm_{timestamp}')
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    train_loader, val_loader, test_loader, stats = create_pkl_dataloaders(
        dataset_path=args.data_path,
        train_subjects=TRAIN_SUBJECTS,
        val_subjects=VAL_SUBJECTS,
        test_subjects=TEST_SUBJECTS,
        input_frames=args.input_frames,
        output_frames=args.output_frames, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode='sequence'
    )
    
    # Model
    model = RiskLSTM(
        input_size=72, 
        hidden_size=512, 
        num_layers=2, 
        num_classes=3,
        output_frames=args.output_frames
    ).to(device)

    train_lstm(args, model, train_loader, val_loader, run_dir, device)
    test_lstm(args, model, test_loader, run_dir, device)
