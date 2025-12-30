import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import random
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.mlp import RiskMLP
from utils.pkl_data_loader import create_pkl_dataloaders
from utils.visualization import (
    save_training_curves,
    save_confusion_matrix,
    save_pr_curves,
    save_metrics_report
)
from utils.skeleton_viz import visualize_missed_collision

# Dataset Split Configuration
TRAIN_SUBJECTS = [
    'S01', 'S05', 'S06', 'S07', 'S08', 'S09',
    'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17'
]
VAL_SUBJECTS = ['S00', 'S04']
TEST_SUBJECTS = ['S02', 'S03', 'S18', 'S19']

RISK_WEIGHTS = [1, 4, 10] 
CLASS_NAMES = ['Safe', 'Near-Collision', 'Collision']


def _write_tuning_metrics(run_dir: str, payload: dict) -> None:
    import json
    path = os.path.join(run_dir, 'tuning_metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def train_mlp(args, train_loader, val_loader, run_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Collision threshold: {args.threshold}")
    print(f"Risk Weights: {RISK_WEIGHTS}")
    
    model = RiskMLP(input_size=72, hidden_sizes=[128, 64], num_classes=3).to(device)
    
    weights = torch.tensor(RISK_WEIGHTS).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)  # Aumentato per regolarizzazione
    
    best_val_loss = float('inf')
    best_collision_recall = 0.0
    best_missed_warnings = float('inf')
    best_epoch = 0
    
    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    best_val_preds = None
    best_val_targets = None
    best_val_probs = None
    
    for epoch in range(args.epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total if total > 0 else 0})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_targets_list = []
        val_probs_list = []
        val_inputs_list = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                val_targets_list.extend(targets.cpu().numpy())
                val_probs_list.extend(probs.cpu().numpy())
                val_inputs_list.extend(inputs.cpu().numpy())
        
        val_probs_array = np.array(val_probs_list)
        val_targets_array = np.array(val_targets_list)
        val_inputs_array = np.array(val_inputs_list)
        collision_probs = val_probs_array[:, 2]
        
        # Apply threshold logic
        val_preds = np.argmax(val_probs_array[:, :2], axis=1)
        val_preds[collision_probs >= args.threshold] = 2
        
        val_correct = np.sum(val_preds == val_targets_array)
        val_total = len(val_targets_array)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0

       # calculate collision recall 
        collision_indices = (val_targets_array == 2)
        if np.sum(collision_indices) > 0:
            current_collision_recall = np.sum(val_preds[collision_indices] == 2) / np.sum(collision_indices)
            
            # identify missed collision samples
            missed_collisions = collision_indices & (val_preds != 2)
            if np.sum(missed_collisions) > 0:
                missed_dir = os.path.join(run_dir, 'missed_collisions')
                os.makedirs(missed_dir, exist_ok=True)
                
                missed_indices = np.where(missed_collisions)[0]
                print(f"Missed {len(missed_indices)} collision(s) at Epoch {epoch+1}:")
                
                for idx in missed_indices:                    
                    viz_path = os.path.join(missed_dir, f'epoch{epoch+1}_sample_{idx}.png')
                    visualize_missed_collision(
                        skeleton_data=val_inputs_array[idx],
                        probs=val_probs_array[idx],
                        true_label=val_targets_array[idx],
                        pred_label=val_preds[idx],
                        class_names=CLASS_NAMES,
                        sample_idx=idx,
                        threshold=args.threshold,
                        save_path=viz_path
                    )
                    
        else:
            current_collision_recall = 0.0
        
        print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}")
        
        # 2. Errori Near->Safe 
        # True Label = 1 (Near), Predicted = 0 (Safe)
        current_missed_warnings = np.sum((val_targets_array == 1) & (val_preds == 0))        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        save_model = False
        
        # LIVELLO 1: Massimizzare la Sicurezza (Collision Recall)
        if current_collision_recall > best_collision_recall:
            save_model = True
            print(f"Improved safety: Recall: {current_collision_recall:.2f}")
            
        # LIVELLO 2: A parità di sicurezza, Minimizzare i Mancati Preavvisi (Near->Safe)
        elif current_collision_recall == best_collision_recall:
            if current_missed_warnings < best_missed_warnings:
                save_model = True
                print(f"Better warnings: Missed Warnings dropped to {current_missed_warnings}")
            
            # LIVELLO 3: A parità di recall E missed warnings, Minimizzare la Loss (Precisione generale)
            elif current_missed_warnings == best_missed_warnings:
                if avg_val_loss < best_val_loss:
                    save_model = True
                    print(f"Optimized loss: Val Loss dropped to {avg_val_loss:.4f}")
    
        if save_model:
            best_collision_recall = current_collision_recall
            best_missed_warnings = current_missed_warnings
            best_val_loss = avg_val_loss
            best_val_preds = val_preds.copy()
            best_val_targets = val_targets_array.copy()
            best_val_probs = val_probs_array.copy()
            torch.save(model.state_dict(), os.path.join(run_dir, 'mlp_best.pth'))
            best_epoch = epoch

            _write_tuning_metrics(
                run_dir,
                {
                    'best_epoch': int(best_epoch + 1),
                    'best_val_loss': float(best_val_loss),
                    'best_collision_recall': float(best_collision_recall),
                    'best_missed_warnings': int(best_missed_warnings),
                    'threshold': float(args.threshold),
                    'risk_weights': list(RISK_WEIGHTS),
                    'epochs': int(args.epochs),
                    'lr': float(args.lr),
                    'batch_size': int(getattr(args, 'batch_size', 0)),
                    'train_subjects': list(TRAIN_SUBJECTS),
                    'val_subjects': list(VAL_SUBJECTS),
                }
            )
    
    # Save visualizations
    save_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, run_dir)
    
    # Prepare info text
    info_val = f"Val Subjects: {VAL_SUBJECTS} | Threshold: {args.threshold} | Epochs: {args.epochs}"
    info_train = f"Train Subjects: {TRAIN_SUBJECTS[:3]}+{len(TRAIN_SUBJECTS)-3} more | Threshold: {args.threshold}"
    
    if best_val_preds is not None:
        save_confusion_matrix(best_val_targets, best_val_preds, CLASS_NAMES, run_dir, 'confusion_matrix_val.png', info_text=info_val)
        save_pr_curves(best_val_targets, best_val_probs, CLASS_NAMES, run_dir, 'pr_curves_val.png', info_text=info_val)
        save_metrics_report(best_val_targets, best_val_preds, CLASS_NAMES, run_dir, 'metrics_report_val.png', info_text=info_val)
    
    # Salva anche le metriche sul training set finale
    print("\nEvaluating on training set...")
    model.eval()
    train_targets_list = []
    train_probs_list = []
    
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            train_targets_list.extend(targets.cpu().numpy())
            train_probs_list.extend(probs.cpu().numpy())
    
    train_targets_array = np.array(train_targets_list)
    train_probs_array = np.array(train_probs_list)
    collision_probs_train = train_probs_array[:, 2]
    train_preds_final = np.argmax(train_probs_array[:, :2], axis=1)
    train_preds_final[collision_probs_train >= args.threshold] = 2
    
    save_confusion_matrix(train_targets_array, train_preds_final, CLASS_NAMES, run_dir, 'confusion_matrix_train.png', info_text=info_train)
    save_pr_curves(train_targets_array, train_probs_array, CLASS_NAMES, run_dir, 'pr_curves_train.png', info_text=info_train)
    save_metrics_report(train_targets_array, train_preds_final, CLASS_NAMES, run_dir, 'metrics_report_train.png', info_text=info_train)
    
    print("Training complete.")


def test_mlp(args, test_loader, run_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTesting on device: {device}")
    print(f"Using Collision threshold: {args.threshold}")

    model = RiskMLP(input_size=72, hidden_sizes=[128, 64], num_classes=3).to(device)
    
    checkpoint_path = os.path.join(run_dir, 'mlp_best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model not found at {checkpoint_path}")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    collision_probs = all_probs[:, 2]
    
    all_preds = np.argmax(all_probs[:, :2], axis=1)
    all_preds[collision_probs >= args.threshold] = 2
    
    # Prepare info text
    info_test = f"Test Subjects: {TEST_SUBJECTS} | Threshold: {args.threshold}"
    
    # Save visualizations
    save_confusion_matrix(all_targets, all_preds, CLASS_NAMES, run_dir, 'confusion_matrix_test.png', info_text=info_test)
    save_pr_curves(all_targets, all_probs, CLASS_NAMES, run_dir, 'pr_curves_test.png', info_text=info_test)
    save_metrics_report(all_targets, all_preds, CLASS_NAMES, run_dir, 'metrics_report_test.png', info_text=info_test)
    
    print("Test complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--threshold', type=float, default=0.1, help="Collision probability threshold")
    args = parser.parse_args()
    
    # Seed per riproducibilità
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', f'mlp_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    train_loader, val_loader, test_loader, stats = create_pkl_dataloaders(
        dataset_path=args.data_path,
        train_subjects=TRAIN_SUBJECTS,
        val_subjects=VAL_SUBJECTS,
        test_subjects=TEST_SUBJECTS,
        input_frames=1,
        output_frames=0,
        batch_size=args.batch_size,
        num_workers=0,
        mode='single_frame',
        use_weighted_sampler=True,
        augment_collision=True,
        augment_factor=50
    )
    
    train_mlp(args, train_loader, val_loader, run_dir)
    test_mlp(args, test_loader, run_dir)