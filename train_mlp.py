"""
MLP Training Script for Risk Classification
"""
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

# Dataset Split Configuration
TRAIN_SUBJECTS = [
    'S01', 'S05', 'S06', 'S07', 'S08', 'S09',
    'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17'
]
VAL_SUBJECTS = ['S00', 'S04']
TEST_SUBJECTS = ['S02', 'S03', 'S18', 'S19']

# Risk Configuration 
RISK_WEIGHTS = [1.0, 2.0, 100.0]
CLASS_NAMES = ['Safe', 'Near-Collision', 'Collision']


def train_mlp(args, train_loader, val_loader, run_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Collision threshold: {args.threshold}")
    
    model = RiskMLP(input_size=72, hidden_sizes=[128, 64], num_classes=3).to(device)
    
    weights = torch.tensor(RISK_WEIGHTS).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # Best validation predictions
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
        
        val_probs_array = np.array(val_probs_list)
        val_targets_array = np.array(val_targets_list)
        collision_probs = val_probs_array[:, 2]
        
        # Apply threshold: if P(Collision) >= threshold -> Collision
        val_preds = np.argmax(val_probs_array[:, :2], axis=1)  # argmax tra Safe e Near-Collision
        val_preds[collision_probs >= args.threshold] = 2       # Se P(Collision) >= threshold -> Collision
        
        val_correct = np.sum(val_preds == val_targets_array)
        val_total = len(val_targets_array)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_preds = val_preds.copy()
            best_val_targets = val_targets_array.copy()
            best_val_probs = val_probs_array.copy()
            torch.save(model.state_dict(), os.path.join(run_dir, 'mlp_best.pth'))
            print(">>> Saved best model.")
    
    # Save visualizations
    save_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, run_dir)
    
    if best_val_preds is not None:
        save_confusion_matrix(best_val_targets, best_val_preds, CLASS_NAMES, run_dir, 'confusion_matrix_val.png')
        save_pr_curves(best_val_targets, best_val_probs, CLASS_NAMES, run_dir, 'pr_curves_val.png')
        save_metrics_report(best_val_targets, best_val_preds, CLASS_NAMES, run_dir, 'metrics_report_val.png')
    
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
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    
    # Apply threshold: if P(Collision) >= threshold -> Collision
    all_preds = np.argmax(all_probs[:, :2], axis=1)  # argmax tra Safe e Near-Collision
    all_preds[collision_probs >= args.threshold] = 2  # Se P(Collision) >= threshold -> Collision
    
    # Save visualizations
    save_confusion_matrix(all_targets, all_preds, CLASS_NAMES, run_dir, 'confusion_matrix_test.png')
    save_pr_curves(all_targets, all_probs, CLASS_NAMES, run_dir, 'pr_curves_test.png')
    save_metrics_report(all_targets, all_preds, CLASS_NAMES, run_dir, 'metrics_report_test.png')
    
    print("Test complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.90, help="Collision probability threshold")
    args = parser.parse_args()
    
    # Seed fisso per riproducibilità
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Setup run directory
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
        num_workers=args.num_workers,
        mode='single_frame',
        use_weighted_sampler=True,
        augment_collision=True,
        augment_factor=50
    )
    
    train_mlp(args, train_loader, val_loader, run_dir)
    test_mlp(args, test_loader, run_dir)
