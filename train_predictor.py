"""Training Script for Pose Forecasting Models

- Training loop with validation
- MPJPE metrics (Mean Per Joint Position Error)
- Checkpointing and logging (PyTorch based)
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, f1_score
from utils.pkl_data_loader import create_pkl_dataloaders
from models import MLP, LSTM
from utils.metrics import compute_mpjpe, compute_metrics, compute_risk


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 test_loader,
                 device: torch.device,
                 stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 save_dir: str = 'checkpoints',
                 experiment_name: Optional[str] = None,
                 extra_config: Optional[Dict[str, Any]] = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Statistics for denormalization
        self.stats = stats
        if self.stats is not None:
            self.mean = torch.from_numpy(self.stats[0]).float().to(device)
            self.std = torch.from_numpy(self.stats[1]).float().to(device)
        else:
            self.mean = None
            self.std = None
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Logging
        if experiment_name is None:
            experiment_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
        self.experiment_name = experiment_name
        # Use 'runs' directory instead of 'checkpoints' for better organization
        self.save_dir = os.path.join('runs', experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_mpjpe': [],
            'val_loss': [],
            'val_mpjpe': [],
            'val_risk_acc': [],
            'val_mpjpe_1sec': []
        }

        # Save configuration
        self.save_config(extra_config or {})
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor using stored stats"""
        if self.mean is None or self.std is None:
            return tensor
        
        # tensor shape: (B, T, 24, 3)
        # mean/std shape: (3,) -> broadcast to (1, 1, 1, 3)
        return tensor * (self.std + 1e-8) + self.mean

    def save_config(self, extra: Dict[str, Any]) -> None:
        """Save training configuration"""
        config = {
            'model': self.model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'optimizer': str(self.optimizer),
            'device': str(self.device),
            'experiment_name': self.experiment_name,
            'lr': self.optimizer.param_groups[0].get('lr', None),
            'weight_decay': self.optimizer.param_groups[0].get('weight_decay', None)
        }

        # Merge any additional experiment metadata (hyperparameters, dataset split, etc.)
        try:
            config.update(extra)
        except Exception:
            pass
    
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()        
        total_loss = 0
        total_mpjpe = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (input_seq, target_seq, input_risk, output_risk, actions, crashes) in enumerate(pbar):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(input_seq)
            if isinstance(outputs, tuple):
                output_seq = outputs[0]
            else:
                output_seq = outputs

            # --- Weighted Loss (Risk-Aware) ---
            # Calculate risk on Ground Truth to assign weights
            with torch.no_grad():
                target_denorm = self.denormalize(target_seq)
                risk_info = compute_risk(target_denorm)
                risk_classes = risk_info['risk_classes']  # (B, P)
                
                # Weights: Safe=1, Medium=5, High=20
                weights = torch.ones_like(risk_classes, dtype=torch.float32)
                weights[risk_classes == 1] = 5.0
                weights[risk_classes == 2] = 20.0
                
                # Expand for broadcasting: (B, P) -> (B, P, 1, 1)
                weights = weights.unsqueeze(-1).unsqueeze(-1)
            
            # Compute weighted MSE
            loss_unreduced = torch.nn.functional.mse_loss(output_seq, target_seq, reduction='none')
            loss = (loss_unreduced * weights).mean()
            # ----------------------------------
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                # Denormalize for metrics
                output_denorm = self.denormalize(output_seq)
                target_denorm = self.denormalize(target_seq)
                mpjpe = compute_mpjpe(output_denorm, target_denorm)
            
            total_loss += loss.item()
            total_mpjpe += mpjpe.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'mpjpe': mpjpe.item()
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mpjpe = total_mpjpe / len(self.train_loader)
        
        return avg_loss, avg_mpjpe
    
    def validate(self, epoch: int) -> Tuple[float, float, float, float]:
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        total_mpjpe = 0
        total_risk_acc = 0
        all_mpjpe_per_frame = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for input_seq, target_seq, input_risk, output_risk, actions, crashes in pbar:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Forward
                outputs = self.model(input_seq)
                if isinstance(outputs, tuple):
                    output_seq = outputs[0]
                else:
                    output_seq = outputs
                
                # Loss
                loss = self.criterion(output_seq, target_seq)
                
                # Metrics
                output_denorm = self.denormalize(output_seq)
                target_denorm = self.denormalize(target_seq)
                metrics = compute_metrics(output_denorm, target_denorm)
                
                # Risk Evaluation
                pred_risk = compute_risk(output_denorm)
                gt_risk = compute_risk(target_denorm)
                
                # Calculate Risk Accuracy (Exact Match)
                risk_acc = (pred_risk['risk_classes'] == gt_risk['risk_classes']).float().mean()
                
                total_loss += loss.item()
                total_mpjpe += metrics['mpjpe']
                total_risk_acc += risk_acc.item()
                all_mpjpe_per_frame.append(metrics['mpjpe_per_frame'])
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mpjpe': metrics['mpjpe'],
                    'risk_acc': risk_acc.item()
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_mpjpe = total_mpjpe / len(self.val_loader)
        avg_risk_acc = total_risk_acc / len(self.val_loader)
        avg_mpjpe_per_frame = np.mean(all_mpjpe_per_frame, axis=0)
        
        mpjpe_1sec = 0.0
        if len(avg_mpjpe_per_frame) >= 25:
            mpjpe_1sec = avg_mpjpe_per_frame[24]
        elif len(avg_mpjpe_per_frame) > 0:
             mpjpe_1sec = avg_mpjpe_per_frame[-1]
        
        return avg_loss, avg_mpjpe, avg_risk_acc, mpjpe_1sec
    
    def compute_saliency(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute saliency map (gradient of loss w.r.t input) for feature importance analysis.
        """
        self.model.eval()
        # Enable gradients for input
        input_seq = input_seq.clone().detach().requires_grad_(True)
        
        # Forward
        outputs = self.model(input_seq)
        if isinstance(outputs, tuple):
            output_seq = outputs[0]
        else:
            output_seq = outputs
            
        # Loss (MSE)
        loss = self.criterion(output_seq, target_seq)
        
        # Backward
        self.model.zero_grad()
        loss.backward()
        
        # Saliency: magnitude of gradient
        # Shape: (B, T, J, 3)
        saliency = input_seq.grad.abs()
        return saliency.detach()

    def plot_results(self, dataset_name, all_risk_gt, all_risk_pred_cls, all_risk_scores, all_inputs, all_targets, all_preds, worst_indices, output_dir):
        """Generate and save all requested plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        risk_gt_flat = all_risk_gt.view(-1).numpy()
        risk_pred_flat = all_risk_pred_cls.view(-1).numpy()
        
        cm = confusion_matrix(risk_gt_flat, risk_pred_flat)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
        plt.xlabel('Predicted Risk')
        plt.ylabel('True Risk')
        plt.title(f'Confusion Matrix ({dataset_name})')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Normalized Confusion Matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
        plt.xlabel('Predicted Risk')
        plt.ylabel('True Risk')
        plt.title(f'Normalized Confusion Matrix ({dataset_name})')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'))
        plt.close()
        
        # 2. PR Curve, F1 Curve, P Curve, R Curve (For High Risk Class)
        # High Risk is Class 2. We use -min_distance as score (lower dist = higher risk)
        y_true_high = (all_risk_gt.view(-1) == 2).float().numpy()
        y_scores_high = -all_risk_scores.view(-1).numpy() # Negate distance so higher score = closer = higher risk
        
        precision, recall, thresholds = precision_recall_curve(y_true_high, y_scores_high)
        
        # PR Curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='High Risk')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({dataset_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'PR_curve.png'))
        plt.close()
        
        # F1 Curve vs Threshold (using distance thresholds implicitly via score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
        plt.xlabel('Score Threshold (-Distance)')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Curve ({dataset_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'F1_curve.png'))
        plt.close()
        
        # Precision Curve
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precision[:-1], label='Precision')
        plt.xlabel('Score Threshold (-Distance)')
        plt.ylabel('Precision')
        plt.title(f'Precision Curve ({dataset_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'P_curve.png'))
        plt.close()
        
        # Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, recall[:-1], label='Recall')
        plt.xlabel('Score Threshold (-Distance)')
        plt.ylabel('Recall')
        plt.title(f'Recall Curve ({dataset_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'R_curve.png'))
        plt.close()
        
        # 3. Visualizations (Worst Cases & Random Good Cases)
        
        # Worst Cases
        for i, idx in enumerate(worst_indices[:5]): # Top 5 worst
            idx = idx.item()
            self._plot_sample(all_inputs[idx], all_targets[idx], all_preds[idx], 
                              f'Worst Case #{i+1}', os.path.join(output_dir, f'worst_case_{i+1}.png'))
            
        # Random Good Cases (Low Error)
        diff = all_preds - all_targets
        sample_errors = torch.norm(diff, dim=-1).mean(dim=(1, 2))
        # Get indices with low error (bottom 10%)
        good_indices = torch.nonzero(sample_errors < torch.quantile(sample_errors, 0.1)).squeeze()
        if len(good_indices.shape) > 0 and len(good_indices) > 0:
             # Pick 3 random good cases
            perm = torch.randperm(len(good_indices))
            selected_good = good_indices[perm[:3]]
            
            for i, idx in enumerate(selected_good):
                idx = idx.item()
                self._plot_sample(all_inputs[idx], all_targets[idx], all_preds[idx], 
                                  f'Good Case #{i+1}', os.path.join(output_dir, f'good_case_{i+1}.png'))

    def _plot_sample(self, inp, tgt, prd, title, save_path):
        plt.figure(figsize=(12, 5))
        
        # View 1: Top View (X-Z plane)
        plt.subplot(1, 2, 1)
        plt.title(f"{title} - Top View (X-Z)")
        plt.scatter(inp[-1, :, 0], inp[-1, :, 2], c='black', label='Input', alpha=0.5)
        plt.scatter(tgt[-1, :, 0], tgt[-1, :, 2], c='green', label='Target', marker='x')
        plt.scatter(prd[-1, :, 0], prd[-1, :, 2], c='red', label='Pred', marker='o')
        plt.xlabel('X (Lateral)')
        plt.ylabel('Z (Depth)')
        plt.legend()
        
        # View 2: Side View (Y-Z plane)
        plt.subplot(1, 2, 2)
        plt.title(f"{title} - Side View (Y-Z)")
        plt.scatter(inp[-1, :, 2], inp[-1, :, 1], c='black', label='Input (Last)', alpha=0.5)
        plt.scatter(tgt[-1, :, 2], tgt[-1, :, 1], c='green', label='Target (Last)', marker='x')
        plt.scatter(prd[-1, :, 2], prd[-1, :, 1], c='red', label='Pred (Last)', marker='o')
        plt.xlabel('Z (Depth)')
        plt.ylabel('Y (Height)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def evaluate_dataset(self, loader, dataset_name: str) -> float:
        """
        Evaluate model on a specific dataset and save ALL data for analysis.
        Saves: Inputs, Targets, Predictions, Hidden States, Risk, Actions, Crashes, Saliency (subset).
        """
        self.model.eval()
        
        total_mpjpe = 0
        all_mpjpe_per_frame = []
        
        # Storage for analysis
        all_inputs_norm = []
        all_inputs = []
        all_targets = []
        all_preds = []
        all_hiddens = []
        all_risk_gt = []
        all_risk_pred = []
        all_risk_scores = []
        all_actions = []
        all_crashes = []
        
        print(f"\nEvaluating {dataset_name} set and collecting data...")
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f'Eval {dataset_name}')
            
            for input_seq, target_seq, input_risk, output_risk, actions, crashes in pbar:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Forward
                outputs = self.model(input_seq)
                if isinstance(outputs, tuple):
                    output_seq = outputs[0]
                    # Capture hidden state (last layer, last timestep)
                    # h_n shape: (num_layers, B, hidden_dim) -> take last layer: (B, hidden_dim)
                    hidden_state = outputs[1][0][-1]
                    all_hiddens.append(hidden_state.cpu())
                else:
                    output_seq = outputs
                    # Dummy hidden for MLP
                    all_hiddens.append(torch.zeros(input_seq.size(0), 1))
                
                # Metrics
                output_denorm = self.denormalize(output_seq)
                target_denorm = self.denormalize(target_seq)
                metrics = compute_metrics(output_denorm, target_denorm)
                
                # Risk
                pred_risk_info = compute_risk(output_denorm)
                gt_risk_info = compute_risk(target_denorm)
                
                pred_risk = pred_risk_info['risk_classes']
                gt_risk = gt_risk_info['risk_classes']
                
                total_mpjpe += metrics['mpjpe']
                all_mpjpe_per_frame.append(metrics['mpjpe_per_frame'])
                
                # Store data
                all_inputs_norm.append(input_seq.cpu())
                all_inputs.append(self.denormalize(input_seq).cpu())
                all_targets.append(target_denorm.cpu())
                all_preds.append(output_denorm.cpu())
                all_risk_gt.append(gt_risk.cpu())
                all_risk_pred.append(pred_risk.cpu())
                all_risk_scores.append(pred_risk_info['min_distances'].cpu())
                all_actions.append(actions) # actions is likely a list or tensor
                all_crashes.append(crashes) # crashes is likely a list or tensor
        
        # Concatenate all data
        all_inputs_norm = torch.cat(all_inputs_norm, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_hiddens = torch.cat(all_hiddens, dim=0)
        all_risk_gt = torch.cat(all_risk_gt, dim=0)
        all_risk_pred = torch.cat(all_risk_pred, dim=0)
        all_risk_scores = torch.cat(all_risk_scores, dim=0)
        
        # Handle actions/crashes which might be lists of strings or tensors
        # Assuming they are tensors or lists of tensors from the loader
        try:
            all_actions = torch.cat(all_actions, dim=0)
        except:
            # If they are lists of strings/ints
            all_actions = np.concatenate(all_actions, axis=0)
            
        try:
            all_crashes = torch.cat(all_crashes, dim=0)
        except:
            all_crashes = np.concatenate(all_crashes, axis=0)
        
        avg_mpjpe = total_mpjpe / len(loader)
        avg_mpjpe_per_frame = np.mean(all_mpjpe_per_frame, axis=0)
        
        print(f"\n=== {dataset_name} Results ===")
        print(f"Average MPJPE: {avg_mpjpe:.2f} mm")
        
        # Identify "Complicated Patterns" (Highest Error Samples)
        diff = all_preds - all_targets
        sample_errors = torch.norm(diff, dim=-1).mean(dim=(1, 2)) # Mean over time and joints
        
        # Get top 50 worst samples
        k = min(50, len(sample_errors))
        worst_values, worst_indices = torch.topk(sample_errors, k)
        
        print(f"\nComputing Saliency Maps for top {k} worst predictions in {dataset_name}...")
        
        # Compute Saliency for worst samples
        saliency_maps = []
        batch_size = 10
        
        for i in range(0, k, batch_size):
            batch_idx = worst_indices[i:i+batch_size]
            batch_inputs = all_inputs_norm[batch_idx].to(self.device)
            
            # Re-construct normalized targets
            batch_targets_denorm = all_targets[batch_idx].to(self.device)
            if self.mean is not None:
                batch_targets_norm = (batch_targets_denorm - self.mean) / (self.std + 1e-8)
            else:
                batch_targets_norm = batch_targets_denorm
                
            saliency = self.compute_saliency(batch_inputs, batch_targets_norm)
            saliency_maps.append(saliency.cpu())
            
        saliency_maps = torch.cat(saliency_maps, dim=0)
        
        # --- Generate Plots & Save Results ---
        output_dir = os.path.join(self.save_dir, dataset_name)
        self.plot_results(dataset_name, all_risk_gt, all_risk_pred, all_risk_scores, 
                          all_inputs, all_targets, all_preds, worst_indices, output_dir)
        
        # Save Raw Data
        save_path = os.path.join(output_dir, f'{dataset_name}_analysis_data.pt')
        print(f"Saving {dataset_name} analysis data to {save_path}...")
        
        torch.save({
            'inputs': all_inputs,           # Denormalized inputs
            'targets': all_targets,         # Denormalized targets
            'predictions': all_preds,       # Denormalized predictions
            'hidden_states': all_hiddens,   # Latent features
            'risk_gt': all_risk_gt,
            'risk_pred': all_risk_pred,
            'actions': all_actions,
            'crashes': all_crashes,
            'worst_indices': worst_indices,
            'worst_saliency_maps': saliency_maps, # Saliency for the worst cases
            'avg_mpjpe': avg_mpjpe,
            'mpjpe_per_frame': all_mpjpe_per_frame
        }, save_path)
        
        return avg_mpjpe
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self, num_epochs: int) -> None:
        """Complete training loop"""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Experiment: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Num parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_mpjpe = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mpjpe, val_risk_acc, val_mpjpe_1sec = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Update History
            self.history['train_loss'].append(float(train_loss))
            self.history['train_mpjpe'].append(float(train_mpjpe))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_mpjpe'].append(float(val_mpjpe))
            self.history['val_risk_acc'].append(float(val_risk_acc))
            self.history['val_mpjpe_1sec'].append(float(val_mpjpe_1sec))
            
            # Save History
            with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=4)
            
            # Logging
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*80}")
            print(f"Train → Loss: {train_loss:.4f} | MPJPE: {train_mpjpe:.2f} mm")
            print(f"Val   → Loss: {val_loss:.4f} | MPJPE: {val_mpjpe:.2f} mm | Risk Acc: {val_risk_acc*100:.1f}% | @1s: {val_mpjpe_1sec:.2f} mm")
            print(f"{'='*80}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        # Final Evaluation on ALL sets
        print("\n" + "="*50)
        print("Running FINAL EVALUATION on Train, Val, and Test sets...")
        print("This will save detailed analysis data for all splits.")
        print("="*50)
        
        self.evaluate_dataset(self.train_loader, "train")
        self.evaluate_dataset(self.val_loader, "val")
        self.evaluate_dataset(self.test_loader, "test")
        
        # self.writer.close() # Removed


if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Train Pose Forecasting Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'lstm'], help='Model architecture')
    args = parser.parse_args()

    # Configuration
    base_path = r"c:\Users\Proprietario\Desktop\human-robot-collaboration"
    dataset_path = os.path.join(base_path, "datasets", "3d_skeletons_risk")
    
    # Dataset split (Paper Protocol)
    # Val: S00, S04
    # Test: S02, S03, S18, S19
    # Train: Rest
    val_subjects = ['S00', 'S04']
    test_subjects = ['S02', 'S03', 'S18', 'S19']
    train_subjects = [f'S{i:02d}' for i in range(20) if f'S{i:02d}' not in val_subjects + test_subjects]
    
    # Hyperparameters
    INPUT_FRAMES = 10
    OUTPUT_FRAMES = 25  # 1.0 seconds @ 25 fps
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = 1e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training Model: {args.model.upper()}")
    
    stats = None
    
    # Create dataloaders for MLP
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, stats = create_pkl_dataloaders(
        dataset_path=dataset_path,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        input_frames=INPUT_FRAMES,
        output_frames=OUTPUT_FRAMES,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    # Create model
    if args.model == 'mlp':
        print("\nCreating MLP model...")
        model = MLP(
            input_frames=INPUT_FRAMES,
            output_frames=OUTPUT_FRAMES,
            num_joints=24,
            hidden_dims=[1024, 512, 256],
            dropout=0.2
        )
    elif args.model == 'lstm':
        print("\nCreating LSTM model con dropout 0.5...")
        model = LSTM(
            input_frames=INPUT_FRAMES,
            output_frames=OUTPUT_FRAMES,
            num_joints=24,
            hidden_dim=1024,
            num_layers=2,
            dropout=0.5
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        stats=stats,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        experiment_name=f'{args.model}_v1',
        extra_config={
            'model_type': args.model,
            'dataset_path': dataset_path,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects,
            'input_frames': INPUT_FRAMES,
            'output_frames': OUTPUT_FRAMES,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
        }
    )
    
    # Train
    trainer.train(num_epochs=NUM_EPOCHS)



