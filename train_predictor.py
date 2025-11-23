"""Training Script for Pose Forecasting Models

- Training loop with validation

- MPJPE metrics (Mean Per Joint Position Error)

- Checkpointing and logging with TensorBoard

- Learning rate scheduling

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import argparse
from utils.pkl_data_loader import create_pkl_dataloaders
from models import MLP
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
                 log_dir: str = 'runs',
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
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
        self.best_val_loss = float('inf')
        self.global_step = 0

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
        
        for batch_idx, (input_seq, target_seq, actions, crashes) in enumerate(pbar):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            output_seq = self.model(input_seq)

            # Loss
            loss = self.criterion(output_seq, target_seq)
            
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
            
            # Logging
            if batch_idx % 100 == 0:
                self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('train/mpjpe_step', mpjpe.item(), self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mpjpe = total_mpjpe / len(self.train_loader)
        
        # Log average metrics for the epoch
        self.writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        self.writer.add_scalar('train/mpjpe_epoch', avg_mpjpe, epoch)
        
        return avg_loss, avg_mpjpe
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        total_mpjpe = 0
        all_mpjpe_per_frame = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for input_seq, target_seq, actions, crashes in pbar:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Forward
                output_seq = self.model(input_seq)
                
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
                all_mpjpe_per_frame.append(metrics['mpjpe_per_frame'])
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mpjpe': metrics['mpjpe'],
                    'risk_acc': risk_acc.item()
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_mpjpe = total_mpjpe / len(self.val_loader)
        avg_mpjpe_per_frame = np.mean(all_mpjpe_per_frame, axis=0)
        
        # Logging
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/mpjpe', avg_mpjpe, epoch)
        self.writer.add_scalar('val/risk_accuracy', risk_acc.item(), epoch)
        
        # MPJPE at last frame
        if len(avg_mpjpe_per_frame) > 0:
            mpjpe_last = avg_mpjpe_per_frame[-1]
            self.writer.add_scalar('val/mpjpe_last_frame', mpjpe_last, epoch)
            print(f"MPJPE @ last frame ({len(avg_mpjpe_per_frame)}): {mpjpe_last:.2f} mm")
        
        # MPJPE at 1 second (25 frames @ 25fps) - only if available
        if len(avg_mpjpe_per_frame) >= 25:
            mpjpe_1sec = avg_mpjpe_per_frame[24]
            self.writer.add_scalar('val/mpjpe_1sec', mpjpe_1sec, epoch)
            print(f"MPJPE @ 1 sec: {mpjpe_1sec:.2f} mm")
        
        return avg_loss, avg_mpjpe
    
    def test(self) -> float:
        """Final test"""
        self.model.eval()
        
        total_mpjpe = 0
        all_mpjpe_per_frame = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for input_seq, target_seq, actions, crashes in pbar:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Forward
                output_seq = self.model(input_seq)
                
                # Metrics
                output_denorm = self.denormalize(output_seq)
                target_denorm = self.denormalize(target_seq)
                metrics = compute_metrics(output_denorm, target_denorm)
                
                total_mpjpe += metrics['mpjpe']
                all_mpjpe_per_frame.append(metrics['mpjpe_per_frame'])
        
        avg_mpjpe = total_mpjpe / len(self.test_loader)
        avg_mpjpe_per_frame = np.mean(all_mpjpe_per_frame, axis=0)
        
        print(f"\n=== Test Results ===")
        print(f"Average MPJPE: {avg_mpjpe:.2f} mm")
        
        if len(avg_mpjpe_per_frame) > 0:
            print(f"MPJPE @ last frame ({len(avg_mpjpe_per_frame)}): {avg_mpjpe_per_frame[-1]:.2f} mm")

        if len(avg_mpjpe_per_frame) >= 25:
            mpjpe_1sec = avg_mpjpe_per_frame[24]
            print(f"MPJPE @ 1 sec: {mpjpe_1sec:.2f} mm")
        
        # Save results
        results = {
            'avg_mpjpe': float(avg_mpjpe),
            'mpjpe_per_frame': avg_mpjpe_per_frame.tolist()
        }
        
        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
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
            val_loss, val_mpjpe = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Logging
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            print(f"Train → Loss: {train_loss:>10.2f} | MPJPE: {train_mpjpe:>6.2f} mm")
            print(f"Val   → Loss: {val_loss:>10.2f} | MPJPE: {val_mpjpe:>6.2f} mm")
            print(f"{'='*70}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        # Final test
        print("\n" + "="*50)
        print("Running final test...")
        self.test()
        
        self.writer.close()



if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Train Pose Forecasting Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    # Configuration
    base_path = r"c:\Users\Proprietario\Desktop\human-robot-collaboration"
    dataset_path = os.path.join(base_path, "datasets", "3d_skeletons")
    
    # Dataset split
    train_subjects = [f'S{i:02d}' for i in range(16)]  # S00-S15
    val_subjects = ['S16', 'S17']
    test_subjects = ['S18', 'S19']
    
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
    print(f"Training Model: MLP")
    
    stats = None
    
    # Create dataloaders for MLP
    print("\nCreating MLP dataloaders...")
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
    
    # Create MLP model
    print("\nCreating MLP model...")
    model = MLP(
        input_frames=INPUT_FRAMES,
        output_frames=OUTPUT_FRAMES,
        num_joints=24,
        hidden_dims=[1024, 512, 256],
        dropout=0.2
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
        experiment_name=f'mlp_v1',
        extra_config={
            'model_type': 'mlp',
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

