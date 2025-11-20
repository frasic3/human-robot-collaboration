import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np

from models.pose_estimator import PoseEstimator
from utils.pose_estimation_data_loader import create_pose_estimation_dataloaders

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    train_subjects = ['S00', 'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    val_subjects = ['S10', 'S11']
    
    train_loader, val_loader = create_pose_estimation_dataloaders(
        base_path=args.data_path,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        batch_size=args.batch_size,
        stride=args.stride,
        num_workers=0 # Windows compatibility
    )
    
    # Model
    model = PoseEstimator(pretrained=True).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for frames, targets in pbar:
            frames = frames.to(device)
            targets = targets.to(device) # (B, 24, 3) in meters
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, targets in val_loader:
                frames = frames.to(device)
                targets = targets.to(device)
                
                outputs = model(frames)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'pose_estimator_best.pth'))
            print("Saved best model.")
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--stride', type=int, default=10, help="Frame stride to reduce dataset size")
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
