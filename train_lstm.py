import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import recall_score

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
RISK_WEIGHTS = [1.0, 5.0, 20.0]

def train_lstm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    # For LSTM, we use sequence mode.
    # Input: 10 frames. Output: Risk of the NEXT 25 frames.
    
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
        hidden_size=128, 
        num_layers=2, 
        num_classes=3,
        output_frames=args.output_frames
    ).to(device)
    
    # Weighted Loss
    weights = torch.tensor(RISK_WEIGHTS).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for inputs, output_risks in pbar:
            inputs = inputs.to(device) # (B, 10, 72)
            
            # Target: Risk of FUTURE frames
            # output_risks shape: (B, 25)
            targets = output_risks.to(device) # (B, 25)
            
            optimizer.zero_grad()
            outputs = model(inputs) # (B, 25, 3)
            
            # Flatten for CrossEntropyLoss
            # outputs: (B*25, 3)
            # targets: (B*25)
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
        
        with torch.no_grad():
            for inputs, output_risks in val_loader:
                inputs = inputs.to(device)
                targets = output_risks.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 3), targets.view(-1))
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 2)
                val_total += targets.numel()
                val_correct += (predicted == targets).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'lstm_best.pth'))
            print("Saved best model.")
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'c:\Users\Proprietario\Desktop\human-robot-collaboration\datasets\3d_skeletons_risk')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--output_frames', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_lstm(args)
