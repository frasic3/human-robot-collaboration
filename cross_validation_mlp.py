"""
Cross-Validation Script - Testa diverse combinazioni Train/Val/Test
"""
import os
import argparse
import numpy as np
import torch
from datetime import datetime
from itertools import combinations
import json

from train_mlp import train_mlp, test_mlp, RISK_WEIGHTS, CLASS_NAMES
from utils.pkl_data_loader import create_pkl_dataloaders
from models.mlp import RiskMLP

# Tutti i soggetti disponibili
ALL_SUBJECTS = [
    'S00', 'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09',
    'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19'
]


def create_cv_splits(subjects, n_folds=5):
    """
    Crea n_folds split diversi per cross-validation.
    Ogni fold usa soggetti diversi per train/val/test.
    """
    n_subjects = len(subjects)
    subjects_shuffled = subjects.copy()
    np.random.shuffle(subjects_shuffled)
    
    splits = []
    fold_size = n_subjects // n_folds
    
    for i in range(n_folds):
        # Test: fold corrente
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else n_subjects
        test_subjects = subjects_shuffled[test_start:test_end]
        
        # Rimangono gli altri per train+val
        remaining = [s for s in subjects_shuffled if s not in test_subjects]
        
        # Val: 20% dei rimanenti (circa 2-3 soggetti)
        n_val = max(1, len(remaining) // 5)
        val_subjects = remaining[:n_val]
        train_subjects = remaining[n_val:]
        
        splits.append({
            'fold': i + 1,
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        })
    
    return splits


def evaluate_model(model, test_loader, threshold, device):
    """Valuta il modello sul test set e ritorna metriche"""
    import torch.nn.functional as F
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    collision_probs = all_probs[:, 2]
    
    # Apply threshold
    all_preds = np.argmax(all_probs[:, :2], axis=1)
    all_preds[collision_probs >= threshold] = 2
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1, 2], zero_division=0
    )
    
    # Collision-specific metrics
    collision_indices = (all_targets == 2)
    collision_recall = 0.0
    if np.sum(collision_indices) > 0:
        collision_recall = np.sum(all_preds[collision_indices] == 2) / np.sum(collision_indices)
    
    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'collision_recall': collision_recall
    }


def run_cross_validation(args):
    """Esegue cross-validation con diverse configurazioni di split"""
    
    # Setup seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Crea cartella principale per CV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv_dir = os.path.join('runs', f'cv_mlp_{timestamp}')
    os.makedirs(cv_dir, exist_ok=True)
    print(f"Cross-Validation directory: {cv_dir}")
    
    # Crea gli split
    splits = create_cv_splits(ALL_SUBJECTS, n_folds=args.n_folds)
    
    # Salva gli split in un file
    with open(os.path.join(cv_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    # Risultati per ogni fold
    fold_results = []
    
    for split in splits:
        fold_num = split['fold']
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{args.n_folds}")
        print(f"{'='*60}")
        print(f"Train: {split['train']}")
        print(f"Val:   {split['val']}")
        print(f"Test:  {split['test']}")
        
        # Crea directory per questo fold
        fold_dir = os.path.join(cv_dir, f'fold_{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Carica i dati
        train_loader, val_loader, test_loader, stats = create_pkl_dataloaders(
            dataset_path=args.data_path,
            train_subjects=split['train'],
            val_subjects=split['val'],
            test_subjects=split['test'],
            input_frames=1,
            output_frames=0,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode='single_frame',
            use_weighted_sampler=True,
            augment_collision=True,
            augment_factor=50
        )
        
        # Training
        print(f"\n--- Training Fold {fold_num} ---")
        train_mlp(args, train_loader, val_loader, fold_dir)
        
        # Test con visualizzazioni complete
        print(f"\n--- Testing Fold {fold_num} (with visualizations) ---")
        test_mlp(args, test_loader, fold_dir)
        
        # Calcola metriche addizionali
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RiskMLP(input_size=72, hidden_sizes=[128, 64], num_classes=3).to(device)
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'mlp_best.pth'), map_location=device, weights_only=True))
        
        metrics = evaluate_model(model, test_loader, args.threshold, device)
        
        # Salva metriche del fold
        fold_results.append({
            'fold': fold_num,
            'train_subjects': split['train'],
            'val_subjects': split['val'],
            'test_subjects': split['test'],
            'metrics': metrics
        })
        
        print(f"\nFold {fold_num} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Collision Recall: {metrics['collision_recall']:.4f}")
        print(f"  F1 Scores: Safe={metrics['f1'][0]:.4f}, Near={metrics['f1'][1]:.4f}, Coll={metrics['f1'][2]:.4f}")
    
    # Calcola statistiche aggregate
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    accuracies = [r['metrics']['accuracy'] for r in fold_results]
    collision_recalls = [r['metrics']['collision_recall'] for r in fold_results]
    f1_safe = [r['metrics']['f1'][0] for r in fold_results]
    f1_near = [r['metrics']['f1'][1] for r in fold_results]
    f1_coll = [r['metrics']['f1'][2] for r in fold_results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Validation for MLP Risk Classification")
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--n_folds', type=int, default=5, help="Number of CV folds")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs per fold")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.1, help="Collision probability threshold")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_cross_validation(args)
