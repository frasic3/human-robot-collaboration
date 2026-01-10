"""Cross-validation per MLP (frame-based) con metriche aggregate.

Questo script esegue una K-fold cross-validation sui soggetti, allenando un MLP
su frame singoli e valutando metriche standard per ogni fold + statistiche
aggregate (mean/std/CI 95%).
"""

import os
import argparse
import json
from datetime import datetime

import numpy as np
import torch

from train_mlp import train_mlp, test_mlp, CLASS_NAMES
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


def _safe_import_sklearn_metrics():
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
            average_precision_score,
        )

        return {
            'accuracy_score': accuracy_score,
            'balanced_accuracy_score': balanced_accuracy_score,
            'confusion_matrix': confusion_matrix,
            'precision_recall_fscore_support': precision_recall_fscore_support,
            'roc_auc_score': roc_auc_score,
            'average_precision_score': average_precision_score,
        }
    except Exception:
        return None


def _apply_collision_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Applica la logica: argmax tra classi (0,1) e forza classe 2 se p(collision) >= threshold."""
    if probs.ndim != 2 or probs.shape[1] < 3:
        raise ValueError('probs must have shape (N, 3)')
    collision_probs = probs[:, 2]
    preds = np.argmax(probs[:, :2], axis=1)
    preds[collision_probs >= float(threshold)] = 2
    return preds.astype(np.int64)


def _collect_targets_and_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch.nn.functional as F

    model.eval()
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_prob = np.concatenate(all_probs, axis=0) if len(all_probs) else np.zeros((0, 3), dtype=np.float32)
    return y_true, y_prob


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, class_names: list[str]):
    sk = _safe_import_sklearn_metrics()
    if sk is None:
        # Fallback minimal (senza sklearn)
        y_pred = _apply_collision_threshold(y_prob, threshold)
        acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
        return {
            'n_samples': int(len(y_true)),
            'accuracy': acc,
            'balanced_accuracy': None,
            'macro_f1': None,
            'weighted_f1': None,
            'per_class': None,
            'confusion_matrix': None,
            'confusion_matrix_norm': None,
            'collision_recall': None,
            'missed_warnings_near_to_safe': None,
            'roc_auc_ovr_macro': None,
            'pr_auc_ovr_macro': None,
        }

    y_pred = _apply_collision_threshold(y_prob, threshold)

    accuracy = float(sk['accuracy_score'](y_true, y_pred)) if len(y_true) else 0.0
    balanced_acc = float(sk['balanced_accuracy_score'](y_true, y_pred)) if len(y_true) else 0.0

    precision, recall, f1, support = sk['precision_recall_fscore_support'](
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if np.sum(support) > 0 else 0.0

    cm = sk['confusion_matrix'](y_true, y_pred, labels=[0, 1, 2]).astype(int)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)

    # Collision recall = recall della classe 2
    collision_recall = float(recall[2]) if len(recall) >= 3 else 0.0

    missed_warnings = int(np.sum((y_true == 1) & (y_pred == 0)))

    # AUC (probabilità raw, non thresholded). Gestione fold con classi mancanti.
    roc_auc = None
    pr_auc = None
    try:
        # One-hot y_true
        y_true_oh = np.eye(3, dtype=np.int64)[y_true]
        present = np.unique(y_true)
        if len(present) >= 2:
            roc_auc = float(sk['roc_auc_score'](y_true_oh, y_prob, average='macro', multi_class='ovr'))

            # PR-AUC macro OVR: media delle AP per classe presente
            per_class_ap = []
            for c in range(3):
                if np.any(y_true == c) and np.any(y_true != c):
                    per_class_ap.append(float(sk['average_precision_score'](y_true_oh[:, c], y_prob[:, c])))
            if len(per_class_ap) > 0:
                pr_auc = float(np.mean(per_class_ap))
    except Exception:
        pass

    per_class = []
    for i, name in enumerate(class_names):
        per_class.append(
            {
                'class': name,
                'index': int(i),
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            }
        )

    # Class distribution
    counts = {
        class_names[i]: int(np.sum(y_true == i)) for i in range(3)
    }

    return {
        'n_samples': int(len(y_true)),
        'class_counts': counts,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_norm': cm_norm.tolist(),
        'collision_recall': collision_recall,
        'missed_warnings_near_to_safe': missed_warnings,
        'roc_auc_ovr_macro': roc_auc,
        'pr_auc_ovr_macro': pr_auc,
    }


def _aggregate_scalar(values: list[float | None]):
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    n = int(arr.size)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    median = float(np.median(arr))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))

    # CI 95% (approssimazione normale) : mean ± 1.96 * std/sqrt(n)
    ci95 = None
    if n >= 2:
        half = 1.96 * std / float(np.sqrt(n))
        ci95 = [float(mean - half), float(mean + half)]

    return {
        'n': n,
        'mean': mean,
        'std': std,
        'median': median,
        'min': vmin,
        'max': vmax,
        'ci95': ci95,
    }


def evaluate_model(model, loader, threshold, device, class_names: list[str]):
    """Valuta il modello su un loader e ritorna metriche standard."""
    y_true, y_prob = _collect_targets_and_probs(model, loader, device)
    return _compute_metrics(y_true, y_prob, threshold, class_names)


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
    fold_results: list[dict] = []
    
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
        
        # Seed per fold (riproducibile ma diverso)
        fold_seed = int(args.seed + fold_num)

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
            use_weighted_sampler=bool(args.use_weighted_sampler),
            augment_collision=bool(args.augment_collision),
            augment_factor=int(args.augment_factor),
            seed=fold_seed,
            train_stride=int(args.train_stride),
            val_stride=int(args.val_stride),
            test_stride=int(args.test_stride),
        )
        
        # Training
        print(f"\n--- Training Fold {fold_num} ---")
        train_mlp(
            args,
            train_loader,
            val_loader,
            fold_dir,
            train_subjects=split['train'],
            val_subjects=split['val'],
        )
        
        # Test con visualizzazioni complete
        print(f"\n--- Testing Fold {fold_num} (with visualizations) ---")
        test_mlp(args, test_loader, fold_dir)
        
        # Calcola metriche addizionali
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RiskMLP(input_size=72, hidden_sizes=[128, 64], num_classes=3).to(device)
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'mlp_best.pth'), map_location=device, weights_only=True))
        
        metrics_test = evaluate_model(model, test_loader, args.threshold, device, CLASS_NAMES)
        metrics_val = evaluate_model(model, val_loader, args.threshold, device, CLASS_NAMES)
        
        # Salva metriche del fold
        fold_payload = {
            'fold': fold_num,
            'train_subjects': split['train'],
            'val_subjects': split['val'],
            'test_subjects': split['test'],
            'threshold': float(args.threshold),
            'metrics': {
                'val': metrics_val,
                'test': metrics_test,
            },
        }
        fold_results.append(fold_payload)

        with open(os.path.join(fold_dir, 'cv_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(fold_payload, f, indent=2)
        
        print(f"\nFold {fold_num} Results:")
        print(f"  [Val ] Acc={metrics_val['accuracy']:.4f} | BalAcc={metrics_val['balanced_accuracy']:.4f} | MacroF1={metrics_val['macro_f1']:.4f} | CollRecall={metrics_val['collision_recall']:.4f}")
        print(f"  [Test] Acc={metrics_test['accuracy']:.4f} | BalAcc={metrics_test['balanced_accuracy']:.4f} | MacroF1={metrics_test['macro_f1']:.4f} | CollRecall={metrics_test['collision_recall']:.4f}")
    
    # Salva tutte le metriche dei fold
    with open(os.path.join(cv_dir, 'cv_metrics_folds.json'), 'w', encoding='utf-8') as f:
        json.dump({'folds': fold_results}, f, indent=2)

    # Calcola statistiche aggregate
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    def _extract(split_name: str, key: str):
        vals = []
        for r in fold_results:
            v = r['metrics'][split_name].get(key, None)
            vals.append(v)
        return vals

    summary = {
        'n_folds': int(args.n_folds),
        'threshold': float(args.threshold),
        'data_path': str(args.data_path),
        'settings': {
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'use_weighted_sampler': bool(args.use_weighted_sampler),
            'augment_collision': bool(args.augment_collision),
            'augment_factor': int(args.augment_factor),
            'train_stride': int(args.train_stride),
            'val_stride': int(args.val_stride),
            'test_stride': int(args.test_stride),
            'seed': int(args.seed),
        },
        'val': {},
        'test': {},
    }

    for split_name in ['val', 'test']:
        summary[split_name]['accuracy'] = _aggregate_scalar(_extract(split_name, 'accuracy'))
        summary[split_name]['balanced_accuracy'] = _aggregate_scalar(_extract(split_name, 'balanced_accuracy'))
        summary[split_name]['macro_f1'] = _aggregate_scalar(_extract(split_name, 'macro_f1'))
        summary[split_name]['weighted_f1'] = _aggregate_scalar(_extract(split_name, 'weighted_f1'))
        summary[split_name]['collision_recall'] = _aggregate_scalar(_extract(split_name, 'collision_recall'))
        summary[split_name]['missed_warnings_near_to_safe'] = _aggregate_scalar(
            [float(x) if x is not None else None for x in _extract(split_name, 'missed_warnings_near_to_safe')]
        )
        summary[split_name]['roc_auc_ovr_macro'] = _aggregate_scalar(_extract(split_name, 'roc_auc_ovr_macro'))
        summary[split_name]['pr_auc_ovr_macro'] = _aggregate_scalar(_extract(split_name, 'pr_auc_ovr_macro'))

    with open(os.path.join(cv_dir, 'cv_metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # CSV compatto per metriche principali (test)
    csv_path = os.path.join(cv_dir, 'cv_metrics_test.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('fold,accuracy,balanced_accuracy,macro_f1,weighted_f1,collision_recall,roc_auc_ovr_macro,pr_auc_ovr_macro,missed_warnings_near_to_safe\n')
        for r in fold_results:
            m = r['metrics']['test']
            f.write(
                f"{r['fold']},{m.get('accuracy', '')},{m.get('balanced_accuracy', '')},{m.get('macro_f1', '')},{m.get('weighted_f1', '')},{m.get('collision_recall', '')},{m.get('roc_auc_ovr_macro', '')},{m.get('pr_auc_ovr_macro', '')},{m.get('missed_warnings_near_to_safe', '')}\n"
            )

    # Print summary (val/test) con mean ± std per le metriche principali
    def _print_mean_std(split_name: str, metric_key: str, label: str):
        pack = summary[split_name].get(metric_key, None)
        if pack is None:
            return
        ci = pack.get('ci95', None)
        ci_str = f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ''
        print(f"{split_name.capitalize()} {label}: {pack['mean']:.4f} ± {pack['std']:.4f}{ci_str}")

    _print_mean_std('val', 'accuracy', 'Accuracy')
    _print_mean_std('val', 'collision_recall', 'Collision Recall')
    _print_mean_std('test', 'accuracy', 'Accuracy')
    _print_mean_std('test', 'collision_recall', 'Collision Recall')

    # Extra (opzionali)
    _print_mean_std('test', 'macro_f1', 'Macro-F1')


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
    parser.add_argument('--use_weighted_sampler', action='store_true', help='Use WeightedRandomSampler for training')
    parser.add_argument('--augment_collision', action='store_true', help='Augment Collision class in training set')
    parser.add_argument('--augment_factor', type=int, default=50, help='Augmentation factor for Collision samples')
    parser.add_argument('--train_stride', type=int, default=10)
    parser.add_argument('--val_stride', type=int, default=20)
    parser.add_argument('--test_stride', type=int, default=30)
    args = parser.parse_args()
    
    run_cross_validation(args)
