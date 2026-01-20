"""Monte Carlo Cross-validation runner for LSTM forecasting.

Subject-level CV (like cross_validation.py):
- Picks a test fold of subjects.
- Uses ~20% of remaining subjects as validation.
- Trains LSTM on train subjects and evaluates on the held-out test subjects.

Saves:
- runs/cv_lstm_<timestamp>/splits.json
- runs/cv_lstm_<timestamp>/fold_i/metrics.json

Per-fold it also saves the same test artifacts produced by train_lstm.py
(plots) for a definitive evaluation.
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from models.lstm import RiskLSTM
from utils.pkl_data_loader import create_pkl_dataloaders

import train_lstm as lstm_train


ALL_SUBJECTS = [
    'S00', 'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09',
    'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19'
]


def create_cv_splits(subjects, n_folds=5, seed=42):
    """Create n_folds splits for cross-validation (reproducible)."""
    rng = np.random.default_rng(int(seed))
    subjects_shuffled = list(subjects)
    rng.shuffle(subjects_shuffled)

    splits = []
    fold_size = max(1, len(subjects_shuffled) // int(n_folds))

    for i in range(int(n_folds)):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else len(subjects_shuffled)
        test_subjects = subjects_shuffled[test_start:test_end]

        remaining = [s for s in subjects_shuffled if s not in test_subjects]
        n_val = max(1, len(remaining) // 5)  # ~20%
        val_subjects = remaining[:n_val]
        train_subjects = remaining[n_val:]

        splits.append({
            'fold': i + 1,
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects,
        })

    return splits


def evaluate_loader_basic(args, model, loader, device, prefix=''):
    """Evaluate a loader (silent) and return basic metrics + arrays."""
    model.eval()
    risk_weights = list(args.risk_weights)
    criterion = lstm_train.TemporalWeightedCrossEntropy(
        class_weights=risk_weights,
        output_frames=lstm_train.OUTPUT_FRAMES,
        urgent_frames=lstm_train.URGENT_FRAMES,
        decay_factor=lstm_train.DECAY_FACTOR,
        device=device,
    )

    avg_loss, all_targets, all_probs = lstm_train._collect_targets_and_probs(
        args,
        model,
        loader,
        device,
        criterion=criterion,
        desc=prefix,
        show_progress=False,
    )
    all_preds = lstm_train.apply_collision_threshold(all_probs, float(args.threshold))
    acc = float(np.mean(all_preds == all_targets))
    k = int(getattr(args, 'eval_frames', lstm_train.EVAL_FRAMES))
    min_r, per_class = lstm_train.min_class_recall_first_k(all_targets, all_preds, k)
    mean_r = lstm_train.mean_class_recall_first_k(all_targets, all_preds, k)
    return {
        'loss': float(avg_loss) if avg_loss is not None else None,
        'accuracy': float(acc),
        'min_recall_first_k': float(min_r),
        'mean_recall_first_k': float(mean_r),
        'per_class_first_k': per_class,
        'eval_frames': int(k),
        'arrays': {
            'targets': all_targets,
            'preds': all_preds,
            'probs': all_probs,
        },
    }


def run_cv(args) -> None:
    seed = int(args.seed)

    # Global seeds (reproducible splits)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv_dir = os.path.join('runs', f'cv_lstm_{timestamp}')
    os.makedirs(cv_dir, exist_ok=True)

    splits = create_cv_splits(ALL_SUBJECTS, n_folds=args.n_folds, seed=seed)
    with open(os.path.join(cv_dir, 'splits.json'), 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)

    fold_results = []

    for split in splits:
        fold = int(split['fold'])
        fold_dir = os.path.join(cv_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        print('\n' + '=' * 70)
        print(f"FOLD {fold}/{args.n_folds}")
        print('=' * 70)
        print('Train:', split['train'])
        print('Val:  ', split['val'])
        print('Test: ', split['test'])

        # Per-fold seed to keep training fully reproducible (but different across folds)
        fold_seed = seed + fold

        # Build loaders
        train_loader, val_loader, test_loader, _ = create_pkl_dataloaders(
            dataset_path=args.data_path,
            train_subjects=split['train'],
            val_subjects=split['val'],
            test_subjects=split['test'],
            input_frames=lstm_train.INPUT_FRAMES,
            output_frames=lstm_train.OUTPUT_FRAMES,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode='sequence',
            use_weighted_sampler=True,
            augment_collision=bool(int(args.augment_factor) > 0),
            augment_factor=int(args.augment_factor),
            sampler_strategy=lstm_train.SAMPLER_STRATEGY,
            sampler_collision_scale=lstm_train.SAMPLER_COLLISION_SCALE,
            sampler_collision_power=lstm_train.SAMPLER_COLLISION_POWER,
            sampler_min_weight=lstm_train.SAMPLER_MIN_WEIGHT,
            train_stride=lstm_train.TRAIN_STRIDE,
            val_stride=lstm_train.VAL_STRIDE,
            test_stride=lstm_train.TEST_STRIDE,
            seed=fold_seed,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        model = RiskLSTM(
            input_size=72,
            hidden_size=512,
            num_layers=2,
            num_classes=3,
            output_frames=lstm_train.OUTPUT_FRAMES,
        ).to(device)

        # Args namespace for train_lstm
        train_args = argparse.Namespace(
            epochs=int(args.epochs),
            lr=float(args.lr),
            patience=int(args.patience),
            threshold=float(args.threshold),
            risk_weights=list(args.risk_weights),
            augment_factor=int(args.augment_factor),
            eval_frames=int(args.eval_frames),
            seed=fold_seed,
        )

        # Seed determinism inside this fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        random.seed(fold_seed)

        # Train
        lstm_train.train_lstm(train_args, model, train_loader, val_loader, fold_dir, device)

        # Evaluate on held-out test subjects
        ckpt = os.path.join(fold_dir, 'lstm_best.pth')
        if not os.path.exists(ckpt):
            raise RuntimeError(f"Missing checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        # Evaluate (silent). For TEST also save full artifact pack.
        metrics_val_pack = evaluate_loader_basic(train_args, model, val_loader, device, prefix='[Val]')
        metrics_test_pack = evaluate_loader_basic(train_args, model, test_loader, device, prefix='[Test]')

        eval_pack = lstm_train.comprehensive_evaluation(
            metrics_test_pack['arrays']['targets'],
            metrics_test_pack['arrays']['preds'],
            metrics_test_pack['arrays']['probs'],
            lstm_train.OUTPUT_FRAMES,
            fold_dir,
            prefix='test_',
        )

        test_results = {
            'test_loss': metrics_test_pack['loss'],
            'test_accuracy': metrics_test_pack['accuracy'],
            'min_recall_first_k': metrics_test_pack['min_recall_first_k'],
            'mean_recall_first_k': metrics_test_pack['mean_recall_first_k'],
            'per_class_first_k': metrics_test_pack['per_class_first_k'],
            **eval_pack,
        }

        fold_record = {
            'fold': fold,
            'train_subjects': split['train'],
            'val_subjects': split['val'],
            'test_subjects': split['test'],
            'config': {
                'threshold': float(args.threshold),
                'risk_weights': list(args.risk_weights),
                'augment_factor': int(args.augment_factor),
                'lr': float(args.lr),
                'epochs': int(args.epochs),
                'patience': int(args.patience),
                'eval_frames': int(args.eval_frames),
                'seed': int(fold_seed),
            },
            'val': {
                'loss': metrics_val_pack['loss'],
                'accuracy': metrics_val_pack['accuracy'],
                'min_recall_first_k': metrics_val_pack['min_recall_first_k'],
                'mean_recall_first_k': metrics_val_pack['mean_recall_first_k'],
                'per_class_first_k': metrics_val_pack['per_class_first_k'],
                'eval_frames': metrics_val_pack['eval_frames'],
            },
            'test': {
                'loss': metrics_test_pack['loss'],
                'accuracy': metrics_test_pack['accuracy'],
                'min_recall_first_k': metrics_test_pack['min_recall_first_k'],
                'mean_recall_first_k': metrics_test_pack['mean_recall_first_k'],
                'per_class_first_k': metrics_test_pack['per_class_first_k'],
                'eval_frames': metrics_test_pack['eval_frames'],
            },
        }

        with open(os.path.join(fold_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(fold_record, f, indent=2)

        fold_results.append(fold_record)

        print("Fold test:",
              f"min@{args.eval_frames}={fold_record['test']['min_recall_first_k']:.3f}",
              f"mean@{args.eval_frames}={fold_record['test']['mean_recall_first_k']:.3f}",
              f"acc={fold_record['test']['accuracy']:.3f}")

    # Summary
    def _collect(key_path: list[str]) -> list[float]:
        out = []
        for r in fold_results:
            d = r
            for k in key_path:
                d = d[k]
            out.append(float(d))
        return out

    test_min = _collect(['test', 'min_recall_first_k'])
    test_mean = _collect(['test', 'mean_recall_first_k'])
    test_acc = _collect(['test', 'accuracy'])
    test_loss = _collect(['test', 'loss'])

    summary = {
        'n_folds': int(args.n_folds),
        'config': {
            'threshold': float(args.threshold),
            'risk_weights': list(args.risk_weights),
            'augment_factor': int(args.augment_factor),
            'lr': float(args.lr),
            'epochs': int(args.epochs),
            'patience': int(args.patience),
            'eval_frames': int(args.eval_frames),
            'seed': int(args.seed),
        },
        'test': {
            'min_recall_first_k_mean': float(np.mean(test_min)),
            'min_recall_first_k_std': float(np.std(test_min)),
            'mean_recall_first_k_mean': float(np.mean(test_mean)),
            'mean_recall_first_k_std': float(np.std(test_mean)),
            'accuracy_mean': float(np.mean(test_acc)),
            'accuracy_std': float(np.std(test_acc)),
            'loss_mean': float(np.mean(test_loss)),
            'loss_std': float(np.std(test_loss)),
        },
    }

    print('\n' + '=' * 70)
    print('CROSS-VALIDATION SUMMARY (TEST)')
    print('=' * 70)
    print(f"min@{args.eval_frames}: {summary['test']['min_recall_first_k_mean']:.3f} ± {summary['test']['min_recall_first_k_std']:.3f}")
    print(f"mean@{args.eval_frames}: {summary['test']['mean_recall_first_k_mean']:.3f} ± {summary['test']['mean_recall_first_k_std']:.3f}")
    print(f"acc: {summary['test']['accuracy_mean']:.3f} ± {summary['test']['accuracy_std']:.3f}")
    print(f"saved: {cv_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Cross-Validation for LSTM Risk Forecasting')
    p.add_argument('--data_path', type=str, default='datasets/3d_skeletons_risk')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--threshold', type=float, default=0.10)
    p.add_argument('--risk_weights', type=float, nargs=3, default=[1.0, 1.0, 5.0])
    p.add_argument('--augment_factor', type=int, default=30)
    p.add_argument('--eval_frames', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    run_cv(args)
