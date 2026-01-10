"""Evaluate a saved checkpoint (MLP or LSTM) without retraining.

Examples:
  # MLP, official split
  python eval_checkpoint.py --model mlp --data_path datasets/3d_skeletons_risk \
      --checkpoint runs/mlp_20251230_115733/mlp_best.pth --threshold 0.10

  # LSTM, official split
  python eval_checkpoint.py --model lstm --data_path datasets/3d_skeletons_risk \
      --checkpoint runs/lstm_20251230_120405/lstm_best.pth --threshold 0.10

  # Cross-validation fold (uses the same train/val/test subjects from splits.json)
  python eval_checkpoint.py --model mlp --data_path datasets/3d_skeletons_risk \
      --checkpoint runs/cv_mlp_20260109_202231/fold_1/mlp_best.pth \
      --splits_json runs/cv_mlp_20260109_202231/splits.json --fold 1

Notes:
- Z-score stats are computed ONLY on the provided train subjects and applied to val/test.
- Output artifacts are saved to --out_dir (defaults to the checkpoint folder).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.mlp import RiskMLP
from models.lstm import RiskLSTM
from utils.pkl_data_loader import create_pkl_dataloaders

# Reuse plotting utilities
from utils.visualization import (
    save_confusion_matrix,
    save_temporal_heatmap,
    save_metrics_summary,
    compute_per_timestep_metrics,
    compute_mean_anticipation_time,
    compute_event_level_recall,
    compute_flicker_rate,
)


DEFAULT_CLASS_NAMES = ['Safe', 'Near-Collision', 'Collision']


def _json_default(obj: Any):
    """JSON serializer for numpy / torch objects."""
    try:
        import numpy as _np

        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass

    try:
        import torch as _torch

        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # Fallback: let json raise the TypeError with a helpful message
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def _resolve_checkpoint_path(model: str, checkpoint: str) -> str:
    """Allow passing either a .pth file or a run directory.

    If a directory is provided, we assume it contains the standard checkpoint name.
    """
    checkpoint = str(checkpoint)
    if os.path.isdir(checkpoint):
        expected = 'mlp_best.pth' if str(model).lower() == 'mlp' else 'lstm_best.pth'
        return os.path.join(checkpoint, expected)
    return checkpoint


def _load_state_dict(path: str, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch versions
        return torch.load(path, map_location=device)


def _resolve_splits(
    args: argparse.Namespace,
    default_train: list[str],
    default_val: list[str],
    default_test: list[str],
) -> tuple[list[str], list[str], list[str]]:
    if args.splits_json:
        with open(args.splits_json, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        fold = int(args.fold)
        match = None
        for s in splits:
            if int(s.get('fold', -1)) == fold:
                match = s
                break
        if match is None:
            raise ValueError(f"Fold {fold} not found in {args.splits_json}")
        return list(match['train']), list(match['val']), list(match['test'])

    train_subjects = list(args.train_subjects) if args.train_subjects else list(default_train)
    val_subjects = list(args.val_subjects) if args.val_subjects else list(default_val)
    test_subjects = list(args.test_subjects) if args.test_subjects else list(default_test)
    return train_subjects, val_subjects, test_subjects


def _ensure_out_dir(args: argparse.Namespace) -> str:
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def eval_mlp(args: argparse.Namespace) -> dict[str, Any]:
    from train_mlp import TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS, CLASS_NAMES

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = _ensure_out_dir(args)

    train_subjects, val_subjects, test_subjects = _resolve_splits(
        args,
        default_train=list(TRAIN_SUBJECTS),
        default_val=list(VAL_SUBJECTS),
        default_test=list(TEST_SUBJECTS),
    )

    _, _, test_loader, _ = create_pkl_dataloaders(
        dataset_path=args.data_path,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        input_frames=1,
        output_frames=0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        mode='single_frame',
        use_weighted_sampler=False,
        augment_collision=False,
        augment_factor=0,
        seed=int(args.seed) if args.seed is not None else None,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.test_stride,
    )

    model = RiskMLP(input_size=72, hidden_sizes=[128, 64], num_classes=3).to(device)
    state = _load_state_dict(args.checkpoint, device)
    model.load_state_dict(state)
    model.eval()

    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_prob = np.concatenate(all_probs, axis=0) if len(all_probs) else np.zeros((0, 3), dtype=np.float32)

    collision_probs = y_prob[:, 2] if y_prob.size else np.zeros((0,), dtype=np.float32)
    y_pred = np.argmax(y_prob[:, :2], axis=1) if y_prob.size else np.zeros((0,), dtype=np.int64)
    y_pred = y_pred.astype(np.int64)
    y_pred[collision_probs >= float(args.threshold)] = 2

    acc = float(np.mean(y_pred == y_true)) if y_true.size else 0.0

    info = (
        f"Checkpoint: {os.path.relpath(args.checkpoint)} | "
        f"Threshold: {args.threshold} | "
        f"Test Subjects: {test_subjects}"
    )

    # Minimal artifact: confusion matrix (official eval file)
    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, out_dir, 'eval_confusion_matrix.png', info_text=info)

    results = {
        'model': 'mlp',
        'checkpoint': os.path.abspath(args.checkpoint),
        'data_path': os.path.abspath(args.data_path),
        'threshold': float(args.threshold),
        'accuracy': acc,
        'subjects': {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects,
        },
        'out_dir': os.path.abspath(out_dir),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }

    with open(os.path.join(out_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=_json_default)

    print(f"[MLP] accuracy={acc:.4f} | saved to: {out_dir}")
    return results


def eval_lstm(args: argparse.Namespace) -> dict[str, Any]:
    import train_lstm as lstm_train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = _ensure_out_dir(args)

    train_subjects, val_subjects, test_subjects = _resolve_splits(
        args,
        default_train=list(lstm_train.TRAIN_SUBJECTS),
        default_val=list(lstm_train.VAL_SUBJECTS),
        default_test=list(lstm_train.TEST_SUBJECTS),
    )

    input_frames = int(args.input_frames)
    output_frames = int(args.output_frames)

    # IMPORTANT: match train_lstm.py defaults unless explicitly overridden.
    # Different strides change both the evaluated test set and the normalization stats.
    train_stride = args.train_stride if args.train_stride is not None else lstm_train.TRAIN_STRIDE
    val_stride = args.val_stride if args.val_stride is not None else lstm_train.VAL_STRIDE
    test_stride = args.test_stride if args.test_stride is not None else lstm_train.TEST_STRIDE

    _, _, test_loader, _ = create_pkl_dataloaders(
        dataset_path=args.data_path,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        mode='sequence',
        use_weighted_sampler=False,
        augment_collision=False,
        augment_factor=0,
        seed=int(args.seed) if args.seed is not None else None,
        train_stride=train_stride,
        val_stride=val_stride,
        test_stride=test_stride,
    )

    model = RiskLSTM(
        input_size=72,
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        num_classes=3,
        output_frames=output_frames,
        dropout=float(args.dropout),
    ).to(device)

    state = _load_state_dict(args.checkpoint, device)
    model.load_state_dict(state)
    model.eval()

    # Same loss family as training (temporal-weighted)
    risk_weights = tuple(float(x) for x in args.risk_weights)
    criterion = lstm_train.TemporalWeightedCrossEntropy(
        class_weights=risk_weights,
        output_frames=output_frames,
        urgent_frames=int(args.urgent_frames),
        decay_factor=float(args.decay_factor),
        device=device,
    )

    avg_loss, all_targets, all_probs = lstm_train._collect_targets_and_probs(
        args,
        model,
        test_loader,
        device,
        criterion=criterion,
        desc='[Eval-Test]',
        show_progress=True,
    )

    all_preds = lstm_train.apply_collision_threshold(all_probs, float(args.threshold))
    acc = float(np.mean(all_preds == all_targets)) if all_targets.size else 0.0

    k = int(args.eval_frames)
    min_r, per_class = lstm_train.min_class_recall_first_k(all_targets, all_preds, k)
    mean_r = lstm_train.mean_class_recall_first_k(all_targets, all_preds, k)

    # Minimal artifacts (requested): temporal heatmap only
    save_temporal_heatmap(
        all_targets,
        all_preds,
        output_frames,
        out_dir,
        filename='temporal_heatmap_eval_test.png',
    )

    # Eval metrics summary image (like training "metrics_summary"), but titled for eval.
    timestep_metrics = compute_per_timestep_metrics(all_targets, all_preds, output_frames)
    mat_stats = compute_mean_anticipation_time(all_targets, all_preds)
    event_stats = compute_event_level_recall(all_targets, all_preds, min_consecutive=1)
    flicker_stats = compute_flicker_rate(all_preds)
    save_metrics_summary(
        timestep_metrics,
        mat_stats,
        event_stats,
        flicker_stats,
        out_dir,
        filename='eval_metrics_summary.png',
        title='Eval Metrics Summary',
        header='EVAL METRICS SUMMARY',
    )

    results = {
        'model': 'lstm',
        'checkpoint': os.path.abspath(args.checkpoint),
        'data_path': os.path.abspath(args.data_path),
        'threshold': float(args.threshold),
        'risk_weights': list(risk_weights),
        'loss': float(avg_loss) if avg_loss is not None else None,
        'accuracy': acc,
        'eval_frames': k,
        'min_recall_first_k': float(min_r),
        'mean_recall_first_k': float(mean_r),
        'per_class_first_k': per_class,
        'subjects': {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects,
        },
        'config': {
            'input_frames': input_frames,
            'output_frames': output_frames,
            'hidden_size': int(args.hidden_size),
            'num_layers': int(args.num_layers),
            'dropout': float(args.dropout),
            'urgent_frames': int(args.urgent_frames),
            'decay_factor': float(args.decay_factor),
        },
        'out_dir': os.path.abspath(out_dir),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }

    with open(os.path.join(out_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=_json_default)

    print(
        f"[LSTM] acc={acc:.4f} | loss={results['loss']} | "
        f"min@{k}={min_r:.3f} mean@{k}={mean_r:.3f} | saved to: {out_dir}"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Evaluate MLP/LSTM from a checkpoint (no training).')

    p.add_argument('--model', type=str, choices=['mlp', 'lstm'], required=True)
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--out_dir', type=str, default=None)

    p.add_argument('--threshold', type=float, default=0.1)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)

    # Optional: reuse CV splits
    p.add_argument('--splits_json', type=str, default=None, help='Path to runs/.../splits.json')
    p.add_argument('--fold', type=int, default=1, help='Fold index to pick from splits.json')

    # Optional manual subjects (ignored if --splits_json is provided)
    p.add_argument('--train_subjects', nargs='*', default=None)
    p.add_argument('--val_subjects', nargs='*', default=None)
    p.add_argument('--test_subjects', nargs='*', default=None)

    # Optional stride overrides (if None, loader defaults apply)
    p.add_argument('--train_stride', type=int, default=None)
    p.add_argument('--val_stride', type=int, default=None)
    p.add_argument('--test_stride', type=int, default=None)

    # LSTM-only params (must match the checkpoint architecture)
    p.add_argument('--input_frames', type=int, default=10)
    p.add_argument('--output_frames', type=int, default=25)
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.5)

    # LSTM loss/eval params
    p.add_argument('--risk_weights', type=float, nargs=3, default=[1.0, 1.0, 5.0])
    p.add_argument('--urgent_frames', type=int, default=15)
    p.add_argument('--decay_factor', type=float, default=0.3)
    p.add_argument('--eval_frames', type=int, default=15)

    return p


def main() -> None:
    args = build_parser().parse_args()

    # Accept passing a run directory instead of a .pth file
    args.checkpoint = _resolve_checkpoint_path(args.model, args.checkpoint)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"data_path not found: {args.data_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    if args.model == 'mlp':
        eval_mlp(args)
    else:
        eval_lstm(args)


if __name__ == '__main__':
    main()
