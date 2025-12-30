import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import random
from tqdm import tqdm
import numpy as np
from datetime import datetime
from models.lstm import RiskLSTM
from utils.pkl_data_loader import create_pkl_dataloaders
from utils.visualization import (
    save_training_curves,
    comprehensive_evaluation,
)

TRAIN_SUBJECTS = [
    'S01', 'S05', 'S06', 'S07', 'S08', 'S09',
    'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17'
]

VAL_SUBJECTS = ['S00', 'S04']

TEST_SUBJECTS = ['S02', 'S03', 'S18', 'S19']

# Risk Configuration
RISK_WEIGHTS = [1.0, 1.0, 5.0]  # Safe, Near, Collision
CLASS_NAMES = ['Safe', 'Near-Collision', 'Collision']
BATCH_SIZE = 64
INPUT_FRAMES = 10
OUTPUT_FRAMES = 25
EVAL_FRAMES = 15
SAMPLER_STRATEGY = 'balanced_batch'
SAMPLER_COLLISION_SCALE = 0.0
SAMPLER_COLLISION_POWER = 1.0
SAMPLER_MIN_WEIGHT = 1.0
URGENT_FRAMES = 15
DECAY_FACTOR = 0.3
TRAIN_STRIDE = 10
VAL_STRIDE = 1
TEST_STRIDE = 1


class TemporalWeightedCrossEntropy(nn.Module):

    def __init__(
        self,
        class_weights,
        output_frames=25,
        urgent_frames=15,
        decay_factor=0.3,
        device='cpu',
    ):

        super().__init__()
        self.class_weights = torch.tensor(class_weights).float().to(device)
        self.output_frames = int(output_frames)
        temporal_weights = torch.ones(self.output_frames, device=device)
        urgent_frames = int(max(0, min(urgent_frames, self.output_frames)))
        if urgent_frames < self.output_frames:
            temporal_weights[urgent_frames:] = float(decay_factor)
        self.temporal_weights = temporal_weights

    def forward(self, inputs, targets):

        # inputs: (B, P, C)
        # targets: (B, P)
        ce_loss = F.cross_entropy(
            inputs.view(-1, inputs.shape[-1]),
            targets.view(-1),
            weight=self.class_weights,
            reduction='none',
        )
        ce_loss = ce_loss.view(inputs.shape[0], self.output_frames)
        weighted_loss = ce_loss * self.temporal_weights
        return weighted_loss.mean()


def mean_class_recall_first_k(targets: np.ndarray, preds: np.ndarray, k: int) -> float:
    """Compute mean-class recall over the first k steps."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    k = int(max(1, k))
    t = targets[:, :k].reshape(-1)
    p = preds[:, :k].reshape(-1)
    recalls = []
    for c in range(3):

        mask = (t == c)
        if mask.sum() > 0:
            recall = np.sum((p == c) & mask) / mask.sum()
            recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0


def min_class_recall_first_k(targets: np.ndarray, preds: np.ndarray, k: int) -> tuple[float, dict]:
    """Compute min-class recall over the first k steps.
    Returns:
        (min_recall, per_class_recalls_dict)
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    k = int(max(1, k))
    t = targets[:, :k].reshape(-1)
    p = preds[:, :k].reshape(-1)
    recalls = {}
    values = []
    for c, name in enumerate(CLASS_NAMES):

        mask = (t == c)
        denom = int(mask.sum())
        if denom == 0:
            r = 0.0
        else:
            r = float((p[mask] == c).sum()) / float(denom)
        recalls[name] = r
        values.append(r)
    return float(min(values)), recalls


def _write_tuning_metrics(run_dir: str, payload: dict) -> None:
    import json
    path = os.path.join(run_dir, 'tuning_metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


# collect targets and probs

def _collect_targets_and_probs(args, model, loader, device, criterion=None, desc='', show_progress=True):

    """Collect (N,P) targets and (N,P,3) probabilities for a loader."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    with torch.no_grad():

        for inputs, output_risks in tqdm(loader, desc=desc, disable=not show_progress):

            inputs = inputs.to(device)
            targets = output_risks.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=2)
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    avg_loss = (total_loss / len(loader)) if (criterion is not None and len(loader) > 0) else None
    return avg_loss, all_targets, all_probs

# save val artifacts

def save_val_artifacts(args, model, val_loader, run_dir, device):

    """Save the same validation artifacts we used to produce (plots + summaries)."""
    best_model_path = os.path.join(run_dir, 'lstm_best.pth')
    if not os.path.exists(best_model_path):

        print(f"Warning: Best model not found at {best_model_path}; skipping val artifacts")
        return None
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    risk_weights = tuple(getattr(args, 'risk_weights', RISK_WEIGHTS))
    criterion = TemporalWeightedCrossEntropy(
        class_weights=risk_weights,
        output_frames=OUTPUT_FRAMES,
        urgent_frames=URGENT_FRAMES,
        decay_factor=DECAY_FACTOR,
        device=device,
    )
    # Silent: this pass is only to save plots/metrics
    avg_val_loss, all_targets, all_probs = _collect_targets_and_probs(
        args, model, val_loader, device, criterion=criterion, desc='[Val]', show_progress=False
    )
    all_preds = apply_collision_threshold(all_probs, args.threshold)
    # Save all plots and summaries (silent by default)
    eval_pack = comprehensive_evaluation(
        all_targets,
        all_preds,
        all_probs,
        OUTPUT_FRAMES,
        run_dir,
        prefix='val_',
    )
    return {
        'val_loss': float(avg_val_loss) if avg_val_loss is not None else None,
        'val_accuracy': float(np.mean(all_preds == all_targets)),
        **eval_pack,
    }

# training 

def apply_collision_threshold(probs, threshold):

    """
    Applica threshold sulla probabilità di Collision.
    Args:
        probs: (N, P, 3) probabilità per ogni classe
        threshold: soglia per classe Collision
    Returns:
        (N, P) predizioni con threshold applicato
    """
    N, P, _ = probs.shape
    # Predizione di default: argmax su Safe e Near
    preds = np.argmax(probs[:, :, :2], axis=2)  # (N, P)
    # Override con Collision se prob >= threshold
    collision_probs = probs[:, :, 2]  # (N, P)
    preds[collision_probs >= threshold] = 2
    return preds


def train_lstm(args, model, train_loader, val_loader, run_dir, device):

    """Main training loop con weighted loss e threshold."""
    risk_weights = tuple(getattr(args, 'risk_weights', RISK_WEIGHTS))
    print(f"Using device: {device}")
    print(f"Collision threshold: {args.threshold}")
    print(f"Risk Weights: {risk_weights}")
    # Loss (always temporal-weighted)
    criterion = TemporalWeightedCrossEntropy(
        class_weights=risk_weights,
        output_frames=OUTPUT_FRAMES,
        urgent_frames=URGENT_FRAMES,
        decay_factor=DECAY_FACTOR,
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_min_recall_first = -1.0
    best_mean_recall_first = -1.0
    best_epoch = 0
    # Training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    # Best model data
    best_val_preds = None
    best_val_targets = None
    best_val_probs = None
    epochs_no_improve = 0
    # Assicurati che la cartella run_dir esista
    os.makedirs(run_dir, exist_ok=True)
    try:
        for epoch in range(args.epochs):

            # --- TRAIN ---
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
            for inputs, output_risks in pbar:
                inputs = inputs.to(device)  # (B, T, 72)
                targets = output_risks.to(device)  # (B, P)
                optimizer.zero_grad()
                outputs = model(inputs)  # (B, P, 3)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # Apply threshold per training accuracy
                probs = F.softmax(outputs, dim=2).detach().cpu().numpy()
                preds = apply_collision_threshold(probs, args.threshold)
                targets_np = targets.cpu().numpy()
                total += targets.numel()
                correct += np.sum(preds == targets_np)
                pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            avg_train_loss = train_loss / len(train_loader)
            train_acc = correct / total
            # --- VALIDATION ---
            model.eval()
            val_loss = 0.0
            val_targets_list = []
            val_probs_list = []
            with torch.no_grad():

                for inputs, output_risks in val_loader:
                    inputs = inputs.to(device)
                    targets = output_risks.to(device)
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=2)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_targets_list.append(targets.cpu().numpy())
                    val_probs_list.append(probs.cpu().numpy())
            # Concatenate all batches
            val_targets_array = np.concatenate(val_targets_list, axis=0)  # (N, P)
            val_probs_array = np.concatenate(val_probs_list, axis=0)  # (N, P, 3)
            # Apply threshold (or just argmax)
            val_preds_array = apply_collision_threshold(val_probs_array, args.threshold)
            # Calculate metrics
            avg_val_loss = val_loss / len(val_loader)
            val_acc = np.mean(val_preds_array == val_targets_array)
            k = EVAL_FRAMES
            min_recall_first, per_class = min_class_recall_first_k(val_targets_array, val_preds_array, k)
            mean_recall_first = mean_class_recall_first_k(val_targets_array, val_preds_array, k)
            print(
                f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f} | "
                f"min-recall@{k}={min_recall_first:.2%} | mean-recall@{k}={mean_recall_first:.2%} | "
                f"(Safe={per_class['Safe']:.1%}, Near={per_class['Near-Collision']:.1%}, Coll={per_class['Collision']:.1%})"
            )
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            # Hierarchical selection: min-recall@k > mean-recall@k > loss
            save_model = False
            if min_recall_first > best_min_recall_first:
                save_model = True
            elif min_recall_first == best_min_recall_first:
                if mean_recall_first > best_mean_recall_first:
                    save_model = True
                elif mean_recall_first == best_mean_recall_first:
                    if avg_val_loss < best_val_loss:
                        save_model = True

            if save_model:
                best_val_loss = avg_val_loss
                best_min_recall_first = float(min_recall_first)
                best_mean_recall_first = float(mean_recall_first)
                best_val_preds = val_preds_array.copy()
                best_val_targets = val_targets_array.copy()
                best_val_probs = val_probs_array.copy()
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(run_dir, 'lstm_best.pth'))
                print("Saved best model")

                _write_tuning_metrics(
                    run_dir,
                    {
                        'best_epoch': int(best_epoch),
                        'best_val_loss': float(best_val_loss),
                        'best_min_recall': float(best_min_recall_first),
                        'best_mean_recall': float(best_mean_recall_first),
                        'eval_frames': int(k),
                        'threshold': float(args.threshold),
                        'risk_weights': list(RISK_WEIGHTS),
                        'augment_factor': int(getattr(args, 'augment_factor', 0)),
                    }
                )
            else:
                epochs_no_improve += 1
                # Early stopping
                if epochs_no_improve >= args.patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs (no improvement for {args.patience} epochs)")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Proceeding with best checkpoint collected so far...")
    # --- SAVE TRAINING VISUALIZATIONS ---
    save_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, run_dir)
    print(f"\nTraining complete. Best model at epoch {best_epoch}")
    return best_epoch


def test_lstm(args, model, test_loader, run_dir, device):

    """Test phase."""
    # Load best model
    best_model_path = os.path.join(run_dir, 'lstm_best.pth')
    if not os.path.exists(best_model_path):

        print(f"Error: Best model not found at {best_model_path}")
        return
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    # Loss (match training): always temporal-weighted
    risk_weights = tuple(getattr(args, 'risk_weights', RISK_WEIGHTS))
    criterion = TemporalWeightedCrossEntropy(
        class_weights=risk_weights,
        output_frames=OUTPUT_FRAMES,
        urgent_frames=URGENT_FRAMES,
        decay_factor=DECAY_FACTOR,
        device=device,
    )
    avg_test_loss, all_targets, all_probs = _collect_targets_and_probs(
        args, model, test_loader, device, criterion=criterion, desc='[Test]', show_progress=True
    )
    # Apply threshold
    all_preds = apply_collision_threshold(all_probs, args.threshold)
    # Calculate basic metrics
    test_acc = np.mean(all_preds == all_targets)
    k = EVAL_FRAMES
    min_first, per_class = min_class_recall_first_k(all_targets, all_preds, k)
    mean_first = mean_class_recall_first_k(all_targets, all_preds, k)
    print(
        f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.4f} | "
        f"min-recall@{k}: {min_first:.2%} | mean-recall@{k}: {mean_first:.2%}"
    )
    # Save all plots and summaries (silent by default)
    eval_pack = comprehensive_evaluation(
        all_targets,
        all_preds,
        all_probs,
        OUTPUT_FRAMES,
        run_dir,
        prefix='test_',
    )
    results = {
        'test_loss': float(avg_test_loss) if avg_test_loss is not None else None,
        'test_accuracy': float(test_acc),
        'min_recall_first_k': float(min_first),
        'mean_recall_first_k': float(mean_first),
        'per_class_first_k': per_class,
        **eval_pack,
    }
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'c:\Users\Proprietario\Desktop\human-robot-collaboration\datasets\3d_skeletons_risk')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--threshold', type=float, default=0.1, help='Collision probability threshold (lower = more sensitive)')
    parser.add_argument('--risk_weights', type=float, nargs=3, default=RISK_WEIGHTS, help='Loss weights for (Safe Near Collision), e.g. --risk_weights 1 4 10')
    parser.add_argument('--augment_factor', type=int, default=30, help='0 disables augmentation; otherwise number of augmented copies per collision sequence')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run folder name under runs/')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Reproducibility
    seed = int(getattr(args, 'seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Setup run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = args.run_name.strip() if isinstance(args.run_name, str) and args.run_name.strip() else f'lstm_{timestamp}'
    run_dir = os.path.join('runs', run_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    train_loader, val_loader, test_loader, stats = create_pkl_dataloaders(
        dataset_path=args.data_path,
        train_subjects=TRAIN_SUBJECTS,
        val_subjects=VAL_SUBJECTS,
        test_subjects=TEST_SUBJECTS,
        input_frames=INPUT_FRAMES,
        output_frames=OUTPUT_FRAMES,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        mode='sequence',
        use_weighted_sampler=True,
        augment_collision=bool(int(args.augment_factor) > 0),
        augment_factor=int(args.augment_factor),
        sampler_strategy=SAMPLER_STRATEGY,
        sampler_collision_scale=SAMPLER_COLLISION_SCALE,
        sampler_collision_power=SAMPLER_COLLISION_POWER,
        sampler_min_weight=SAMPLER_MIN_WEIGHT,
        seed=seed,
        train_stride=TRAIN_STRIDE,
        val_stride=VAL_STRIDE,
        test_stride=TEST_STRIDE
    )
    
    # Create model
    model = RiskLSTM(
        input_size=72,
        hidden_size=512,
        num_layers=2,
        num_classes=3,
        output_frames=OUTPUT_FRAMES
    ).to(device)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train and Test
    train_lstm(args, model, train_loader, val_loader, run_dir, device)
    save_val_artifacts(args, model, val_loader, run_dir, device)
    test_lstm(args, model, test_loader, run_dir, device)
    print(f"Results saved to: {run_dir}")