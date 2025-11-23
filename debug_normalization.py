"""Debug script to verify normalization behavior in CHICODataset.

Run this to inspect:
- Computed training stats (mean/std)
- Per-axis mean/std of a normalized batch
- Reconstructed (denormalized) batch stats

Expected:
- Normalized batch per-axis mean ≈ 0, std ≈ 1 (not exact, batch sampling noise)
- Denormalized batch per-axis mean close to original dataset mean

Usage (Windows PowerShell):
    & .venv/Scripts/python.exe debug_normalization.py
"""

import os
import numpy as np
import torch
from utils.pkl_data_loader import create_pkl_dataloaders, CHICODataset

def tensor_stats(x: torch.Tensor):
    """Return per-axis mean/std for last dim (coordinates)."""
    # x shape: (..., 3)
    mean = x.mean(dim=tuple(range(x.ndim - 1)))  # (3,)
    std = x.std(dim=tuple(range(x.ndim - 1)))    # (3,)
    return mean.cpu().numpy(), std.cpu().numpy()


def main():
    base_path = r"c:\Users\Proprietario\Desktop\human-robot-collaboration\datasets\3d_skeletons"

    # Smaller subject subset for speed
    train_subjects = ["S00", "S01"]
    val_subjects = ["S16"]
    test_subjects = ["S18"]

    train_loader, val_loader, test_loader, stats = create_pkl_dataloaders(
        dataset_path=base_path,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        input_frames=10,
        output_frames=25,
        batch_size=8,
        num_workers=0,
    )

    mean, std = stats
    print("=== Global Training Stats (root-relative) ===")
    print(f"Mean (x,y,z): {mean}")
    print(f"Std  (x,y,z): {std}\n")

    batch = next(iter(train_loader))
    input_seq, target_seq, actions, crashes = batch  # input (B,T,24,3) target (B,P,24,3)

    # Concatenate for overall batch stats
    concat_norm = torch.cat([input_seq, target_seq], dim=1)  # (B, T+P, 24, 3)
    norm_mean, norm_std = tensor_stats(concat_norm)
    print("=== Normalized Batch Stats ===")
    print(f"Per-axis mean ≈ 0: {norm_mean}")
    print(f"Per-axis std  ≈ 1: {norm_std}\n")

    # Denormalize batch
    device = concat_norm.device
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)
    denorm = concat_norm * (std_t + 1e-8) + mean_t  # (B, T+P, 24, 3)
    denorm_mean, denorm_std = tensor_stats(denorm)
    print("=== Reconstructed (Denormalized) Batch Stats ===")
    print(f"Mean close to dataset mean: {denorm_mean}")
    print(f"Std close to dataset std  : {denorm_std}\n")

    # Sanity check differences
    print("=== Differences ===")
    print(f"Mean diff: {denorm_mean - mean}")
    print(f"Std  diff: {denorm_std - std}")

    print("\nNote: Small deviations are normal due to batch sampling. Large deviations would indicate a normalization issue.")


if __name__ == "__main__":
    main()
