"""
Utility functions for CHICO dataset and training
"""
import numpy as np
import torch
from typing import Dict, Any


def compute_mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Per Joint Position Error (MPJPE) in millimeters
    
    Calculates the mean Euclidean distance between predicted and real poses.
    This is the standard metric for evaluating pose forecasting.
    
    Args:
        predicted: (B, P, 24, 3) - Predicted poses
        target: (B, P, 24, 3) - Ground truth poses
    
    Returns:
        mpjpe: float - Mean error in millimeters
    
    Formula:
        MPJPE = mean(||predicted_joint - target_joint||₂)
    """
    diff = predicted - target  # (B, P, 24, 3)
    dist = torch.norm(diff, dim=-1)  # (B, P, 24) - Euclidean distance
    mpjpe = dist.mean()
    return mpjpe


def compute_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    """
    Calculate detailed metrics for evaluation
    
    Returns:
        dict with:
        - mpjpe: global mean error
        - mpjpe_per_frame: error for each temporal frame
        - mpjpe_per_joint: error for each joint
    """
    # Global MPJPE
    mpjpe = compute_mpjpe(predicted, target)
    
    # MPJPE per frame (how error grows over time)
    diff = predicted - target  # (B, P, 24, 3)
    dist = torch.norm(diff, dim=-1)  # (B, P, 24)
    mpjpe_per_frame = dist.mean(dim=(0, 2))  # (P,)
    
    # MPJPE per joint (which joints are harder to predict)
    mpjpe_per_joint = dist.mean(dim=(0, 1))  # (24,)
    
    return {
        'mpjpe': mpjpe.item(),
        'mpjpe_per_frame': mpjpe_per_frame.cpu().numpy(),
        'mpjpe_per_joint': mpjpe_per_joint.cpu().numpy()
    }


def loss_to_mpjpe(loss_mse: float) -> float:
    """
    Convert MSE loss to approximate MPJPE
    
    Args:
        loss_mse: Mean Squared Error loss
    
    Returns:
        mpjpe_approx: Approximate MPJPE in mm
    """
    return float(np.sqrt(loss_mse))


def mpjpe_to_loss(mpjpe: float) -> float:
    """
    Convert MPJPE to approximate MSE loss
    
    Args:
        mpjpe: Mean Per Joint Position Error in mm
    
    Returns:
        loss_mse: Approximate MSE loss
    """
    return float(mpjpe ** 2)


def print_training_summary(epoch: int, num_epochs: int, train_loss: float, train_mpjpe: float, val_loss: float, val_mpjpe: float) -> None:
    """
    Print formatted training summary
    """
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{num_epochs}")
    print(f"{'='*60}")
    print(f"Train → Loss: {train_loss:>10.2f} | MPJPE: {train_mpjpe:>6.2f} mm")
    print(f"Val   → Loss: {val_loss:>10.2f} | MPJPE: {val_mpjpe:>6.2f} mm")
    print(f"{'='*60}")


def compute_risk(poses: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Computes risk level based on minimum Human-Robot distance.
    
    Args:
        poses: (B, P, 24, 3) - Denormalized poses in mm
        
    Returns:
        dict containing:
        - min_distances: (B, P) in mm
        - risk_classes: (B, P) - 0 (Safe), 1 (Medium), 2 (High)
    """
    # Split Human (0-14) and Robot (15-23)
    # poses shape: (Batch, Prediction_Horizon, Joints, Coordinates)
    human = poses[:, :, :15, :]  # (B, P, 15, 3)
    robot = poses[:, :, 15:, :]  # (B, P, 9, 3)
    
    # Compute pairwise distances
    # Expand dims for broadcasting: (B, P, 15, 1, 3) - (B, P, 1, 9, 3)
    diff = human.unsqueeze(3) - robot.unsqueeze(2)  # (B, P, 15, 9, 3)
    dist = torch.norm(diff, dim=-1)  # (B, P, 15, 9)
    
    # Find minimum distance for each timestamp (worst case)
    # min over robot joints, then min over human joints
    min_dist = dist.min(dim=-1)[0].min(dim=-1)[0]  # (B, P)
    
    # Classify Risk
    # Safe (> 300mm): 0
    # Medium (200-300mm): 1
    # High (< 200mm): 2
    
    risk = torch.zeros_like(min_dist, dtype=torch.long)
    
    # Default is 0 (Safe)
    # Apply thresholds
    risk[min_dist < 300] = 1  # Medium
    risk[min_dist < 200] = 2  # High
    
    return {
        'min_distances': min_dist,
        'risk_classes': risk
    }
