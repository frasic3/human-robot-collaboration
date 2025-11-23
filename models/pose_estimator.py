import torch
import torch.nn as nn
import torchvision.models as models

class PoseEstimator(nn.Module):
    """
    Single-Frame Pose Estimator
    Input: Single Video Frame (3, H, W)
    Output: 3D Skeleton (24 joints * 3 coords)
    """
    def __init__(self, num_joints=24, pretrained=True):
        super(PoseEstimator, self).__init__()
        
        # Use ResNet18/50 as backbone
        # ResNet is great for extracting features from images
        backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the last classification layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # New regression head to predict 3D coordinates
        # Input: 512 features from ResNet18
        # Output: 24 * 3 = 72 coordinates
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_joints * 3)
        )
        
    def forward(self, x):
        # x shape: (Batch, 3, 224, 224)
        
        # Extract features
        x = self.features(x) # (Batch, 512, 1, 1)
        x = x.flatten(1)     # (Batch, 512)
        
        # Predict coordinates
        x = self.regressor(x) # (Batch, 72)
        
        # Reshape to (Batch, 24, 3)
        return x.view(-1, 24, 3)
