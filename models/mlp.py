import torch
import torch.nn as nn

class RiskMLP(nn.Module):
    def __init__(self, input_size=72, hidden_sizes=[128, 64], num_classes=3):
        super(RiskMLP, self).__init__()
        
        layers = []
        in_dim = input_size
        
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # x is already flattened by DataLoader: (Batch, 72)
        return self.model(x)