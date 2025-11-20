import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):

    """

    MLP for pose forecasting with residual prediction

    Predicts the delta relative to the last input frame

    Input: sequence of T frames (T, 24, 3)

    Output: sequence of P frames (P, 24, 3)

    """
    
    def __init__(self,

                 input_frames: int = 10,

                 output_frames: int = 25,

                 num_joints: int = 24,

                 hidden_dims: Optional[List[int]] = None,

                 dropout: float = 0.2):

        super(MLP, self).__init__()
        

        self.input_frames = input_frames

        self.output_frames = output_frames

        self.num_joints = num_joints

        if hidden_dims is None:

            hidden_dims = [1024, 512, 256]
        

        # MLP to predict deltas

        input_dim = input_frames * num_joints * 3

        output_dim = output_frames * num_joints * 3
        

        layers = []

        prev_dim = input_dim
        

        for hidden_dim in hidden_dims:

            layers.append(nn.Linear(prev_dim, hidden_dim))

            layers.append(nn.ReLU())

            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim
        

        layers.append(nn.Linear(prev_dim, output_dim))
        

        self.mlp = nn.Sequential(*layers)
        

        self._init_weights()
    

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

                if m.bias is not None:

                    nn.init.zeros_(m.bias)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """

        Args:

            x: (B, T, 24, 3) - batch of input sequences

        Returns:

            output: (B, P, 24, 3) - batch of predicted sequences

        """

        B, T, J, C = x.shape
        

        # Last input frame as reference

        last_frame = x[:, -1:, :, :]  # (B, 1, 24, 3)
        

        # Flatten input

        x_flat = x.view(B, -1)  # (B, T*24*3)
        

        # Predict deltas (changes relative to last frame)

        delta = self.mlp(x_flat)  # (B, P*24*3)

        delta = delta.view(B, self.output_frames, J, C)  # (B, P, 24, 3)
        

        # Add delta to last frame to get absolute poses

        output = last_frame + delta
        
        return output