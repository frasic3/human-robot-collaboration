import torch
import torch.nn as nn

from typing import Tuple

class LSTM(nn.Module):
    def __init__(self, 

                 input_frames: int = 10,

                 output_frames: int = 25,

                 num_joints: int = 24,

                 hidden_dim: int = 1024,

                 num_layers: int = 2,

                 dropout: float = 0.2):
        super(LSTM, self).__init__()
        

        self.input_frames = input_frames

        self.output_frames = output_frames

        self.num_joints = num_joints

        self.input_dim = num_joints * 3  # 24 joints * 3 coords (x,y,z)
        

        # Encoder LSTM

        # Takes sequence of poses and produces a hidden state

        self.lstm = nn.LSTM(

            input_size=self.input_dim,

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout if num_layers > 1 else 0
        )
        

        # Decoder (Predictor)

        # Maps the final hidden state to the full future sequence

        # We use a simple MLP decoder here to map from hidden state to all future frames at once

        # Alternatively, we could use an autoregressive decoder, but this is faster and often more stable

        self.decoder = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_frames * self.input_dim)
        )
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        """

        Args:

            x: Input tensor of shape (Batch, Input_Frames, Joints, 3)
            

        Returns:

            prediction: Predicted future poses (Batch, Output_Frames, Joints, 3)

            (h_n, c_n): Final hidden and cell states from the LSTM. 

                        Useful for analysis (latent space visualization, etc.)

        """

        batch_size = x.size(0)
        

        # Flatten joints: (B, T, J, 3) -> (B, T, J*3)

        x = x.view(batch_size, self.input_frames, -1)
        

        # LSTM Forward

        # out: (B, T, hidden_dim) - output features for each frame

        # (h_n, c_n): (num_layers, B, hidden_dim) - final states

        lstm_out, (h_n, c_n) = self.lstm(x)
        

        # We use the output of the last time step to predict the future

        last_time_step_feature = lstm_out[:, -1, :]  # (B, hidden_dim)
        

        # Decode to future frames

        # (B, hidden_dim) -> (B, Output_Frames * J * 3)

        prediction_flat = self.decoder(last_time_step_feature)
        

        # Reshape back to (B, Output_Frames, J, 3)

        prediction = prediction_flat.view(batch_size, self.output_frames, self.num_joints, 3)
        
        return prediction, (h_n, c_n)

