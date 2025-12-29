import torch
import torch.nn as nn

class RiskLSTM(nn.Module):
    def __init__(
        self,
        input_size=72,
        hidden_size=128,
        num_layers=2,
        num_classes=3,
        output_frames=25,
        dropout=0.5,
    ):
        super(RiskLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_frames = output_frames
        self.num_classes = num_classes
        
        # Encoder
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0, # No dropout if single layer
        )

        # Decoder (Predicts sequence of risks from final hidden state)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, output_frames * num_classes)
        )
        
    def forward(self, x):
        B = x.size(0)
        _, (h_n, _) = self.lstm(x)
        final_memory = h_n[-1]
        flattened_prediction = self.decoder(final_memory)
        return flattened_prediction.view(B, self.output_frames, self.num_classes)