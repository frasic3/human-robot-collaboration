import torch
import time
import numpy as np
from models.mlp import RiskMLP
from models.lstm import RiskLSTM

# Config
device = torch.device('cpu') # Test su CPU (scenario peggiore/reale per robot industriali)
input_frames = 10
features = 72

# 1. Measure MLP
mlp = RiskMLP(input_size=72, hidden_sizes=[128, 64], num_classes=3).to(device)
mlp.eval()
dummy_input = torch.randn(1, 72).to(device) # Batch=1

# Warmup
for _ in range(100): _ = mlp(dummy_input)

start = time.perf_counter()
iters = 1000
with torch.no_grad():
    for _ in range(iters):
        _ = mlp(dummy_input)
end = time.perf_counter()
mlp_latency = (end - start) / iters * 1000 # in ms

# 2. Measure LSTM
lstm = RiskLSTM(input_size=72, hidden_size=512, num_layers=2, num_classes=3, output_frames=25).to(device)
lstm.eval()
dummy_seq = torch.randn(1, 10, 72).to(device) # Batch=1, Seq=10

# Warmup
for _ in range(100): _ = lstm(dummy_seq)

start = time.perf_counter()
with torch.no_grad():
    for _ in range(iters):
        _ = lstm(dummy_seq)
end = time.perf_counter()
lstm_latency = (end - start) / iters * 1000 # in ms

print(f"MLP Latency (CPU, Batch=1): {mlp_latency:.4f} ms")
print(f"LSTM Latency (CPU, Batch=1): {lstm_latency:.4f} ms")
print(f"Total Pipeline Latency: {mlp_latency + lstm_latency:.4f} ms")
print(f"Max Frequency: {1000 / (mlp_latency + lstm_latency):.2f} Hz")