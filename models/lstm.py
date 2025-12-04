import torch
import torch.nn as nn

class RiskLSTM(nn.Module):
    def __init__(self, input_size=72, hidden_size=128, num_layers=2, num_classes=3, output_frames=25):
        super(RiskLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_frames = output_frames
        self.num_classes = num_classes
        
        # Encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Decoder (Predicts sequence of risks from final hidden state)
        # Maps hidden_size -> output_frames * num_classes
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_frames * num_classes)
        )
        
    def forward(self, x):
        """
        Spiegazione passo-passo del flusso dei dati:
        x: Input Batch. Shape: (Batch_Size, T_input, 72)
           Esempio: (32, 10, 72) -> 32 sequenze, lunghe 10 frame, con 72 features ciascuna.
        """
        B = x.size(0) # Batch size (es. 32)
        
        # -------------------------------------------------------
        # 1. ENCODER (La LSTM "guarda" la sequenza passata)
        # -------------------------------------------------------
        # La LSTM processa internamente i 10 frame uno alla volta.
        # Non serve un ciclo for esplicito, PyTorch lo fa in C++ (molto veloce).
        #
        # lstm_out: contiene gli stati intermedi per ogni istante t (da 1 a 10).
        #           Shape: (Batch, 10, 128) -> Non ci serve per la predizione futura.
        #
        # (h_n, c_n): sono lo stato della memoria ALLA FINE della sequenza (dopo il 10° frame).
        #             h_n Shape: (Num_Layers, Batch, 128)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Prendiamo l'hidden state dell'ultimo layer LSTM.
        # Questo vettore è il "RIASSUNTO" compresso di tutto ciò che è successo nei 10 frame.
        final_memory = h_n[-1] # Shape: (Batch, 128)
        
        # -------------------------------------------------------
        # 2. DECODER (Dalla memoria, predice il futuro)
        # -------------------------------------------------------
        # Il decoder prende SOLO il riassunto finale (128 numeri) e deve
        # "srotolarlo" per generare 25 frame futuri di rischio.
        
        # Passaggio attraverso i layer lineari del decoder
        # Input: (Batch, 128) -> Output: (Batch, 25 * 3)
        flattened_prediction = self.decoder(final_memory)
        
        # -------------------------------------------------------
        # 3. RESHAPE (Mettiamo in ordine l'output)
        # -------------------------------------------------------
        # L'output è un vettore piatto lungo 75 (25 frame * 3 classi).
        # Lo ritrasformiamo in una sequenza temporale.
        # Shape finale: (Batch, 25, 3)
        prediction = flattened_prediction.view(B, self.output_frames, self.num_classes)
        
        return prediction

