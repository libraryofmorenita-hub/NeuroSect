"""
NEUROSECT — Stage 2: EEG Temporal Decoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY EEG IN THIS PIPELINE?
    fMRI has great spatial resolution but terrible temporal resolution.
    EEG has millisecond temporal resolution — it catches the moment
    a "candidate future" emerges in the precog's mind.

    In our simulation: EEG complements fMRI by providing temporal dynamics.
    The EEG decoder extracts frequency-band signatures (delta, theta, alpha,
    beta, gamma) and produces a temporal context embedding.

ARCHITECTURE: LSTM → Linear
    WHY LSTM: EEG is a time series. LSTMs are designed to capture
              temporal dependencies — "what happened at t-1 affects t."

    EEG input shape: (batch, time_steps, n_channels)
                     e.g. (32, 250, 128) = 32 trials, 1 second at 250Hz, 128 electrodes

LEARN — New concept: LSTM
    LSTM (Long Short-Term Memory) is a type of RNN (Recurrent Neural Network).
    It maintains a "hidden state" that gets updated at each time step.
    The final hidden state summarizes the entire sequence.
    Perfect for EEG, speech, any sequential data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class EEGDecoder(nn.Module):
    """
    LSTM-based decoder: EEG time series → semantic embedding.

    Processes per-subject EEG to extract temporal neural dynamics.
    Outputs an embedding in the same CLIP space as the fMRI decoder
    so they can be fused in Stage 4.
    """

    def __init__(
        self,
        n_channels: int = 128,     # EEG electrodes
        n_timepoints: int = 250,   # 1 second at 250 Hz
        embed_dim: int = 768,      # CLIP embedding dim (must match fMRI decoder)
        hidden_dim: int = 256,     # LSTM hidden size
        n_layers: int = 2,         # LSTM depth
        dropout: float = 0.3,
    ):
        super().__init__()

        # ─── Temporal feature extraction ────────────────────────────────────
        # LEARN: LSTM processes the sequence step by step.
        #        batch_first=True means input shape is (batch, time, channels)
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True    # WHY bidirectional: looks at sequence forward AND backward
        )

        # LEARN: bidirectional LSTM doubles the output dimension
        lstm_out_dim = hidden_dim * 2  # forward + backward

        # ─── Projection to CLIP space ────────────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        ARGS:
            eeg: (batch, time_steps, n_channels) — EEG time series

        RETURNS:
            embedding: (batch, embed_dim) — temporal semantic embedding
        """
        # LEARN: lstm returns (output, (h_n, c_n))
        #        output shape: (batch, time_steps, hidden*2)
        #        h_n shape:    (n_layers*2, batch, hidden) — final hidden state
        lstm_out, (h_n, _) = self.lstm(eeg)

        # Use the last time step's output as sequence summary
        # WHY last timestep: it has "seen" the entire sequence
        last_out = lstm_out[:, -1, :]   # (batch, hidden*2)

        embedding = self.projection(last_out)

        # L2 normalize — match CLIP's unit-sphere embedding space
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding

    def decode_with_bands(self, eeg: torch.Tensor) -> dict:
        """
        Decompose EEG into frequency bands before decoding.

        WHY FREQUENCY BANDS?
            Different bands carry different information:
            - Delta (0.5-4 Hz):  deep sleep, unconscious processing
            - Theta (4-8 Hz):    working memory, attention
            - Alpha (8-12 Hz):   relaxed awareness, inhibition
            - Beta (12-30 Hz):   active cognition, motor planning
            - Gamma (30-100 Hz): high-level binding, conscious perception

        TODO: Implement bandpass filtering using scipy.signal.butter
              This is where the real neuroscience lives.

        LEARN: scipy.signal.butter(N, Wn, btype='bandpass', fs=250)
               creates a Butterworth filter. Apply with sosfilt().
        """
        # TODO: Apply bandpass filters for each band
        # For now, just run the full-band decoder
        embedding = self.forward(eeg)
        return {
            "embedding": embedding,
            "bands": {}  # TODO: add per-band features
        }


if __name__ == "__main__":
    print("Testing EEGDecoder...")
    model = EEGDecoder()

    fake_eeg = torch.randn(8, 250, 128)  # 8 trials, 1s at 250Hz, 128 channels
    embedding = model(fake_eeg)
    print(f"  Input:     {fake_eeg.shape}")
    print(f"  Embedding: {embedding.shape}")   # (8, 768)
    print(f"  Norms:     {embedding.norm(dim=-1)}")  # ~1.0
    print("EEGDecoder OK.")
