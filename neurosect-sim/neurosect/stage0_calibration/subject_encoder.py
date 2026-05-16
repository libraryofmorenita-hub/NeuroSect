"""
NEUROSECT — Stage 0: Subject-Specific Encoder (Calibration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY A SEPARATE ENCODER PER SUBJECT?
    Every brain is different. Alice's visual cortex responds to color
    differently than Ben's or Cal's. If you train one model for all three,
    the individual differences cancel out and accuracy drops.

    One encoder per subject = each precog has their own neural fingerprint model.
    This is the "lifetime calibration" from the NeuroSect paper.

ARCHITECTURE: MLP (Multi-Layer Perceptron)
    fMRI voxels (15,724 dims) → hidden layers → CLIP embedding (768 dims)

    WHY MLP and not CNN?
        fMRI voxels don't have spatial structure like pixels do —
        each voxel is independent. CNNs exploit spatial locality.
        MLPs treat every input dimension equally. Right tool for the job.

LEARN — Key PyTorch concepts used here:
    nn.Module        — base class for all PyTorch models
    nn.Linear        — fully connected layer (y = Wx + b)
    nn.LayerNorm     — normalizes activations (stabilizes training)
    nn.GELU          — activation function (smoother than ReLU)
    nn.Dropout       — randomly zeros neurons during training (prevents overfitting)
    forward()        — defines the computation graph
"""

import torch
import torch.nn as nn
from typing import Tuple


class SubjectEncoder(nn.Module):
    """
    Per-subject MLP encoder: fMRI voxels → CLIP embedding space.

    One of these is trained separately for Alice, Ben, and Cal.
    After training, each encoder "knows" that subject's neural fingerprint.
    """

    def __init__(
        self,
        n_voxels: int = 15724,    # Number of visual cortex voxels in NSD ROI
        embed_dim: int = 768,      # CLIP embedding dimension (must match CLIP model)
        hidden_dim: int = 1024,    # Hidden layer width — larger = more capacity
        dropout: float = 0.5,      # Dropout rate — 0.5 is aggressive, good for fMRI
        subject_name: str = "subj01"
    ):
        super().__init__()
        self.subject_name = subject_name
        self.embed_dim = embed_dim

        # ─── Architecture ───────────────────────────────────────────────────
        # LEARN: nn.Sequential chains layers — input flows through each in order
        # WHY 3 layers: enough depth to learn non-linear brain→CLIP mapping
        #               without being so deep it's hard to train on small data
        self.encoder = nn.Sequential(
            # Layer 1: voxels → hidden
            nn.Linear(n_voxels, hidden_dim),
            nn.LayerNorm(hidden_dim),   # WHY LayerNorm: stabilizes training, faster convergence
            nn.GELU(),                   # WHY GELU: smoother gradient flow than ReLU
            nn.Dropout(dropout),         # WHY Dropout: fMRI datasets are small → overfitting risk

            # Layer 2: hidden → hidden
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 3: hidden → CLIP space
            nn.Linear(hidden_dim, embed_dim),
        )

        # Final normalization — CLIP embeddings live on the unit sphere
        # WHY: Cosine similarity (used by CLIP) requires unit-norm vectors
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: brain activations → predicted CLIP embedding.

        ARGS:
            voxels: (batch_size, n_voxels) — fMRI betas for this subject

        RETURNS:
            embedding: (batch_size, embed_dim) — predicted CLIP embedding
                       normalized to unit sphere

        LEARN: This is called automatically when you do model(input)
        """
        x = self.encoder(voxels)
        x = self.norm(x)
        # L2 normalize to unit sphere — matches CLIP's embedding space
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def forward_with_uncertainty(
        self, voxels: torch.Tensor, n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout uncertainty estimation.

        WHY: We need uncertainty for the Bayesian consensus in Stage 4.
             More uncertain prediction → lower weight in fusion.
             Standard forward pass uses mean of dropout masks = too confident.
             MC Dropout: keep dropout ON at test time, run N passes, measure variance.

        LEARN: MC Dropout (Gal & Ghahramani, 2016) turns Dropout into
               approximate Bayesian inference — cheap uncertainty for free.

        ARGS:
            voxels:   (batch_size, n_voxels)
            n_samples: how many stochastic forward passes

        RETURNS:
            mean_embed:  (batch_size, embed_dim) — mean prediction
            uncertainty: (batch_size,) — scalar uncertainty per sample
        """
        # LEARN: train() keeps Dropout active even at test time
        self.train()

        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                samples.append(self.forward(voxels))

        # Stack: (n_samples, batch_size, embed_dim)
        samples = torch.stack(samples, dim=0)

        mean_embed = samples.mean(dim=0)          # Average prediction
        # Variance across samples = uncertainty estimate
        uncertainty = samples.var(dim=0).mean(dim=-1)  # (batch_size,)

        # Back to eval mode (no more MC Dropout)
        self.eval()
        return mean_embed, uncertainty


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing SubjectEncoder...")
    model = SubjectEncoder(subject_name="alice")

    # Fake batch: 4 trials, 15724 voxels each
    fake_voxels = torch.randn(4, 15724)

    # Standard forward pass
    embedding = model(fake_voxels)
    print(f"  Input:  {fake_voxels.shape}")
    print(f"  Output: {embedding.shape}")   # Should be (4, 768)
    print(f"  Norms:  {embedding.norm(dim=-1)}")  # Should be all ~1.0

    # Uncertainty pass
    mean, uncertainty = model.forward_with_uncertainty(fake_voxels, n_samples=10)
    print(f"  Uncertainty shape: {uncertainty.shape}")  # (4,)
    print("SubjectEncoder OK.")
