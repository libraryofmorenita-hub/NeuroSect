"""
NEUROSECT — Stage 2: fMRI Pattern Decoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS DOES:
    Maps per-subject fMRI betas → CLIP embedding + uncertainty.
    This IS the SubjectEncoder from Stage 0, used at inference time.

    The decoder produces:
    1. A CLIP embedding (the "semantic latent vector" of what they perceived)
    2. An uncertainty estimate (used by the Consensus Engine to weight the vote)

NOTE: At inference time, we load the Stage 0 checkpoint and run it here.
      The training happened in Stage 0 — Stage 2 is the inference wrapper.

LEARN: This separation (train in Stage 0, infer in Stage 2) reflects real
       ML systems where training and inference are separate concerns.
"""

import torch
import numpy as np
from pathlib import Path
from neurosect.stage0_calibration.subject_encoder import SubjectEncoder
from typing import Tuple, Optional


class FMRIDecoder:
    """
    Wraps a trained SubjectEncoder for inference in the NeuroSect pipeline.
    Handles loading checkpoints and running uncertainty-aware forward passes.
    """

    def __init__(
        self,
        subject_name: str,
        checkpoint_path: Optional[str] = None,
        config: Optional[dict] = None,
        device: str = "cpu"
    ):
        self.subject_name = subject_name
        self.device = torch.device(device)

        n_voxels = config["data"]["n_voxels"] if config else 15724
        embed_dim = config["model"]["embed_dim"] if config else 768
        hidden_dim = config["model"]["hidden_dim"] if config else 1024
        dropout = config["model"]["dropout"] if config else 0.5

        self.model = SubjectEncoder(
            n_voxels=n_voxels,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            subject_name=subject_name
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            # LEARN: load_state_dict loads saved weights into the model
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[WARN] No checkpoint for {subject_name} — using random weights")

        self.model.eval()

    def decode(
        self,
        voxels: np.ndarray,
        n_mc_samples: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the decoder with MC Dropout uncertainty estimation.

        ARGS:
            voxels:       (n_trials, n_voxels) fMRI betas for this subject
            n_mc_samples: number of MC Dropout passes for uncertainty

        RETURNS:
            embeddings:  (n_trials, 768) predicted CLIP embeddings
            uncertainty: (n_trials,) uncertainty per trial (higher = less confident)

        WHY UNCERTAINTY MATTERS:
            If Alice sees something clearly (low noise trial), her prediction
            should get more weight in the Consensus Engine than a noisy trial.
            MC Dropout gives us this per-trial confidence for free.
        """
        voxels_t = torch.from_numpy(voxels).float().to(self.device)

        mean_embed, uncertainty = self.model.forward_with_uncertainty(
            voxels_t, n_samples=n_mc_samples
        )

        return (
            mean_embed.cpu().numpy(),
            uncertainty.cpu().numpy()
        )


if __name__ == "__main__":
    print("Testing FMRIDecoder...")
    decoder = FMRIDecoder(subject_name="alice")

    fake_voxels = np.random.randn(8, 15724).astype(np.float32)
    embeddings, uncertainty = decoder.decode(fake_voxels)
    print(f"  Embeddings: {embeddings.shape}")   # (8, 768)
    print(f"  Uncertainty: {uncertainty.shape}") # (8,)
    print(f"  Uncertainty range: {uncertainty.min():.4f} – {uncertainty.max():.4f}")
    print("FMRIDecoder OK.")
