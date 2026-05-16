"""
NEUROSECT — Stage 1: Signal Separator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS DOES:
    Takes the "shared scanner field" (all three subjects' fMRI mixed together
    in PCA space) and classifies which subject each activation pattern belongs to.

    In the real NeuroSect: one MRI field, three brains.
    In our simulation: PCA projection of all three subjects into one space,
    then a classifier that must separate them back out.

WHY THIS MATTERS FOR YOUR PORTFOLIO:
    This is the technical proof that individual neural fingerprints are
    separable — that Alice's brain looks different from Ben's even when
    they're viewing the same image. That's a real neuroscience result.

ARCHITECTURE: Small CNN → Linear classifier
    WHY CNN: The PCA components have local structure (nearby components
             often covary). A CNN can exploit this better than a flat MLP.

LEARN — New concepts here:
    nn.Conv1d        — 1D convolution (for sequence-like PCA components)
    nn.AdaptiveAvgPool1d — pools variable-length sequences to fixed size
    CrossEntropyLoss — standard multi-class classification loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SignalSeparator(nn.Module):
    """
    Classifies which subject (Alice/Ben/Cal) a given fMRI pattern belongs to.

    Input:  (batch, n_pca_components) — shared PCA voxel space
    Output: (batch, 3) — logits for [Alice, Ben, Cal]
    """

    # Class labels for reference
    SUBJECTS = {0: "alice", 1: "ben", 2: "cal"}

    def __init__(self, n_components: int = 512, n_subjects: int = 3):
        super().__init__()
        self.n_components = n_components

        # ─── Feature extraction ─────────────────────────────────────────────
        # LEARN: Conv1d treats our PCA components like a 1D sequence.
        #        kernel_size=8 means each filter looks at 8 consecutive components.
        #        This captures local covariance structure in PCA space.
        self.feature_extractor = nn.Sequential(
            # (batch, 1, 512) → (batch, 64, ~256)
            nn.Conv1d(1, 64, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(64),   # WHY BatchNorm: normalizes across batch, faster training
            nn.GELU(),

            # (batch, 64, ~256) → (batch, 128, ~128)
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Pool to fixed size regardless of input length
            nn.AdaptiveAvgPool1d(32),  # → (batch, 128, 32)
        )

        # ─── Classifier head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (batch, 128*32) = (batch, 4096)
            nn.Linear(128 * 32, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_subjects),      # (batch, 3) logits
        )

    def forward(self, pca_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        ARGS:
            pca_features: (batch, n_components) shared PCA voxel activations

        RETURNS:
            logits: (batch, 3) — raw scores for each subject
                    Apply softmax to get probabilities.

        LEARN: We return logits (pre-softmax) because CrossEntropyLoss
               applies softmax internally — more numerically stable.
        """
        # Add channel dimension for Conv1d: (batch, n_comp) → (batch, 1, n_comp)
        x = pca_features.unsqueeze(1)
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits

    def predict_subject(self, pca_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict which subject + confidence.

        RETURNS:
            subject_ids: (batch,) — integer 0/1/2 for Alice/Ben/Cal
            confidence:  (batch,) — softmax probability for predicted class
        """
        logits = self.forward(pca_features)
        probs = F.softmax(logits, dim=-1)
        subject_ids = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        return subject_ids, confidence

    def separate_batch(
        self, pca_features: torch.Tensor
    ) -> dict:
        """
        Take a mixed batch and return per-subject data packages.

        WHY: This is Stage 1's core output — routing each activation
             to the right subject's decoder in Stage 2.

        RETURNS:
            dict with keys "alice", "ben", "cal"
            each value is the features classified as that subject
        """
        subject_ids, confidence = self.predict_subject(pca_features)

        separated = {}
        for idx, name in self.SUBJECTS.items():
            mask = (subject_ids == idx)
            separated[name] = {
                "features": pca_features[mask],
                "confidence": confidence[mask],
                "indices": mask.nonzero(as_tuple=True)[0]
            }

        return separated


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing SignalSeparator...")
    model = SignalSeparator(n_components=512)

    # Simulate a mixed batch from 3 subjects
    fake_shared = torch.randn(12, 512)  # 12 trials in shared PCA space
    logits = model(fake_shared)
    print(f"  Input:  {fake_shared.shape}")
    print(f"  Logits: {logits.shape}")   # (12, 3)

    separated = model.separate_batch(fake_shared)
    for subj, data in separated.items():
        print(f"  {subj}: {len(data['features'])} trials classified here")
    print("SignalSeparator OK.")
