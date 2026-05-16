"""
NEUROSECT — fMRI Preprocessing Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEARN: fMRI preprocessing is a pipeline of signal cleaning steps.
       The goal is to get from raw BOLD signal → clean voxel activations.
       NSD already does most of this for you (that's why we use it).
       This module handles the final normalization and dimensionality reduction.

PIPELINE:
    1. Z-score normalize per voxel (remove baseline drift)
    2. PCA to shared voxel space (for Stage 1 Signal Separator)
    3. Extract CLIP target embeddings (for Stage 0 training targets)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict


def zscore_normalize(betas: np.ndarray) -> np.ndarray:
    """
    Z-score normalize fMRI betas across trials (per voxel).

    WHY: Different brain regions have different baseline activity levels.
         Z-scoring puts all voxels on the same scale — mean=0, std=1.
         Without this, high-activation regions dominate the model.

    LEARN: StandardScaler from sklearn does this automatically.
           It fits on training data, then transforms train + test separately.
           NEVER fit on test data — that's data leakage.

    ARGS:
        betas: (n_trials, n_voxels) raw fMRI responses

    RETURNS:
        normalized betas: same shape, each voxel column has mean=0, std=1
    """
    # TODO: Replace this stub with real normalization
    scaler = StandardScaler()
    normalized = scaler.fit_transform(betas)
    return normalized.astype(np.float32)


def build_shared_voxel_space(
    subject_betas: Dict[str, np.ndarray],
    n_components: int = 512
) -> Tuple[np.ndarray, PCA]:
    """
    Project all subjects into a shared PCA voxel space.

    WHY: This is how we simulate the "shared scanner environment."
         In the real NeuroSect, three precogs are in the same MRI field.
         Here, we concatenate all subjects' betas and project to a common space.
         The Signal Separator (Stage 1) then learns to identify which subject
         each activation pattern came from.

    LEARN: PCA (Principal Component Analysis) finds the directions of maximum
           variance in your data. n_components=512 keeps the 512 most informative
           dimensions out of 15,724 voxels — massive compression with little loss.

    ARGS:
        subject_betas: dict {"subj01": (n_trials, n_voxels), ...}
        n_components:  how many PCA dimensions to keep

    RETURNS:
        shared_betas: (n_all_trials, n_components) — all subjects in shared space
        pca:          fitted PCA object (save this for inference)
    """
    # Stack all subjects' betas
    all_betas = np.concatenate(list(subject_betas.values()), axis=0)

    # Fit PCA on the combined data
    pca = PCA(n_components=n_components, random_state=42)
    shared_betas = pca.fit_transform(all_betas)

    print(f"Shared voxel space: {all_betas.shape} → {shared_betas.shape}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    return shared_betas.astype(np.float32), pca


def split_train_test(
    betas: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    LEARN: CRITICAL — always split before any fitting/normalization.
           The test set must be completely unseen during training.
           We use a fixed random_state so results are reproducible.

    WHY test_size=0.1:
        NSD has ~10,000 trials per subject. 10% = 1,000 test trials.
        That's enough for reliable evaluation metrics.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(
        betas, labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )


def extract_clip_targets(images: list, clip_model, clip_processor, device: str) -> np.ndarray:
    """
    Extract CLIP image embeddings — these are our training targets.

    WHY: Our fMRI decoder learns to map brain activations → CLIP embeddings.
         Then we use those CLIP embeddings to condition Stable Diffusion.
         CLIP is the bridge between brain space and image generation space.

    LEARN: CLIP (Contrastive Language-Image Pretraining, OpenAI 2021) encodes
           images and text into the same 768-dim vector space.
           Images that look similar have similar CLIP embeddings.
           This is what makes it useful as a "semantic" target.

    ARGS:
        images:         list of PIL Images
        clip_model:     loaded CLIPModel from transformers
        clip_processor: loaded CLIPProcessor
        device:         "cuda", "mps", or "cpu"

    RETURNS:
        embeddings: (n_images, 768) float32 array
    """
    import torch

    embeddings = []
    batch_size = 32

    clip_model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = clip_processor(images=batch, return_tensors="pt").to(device)
            # LEARN: image_embeds is the visual feature vector, shape (batch, 768)
            feats = clip_model.get_image_features(**inputs)
            # Normalize to unit sphere — CLIP uses cosine similarity
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)
