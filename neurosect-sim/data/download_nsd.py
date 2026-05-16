"""
NEUROSECT — Data Download: Natural Scenes Dataset (NSD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY NSD?
    The NSD dataset (Allen et al., 2022) is the perfect Neurosect simulator:
    - 8 subjects viewed 73,000 natural images while in an fMRI scanner
    - We pick 3 subjects → Alice (subj01), Ben (subj02), Cal (subj03)
    - Same images, different neural responses = the multi-precog setup
    - The "shared scanner environment" is built into the dataset structure

    Paper: https://www.nature.com/articles/s41593-021-00962-x
    Dataset: https://naturalscenesdataset.org/

LEARN: fMRI data comes as NIfTI files (.nii.gz) — 4D arrays:
    (x, y, z, time) where x/y/z are brain voxels and time is TRs (scans)
    nibabel loads these: img = nibabel.load('file.nii.gz')
    nilearn helps you extract signals from regions of interest (ROIs)

SETUP REQUIRED:
    1. Register at https://naturalscenesdataset.org/
    2. Download the "betas" (preprocessed fMRI responses) for subjects 1-3
    3. Download the stimulus images (COCO subset)
    4. Place in data/nsd/ following the structure below

STRUCTURE EXPECTED:
    data/nsd/
    ├── subj01/
    │   ├── betas_fithrf_GLMdenoise_RR/   ← fMRI responses to each image
    │   └── roi_masks/                     ← which voxels are visual cortex
    ├── subj02/
    ├── subj03/
    └── stimuli/
        └── images/                        ← the COCO images shown to subjects
"""

import os
import json
import numpy as np
from pathlib import Path


def verify_nsd_structure(nsd_root: str) -> bool:
    """
    Check that the NSD data is in the expected place.

    LEARN: Always validate your data directory before training.
           Silent missing-file errors cause mysterious crashes later.
    """
    nsd_root = Path(nsd_root)
    subjects = ["subj01", "subj02", "subj03"]
    all_good = True

    for subj in subjects:
        subj_path = nsd_root / subj
        if not subj_path.exists():
            print(f"  MISSING: {subj_path}")
            all_good = False
        else:
            print(f"  OK: {subj_path}")

    stimuli_path = nsd_root / "stimuli" / "images"
    if not stimuli_path.exists():
        print(f"  MISSING: {stimuli_path}")
        all_good = False
    else:
        n_images = len(list(stimuli_path.glob("*.jpg")))
        print(f"  OK: {stimuli_path} ({n_images} images)")

    return all_good


def load_subject_betas(nsd_root: str, subject: str, n_sessions: int = 37) -> np.ndarray:
    """
    Load the fMRI beta coefficients for one subject.

    WHY BETAS?
        Raw fMRI is a 4D time series. NSD preprocesses this into "betas" —
        one coefficient per image per voxel — already denoised and aligned.
        This saves you weeks of preprocessing work.

    ARGS:
        nsd_root:   path to your NSD data directory
        subject:    "subj01", "subj02", or "subj03"
        n_sessions: NSD has up to 37 sessions per subject

    RETURNS:
        betas: np.ndarray of shape (n_images, n_voxels)
               Each row is the brain's response to one image.

    TODO: Implement this once you have the data downloaded.
          The NSD betas are stored as .hdf5 files — use h5py to load them.
          Hint: import h5py; f = h5py.File(path, 'r'); betas = f['betas'][:]
    """
    # TODO: Replace this with real NSD loading code
    # For now, returns synthetic data so the pipeline can be tested
    print(f"[STUB] Loading betas for {subject} — replace with real NSD loader")
    n_images = 1000   # NSD has ~10,000 per subject; use 1000 for prototyping
    n_voxels = 15724  # Visual cortex ROI voxel count
    return np.random.randn(n_images, n_voxels).astype(np.float32)


def load_stimulus_images(nsd_root: str, image_ids: list) -> list:
    """
    Load the COCO images that were shown to subjects.

    WHY: We need the ground truth images to:
         1. Extract CLIP embeddings (training targets for our decoders)
         2. Evaluate reconstruction quality (compare output to original)

    LEARN: CLIP (Contrastive Language-Image Pretraining) maps images and text
           into the same embedding space. We'll train our fMRI decoder to
           predict CLIP embeddings, then use those to condition Stable Diffusion.

    TODO: Load real images from data/nsd/stimuli/images/
    """
    print(f"[STUB] Loading {len(image_ids)} stimulus images")
    # TODO: Load actual COCO images using PIL
    # from PIL import Image
    # images = [Image.open(f"{nsd_root}/stimuli/images/{img_id:05d}.jpg") for img_id in image_ids]
    return image_ids  # placeholder


if __name__ == "__main__":
    nsd_root = "data/nsd"
    print("Checking NSD data structure...")
    ok = verify_nsd_structure(nsd_root)
    if ok:
        print("\nAll good! Run the preprocessing pipeline next.")
    else:
        print("\nDownload missing files from https://naturalscenesdataset.org/")
