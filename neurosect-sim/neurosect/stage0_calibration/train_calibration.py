"""
NEUROSECT — Stage 0: Calibration Training Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS DOES:
    Trains one SubjectEncoder per subject (Alice, Ben, Cal).
    The encoder learns to map fMRI voxels → CLIP image embeddings.
    After training, each encoder "knows" that subject's neural fingerprint.

LOSS FUNCTION: Cosine Similarity Loss
    We want the predicted CLIP embedding to point in the same direction
    as the true CLIP embedding of the image the person was viewing.
    Cosine similarity measures angular distance — perfect for unit-sphere embeddings.

LEARN — PyTorch training loop anatomy:
    1. Forward pass:  predictions = model(inputs)
    2. Loss:          loss = criterion(predictions, targets)
    3. Backward pass: loss.backward()  ← computes gradients
    4. Update:        optimizer.step() ← adjusts weights
    5. Zero grads:    optimizer.zero_grad() ← reset for next batch
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb
import yaml
from pathlib import Path
from neurosect.stage0_calibration.subject_encoder import SubjectEncoder


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss.

    WHY: Our encoder predicts CLIP embeddings on the unit sphere.
         The best metric for unit vectors is cosine similarity.
         Loss = 1 - cosine_similarity (so loss=0 means perfect alignment)

    LEARN: torch.nn.functional.cosine_similarity computes the dot product
           of L2-normalized vectors — the cosine of the angle between them.
           Range: [-1, 1]. We want 1 (same direction), so loss = 1 - similarity.
    """
    return 1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()


def train_subject_encoder(
    subject_name: str,
    voxels_train: np.ndarray,        # (n_train, n_voxels) fMRI betas
    clips_train: np.ndarray,         # (n_train, 768) CLIP target embeddings
    voxels_test: np.ndarray,
    clips_test: np.ndarray,
    config: dict,
    save_dir: str = "checkpoints/"
):
    """
    Full training loop for one subject's encoder.

    TODO: Run this once per subject:
        train_subject_encoder("alice", alice_voxels_train, alice_clips_train, ...)
        train_subject_encoder("ben",   ben_voxels_train,   ben_clips_train,   ...)
        train_subject_encoder("cal",   cal_voxels_train,   cal_clips_train,   ...)
    """
    device = torch.device(config["training"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"\nTraining encoder for {subject_name} on {device}")

    # ─── Model ──────────────────────────────────────────────────────────────
    model = SubjectEncoder(
        n_voxels=config["data"]["n_voxels"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
        subject_name=subject_name
    ).to(device)

    # ─── Data ───────────────────────────────────────────────────────────────
    # LEARN: TensorDataset pairs inputs with targets
    #        DataLoader batches them and shuffles between epochs
    train_ds = TensorDataset(
        torch.from_numpy(voxels_train),
        torch.from_numpy(clips_train)
    )
    test_ds = TensorDataset(
        torch.from_numpy(voxels_test),
        torch.from_numpy(clips_test)
    )
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=4)

    # ─── Optimizer ──────────────────────────────────────────────────────────
    # WHY AdamW: Adam with weight decay — prevents overfitting
    # WHY CosineAnnealingLR: learning rate decays smoothly → better convergence
    optimizer = AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # ─── Logging ────────────────────────────────────────────────────────────
    # LEARN: wandb (Weights & Biases) tracks metrics, saves plots, lets you
    #        compare runs visually. Run `wandb login` once before training.
    wandb.init(project="neurosect", name=f"calibration_{subject_name}", config=config)

    # ─── Training Loop ──────────────────────────────────────────────────────
    best_test_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        # ── Train ──
        model.train()
        train_losses = []

        for voxels_batch, clips_batch in train_loader:
            voxels_batch = voxels_batch.to(device)
            clips_batch = clips_batch.to(device)

            # LEARN: This is the core training step — memorize this pattern
            optimizer.zero_grad()                          # 1. Zero gradients
            pred_clips = model(voxels_batch)               # 2. Forward pass
            loss = cosine_loss(pred_clips, clips_batch)    # 3. Compute loss
            loss.backward()                                 # 4. Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()                               # 5. Update weights

            train_losses.append(loss.item())

        scheduler.step()

        # ── Evaluate ──
        model.eval()
        test_losses = []

        with torch.no_grad():  # LEARN: no_grad() saves memory — no gradients needed
            for voxels_batch, clips_batch in test_loader:
                voxels_batch = voxels_batch.to(device)
                clips_batch = clips_batch.to(device)
                pred_clips = model(voxels_batch)
                loss = cosine_loss(pred_clips, clips_batch)
                test_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)

        wandb.log({
            f"{subject_name}/train_loss": train_loss,
            f"{subject_name}/test_loss": test_loss,
            "epoch": epoch
        })

        print(f"  Epoch {epoch+1:3d} | train: {train_loss:.4f} | test: {test_loss:.4f}")

        # Save best checkpoint
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_path = Path(save_dir) / f"encoder_{subject_name}_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best checkpoint: {save_path}")

    wandb.finish()
    print(f"\nTraining complete for {subject_name}. Best test loss: {best_test_loss:.4f}")
    return model


if __name__ == "__main__":
    # TODO: Replace with real data loading
    cfg = yaml.safe_load(open("config/config.yaml"))

    # Synthetic data for testing the training loop
    n_train, n_test = 900, 100
    n_voxels = cfg["data"]["n_voxels"]
    embed_dim = cfg["model"]["embed_dim"]

    for subject in ["alice", "ben", "cal"]:
        train_subject_encoder(
            subject_name=subject,
            voxels_train=np.random.randn(n_train, n_voxels).astype(np.float32),
            clips_train=np.random.randn(n_train, embed_dim).astype(np.float32),
            voxels_test=np.random.randn(n_test, n_voxels).astype(np.float32),
            clips_test=np.random.randn(n_test, embed_dim).astype(np.float32),
            config=cfg
        )
