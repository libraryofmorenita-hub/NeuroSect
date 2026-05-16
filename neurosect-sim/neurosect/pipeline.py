"""
NEUROSECT — Full Pipeline Orchestrator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THIS FILE ties all five stages together into one callable pipeline.
After you implement the individual stages, this is how you run the full system.

USAGE:
    pipeline = NeurosectPipeline(config, checkpoints_dir="checkpoints/")
    result = pipeline.run(alice_voxels, ben_voxels, cal_voxels)
    print(result.top_futures)
    result.consensus_image.save("output.png")

LEARN: This pattern (one Orchestrator class that coordinates many components)
       is called the "Facade" design pattern — it hides complexity behind
       a clean interface. Very common in production ML systems.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

from neurosect.stage1_separator.signal_separator import SignalSeparator
from neurosect.stage2_decoder.fmri_decoder import FMRIDecoder
from neurosect.stage4_consensus.bayesian_fusion import BayesianFusion, SubjectOutput
from neurosect.stage4_consensus.bayesian_fusion import ConsensusOutput
from neurosect.stage4_consensus.quantum_probability import QuantumProbabilityCalculator
from neurosect.stage3_synthesizer.image_synthesizer import ImageSynthesizer


@dataclass
class PipelineResult:
    """
    Complete output of one NeuroSect inference run.
    """
    consensus_embedding: np.ndarray       # (768,) fused latent vector
    top_futures: List[Dict]               # Top-15 ranked COCO categories
    consensus_output: ConsensusOutput     # Full Bayesian fusion result
    consensus_image: Optional[object]     # PIL Image (None if synthesizer off)


class NeurosectPipeline:
    """
    Runs the full five-stage NeuroSect pipeline.

    Stages:
        0. Calibration  — already done (checkpoints loaded)
        1. Separator    — classify subject from shared voxel space
        2. Decoder      — fMRI betas → CLIP embeddings per subject
        3. Consensus    — Bayesian fusion → ranked futures
        4. Synthesizer  — consensus embedding → image
    """

    def __init__(
        self,
        config: dict,
        checkpoints_dir: str = "checkpoints/",
        enable_synthesizer: bool = False,  # Off by default — needs GPU + SD model
        device: str = "cpu"
    ):
        self.config = config
        self.device = device
        checkpoints = Path(checkpoints_dir)

        print("Initializing NeuroSect pipeline...")

        # ─── Stage 1: Signal Separator ──────────────────────────────────────
        self.separator = SignalSeparator(
            n_components=config.get("n_pca_components", 512)
        )
        sep_ckpt = checkpoints / "separator_best.pt"
        if sep_ckpt.exists():
            import torch
            self.separator.load_state_dict(torch.load(sep_ckpt, map_location=device))
        print("  Stage 1 (Separator): loaded")

        # ─── Stage 2: fMRI Decoders (one per subject) ───────────────────────
        self.decoders = {
            "alice": FMRIDecoder("alice", str(checkpoints / "encoder_alice_best.pt"), config, device),
            "ben":   FMRIDecoder("ben",   str(checkpoints / "encoder_ben_best.pt"),   config, device),
            "cal":   FMRIDecoder("cal",   str(checkpoints / "encoder_cal_best.pt"),   config, device),
        }
        print("  Stage 2 (Decoders): loaded")

        # ─── Stage 4: Consensus Engine ──────────────────────────────────────
        self.fusion = BayesianFusion(
            embed_dim=config["model"]["embed_dim"],
            minority_threshold=config["consensus"]["minority_report_threshold"]
        )
        # QuantumProbabilityCalculator needs CLIP — skip if not available
        self.prob_calculator = None  # TODO: init with CLIP model
        print("  Stage 4 (Consensus): initialized")

        # ─── Stage 3: Image Synthesizer (optional) ──────────────────────────
        self.synthesizer = None
        if enable_synthesizer:
            self.synthesizer = ImageSynthesizer(device=device)
            print("  Stage 3 (Synthesizer): initialized")
        else:
            print("  Stage 3 (Synthesizer): disabled (set enable_synthesizer=True)")

        print("Pipeline ready.\n")

    def run(
        self,
        alice_voxels: np.ndarray,   # (n_trials, n_voxels)
        ben_voxels: np.ndarray,
        cal_voxels: np.ndarray,
        generate_image: bool = False,
        n_mc_samples: int = 20,
    ) -> PipelineResult:
        """
        Run the full pipeline on one batch of trials.

        STAGE ORDER:
            2 → 4a → 4b → 3
            (We skip Stage 1 here — voxels are already separated by subject)

        NOTE: Stage 1 (Signal Separator) would be called first if you
              had a genuinely mixed shared-scanner input. In practice,
              we pre-separate by subject using NSD's structure.

        TODO: Add Stage 1 call when running in "shared scanner" mode.
        """
        print("Running NeuroSect pipeline...")

        # ─── Stage 2: Decode each subject ───────────────────────────────────
        print("  Stage 2: Decoding...")
        alice_embed, alice_uncertainty = self.decoders["alice"].decode(alice_voxels, n_mc_samples)
        ben_embed, ben_uncertainty = self.decoders["ben"].decode(ben_voxels, n_mc_samples)
        cal_embed, cal_uncertainty = self.decoders["cal"].decode(cal_voxels, n_mc_samples)

        # Average across trials if batch
        alice_embed = alice_embed.mean(axis=0)
        ben_embed = ben_embed.mean(axis=0)
        cal_embed = cal_embed.mean(axis=0)
        alice_unc = float(alice_uncertainty.mean())
        ben_unc = float(ben_uncertainty.mean())
        cal_unc = float(cal_uncertainty.mean())

        print(f"    Alice: uncertainty={alice_unc:.4f}")
        print(f"    Ben:   uncertainty={ben_unc:.4f}")
        print(f"    Cal:   uncertainty={cal_unc:.4f}")

        # ─── Stage 4a: Bayesian Fusion ──────────────────────────────────────
        print("  Stage 4a: Bayesian fusion...")
        alice_out = SubjectOutput("alice", alice_embed, alice_unc, is_primary=True)
        ben_out   = SubjectOutput("ben",   ben_embed,   ben_unc)
        cal_out   = SubjectOutput("cal",   cal_embed,   cal_unc)

        consensus_out = self.fusion(alice_out, ben_out, cal_out)
        print(f"    Consensus confidence: {consensus_out.confidence:.3f}")
        print(f"    Weights: {consensus_out.weights}")
        if consensus_out.minority_reports:
            print(f"    MINORITY REPORTS: {consensus_out.minority_reports}")

        # ─── Stage 4b: Quantum Probability Ranking ──────────────────────────
        top_futures = []
        if self.prob_calculator is not None:
            print("  Stage 4b: Computing probability rankings...")
            top_futures = self.prob_calculator.get_top_k_futures(
                consensus_out.consensus_embedding, k=15
            )
            print(f"    Top prediction: {top_futures[0]['event']} "
                  f"(p={top_futures[0]['probability']:.3f})")
        else:
            print("  Stage 4b: Skipped (CLIP not loaded)")

        # ─── Stage 3: Image Synthesis ────────────────────────────────────────
        consensus_image = None
        if generate_image and self.synthesizer and top_futures:
            print("  Stage 3: Synthesizing image...")
            consensus_image = self.synthesizer.synthesize(
                consensus_out.consensus_embedding, top_futures
            )

        print("Pipeline complete.\n")
        return PipelineResult(
            consensus_embedding=consensus_out.consensus_embedding,
            top_futures=top_futures,
            consensus_output=consensus_out,
            consensus_image=consensus_image,
        )


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config/config.yaml"))

    pipeline = NeurosectPipeline(cfg)

    # Test with synthetic data
    n_trials = 4
    n_voxels = cfg["data"]["n_voxels"]

    result = pipeline.run(
        alice_voxels=np.random.randn(n_trials, n_voxels).astype(np.float32),
        ben_voxels=np.random.randn(n_trials, n_voxels).astype(np.float32),
        cal_voxels=np.random.randn(n_trials, n_voxels).astype(np.float32),
    )

    print(f"Consensus embedding shape: {result.consensus_embedding.shape}")
    print(f"Confidence: {result.consensus_output.confidence:.3f}")
    print(f"Minority reports: {result.consensus_output.minority_reports}")
