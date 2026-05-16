"""
NEUROSECT — Stage 4: Bayesian Consensus Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THIS IS YOUR NOVEL CONTRIBUTION.
    MindEye2, Brain-Diffuser, and others do single-subject fMRI decoding.
    No one has built the multi-subject Bayesian consensus layer on top.
    This is what makes NeuroSect a real research contribution.

WHAT IT DOES:
    Takes three latent vectors (Alice, Ben, Cal) + their uncertainties.
    Fuses them into one consensus embedding using inverse-uncertainty weighting.
    Flags "minority reports" when one subject diverges significantly.

THE MATH:
    Standard Bayesian fusion with inverse-variance weighting:

    w_i = 1 / uncertainty_i           (less uncertain → higher weight)
    w_i = w_i / sum(w_j)              (normalize weights to sum=1)
    consensus = sum(w_i * embedding_i) (weighted average)

    This IS P(E|A,B,C) implemented — the Bayesian posterior over the
    three independent observational channels.

LEARN: Bayesian inference is about combining evidence from multiple sources.
       Each source has a likelihood (what it "says") and a reliability (how certain).
       More reliable sources get more weight. Simple but powerful.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SubjectOutput:
    """
    Package from one subject's decoder.
    One of these per subject per inference call.
    """
    subject_name: str                    # "alice", "ben", or "cal"
    embedding: np.ndarray                # (embed_dim,) CLIP latent vector
    uncertainty: float                   # Scalar uncertainty from MC Dropout
    is_primary: bool = False             # True only for Alice


@dataclass
class ConsensusOutput:
    """
    Final output of the Consensus Engine.
    """
    consensus_embedding: np.ndarray      # (embed_dim,) fused latent vector
    weights: Dict[str, float]            # {"alice": 0.5, "ben": 0.3, "cal": 0.2}
    minority_reports: Dict[str, float]   # Subjects that diverged: {name: divergence_score}
    confidence: float                    # Overall consensus confidence


class BayesianFusion(torch.nn.Module):
    """
    Fuses three subject embeddings into one consensus estimate.

    WHY NOT JUST AVERAGE?
        Simple averaging treats all subjects equally.
        But if Alice is very confident and Ben is very uncertain,
        Alice should dominate the consensus.
        Inverse-uncertainty weighting does this automatically.
    """

    def __init__(self, embed_dim: int = 768, minority_threshold: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        # LEARN: This threshold is a hyperparameter you tune.
        #        If cosine distance > threshold, flag minority report.
        self.minority_threshold = minority_threshold

    def forward(
        self,
        alice: SubjectOutput,
        ben: SubjectOutput,
        cal: SubjectOutput
    ) -> ConsensusOutput:
        """
        Fuse three subject outputs into one consensus.

        ARGS:
            alice, ben, cal: SubjectOutput from each subject's decoder

        RETURNS:
            ConsensusOutput with fused embedding and any minority reports
        """
        subjects = [alice, ben, cal]

        # ─── Step 1: Compute inverse-uncertainty weights ─────────────────────
        # WHY: More uncertain → smaller weight in the consensus
        # Add small epsilon to avoid division by zero
        uncertainties = np.array([s.uncertainty for s in subjects]) + 1e-6
        weights_raw = 1.0 / uncertainties          # Inverse uncertainty
        weights = weights_raw / weights_raw.sum()  # Normalize to sum=1

        weight_dict = {
            alice.subject_name: float(weights[0]),
            ben.subject_name: float(weights[1]),
            cal.subject_name: float(weights[2])
        }

        # ─── Step 2: Weighted fusion ─────────────────────────────────────────
        embeddings = np.stack([s.embedding for s in subjects], axis=0)  # (3, embed_dim)
        consensus_embed = (weights[:, None] * embeddings).sum(axis=0)  # (embed_dim,)

        # L2 normalize — stay on the unit sphere
        consensus_embed = consensus_embed / (np.linalg.norm(consensus_embed) + 1e-6)

        # ─── Step 3: Detect minority reports ────────────────────────────────
        minority_reports = self._detect_minority_reports(subjects, consensus_embed)

        # ─── Step 4: Compute overall confidence ─────────────────────────────
        # High confidence = high agreement between subjects
        confidence = self._compute_confidence(embeddings)

        return ConsensusOutput(
            consensus_embedding=consensus_embed,
            weights=weight_dict,
            minority_reports=minority_reports,
            confidence=float(confidence)
        )

    def _detect_minority_reports(
        self,
        subjects: list,
        consensus_embed: np.ndarray
    ) -> Dict[str, float]:
        """
        Flag subjects whose embedding diverges significantly from consensus.

        LEARN: Cosine distance = 1 - cosine_similarity
               Range: [0, 2]. 0 = same direction, 2 = opposite.
               We use 0.3 as threshold (tunable).

        WHY THIS MATTERS:
            In NeuroSect, a minority report means one precog saw a
            different future. Here it means one subject's decoder
            is producing a qualitatively different semantic embedding —
            worth flagging for evaluation.
        """
        minority_reports = {}

        for subj in subjects:
            # Cosine similarity between subject embedding and consensus
            cos_sim = np.dot(subj.embedding, consensus_embed) / (
                np.linalg.norm(subj.embedding) * np.linalg.norm(consensus_embed) + 1e-6
            )
            cos_dist = 1.0 - cos_sim

            if cos_dist > self.minority_threshold:
                minority_reports[subj.subject_name] = float(cos_dist)
                print(f"  [MINORITY REPORT] {subj.subject_name} diverged: "
                      f"distance={cos_dist:.3f}")

        return minority_reports

    def _compute_confidence(self, embeddings: np.ndarray) -> float:
        """
        Overall confidence = mean pairwise agreement between subjects.

        Perfect agreement (all embeddings identical) → confidence = 1.0
        Complete disagreement → confidence ≈ 0.0

        LEARN: This is a measure of inter-rater reliability —
               same concept used in psychology and annotation tasks.
        """
        n = len(embeddings)
        total_sim = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-6
                )
                total_sim += sim
                count += 1

        return total_sim / count if count > 0 else 0.0


if __name__ == "__main__":
    print("Testing BayesianFusion...")
    fusion = BayesianFusion()

    alice = SubjectOutput("alice", np.random.randn(768), uncertainty=0.1, is_primary=True)
    ben   = SubjectOutput("ben",   np.random.randn(768), uncertainty=0.3)
    cal   = SubjectOutput("cal",   np.random.randn(768), uncertainty=0.5)

    result = fusion(alice, ben, cal)
    print(f"  Consensus shape:  {result.consensus_embedding.shape}")
    print(f"  Weights:          {result.weights}")
    print(f"  Confidence:       {result.confidence:.3f}")
    print(f"  Minority reports: {result.minority_reports}")
    print("BayesianFusion OK.")
