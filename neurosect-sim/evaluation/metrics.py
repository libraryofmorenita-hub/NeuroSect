"""
NEUROSECT — Evaluation Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW WE PROVE THE CONSENSUS WORKS:
    The core experiment: compare reconstruction quality for
    - Single subject (Alice alone)
    - Two subjects (Alice + Ben)
    - Three subjects (Alice + Ben + Cal — full NeuroSect)

    If consensus outperforms single-subject, the thesis holds.

METRICS:
    1. CLIP Cosine Similarity — semantic similarity to ground truth
    2. Top-1 / Top-5 Accuracy — did the right COCO category rank first?
    3. Minority Report Rate   — how often does one subject diverge?

LEARN: These are standard metrics in brain decoding research.
       The UT Austin paper (Tang et al., 2023) uses similar evaluation.
"""

import numpy as np
from typing import List, Dict


def clip_cosine_similarity(
    predicted: np.ndarray,   # (n_trials, embed_dim)
    ground_truth: np.ndarray # (n_trials, embed_dim)
) -> float:
    """
    Average cosine similarity between predicted and true CLIP embeddings.

    Range: [-1, 1]. Higher is better. Random baseline ≈ 0.
    Good brain decoding: > 0.3 is meaningful, > 0.5 is strong.

    WHY THIS METRIC:
        It measures semantic accuracy — did we decode the right concept?
        Better than pixel-level metrics (SSIM, PSNR) for brain decoding
        because the brain encodes semantics, not pixels.
    """
    # Normalize (should already be normalized, but safety check)
    pred_norm = predicted / np.linalg.norm(predicted, axis=1, keepdims=True)
    gt_norm = ground_truth / np.linalg.norm(ground_truth, axis=1, keepdims=True)

    # Cosine similarity = dot product of unit vectors
    similarities = (pred_norm * gt_norm).sum(axis=1)  # (n_trials,)
    return float(similarities.mean())


def top_k_accuracy(
    predicted_embeddings: np.ndarray,  # (n_trials, embed_dim)
    ground_truth_embeddings: np.ndarray,
    k: int = 5
) -> float:
    """
    Top-K retrieval accuracy.

    For each trial: is the true embedding in the top-K most similar predictions?

    WHY: Even if the exact embedding isn't perfect, being in the right
         neighborhood (top-5 most similar) is meaningful.

    LEARN: This is a standard "retrieval accuracy" metric.
           Used in CLIP's original paper and most brain decoding work.
    """
    n = len(predicted_embeddings)
    correct = 0

    for i in range(n):
        # Compute similarity between prediction[i] and ALL ground truths
        sims = predicted_embeddings[i] @ ground_truth_embeddings.T  # (n,)
        top_k_indices = np.argsort(sims)[::-1][:k]
        if i in top_k_indices:
            correct += 1

    return correct / n


def minority_report_rate(minority_report_log: List[Dict]) -> Dict:
    """
    Analyze minority report statistics.

    HOW OFTEN does one subject see something different?
    Is Alice (primary) more stable than Ben/Cal?
    """
    if not minority_report_log:
        return {"rate": 0.0, "per_subject": {}}

    n_trials = len(minority_report_log)
    total_reports = 0
    per_subject = {"alice": 0, "ben": 0, "cal": 0}

    for trial in minority_report_log:
        reports = trial.get("minority_reports", {})
        for subj in reports:
            total_reports += 1
            if subj in per_subject:
                per_subject[subj] += 1

    return {
        "rate": total_reports / n_trials,
        "per_subject": {k: v / n_trials for k, v in per_subject.items()}
    }


def compare_single_vs_consensus(
    single_subject_embeds: np.ndarray,   # Alice alone
    consensus_embeds: np.ndarray,         # All three
    ground_truth_embeds: np.ndarray,
) -> Dict:
    """
    THE KEY EXPERIMENT: Does consensus beat single subject?

    LEARN: This is your ablation study — standard in ML papers.
           Ablations remove components one by one to show each contributes.
           Here: does adding Ben and Cal actually help Alice?
    """
    single_sim = clip_cosine_similarity(single_subject_embeds, ground_truth_embeds)
    consensus_sim = clip_cosine_similarity(consensus_embeds, ground_truth_embeds)

    single_top5 = top_k_accuracy(single_subject_embeds, ground_truth_embeds, k=5)
    consensus_top5 = top_k_accuracy(consensus_embeds, ground_truth_embeds, k=5)

    improvement = consensus_sim - single_sim
    print(f"\n{'='*50}")
    print(f"RESULTS: Single Subject vs Consensus")
    print(f"{'='*50}")
    print(f"CLIP Cosine Similarity:")
    print(f"  Alice (single): {single_sim:.4f}")
    print(f"  Consensus (3):  {consensus_sim:.4f}")
    print(f"  Improvement:    {improvement:+.4f}")
    print(f"\nTop-5 Accuracy:")
    print(f"  Alice (single): {single_top5:.2%}")
    print(f"  Consensus (3):  {consensus_top5:.2%}")
    print(f"{'='*50}\n")

    return {
        "single_clip_sim": single_sim,
        "consensus_clip_sim": consensus_sim,
        "improvement": improvement,
        "single_top5": single_top5,
        "consensus_top5": consensus_top5,
    }


if __name__ == "__main__":
    print("Testing evaluation metrics with random data...")
    n, d = 100, 768

    single = np.random.randn(n, d)
    single /= np.linalg.norm(single, axis=1, keepdims=True)

    # Consensus slightly closer to ground truth (simulate improvement)
    gt = np.random.randn(n, d)
    gt /= np.linalg.norm(gt, axis=1, keepdims=True)
    consensus = gt + np.random.randn(n, d) * 0.5  # Closer to gt
    consensus /= np.linalg.norm(consensus, axis=1, keepdims=True)

    results = compare_single_vs_consensus(single, consensus, gt)
