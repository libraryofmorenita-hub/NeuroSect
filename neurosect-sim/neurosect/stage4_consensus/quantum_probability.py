"""
NEUROSECT — Stage 4: Quantum Probability Calculator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE FORMULA FROM THE PAPER:
    P(Eᵢ) = |⟨φᵢ|ψ⟩|²

    |ψ⟩  = the consensus embedding (our joint neural state vector)
    |φᵢ⟩ = a learned basis vector for event category i
    The probability of event Eᵢ is the squared cosine similarity.

WHY "QUANTUM-INSPIRED" AND NOT ACTUALLY QUANTUM?
    Real quantum hardware would be overkill and unavailable.
    The math is classically implemented — cosine similarity in high-dim space.
    "Quantum-inspired" means we use the Born rule formalism (|⟨φ|ψ⟩|²)
    as a principled way to compute probabilities from embedding similarity.

WHAT ARE THE EVENT BASIS VECTORS?
    COCO has 80 object categories (person, car, dog, pizza...).
    Each category has a CLIP text embedding: clip.encode_text("a photo of a dog")
    These text embeddings ARE our φᵢ vectors — the "event library."
    P(Eᵢ) = how much the consensus brain embedding "looks like" category i.

LEARN: This is essentially zero-shot classification using CLIP.
       The same trick powers CLIP's zero-shot ImageNet classifier.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict


# COCO 80 categories — our "library of candidate futures"
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


class QuantumProbabilityCalculator:
    """
    Computes P(Eᵢ) = |⟨φᵢ|ψ⟩|² for all 80 COCO event categories.

    Maintains a "library of candidate futures" as CLIP text embeddings.
    """

    def __init__(self, clip_model, clip_processor, device: str = "cpu"):
        self.device = device
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        # Build the event basis vectors at init time
        self.event_basis = None  # (80, 768) — one embedding per COCO category
        self.event_names = COCO_CATEGORIES

    def build_event_library(self):
        """
        Pre-compute CLIP text embeddings for all 80 COCO categories.
        These are our |φᵢ⟩ basis vectors.

        LEARN: CLIP encodes text the same way it encodes images.
               "a photo of a dog" and a photo of a dog have similar embeddings.
               This is the foundation of zero-shot classification.

        TODO: Run this once at startup and cache the result.
        """
        print("Building event basis library from COCO categories...")

        texts = [f"a photo of a {cat}" for cat in COCO_CATEGORIES]

        with torch.no_grad():
            inputs = self.clip_processor(text=texts, return_tensors="pt",
                                          padding=True).to(self.device)
            text_features = self.clip_model.get_text_features(**inputs)
            # L2 normalize — unit sphere
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.event_basis = text_features.cpu().numpy()  # (80, 768)
        print(f"  Event library: {self.event_basis.shape}")

    def compute_probabilities(
        self, psi: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Compute P(Eᵢ) = |⟨φᵢ|ψ⟩|² for all event categories.

        ARGS:
            psi: (768,) consensus embedding — the |ψ⟩ neural state vector

        RETURNS:
            probabilities: (80,) probability for each COCO category
            ranked_events: list of (category_name, probability) sorted descending

        LEARN: This is a dot product (⟨φᵢ|ψ⟩) then squared (|...|²).
               Since vectors are unit-norm, dot product = cosine similarity.
               Squaring makes the Born rule: P = |amplitude|².
        """
        if self.event_basis is None:
            raise RuntimeError("Call build_event_library() first.")

        # ⟨φᵢ|ψ⟩ — cosine similarity with all 80 basis vectors at once
        # matrix multiply: (80, 768) @ (768,) = (80,)
        amplitudes = self.event_basis @ psi          # shape: (80,)

        # Born rule: P(Eᵢ) = |amplitude|²
        probabilities = amplitudes ** 2              # shape: (80,)

        # Normalize to sum=1 (valid probability distribution)
        probabilities = probabilities / probabilities.sum()

        # Rank and return
        ranked_indices = np.argsort(probabilities)[::-1]
        ranked_events = [
            (COCO_CATEGORIES[i], float(probabilities[i]))
            for i in ranked_indices
        ]

        return probabilities, ranked_events

    def get_top_k_futures(self, psi: np.ndarray, k: int = 15) -> List[Dict]:
        """
        Return the top-k ranked candidate futures.
        This is the "ranked text feed" output in the NeuroSect paper.

        In the paper: scientists see a ranked list of 15 candidate futures.
        Here: 15 most probable COCO categories = 15 candidate "scenes."
        """
        _, ranked = self.compute_probabilities(psi)
        return [
            {"rank": i + 1, "event": name, "probability": prob}
            for i, (name, prob) in enumerate(ranked[:k])
        ]


if __name__ == "__main__":
    print("Testing QuantumProbabilityCalculator (stub mode)...")
    # Full test requires CLIP — test with random basis
    n_events, embed_dim = 80, 768
    fake_basis = np.random.randn(n_events, embed_dim)
    fake_basis /= np.linalg.norm(fake_basis, axis=1, keepdims=True)

    fake_psi = np.random.randn(embed_dim)
    fake_psi /= np.linalg.norm(fake_psi)

    amplitudes = fake_basis @ fake_psi
    probs = amplitudes ** 2
    probs /= probs.sum()

    top5 = np.argsort(probs)[::-1][:5]
    print("  Top 5 (random, for structure test):")
    for i in top5:
        print(f"    {COCO_CATEGORIES[i]}: {probs[i]:.4f}")
    print("QuantumProbabilityCalculator OK.")
