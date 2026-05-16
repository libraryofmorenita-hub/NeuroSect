"""
NEUROSECT — Stage 3: Image Synthesizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS DOES:
    Takes the consensus CLIP embedding and generates a visual image
    using Stable Diffusion conditioned on that embedding.

    This is the "video output" from the paper — the reconstructed scene
    that scientists view on the large display screen.

HOW STABLE DIFFUSION CONDITIONING WORKS:
    Standard SD: text prompt → CLIP text embedding → image
    Our version:  brain signal → fMRI CLIP embedding → image

    We're replacing the text prompt with our brain-decoded embedding.
    IP-Adapter is the bridge — it injects image embeddings into SD's
    cross-attention layers, same slot where text embeddings normally go.

LEARN — Diffusion Models:
    Stable Diffusion works by:
    1. Starting with pure noise (random pixel values)
    2. Iteratively "denoising" guided by the CLIP embedding
    3. After ~50 steps, you have a clean image

    The CLIP embedding steers WHAT the image looks like.
    Our brain embedding steers it toward what the subject perceived.

PAPER REFERENCE:
    Takagi & Nishimoto (2023): "Improving visual image reconstruction
    from human brain activity using latent diffusion models"
    arxiv.org/abs/2306.11536

TODO: This module requires a GPU and ~10GB VRAM.
      On CPU it will work but take 10+ minutes per image.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional


class ImageSynthesizer:
    """
    Generates images from consensus CLIP embeddings using Stable Diffusion.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda",
        guidance_scale: float = 7.5,
        num_steps: int = 50,
        seed: int = 42,
    ):
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_steps = num_steps
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.pipeline = None  # Loaded lazily to save memory

    def load_pipeline(self):
        """
        Load Stable Diffusion pipeline.

        WHY LAZY LOADING:
            SD takes ~5-8GB GPU memory.
            Only load it when you actually need to generate images.

        TODO: For better results, add IP-Adapter for CLIP image conditioning.
              pip install ip-adapter
              https://github.com/tencent-ailab/IP-Adapter

        LEARN: Hugging Face diffusers makes loading SD pipelines one-liner.
        """
        from diffusers import StableDiffusionPipeline

        print(f"Loading Stable Diffusion from {self.pipeline}...")
        # TODO: Switch to IP-Adapter for direct CLIP embedding conditioning
        # For now, we'll use the embedding to guide a text prompt
        print("  [NOTE] Direct CLIP embedding conditioning requires IP-Adapter.")
        print("  [NOTE] Using text-guided generation as placeholder.")

    def embedding_to_prompt(self, clip_embedding: np.ndarray, top_events: list) -> str:
        """
        Convert a CLIP embedding to a text prompt for generation.

        WHY: Full IP-Adapter conditioning is complex to set up.
             As a first working version, we convert the top predicted
             COCO categories into a descriptive prompt.

        This is a valid fallback — the paper eventually uses direct
        embedding conditioning, but text-guided generation validates
        that the pipeline produces sensible outputs.

        ARGS:
            clip_embedding: (768,) consensus embedding
            top_events:     list of {"event": str, "probability": float}

        RETURNS:
            prompt string for Stable Diffusion
        """
        top_3 = top_events[:3]
        scene_elements = ", ".join([e["event"] for e in top_3])
        prompt = (
            f"A photorealistic scene featuring {scene_elements}. "
            f"Natural lighting, high detail, DSLR photography."
        )
        return prompt

    def synthesize(
        self,
        consensus_embedding: np.ndarray,
        top_events: list,
        negative_prompt: str = "blurry, low quality, cartoon, illustration"
    ) -> Image.Image:
        """
        Generate an image from the consensus embedding.

        ARGS:
            consensus_embedding: (768,) fused CLIP embedding from Stage 4
            top_events:          ranked list from QuantumProbabilityCalculator
            negative_prompt:     what to avoid in the generated image

        RETURNS:
            PIL Image

        TODO: Replace text prompt with direct IP-Adapter conditioning
              for true brain-to-image generation.
        """
        if self.pipeline is None:
            self.load_pipeline()

        prompt = self.embedding_to_prompt(consensus_embedding, top_events)
        print(f"  Generating: '{prompt}'")

        # TODO: Replace with:
        # image = self.pipeline(
        #     ip_adapter_image_embeds=consensus_embedding,
        #     negative_prompt=negative_prompt,
        #     ...
        # ).images[0]

        # Placeholder: return a blank image until pipeline is connected
        print("  [STUB] Returning placeholder image — connect SD pipeline to generate real outputs")
        placeholder = Image.new("RGB", (512, 512), color=(200, 200, 200))
        return placeholder

    def synthesize_ranked(
        self,
        consensus_embedding: np.ndarray,
        top_events: list,
        n_images: int = 3
    ) -> list:
        """
        Generate multiple images for the top N candidate futures.

        In NeuroSect: scientists see images for the top predictions.
        Here: generate one image per top COCO category for comparison.
        """
        images = []
        for i, event in enumerate(top_events[:n_images]):
            print(f"  Synthesizing candidate {i+1}: {event['event']} "
                  f"(p={event['probability']:.3f})")
            img = self.synthesize(consensus_embedding, [event])
            images.append({"event": event, "image": img})
        return images


if __name__ == "__main__":
    print("Testing ImageSynthesizer (stub mode)...")
    synth = ImageSynthesizer(device="cpu")

    fake_embedding = np.random.randn(768)
    fake_embedding /= np.linalg.norm(fake_embedding)

    fake_events = [
        {"rank": 1, "event": "dog", "probability": 0.25},
        {"rank": 2, "event": "person", "probability": 0.18},
        {"rank": 3, "event": "couch", "probability": 0.12},
    ]

    prompt = synth.embedding_to_prompt(fake_embedding, fake_events)
    print(f"  Generated prompt: '{prompt}'")
    print("ImageSynthesizer OK (stub).")
