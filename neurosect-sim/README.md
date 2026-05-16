# Neurosect Simulation

**Multi-Subject Bayesian Neural Decoding Pipeline**  
A simulation of the Neurosect speculative multimodal medical imaging system, implemented as a machine learning research project.

**Authors:** Amelia Arabe · Irene Ronda Gómez · Alan Aquino  
**Course:** ECE 187 Biomedical Imaging and Sensing · UC San Diego

---

## The Thesis

Three subjects viewing the same images produce correlated but individually distinct neural responses. Bayesian consensus fusion of their decoders produces **more accurate scene reconstruction than any single decoder alone**.

This is the engineering proof that NeuroSect's multi-precog consensus architecture adds measurable value over single-subject decoding.

---

## Pipeline

```
Stage 0: Calibration    — per-subject fMRI encoder, trained to map voxels → CLIP space
Stage 1: Separator      — classify which subject activation pattern belongs to (shared scanner sim)
Stage 2: Decoder        — fMRI + EEG → CLIP embedding + uncertainty (MC Dropout)
Stage 3: Synthesizer    — consensus CLIP embedding → image (Stable Diffusion)
Stage 4: Consensus      — Bayesian fusion + P(Eᵢ) = |⟨φᵢ|ψ⟩|² + minority report detection
```

## Dataset

- **fMRI:** Natural Scenes Dataset (NSD) — 3 subjects (Alice=subj01, Ben=subj02, Cal=subj03)
- **EEG:** THINGS-EEG dataset
- **Stimuli:** COCO image subset (80 categories = event library)

Register for NSD: https://naturalscenesdataset.org/

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
wandb login  # for experiment tracking
```

## Build Order

1. Download NSD data → `python data/download_nsd.py`
2. Preprocess → `data/preprocess/fmri_preprocess.py`
3. Train encoders → `python -m neurosect.stage0_calibration.train_calibration`
4. Train separator → `python -m neurosect.stage1_separator.train_separator`
5. Run full pipeline → `python neurosect/pipeline.py`
6. Evaluate → `python evaluation/evaluate.py`

## Key Result

| Method | CLIP Cosine Sim | Top-5 Accuracy |
|--------|----------------|----------------|
| Alice (single) | baseline | baseline |
| Alice + Ben | +Δ | +Δ |
| Full Consensus (3) | **best** | **best** |

---

*"The components of a precognitive imaging system are not waiting to be invented. They are waiting to be assembled."*
