# NeuronAI 

[![DOI](https://zenodo.org/badge/1377715284.svg)](https://doi.org/10.5281/zenodo.22851741)

A Biophysical Neural Mass Model for Personalized Pharmacological Simulation

## Description
NeuronAI is a Python framework that combines an extended biophysical neural mass model of EEG dynamics with an artificial intelligence system. It predicts personalized pharmacological modulations to transform a pathological EEG (e.g., from an ASD subject) into a reference state (e.g., from a typically developing subject).

The model integrates:
- Differentiated glutamatergic receptors (AMPA, NMDA NR2A, NMDA NR2B).
- Neuromodulatory systems (dopamine, serotonin, noradrenaline).
- Kuramoto order parameter and Phase Locking Value (PLV) as biomarkers for optimization.

## Key Features
- Creates a digital twin of an EEG recording.
- Uses AI to predict optimal parameter modulations.
- Validated through a seven-level protocol (static analysis, unit testing, property-based testing, coverage, contracts, mutation testing, and fuzz testing).

## Installation

You can run the code directly in Google Colab:

**Current version (v1.0.2):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lgonzalezpaz/NeuronAI/blob/main/NeuronAI_v1.0.2.ipynb)

**Previous version (v1.0.1):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lgonzalezpaz/NeuronAI/blob/main/NeuronAI_v1.0.1.ipynb)


Version 1.0.2 introduces three specific changes that improve robustness across atypical subjects. All original functionalities of version 1.0.1 are preserved.

### Change 1 — Scale-aligned spectrogram

**In v1.0.2:** The `spectrogram_db` function now standardizes each signal
to unit variance before computing the spectrogram. This preserves spectral
shape (the only thing that matters for the distance) and removes the arbitrary
absolute offset. All callers automatically benefit without any API change.

### Change 2 — Base Model Anchored to TD

**In version 1.0.2:** Three handlers (`on_optimize_click`, `on_simulate_click`,
`on_apply_ai_click`) now use the TD-fitted parameters (`td_params_fitted_global`) as the model base instead of fitting them to the subject. This provides a stable base approach, where if a subject's spectrum falls outside the model's repertoire, the platform now reports "no effective modulation" instead of forcing spurious modulation.

### Change 3 — Line-noise cleanup (50 + 60 Hz)

**In v1.0.2:** A new **CLEAN LINE NOISE** button applies a strong IIR
notch (Q = 10, two passes) at both 50 and 60 Hz, symmetrically to subject and TD. The tool reports the ratio before and after, so the user can verify that the interference has been removed.

### New button in the UI

- **1. CLEAN LINE NOISE (50+60 Hz)** — Applied to subject and TD already   in memory. Reports the ratio before and after.

For local installation:
```bash
pip install -r requirements.txt
