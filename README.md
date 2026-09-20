# NeuronAI v2.1: A Biophysical Neural Mass Model for Personalized Pharmacological Simulation

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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/NeuronAI/blob/main/NeuronAI_v2.1.ipynb)

For local installation:
```bash
pip install -r requirements.txt