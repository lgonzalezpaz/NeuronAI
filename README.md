# NeuronAI v2.0

**Biophysical EEG digital twins with simulation-based inference for personalised in silico neuropharmacology**

NeuronAI fits a biophysically interpretable *digital twin* to each scalp EEG recording and uses it to
explore, in silico, how receptor-level and neuromodulatory interventions would be expected to change
that recording. It accompanies the article:

> Alvarado YJ, Quintero M, Cardozo-Urdaneta A, Lossada C, Marrero-Ponce Y, Martinez-Rios F,
> Pérez-Castillo Y, Delgado A, González-Paz L. *NeuronAI: biophysical EEG digital twins and
> simulation-based inference for personalised in silico neuropharmacology in autism* (submitted).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22851741.svg)](https://doi.org/10.5281/zenodo.22851741)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lgonzalezpaz/NeuronAI/blob/main/NeuronAI_v2_Colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is inside

| Component | File | Description |
|---|---|---|
| Neural mass model | `neuronai/model.py` | 17-column extension of Wendling et al. (2002): AMPA, NMDA-GluN2A, NMDA-GluN2B (Mg²⁺ block), NMDA on fast interneurons, dopamine/serotonin/noradrenaline gain control, long-range coupling, volume conduction; exact exponential integrator; batch-vectorised, seeded, bit-wise reproducible |
| EEG descriptors | `neuronai/features.py` | unit-invariant relative band power (δ θ α β), individual alpha frequency, aperiodic exponent, alpha PLV, debiased wPLI, Kuramoto order parameter |
| Data I/O and QC | `neuronai/data.py` | readers for GDF 1.x, EDF, EEGLAB .set/.fdt, .mat, CSV (MNE optional); pre-specified preprocessing and quality control |
| Simulation banks | `neuronai/bank.py` | scrambled-Sobol simulation banks |
| AI components | `neuronai/inference.py` | inverse deep ensemble with split-conformal intervals; forward emulator ensemble |
| In silico pharmacology | `neuronai/intervention.py` | sparse search over 129 target combinations, residual-preserving prediction, Mahalanobis normative distance, random-intervention null |
| Statistics | `neuronai/stats.py` | cluster bootstrap, child-level permutation tests, ICC, FDR |
| Pipeline | `scripts/01…09_*.py` | regenerates every number, table and figure of the article |
| Tests | `tests/test_neuronai.py` | 35 unit/property tests |
| Colab | `notebooks/NeuronAI_v2_Colab.ipynb`, `colab/NeuronAI_v2_Colab_single_cell.py` | runs the full pipeline in Google Colab |

## Quick start (Google Colab, no installation)

1. Mendeley Data protects downloads with a browser check, so obtain the two "Download All" links once in
   your browser (open each dataset page, click "Download All", then copy the link address from the browser's
   downloads page) and paste them into `DIRECT_URLS` at the top of the Colab cell. Alternatively, place the
   two zip files in Google Drive and use `MANUAL_ZIPS`.
2. Open `notebooks/NeuronAI_v2_Colab.ipynb` in Colab (badge above) **or** paste the content of
   `colab/NeuronAI_v2_Colab_single_cell.py` into one cell. `MODE = "quick"` checks the pipeline;
   `MODE = "paper"` reproduces the article (≈ 8–10 h on two CPU cores).
3. To obtain only the descriptors of every recording (≈ 30 min), use
   `colab/NeuronAI_download_full_cohort_Colab.py`; it returns a small archive that can be analysed locally
   (copy its files into `results/` and run `scripts/02c_assign_participants.py --meta <metadata folder>` and the
   following steps).

## Local installation

```bash
git clone https://github.com/lgonzalezpaz/NeuronAI.git
cd NeuronAI
pip install -r requirements.txt
pytest -q                       # 35 tests
bash run_all.sh /path/to/ASD_download.zip /path/to/TD_download.zip     # full pipeline
```

## Reproducing the article

```
scripts/01_generate_banks.py        simulation banks I and E (2 x 8,192 networks)
scripts/00_prepare_data.py          unique recording ids and metadata from the Mendeley downloads
scripts/02_preprocess_features.py   preprocessing, QC, artefact removal and descriptors of the real EEG
scripts/02c_assign_participants.py  participant / condition assignment, age, sex, ASRS
scripts/03_train_models.py          inverse ensemble + conformal calibration; descriptor and spectral emulators
scripts/03b_noise_ceiling.py        intrinsic simulator stochasticity
scripts/03c_benchmark_inference.py  comparison with kNN-ABC and random forest (Table 4)
scripts/04_fit_subjects.py          digital twins (amortised + refined), posterior-predictive and out-of-sample simulations
scripts/05_pharmaco_validity.py     pre-specified pharmaco-EEG face-validity benchmarks
scripts/05b_control_fingerprints.py mechanistic fingerprints of each control; post hoc drug models
scripts/06_intervention.py          emulator-proposed, simulator-confirmed search with matched random-search control
scripts/08_single_case.py           pre-specified single-case analysis (ASD participant 1, rest, vs. TD rest)
scripts/07_validation_stats.py      validation statistics with child-level inference, ASRS association, attentional load
scripts/fig1_schematic.py, 09_figures.py, fig6_clinical_load.py, fig_graphical_abstract.py   figures
tools/build_colab.py                regenerates the Colab notebook / single-cell script from the sources
```

All random number generators are seeded; the simulator, the networks (scikit-learn, CPU) and all
statistics are deterministic, so repeated runs on the same platform give identical numbers. Minor
floating-point differences across CPU architectures/BLAS libraries can propagate through the
stochastic simulator; they do not change any conclusion (see `results/` for the reference outputs).

## Data

The EEG data are openly available from Mendeley Data (Duville et al., 2022; 2023):

* TD children: https://doi.org/10.17632/spwnt8t25y.1
* Children with ASD: https://doi.org/10.17632/7zf99hxxc9.1

Each dataset contains 31 children (EEG/PartNN/), each with 19 recordings: one 2-min eyes-open resting state
(`restEO`) and six blocks under each of three attentional loads (`IAPS`, `4discs`, `8discs`), plus
spreadsheets with age, sex, ASRS scores and tracking capacity. `scripts/00_prepare_data.py` gives every
recording a unique id and `scripts/02c_assign_participants.py` assigns participants and conditions and builds
`results/participants.csv`. In the code and result files the cohort labels are `ASD` and `TD`.

## Intended use and limitations

NeuronAI is a research tool for **hypothesis generation**. Its outputs are model-conditional predictions,
not clinical recommendations, and must not be used to make treatment decisions. See the article
(Discussion and Table 7) for the validated scope and known failure modes, including pharmaco-EEG
benchmarks the current model does not reproduce.

## Citation

Please cite the article and the archived software (`CITATION.cff`).
