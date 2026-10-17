#!/usr/bin/env bash
# Full NeuronAI v2.0 pipeline on the complete Duville et al. datasets.
# Usage: bash run_all.sh <ASD download (folder or zip)> <TD download (folder or zip)> [N_BANK]
set -euo pipefail
ASD=${1:?ASD dataset}; TD=${2:?TD dataset}; N=${3:-8192}
export OMP_NUM_THREADS=1
mkdir -p results figures
python scripts/00_prepare_data.py --asd "$ASD" --td "$TD"
python scripts/01_generate_banks.py --kind I --n "$N"
python scripts/01_generate_banks.py --kind E --n "$N"
python scripts/02_preprocess_features.py --asd data/ASD --td data/TD
python scripts/02c_assign_participants.py --meta data_meta
python scripts/03_train_models.py
python scripts/03b_noise_ceiling.py
python scripts/03c_benchmark_inference.py
# the three heaviest steps can be split across processes: --shard k --nshards n
python scripts/04_fit_subjects.py
python scripts/05_pharmaco_validity.py
python scripts/05b_control_fingerprints.py
python scripts/06_intervention.py
python scripts/08_single_case.py
python scripts/07_validation_stats.py
python scripts/fig1_schematic.py
python scripts/09_figures.py
python scripts/fig6_clinical_load.py
python scripts/fig_graphical_abstract.py
echo "Done. Results in ./results, figures in ./figures"
