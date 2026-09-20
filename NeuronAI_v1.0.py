# -*- coding: utf-8 -*-
"""NeuronAI v1.0

A biophysical neural mass model of EEG dynamics incorporating glutamatergic
and neuromodulatory mechanisms for personalized pharmacological simulation.

This script is designed to be run in Google Colab.
"""

# =============================================================================
# 1. INSTALLATION AND IMPORTS
# =============================================================================
import sys
import subprocess
import warnings
import tempfile
import os
import time
import zipfile
import hashlib
from itertools import combinations
warnings.filterwarnings('ignore')

def install(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

for pkg in ['numpy', 'scipy', 'matplotlib', 'ipywidgets', 'mne', 'pandas',
            'tensorflow', 'scikit-learn', 'h5py']:
    install(pkg)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import mne
mne.set_log_level('ERROR')

# =============================================================================
# 2. IMPORTS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hilbert, butter, filtfilt
from scipy.optimize import minimize
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from google.colab import files
import pandas as pd
from tensorflow.keras import layers, Model, regularizers
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 3. GLOBAL PARAMETERS
# =============================================================================
FS = 200
DURATION = 10.0
NPERSEG = 256
NOVERLAP = 128
BANDS = {'δ':(0.5,4), 'θ':(4,8), 'α':(8,13), 'β':(13,30), 'γ':(30,80)}
EEG_CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8']

ASD_PARAMS = {'g_nmda_ei': 0.6, 'g_nmda_ee': 0.8, 'low_freq_amp': 0.15}
TD_PARAMS = {'g_nmda_ei': 1.0, 'g_nmda_ee': 1.0, 'low_freq_amp': 0.0}

PROTEINS = [
    {"name": "GABA_A",        "param": "g_i",         "base_delta": 0.3},
    {"name": "GABA_B",        "param": "gaba_b",      "base_delta": 0.25},
    {"name": "NMDA (NR2A)",   "param": "g_e",         "base_delta": 0.3},
    {"name": "NMDA (NR2B)",   "param": "nmda_nr2b",   "base_delta": 0.2},
    {"name": "AMPA",          "param": "g_e",         "base_delta": 0.3},
    {"name": "D2_dopamine",   "param": "p_base",      "base_delta": 0.2},
    {"name": "5HT2A",         "param": "g_e",         "base_delta": 0.3},
    {"name": "alpha7_nAChR",  "param": "g_nmda_ee",   "base_delta": 0.3}
]

# =============================================================================
# 4. DICTIONARY FOR PHARMACOLOGICAL INTERPRETATION
# =============================================================================
PARAM_INFO = {
    'g_i': {
        'name': 'GABA_A',
        'type': 'inhibition',
        'desc': 'GABA_A receptor (Cl⁻ channel)',
        'drugs': 'Benzodiazepines, barbiturates, zolpidem (agonists); flumazenil (antagonist)'
    },
    'g_e': {
        'name': 'AMPA / NMDA NR2A / 5HT2A',
        'type': 'excitation',
        'desc': 'Glutamate AMPA and NMDA (NR2A subunit) receptors and 5HT2A serotonin receptors',
        'drugs': 'Ketamine, memantine (NMDA antagonists); psilocybin (5HT2A agonist); AMPAkines'
    },
    'p_base': {
        'name': 'D2 dopamine',
        'type': 'input modulation',
        'desc': 'D2 dopamine receptor (modulates tonic input)',
        'drugs': 'Haloperidol, risperidone (antagonists); bromocriptine, pramipexole (agonists)'
    },
    'g_nmda_ee': {
        'name': 'NMDA NR2B',
        'type': 'excitation',
        'desc': 'NMDA receptor with NR2B subunit (excitation)',
        'drugs': 'Ifenprodil, eliprodil (NR2B antagonists); D-cycloserine (modulator)'
    },
    'gaba_b': {
        'name': 'GABA_B',
        'type': 'inhibition',
        'desc': 'GABA_B receptor (metabotropic)',
        'drugs': 'Baclofen (agonist); saclofen (antagonist)'
    },
    'nmda_nr2b': {
        'name': 'NMDA NR2B',
        'type': 'excitation',
        'desc': 'NMDA receptor with NR2B subunit (excitation)',
        'drugs': 'Ifenprodil, eliprodil (NR2B antagonists); D-cycloserine (modulator)'
    },
    'dopamine': {
        'name': 'Dopamine',
        'type': 'neuromodulator',
        'desc': 'Dopaminergic neuromodulator (mesolimbic, nigrostriatal pathways)',
        'drugs': 'Levodopa, dopaminergic agonists; antipsychotics (antagonists)'
    },
    'serotonin': {
        'name': 'Serotonin',
        'type': 'neuromodulator',
        'desc': 'Serotonergic neuromodulator (raphe pathways)',
        'drugs': 'SSRIs, buspirone (partial agonists); cyproheptadine (antagonist)'
    },
    'norepinephrine': {
        'name': 'Norepinephrine',
        'type': 'neuromodulator',
        'desc': 'Noradrenergic neuromodulator (locus coeruleus)',
        'drugs': 'Atomoxetine, reboxetine (reuptake inhibitors); clonidine (α2 agonist)'
    }
}

def interpret_combo(combo_name, deltas_dict):
    if not combo_name:
        return "No modulation"
    params = combo_name.split('+')
    desc_parts = []
    for p in params:
        if p in PARAM_INFO:
            info = PARAM_INFO[p]
            delta = deltas_dict.get(p, 0.0)
            if delta > 0.1:
                action = "↑ Increase"
            elif delta < -0.1:
                action = "↓ Decrease"
            else:
                action = "≈ No significant change"
            desc_parts.append(f"{action} of {info['name']} ({info['type']})")
        else:
            desc_parts.append(f"Parameter {p}")
    if not desc_parts:
        return "No modulation"
    return "; ".join(desc_parts)

# =============================================================================
# 5. NEURAL MASS MODEL
# =============================================================================
class NeuralMass:
    def __init__(self, g_nmda_ei=1.0, g_nmda_ee=1.0, low_freq_amp=0.0,
                 g_i_delta=0.0, g_e_delta=0.0, p_base_delta=0.0,
                 g_nmda_ee_delta=0.0, gaba_b_delta=0.0, nmda_nr2b_delta=0.0,
                 dopamine_mod=0.0, serotonin_mod=0.0, norepinephrine_mod=0.0,
                 w_ee=15.0, w_ei1_factor=1.0, tau_e=0.005, fs=FS):
        self.fs = fs
        self.dt = 1.0/fs
        self.g_nmda_ei = g_nmda_ei + g_i_delta
        self.g_nmda_ee = g_nmda_ee + g_nmda_ee_delta + g_e_delta
        self.gaba_b = 1.0 + gaba_b_delta
        self.nmda_nr2b = 1.0 + nmda_nr2b_delta
        self.dopamine = dopamine_mod
        self.serotonin = serotonin_mod
        self.norepinephrine = norepinephrine_mod

        p_base_extra = 0.0
        if self.dopamine != 0:
            p_base_extra += 0.15 * self.dopamine
            self.g_nmda_ei -= 0.1 * self.dopamine
        if self.serotonin != 0:
            self.g_nmda_ee += 0.2 * self.serotonin
            self.g_nmda_ei += 0.1 * self.serotonin
        if self.norepinephrine != 0:
            self.g_nmda_ee += 0.15 * self.norepinephrine
            tau_e = tau_e * (1.0 - 0.2 * self.norepinephrine)

        self.low_freq_amp = low_freq_amp + p_base_delta + p_base_extra
        self.tau_e = max(0.002, min(0.02, tau_e))
        self.tau_i1 = 0.008
        self.tau_i2 = 0.050
        self.w_ee = w_ee
        self.w_ei1 = 12.0 * self.g_nmda_ei * w_ei1_factor
        self.w_ei2 = 6.0
        self.w_i1e = -10.0
        self.w_i2e = -4.0
        self.w_ei1 = self.w_ei1 * self.gaba_b
        self.p_base = 3.0 + p_base_delta + p_base_extra
        self.sigma_p = 0.8
        self.noise_amp = 0.4

    def derivs(self, state, t):
        E, I1, I2 = state
        def sig(x, thr=2.5):
            return 1.0/(1.0+np.exp(-(x-thr)))
        I_slow = self.low_freq_amp * np.sin(2*np.pi*2.0*t)
        p = self.p_base + self.sigma_p*np.random.randn() + I_slow
        I_syn_E = self.w_ee * self.g_nmda_ee * sig(E) + self.w_i1e*sig(I1) + self.w_i2e*sig(I2) + p
        I_syn_I1 = self.w_ei1 * sig(E)
        I_syn_I2 = self.w_ei2 * sig(E)
        dE = (-E + I_syn_E)/self.tau_e + self.noise_amp*np.random.randn()
        dI1 = (-I1 + I_syn_I1)/self.tau_i1 + self.noise_amp*np.random.randn()
        dI2 = (-I2 + I_syn_I2)/self.tau_i2 + self.noise_amp*np.random.randn()
        return np.array([dE, dI1, dI2])

    def simulate(self, duration_sec=DURATION, seed=42, n_runs=5, progress_callback=None):
        np.random.seed(seed)
        steps = int(duration_sec*self.fs)
        all_eeg = []
        for run in range(n_runs):
            state = np.array([0.1, 0.0, 0.0])
            eeg = np.zeros(steps)
            for i in range(steps):
                eeg[i] = state[0]
                state = state + self.dt * self.derivs(state, i*self.dt) + np.sqrt(self.dt) * self.noise_amp * np.random.randn(3)
                if progress_callback and (i % max(1, steps//20) == 0 or i == steps-1):
                    progress_callback(int(100*i/steps))
            all_eeg.append(eeg)
            np.random.seed(seed + run + 1)
        return np.mean(all_eeg, axis=0)

# =============================================================================
# 6. AUXILIARY FUNCTIONS
# =============================================================================
def spectrogram_db(signal):
    f, t, Sxx = spectrogram(signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    return 10*np.log10(Sxx+1e-10), f, t

def band_power_temporal(spec, f, t):
    pow_dict = {}
    for band, (fmin,fmax) in BANDS.items():
        idx = (f>=fmin) & (f<fmax)
        if np.any(idx):
            pow_dict[band] = np.mean(spec[idx, :], axis=0)
        else:
            pow_dict[band] = np.zeros(spec.shape[1])
    return pow_dict

def band_power_mean(spec, f, t):
    pow_temp = band_power_temporal(spec, f, t)
    return {b: np.mean(p) for b, p in pow_temp.items()}

def distance_spectral(pow_mean_a, pow_mean_b):
    diff = 0.0
    for band in BANDS:
        va = pow_mean_a.get(band, 0.0)
        vb = pow_mean_b.get(band, 0.0)
        diff += abs(va - vb)
    return diff

def compute_plv_and_r(signals, fs=FS):
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    n_ch, n_samp = signals.shape
    if n_ch < 2:
        return 0.0, 0.0
    b, a = butter(4, [0.5, 45], btype='band', fs=fs)
    filtered = filtfilt(b, a, signals, axis=1)
    phases = np.zeros_like(filtered)
    for ch in range(n_ch):
        phases[ch, :] = np.angle(hilbert(filtered[ch, :]))
    plv_sum = 0.0
    n_pairs = 0
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            diff_phase = phases[i, :] - phases[j, :]
            plv = np.abs(np.mean(np.exp(1j * diff_phase)))
            plv_sum += plv
            n_pairs += 1
    plv_avg = plv_sum / n_pairs if n_pairs > 0 else 0.0
    r_avg = np.mean(np.abs(np.mean(np.exp(1j * phases), axis=0)))
    return plv_avg, r_avg

def compute_combined_distance(eeg_a, eeg_b, fs=FS, weight_kuramoto=0.5):
    if eeg_a.ndim == 1:
        eeg_a = eeg_a.reshape(1, -1)
    if eeg_b.ndim == 1:
        eeg_b = eeg_b.reshape(1, -1)
    sig_a = np.mean(eeg_a, axis=0)
    sig_b = np.mean(eeg_b, axis=0)
    spec_a, f, t = spectrogram_db(sig_a)
    spec_b, _, _ = spectrogram_db(sig_b)
    pow_a = band_power_mean(spec_a, f, t)
    pow_b = band_power_mean(spec_b, f, t)
    d_esp = distance_spectral(pow_a, pow_b)
    plv_a, r_a = compute_plv_and_r(eeg_a, fs)
    plv_b, r_b = compute_plv_and_r(eeg_b, fs)
    d_kur = abs(plv_a - plv_b) + abs(r_a - r_b)
    return d_esp, d_kur * weight_kuramoto, d_esp + d_kur * weight_kuramoto

# =============================================================================
# 7. FUNCTION FOR CONFIDENCE INTERVAL ESTIMATION
# =============================================================================
def compute_improvement_ci(base_params, td_eeg, deltas_vec, d_tot_base, n_reps=10, n_runs_ci=2, fs=FS, weight_kuramoto=0.5):
    g_ei, g_ee, lf = base_params
    improvements = []
    for rep in range(n_reps):
        model_mod = NeuralMass(
            g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf,
            g_i_delta=deltas_vec[0], g_e_delta=deltas_vec[1],
            p_base_delta=deltas_vec[2], g_nmda_ee_delta=deltas_vec[3],
            gaba_b_delta=deltas_vec[4], nmda_nr2b_delta=deltas_vec[5],
            dopamine_mod=deltas_vec[6], serotonin_mod=deltas_vec[7],
            norepinephrine_mod=deltas_vec[8], fs=fs
        )
        eeg_mod = model_mod.simulate(seed=123 + rep, n_runs=n_runs_ci)
        _, _, d_tot = compute_combined_distance(eeg_mod, td_eeg, fs, weight_kuramoto)
        improvement = (d_tot_base - d_tot) / d_tot_base * 100 if d_tot_base > 0 else 0
        improvements.append(improvement)
    mean_imp = np.mean(improvements)
    std_imp = np.std(improvements)
    n = len(improvements)
    ic_lower = mean_imp - 1.96 * (std_imp / np.sqrt(n))
    ic_upper = mean_imp + 1.96 * (std_imp / np.sqrt(n))
    return mean_imp, std_imp, ic_lower, ic_upper

# =============================================================================
# 8. EEG LOADING
# =============================================================================
def load_eeg_multichannel_list(uploaded_data, fs_target=FS, verbose=True):
    all_signals = []
    all_names = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in uploaded_data.items():
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.zip':
                zip_path = os.path.join(tmpdir, filename)
                with open(zip_path, 'wb') as f:
                    f.write(content)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(tmpdir)
                for root, _, files in os.walk(tmpdir):
                    for fname in files:
                        f_ext = os.path.splitext(fname)[1].lower()
                        if f_ext in ['.set', '.gdf', '.edf']:
                            full_path = os.path.join(root, fname)
                            try:
                                if f_ext == '.set':
                                    raw = mne.io.read_raw_eeglab(full_path, preload=True, verbose=False)
                                elif f_ext == '.gdf':
                                    raw = mne.io.read_raw_gdf(full_path, preload=True, verbose=False)
                                elif f_ext == '.edf':
                                    raw = mne.io.read_raw_edf(full_path, preload=True, verbose=False)
                                else:
                                    continue
                                raw.resample(fs_target)
                                raw.filter(0.5, 45, fir_design='firwin', verbose=False)
                                raw.notch_filter(50, verbose=False)
                                ch_names = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
                                if not ch_names:
                                    ch_names = raw.ch_names
                                raw.pick(ch_names)
                                data = raw.get_data()
                                need = int(DURATION * fs_target)
                                if data.shape[1] < need:
                                    pad = need - data.shape[1]
                                    data = np.pad(data, ((0,0),(0,pad)), mode='constant')
                                else:
                                    data = data[:, :need]
                                all_signals.append(data)
                                all_names.append(fname)
                            except Exception as e:
                                if verbose:
                                    print(f"   Error processing {fname} inside ZIP: {e}")
            else:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    if ext == '.set':
                        raw = mne.io.read_raw_eeglab(tmp_path, preload=True, verbose=False)
                    elif ext == '.gdf':
                        raw = mne.io.read_raw_gdf(tmp_path, preload=True, verbose=False)
                    elif ext == '.edf':
                        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                    else:
                        os.unlink(tmp_path)
                        continue
                    raw.resample(fs_target)
                    raw.filter(0.5, 45, fir_design='firwin', verbose=False)
                    raw.notch_filter(50, verbose=False)
                    ch_names = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
                    if not ch_names:
                        ch_names = raw.ch_names
                    raw.pick(ch_names)
                    data = raw.get_data()
                    need = int(DURATION * fs_target)
                    if data.shape[1] < need:
                        pad = need - data.shape[1]
                        data = np.pad(data, ((0,0),(0,pad)), mode='constant')
                    else:
                        data = data[:, :need]
                    all_signals.append(data)
                    all_names.append(filename)
                except Exception as e:
                    if verbose:
                        print(f"   Error processing {filename}: {e}")
                finally:
                    os.unlink(tmp_path)
    if not all_signals:
        return [], [], 0
    return all_signals, all_names, len(all_signals)

# =============================================================================
# 9. TRAINING FUNCTIONS
# =============================================================================
def generate_synthetic_training_data(n_samples=300):
    np.random.seed(42)
    X, y = [], []
    for _ in range(n_samples):
        vec = np.random.rand(7) * 0.5 + 0.1
        d = np.random.uniform(-0.5, 0.5, 9)
        X.append(vec)
        y.append(d)
    return np.array(X), np.array(y)

# =============================================================================
# fit_model_to_real (MODIFIED: fixed seed)
# =============================================================================
def fit_model_to_real(eeg_signal, fs=FS, duration=DURATION):
    np.random.seed(42)  # Fix global seed for determinism
    if eeg_signal.ndim == 2:
        sig = np.mean(eeg_signal, axis=0)
    else:
        sig = eeg_signal
    spec_real, f, t = spectrogram_db(sig)
    pow_real = band_power_mean(spec_real, f, t)

    def objective(params):
        g_ei, g_ee, lf = params
        model = NeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf, fs=fs)
        eeg_sim = model.simulate(duration_sec=duration, seed=42, n_runs=2)
        spec_sim, _, _ = spectrogram_db(eeg_sim)
        pow_sim = band_power_mean(spec_sim, f, t)
        return distance_spectral(pow_sim, pow_real)

    bounds = [(0.1, 2.0), (0.1, 2.0), (0.0, 0.8)]
    best_res = None
    best_val = np.inf
    for seed in range(4):
        np.random.seed(seed + 42)
        x0 = [np.random.uniform(0.3, 1.7), np.random.uniform(0.3, 1.7), np.random.uniform(0.0, 0.5)]
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 50, 'ftol': 1e-5})
        if res.fun < best_val:
            best_val = res.fun
            best_res = res.x
    return best_res

def estimate_kuramoto_weight(eeg_td, n_samples_est=20, fs=FS, duration=DURATION):
    g_ei_td, g_ee_td, lf_td = fit_model_to_real(eeg_td, fs, duration)
    model_td = NeuralMass(g_nmda_ei=g_ei_td, g_nmda_ee=g_ee_td, low_freq_amp=lf_td, fs=fs)
    eeg_td_sim = model_td.simulate(seed=999, n_runs=2)
    sig_td = np.mean(eeg_td_sim, axis=0) if eeg_td_sim.ndim == 2 else eeg_td_sim
    spec_td, f, t = spectrogram_db(sig_td)
    pow_td = band_power_mean(spec_td, f, t)
    plv_td, r_td = compute_plv_and_r(eeg_td_sim, fs)

    d_esp_list = []
    d_kur_list = []
    for i in range(n_samples_est):
        g_ei_var = np.random.uniform(0.3, 1.2)
        g_ee_var = np.random.uniform(0.3, 1.2)
        lf_var = np.random.uniform(0.0, 0.4)
        model_var = NeuralMass(g_nmda_ei=g_ei_var, g_nmda_ee=g_ee_var, low_freq_amp=lf_var, fs=fs)
        eeg_var = model_var.simulate(seed=100+i, n_runs=2)

        sig_var = np.mean(eeg_var, axis=0) if eeg_var.ndim == 2 else eeg_var
        spec_var, _, _ = spectrogram_db(sig_var)
        pow_var = band_power_mean(spec_var, f, t)
        plv_var, r_var = compute_plv_and_r(eeg_var, fs)

        d_esp = distance_spectral(pow_var, pow_td)
        d_kur = abs(plv_var - plv_td) + abs(r_var - r_td)
        d_esp_list.append(d_esp)
        d_kur_list.append(d_kur)

    mean_esp = np.mean(d_esp_list)
    mean_kur = np.mean(d_kur_list)
    if mean_kur == 0:
        weight_opt = 1.0
    else:
        weight_opt = mean_esp / mean_kur
    weight_opt = np.clip(weight_opt, 0.1, 5.0)
    return weight_opt

def generate_training_data_anchored_to_td(eeg_td, mode='standard', fs=FS, duration=DURATION,
                                          asd_base_params=None, weight_kuramoto=0.5,
                                          auto_balance=True, cv_mode=False):
    if auto_balance:
        print("   Estimating automatic Kuramoto weight (no fixed limit)...")
        w_est = estimate_kuramoto_weight(eeg_td, fs=fs, duration=duration)
        print(f"   Estimated Kuramoto weight (dynamic): {w_est:.2f}")
        weight_kuramoto = w_est

    configs = {
        'quick': {'n_samples': 40, 'maxiter': 8, 'n_attempts': 2, 'n_runs_opt': 1},
        'standard': {'n_samples': 80, 'maxiter': 15, 'n_attempts': 3, 'n_runs_opt': 2},
        'precise': {'n_samples': 100, 'maxiter': 25, 'n_attempts': 4, 'n_runs_opt': 2}
    }
    cfg = configs[mode]

    if cv_mode:
        n_samples_cv = max(30, cfg['n_samples'] // 2)
        print(f"   CV mode: reduced samples from {cfg['n_samples']} to {n_samples_cv}")
        cfg['n_samples'] = n_samples_cv

    n_samples = cfg['n_samples']
    maxiter = cfg['maxiter']
    n_attempts = cfg['n_attempts']
    n_runs_opt = cfg['n_runs_opt']

    print(f"   Mode: {mode.upper()} (samples={n_samples}, iter={maxiter}, attempts={n_attempts}, runs={n_runs_opt})")
    print("   Fitting model to TD EEG (reference)...")
    g_ei_td, g_ee_td, lf_td = fit_model_to_real(eeg_td, fs, duration)
    print(f"   TD fit: g_nmda_ei={g_ei_td:.3f}, g_nmda_ee={g_ee_td:.3f}, low_freq={lf_td:.3f}")

    model_td = NeuralMass(g_nmda_ei=g_ei_td, g_nmda_ee=g_ee_td, low_freq_amp=lf_td, fs=fs)
    eeg_td_sim = model_td.simulate(seed=999, n_runs=2)
    sig_td = np.mean(eeg_td_sim, axis=0) if eeg_td_sim.ndim == 2 else eeg_td_sim
    spec_td, f, t = spectrogram_db(sig_td)
    pow_td = band_power_mean(spec_td, f, t)
    plv_td, r_td = compute_plv_and_r(eeg_td_sim, fs)

    if asd_base_params is not None:
        asd_base = asd_base_params
    else:
        asd_base = [0.6, 0.8, 0.15]
    print(f"   ASD base for generating variations: g_nmda_ei={asd_base[0]:.3f}, g_nmda_ee={asd_base[1]:.3f}, low_freq={asd_base[2]:.3f}")

    X_list, y_list = [], []

    for idx in range(n_samples):
        g_ei_var = np.clip(asd_base[0] + np.random.uniform(-0.3, 0.3), 0.2, 1.5)
        g_ee_var = np.clip(asd_base[1] + np.random.uniform(-0.3, 0.3), 0.2, 1.5)
        lf_var = np.clip(asd_base[2] + np.random.uniform(-0.1, 0.1), 0.0, 0.5)

        model_var = NeuralMass(g_nmda_ei=g_ei_var, g_nmda_ee=g_ee_var, low_freq_amp=lf_var, fs=fs)
        eeg_var = model_var.simulate(seed=idx+100, n_runs=2)

        sig_var = np.mean(eeg_var, axis=0) if eeg_var.ndim == 2 else eeg_var
        spec_var, f, t = spectrogram_db(sig_var)
        pow_var = band_power_mean(spec_var, f, t)
        vec_spec = [pow_var.get(b, 0.0) for b in BANDS.keys()]
        plv_var, r_var = compute_plv_and_r(eeg_var, fs)
        features = np.concatenate([vec_spec, [plv_var, r_var]])

        best_deltas = None
        best_cost = np.inf
        for attempt in range(n_attempts):
            x0 = np.random.uniform(-0.2, 0.2, 9)
            def cost(deltas_vec):
                model_mod = NeuralMass(
                    g_nmda_ei=g_ei_var, g_nmda_ee=g_ee_var, low_freq_amp=lf_var,
                    g_i_delta=deltas_vec[0], g_e_delta=deltas_vec[1],
                    p_base_delta=deltas_vec[2], g_nmda_ee_delta=deltas_vec[3],
                    gaba_b_delta=deltas_vec[4], nmda_nr2b_delta=deltas_vec[5],
                    dopamine_mod=deltas_vec[6], serotonin_mod=deltas_vec[7],
                    norepinephrine_mod=deltas_vec[8], fs=fs
                )
                eeg_mod = model_mod.simulate(seed=idx+200+attempt, n_runs=n_runs_opt)
                sig_mod = np.mean(eeg_mod, axis=0) if eeg_mod.ndim == 2 else eeg_mod
                spec_mod, _, _ = spectrogram_db(sig_mod)
                pow_mod = band_power_mean(spec_mod, f, t)
                d_esp = distance_spectral(pow_mod, pow_td)
                plv_mod, r_mod = compute_plv_and_r(eeg_mod, fs)
                d_kur = abs(plv_mod - plv_td) + abs(r_mod - r_td)
                d_tot = d_esp + weight_kuramoto * d_kur
                penalty = 0.01 * np.sum(deltas_vec**2)
                return d_tot + penalty

            bounds = [(-0.8, 0.8)]*6 + [(-1.0, 1.0)]*3
            res = minimize(cost, x0, method='COBYLA', bounds=bounds,
                           options={'maxiter': maxiter, 'tol': 1e-4})
            if res.fun < best_cost:
                best_cost = res.fun
                best_deltas = res.x

        X_list.append(features)
        y_list.append(best_deltas)

        if (idx+1) % 20 == 0:
            print(f"   Generated {idx+1}/{n_samples} samples (cost: {best_cost:.3f})")

    td_params_fitted = (g_ei_td, g_ee_td, lf_td)
    return np.array(X_list), np.array(y_list), td_params_fitted, weight_kuramoto

# =============================================================================
# 10. NEURONAI CLASS
# =============================================================================
class NeuronAI:
    def __init__(self, input_dim=7, output_dim=9, complexity='standard'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.complexity = complexity
        self.model = None
        self.scaler = StandardScaler()
        self.ready = False
        self.td_params_fitted = None

    def build_model(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs

        if self.complexity == 'basico':
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(32, activation='relu')(x)
            epochs = 40
            batch_size = 32
            lr = 0.001
            output_scale = 0.25
        elif self.complexity == 'avanzado':
            x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(32, activation='relu')(x)
            epochs = 80
            batch_size = 16
            lr = 0.0005
            output_scale = 0.25
        else:  # 'standard'
            x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(32, activation='relu')(x)
            epochs = 60
            batch_size = 16
            lr = 0.0005
            output_scale = 0.25

        tanh_out = layers.Dense(self.output_dim, activation='tanh')(x)
        outputs = layers.Lambda(lambda z: z * output_scale)(tanh_out)

        model = Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')

        self._epochs = epochs
        self._batch_size = batch_size
        self.model = model
        return model

    def train(self, X, y, epochs=None, batch_size=None, validation_split=0.2):
        X_scaled = self.scaler.fit_transform(X)
        if self.model is None:
            self.build_model()
        if epochs is None:
            epochs = getattr(self, '_epochs', 60)
        if batch_size is None:
            batch_size = getattr(self, '_batch_size', 16)
        history = self.model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, verbose=0)
        self.ready = True
        return history

    def predict(self, X, return_scaled=False):
        if not self.ready:
            raise ValueError("Model not trained.")
        X_scaled = self.scaler.transform(np.array(X).reshape(1, -1))
        pred = self.model.predict(X_scaled, verbose=0)[0]
        if return_scaled:
            return pred, X_scaled.flatten()
        return pred

# =============================================================================
# 11. AUXILIARY FUNCTION FOR MANUAL EQUIVALENT (MODIFIED: threshold 0.05)
# =============================================================================
def get_manual_equivalent(delta):
    """
    Translates a continuous delta into the equivalent in the manual widgets.
    Returns: (label, value)
    """
    if delta > 0.05:
        return "Agonist", delta
    elif delta < -0.05:
        return "Inhibitor", delta
    else:
        return "None", 0.0

# =============================================================================
# NEW SEED CONSTANTS (for unify)
# =============================================================================
SEED_BASE = 42
SEED_TD = 44
SEED_MOD = 123
N_RUNS_EVAL = 30   # Number of replicates for evaluating combinations

# =============================================================================
# NEW AUXILIARY FUNCTION: simulate with given deltas (using unified seeds)
# =============================================================================
def simulate_with_deltas(g_ei, g_ee, lf, deltas_vec, td_eeg, weight_k, n_runs=N_RUNS_EVAL):
    """
    Simulates the fitted model with the given deltas and computes distance and improvement.
    Returns: d_tot_base, d_tot_mod, improvement (in %)
    """
    # Base model (no modulation)
    model_base = NeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf, fs=FS)
    eeg_base = model_base.simulate(seed=SEED_BASE, n_runs=n_runs)

    # Modulated model
    model_mod = NeuralMass(
        g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf,
        g_i_delta=deltas_vec[0], g_e_delta=deltas_vec[1],
        p_base_delta=deltas_vec[2], g_nmda_ee_delta=deltas_vec[3],
        gaba_b_delta=deltas_vec[4], nmda_nr2b_delta=deltas_vec[5],
        dopamine_mod=deltas_vec[6], serotonin_mod=deltas_vec[7],
        norepinephrine_mod=deltas_vec[8], fs=FS
    )
    eeg_mod = model_mod.simulate(seed=SEED_MOD, n_runs=n_runs)

    # Distances
    _, _, d_tot_base = compute_combined_distance(eeg_base, td_eeg, FS, weight_k)
    _, _, d_tot_mod = compute_combined_distance(eeg_mod, td_eeg, FS, weight_k)
    improvement = (d_tot_base - d_tot_mod) / d_tot_base * 100 if d_tot_base > 0 else 0
    return d_tot_base, d_tot_mod, improvement, eeg_base, eeg_mod

# =============================================================================
# 12. USER INTERFACE
# =============================================================================
train_mode = widgets.RadioButtons(
    options=[('Synthetic data (fast, generic)', 'sintetico'),
             ('Real data (anchored to a specific TD EEG)', 'real')],
    value='sintetico',
    description='Training mode:',
    layout=widgets.Layout(width='500px')
)

training_quality = widgets.Dropdown(
    options=[
        ('⚡ Fast (~1 min)', 'quick'),
        ('⚖️ Standard (~3 min)', 'standard'),
        ('🎯 Precise (~6 min)', 'precise')
    ],
    value='standard',
    description='Quality (real):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

model_complexity = widgets.Dropdown(
    options=[
        ('🧩 Basic (fast)', 'basico'),
        ('⚖️ Standard (recommended)', 'standard'),
        ('🚀 Advanced (more precise)', 'avanzado')
    ],
    value='standard',
    description='AI complexity:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

auto_balance_checkbox = widgets.Checkbox(
    value=True,
    description='Automatic Kuramoto balance (dynamic, no fixed limit)',
    layout=widgets.Layout(width='400px')
)

kuramoto_weight_slider = widgets.FloatSlider(
    value=0.5, min=0.1, max=5.0, step=0.1,
    description='Kuramoto weight (manual):',
    layout=widgets.Layout(width='300px')
)

btn_load_td_for_training = widgets.Button(description='📂 Load Reference EEG (TD) for training (files or .zip)',
                                          button_style='primary', layout=widgets.Layout(width='400px'))
td_upload_status = widgets.HTML(value="<i>No TD has been loaded</i>")

btn_train = widgets.Button(description='🤖 Train AI', button_style='info', layout=widgets.Layout(width='200px'))

btn_calc_ci = widgets.Checkbox(
    value=False,
    description='Compute 95% CI (slower)',
    layout=widgets.Layout(width='300px')
)

btn_cross_val = widgets.Button(
    description='📊 Cross-validation (real)',
    button_style='info',
    layout=widgets.Layout(width='250px')
)

btn_load_asd = widgets.Button(description='📂 Load Problem EEG (ASD or case) (files or .zip)',
                              button_style='primary', layout=widgets.Layout(width='350px'))
asd_upload_status = widgets.HTML(value="<i>No problem EEG has been loaded</i>")

btn_reset_analysis = widgets.Button(
    description='🧹 Reset analysis (clears problem EEG and settings)',
    button_style='danger',
    layout=widgets.Layout(width='350px')
)

btn_simulate_ai_deltas = widgets.Button(
    description='✅ Verify stability of best combination',
    button_style='info',
    layout=widgets.Layout(width='300px')
)

prot_widgets = {}
for prot in PROTEINS:
    prot_widgets[prot['name']] = widgets.Dropdown(
        options=['None', 'Agonist', 'Inhibitor'],
        value='None',
        description=prot['name'],
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='280px')
    )

neuromod_widgets = {}
neuromod_widgets['Dopamine'] = widgets.FloatSlider(min=-1.0, max=1.0, step=0.05, value=0.0,
                                                   description='Dopamine', layout=widgets.Layout(width='300px'))
neuromod_widgets['Serotonin'] = widgets.FloatSlider(min=-1.0, max=1.0, step=0.05, value=0.0,
                                                     description='Serotonin', layout=widgets.Layout(width='300px'))
neuromod_widgets['Norepinephrine'] = widgets.FloatSlider(min=-1.0, max=1.0, step=0.05, value=0.0,
                                                        description='Norepinephrine', layout=widgets.Layout(width='300px'))

btn_simulate = widgets.Button(description='🔬 SIMULATE MANUAL MODULATION', button_style='success', layout=widgets.Layout(width='350px', height='45px'))
btn_optimize = widgets.Button(description='🎯 PREDICT BEST MODULATION (AI + combinatorial search)', button_style='warning', layout=widgets.Layout(width='400px'))

output = widgets.Output()
progress_bar = widgets.IntProgress(value=0, min=0, max=100, description='Progress:', bar_style='info', layout=widgets.Layout(width='100%'))
progress_text = widgets.HTML(value="Waiting for action...")

# Global variables
ai_model = None
eeg_asd = None
td_eeg_for_training = None
td_params_fitted_global = None
asd_filename = ""
td_filename = ""
prev_hash = None

td_eeg_list: list = []
asd_eeg_list: list = []
td_names: list = []
asd_names: list = []
td_promedio = None

# Variables to store the last AI prediction
last_ai_deltas = None
last_ai_combo = None
last_ai_base_params = None
last_ai_td_eeg = None
last_ai_d_tot_base = None
last_ai_weight_k = None

# =============================================================================
# 13. CALLBACK FUNCTIONS
# =============================================================================
def get_deltas(selections, neuromod_selections):
    deltas = {'g_i':0, 'g_e':0, 'p_base':0, 'g_nmda_ee':0, 'gaba_b':0, 'nmda_nr2b':0}
    for p in PROTEINS:
        sel = selections.get(p['name'], 'None')
        if sel != 'None':
            sign = 1.0 if sel=='Agonist' else -1.0
            param = p['param']
            if param == 'g_i': deltas['g_i'] += sign * p['base_delta']
            elif param == 'g_e': deltas['g_e'] += sign * p['base_delta']
            elif param == 'p_base': deltas['p_base'] += sign * p['base_delta']
            elif param == 'g_nmda_ee': deltas['g_nmda_ee'] += sign * p['base_delta']
            elif param == 'gaba_b': deltas['gaba_b'] += sign * p['base_delta']
            elif param == 'nmda_nr2b': deltas['nmda_nr2b'] += sign * p['base_delta']
    neuromod = {
        'dopamine': neuromod_selections.get('Dopamine', 0),
        'serotonin': neuromod_selections.get('Serotonin', 0),
        'norepinephrine': neuromod_selections.get('Norepinephrine', 0)
    }
    return deltas, neuromod

def reset_widgets():
    for w in prot_widgets.values():
        w.value = 'None'
    for w in neuromod_widgets.values():
        w.value = 0.0

def on_reset_analysis_click(b):
    global eeg_asd, asd_filename, prev_hash, asd_eeg_list, asd_names
    with output:
        clear_output(wait=True)
        eeg_asd = None
        asd_filename = ""
        prev_hash = None
        asd_eeg_list = []
        asd_names = []
        asd_upload_status.value = "<i>No problem EEG has been loaded</i>"
        reset_widgets()
        print("🧹 Analysis reset. You can load a new problem EEG and adjust parameters manually.")
        print("   The trained AI model (TD) remains intact.")

def on_simulate_ai_deltas_click(b):
    global last_ai_deltas, last_ai_combo, last_ai_base_params, last_ai_td_eeg, last_ai_d_tot_base, last_ai_weight_k
    with output:
        clear_output(wait=True)
        if last_ai_deltas is None or last_ai_base_params is None:
            print("⚠️ First run the AI prediction (button 'PREDICT BEST MODULATION').")
            return
        if eeg_asd is None:
            print("⚠️ Load a Problem EEG (ASD).")
            return

        g_ei, g_ee, lf = last_ai_base_params
        weight_k = last_ai_weight_k if last_ai_weight_k is not None else 0.5
        td_eeg = last_ai_td_eeg

        model_mod = NeuralMass(
            g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf,
            g_i_delta=last_ai_deltas[0], g_e_delta=last_ai_deltas[1],
            p_base_delta=last_ai_deltas[2], g_nmda_ee_delta=last_ai_deltas[3],
            gaba_b_delta=last_ai_deltas[4], nmda_nr2b_delta=last_ai_deltas[5],
            dopamine_mod=last_ai_deltas[6], serotonin_mod=last_ai_deltas[7],
            norepinephrine_mod=last_ai_deltas[8], fs=FS
        )
        eeg_mod = model_mod.simulate(seed=123, n_runs=N_RUNS_EVAL)

        model_base = NeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf, fs=FS)
        eeg_base = model_base.simulate(seed=42, n_runs=N_RUNS_EVAL)

        _, _, d_tot_base = compute_combined_distance(eeg_base, td_eeg, FS, weight_k)
        _, _, d_tot_mod = compute_combined_distance(eeg_mod, td_eeg, FS, weight_k)
        improvement = (d_tot_base - d_tot_mod) / d_tot_base * 100 if d_tot_base > 0 else 0

        print("\n🔄 SIMULATION WITH AI DELTAS (consistency)")
        print(f"   Initial distance (ASD-TD): {d_tot_base:.3f}")
        print(f"   Final distance (modulated - TD): {d_tot_mod:.3f}")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Combination used: {last_ai_combo if last_ai_combo else 'complete'}")
        print("\n✅ This simulation uses exactly the same deltas that the AI predicted.")
        print("   It should match the AI prediction results (small variations due to randomness in seeds).")

        sig_base_plot = np.mean(eeg_base, axis=0) if eeg_base.ndim == 2 else eeg_base
        sig_mod_plot = np.mean(eeg_mod, axis=0) if eeg_mod.ndim == 2 else eeg_mod
        sig_td_plot = np.mean(td_eeg, axis=0) if td_eeg.ndim == 2 else td_eeg
        spec_base_p, f, t = spectrogram_db(sig_base_plot)
        spec_mod_p, _, _ = spectrogram_db(sig_mod_plot)
        spec_td_p, _, _ = spectrogram_db(sig_td_plot)

        plt.figure(figsize=(10,6))
        plt.plot(f, np.mean(spec_base_p, axis=1), label='Problem (fitted)', linestyle='--', alpha=0.7)
        plt.plot(f, np.mean(spec_mod_p, axis=1), label=f'Modulated (AI)', linewidth=2)
        plt.plot(f, np.mean(spec_td_p, axis=1), label='TD (reference)', linestyle=':', alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.legend()
        plt.title(f'Spectrum - Simulation with AI deltas (improvement: {improvement:.1f}%)')
        plt.grid(True)
        plt.show()

def on_load_td_for_training_click(b):
    global td_eeg_for_training, td_filename, td_eeg_list, td_names, td_promedio
    with output:
        clear_output(wait=True)
        print("📂 Upload EEG Reference (TD) file(s). Can be a ZIP with multiple files.")
        uploaded = files.upload()
        if uploaded:
            signals, names, n_files = load_eeg_multichannel_list(uploaded, verbose=False)
            if signals:
                td_eeg_list.extend(signals)
                td_names.extend(names)
                td_promedio = np.mean(td_eeg_list, axis=0)
                td_eeg_for_training = td_promedio
                if n_files == 1:
                    td_filename = names[0]
                else:
                    td_filename = f"average_of_{len(td_eeg_list)}_files"
                td_upload_status.value = f"✅ TD loaded: {len(td_eeg_list)} files (channels: {td_promedio.shape[0]})"
                print(f"✅ TD loaded: {len(td_eeg_list)} files")
                if n_files == 1:
                    print(f"   Single file: {names[0]}")
                else:
                    print(f"   Names: {', '.join(names[:3])}{'...' if len(names)>3 else ''}")
            else:
                td_upload_status.value = "❌ Error loading TD (no valid EEG found)"
                print("❌ Error loading (possibly no valid EEG files found).")
        else:
            print("No file was selected.")

def on_train_click(b):
    global ai_model, td_params_fitted_global
    with output:
        clear_output(wait=True)
        print("🤖 Starting AI training...")
        if train_mode.value == 'sintetico':
            print("   Synthetic mode: generating generic data...")
            X, y = generate_synthetic_training_data(n_samples=300)
            td_params_fitted_global = None
        else:
            print("   Real mode: a loaded TD EEG is required.")
            if td_eeg_for_training is None:
                print("⚠️ First load a Reference EEG (TD) using the corresponding button.")
                return
            quality = training_quality.value
            auto_balance = auto_balance_checkbox.value
            if auto_balance:
                print("   Automatic Kuramoto balance ENABLED (dynamic, no fixed limit).")
                weight_k = None
            else:
                weight_k = kuramoto_weight_slider.value
                print(f"   Automatic balance DISABLED. Using manual weight: {weight_k:.2f}")

            complexity = model_complexity.value
            print(f"   AI complexity: {complexity.upper()}")

            print(f"   Generating data anchored to the loaded TD (mode {quality.upper()} optimized)...")
            X, y, td_params, w_used = generate_training_data_anchored_to_td(
                td_eeg_for_training, mode=quality,
                weight_kuramoto=weight_k if not auto_balance else 0.5,
                auto_balance=auto_balance,
                cv_mode=False
            )
            td_params_fitted_global = td_params
            print(f"   Fitted TD parameters: g_nmda_ei={td_params[0]:.3f}, g_nmda_ee={td_params[1]:.3f}, low_freq={td_params[2]:.3f}")
            if auto_balance:
                print(f"   Kuramoto weight used: {w_used:.2f}")

        ai_model = NeuronAI(input_dim=7, output_dim=9, complexity=complexity)
        history = ai_model.train(X, y, validation_split=0.25)
        print("✅ Training completed.")
        plt.figure(figsize=(8,4))
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss during training')
        plt.legend()
        plt.show()

def on_load_asd_click(b):
    global eeg_asd, asd_filename, prev_hash, asd_eeg_list, asd_names
    with output:
        clear_output(wait=True)
        print("📂 Upload Problem EEG (ASD or case) file(s). Can be a ZIP with multiple files.")
        uploaded = files.upload()
        if uploaded:
            signals, names, n_files = load_eeg_multichannel_list(uploaded, verbose=True)
            if signals:
                asd_eeg_list.extend(signals)
                asd_names.extend(names)
                eeg_asd = signals[0] if signals else None
                if n_files == 1:
                    asd_filename = names[0]
                else:
                    asd_filename = f"average_of_{len(asd_eeg_list)}_files"
                flat = signals[0].flatten()[:100]
                hash_val = hashlib.md5(flat.tobytes()).hexdigest()[:8]
                prev_hash = hash_val
                asd_upload_status.value = f"✅ Problem loaded: {asd_filename} (channels: {signals[0].shape[0]}, hash: {hash_val})"
                print(f"✅ Problem loaded: {asd_filename} (channels: {signals[0].shape[0]}, hash: {hash_val})")
                if n_files > 1:
                    print(f"   ({n_files} files were averaged)")
                    print(f"   Names: {', '.join(names[:3])}{'...' if len(names)>3 else ''}")
                else:
                    print(f"   Single file: {names[0]}")
            else:
                asd_upload_status.value = "❌ Error loading problem (no valid EEG found)"
                print("❌ Error loading (possibly no valid EEG files found).")
        else:
            print("No file was selected.")

def on_cross_val_click(b):
    with output:
        clear_output(wait=True)
        print("📊 Starting cross-validation...")

        # ============================================================
        # QUICK CONFIGURATION (change these values as needed)
        # ============================================================
        CV_MODE = False          # True = reduce samples by half (faster), False = use all (more precise)
        N_SAMPLES_CV = 80        # Number of synthetic samples when CV_MODE=False (ignored if True, uses half)
        PRINT_DIAGNOSTICS = True # Print features and deltas of the first ASD subjects
        # ============================================================

        # Verify enough ASD and TD
        if len(asd_eeg_list) < 2:
            print("⚠️ At least 2 ASD EEGs are needed for cross-validation (recommended ≥3).")
            print(f"   Currently: {len(asd_eeg_list)} ASD loaded.")
            return
        if len(td_eeg_list) < 1:
            print("⚠️ At least 1 TD EEG is needed for cross-validation (recommended ≥3).")
            print(f"   Currently: {len(td_eeg_list)} TD loaded.")
            return

        print(f"   Using {len(asd_eeg_list)} ASD and {len(td_eeg_list)} TD for CV.")
        quality = training_quality.value
        auto_balance = auto_balance_checkbox.value
        if auto_balance:
            print("   Automatic Kuramoto balance ENABLED (dynamic, no fixed limit).")
            weight_k = None
        else:
            weight_k = kuramoto_weight_slider.value
            print(f"   Automatic balance DISABLED. Using manual weight: {weight_k:.2f}")
        print(f"   Training quality: {quality.upper()}")
        print(f"   CV mode: {'Reduced (fast)' if CV_MODE else 'Complete (precise)'}")

        complexity = model_complexity.value
        print(f"   AI complexity: {complexity.upper()}")

        td_prom = np.mean(td_eeg_list, axis=0)

        asd_base_params = [0.6, 0.8, 0.15]
        print(f"   ASD base (typical fixed): g_nmda_ei={asd_base_params[0]:.3f}, g_nmda_ee={asd_base_params[1]:.3f}, low_freq={asd_base_params[2]:.3f}")

        print("   Generating training data with the average TD...")

        # Configure CV mode according to the variable
        if CV_MODE:
            X, y, td_params, w_used = generate_training_data_anchored_to_td(
                td_prom, mode=quality,
                asd_base_params=asd_base_params,
                weight_kuramoto=weight_k if not auto_balance else 0.5,
                auto_balance=auto_balance,
                cv_mode=True
            )
        else:
            # Use a fixed number of samples (N_SAMPLES_CV) without halving
            # To do this, pass cv_mode=False but force the number of samples
            # by temporarily redefining the configs
            original_configs = {
                'quick': {'n_samples': 40, 'maxiter': 8, 'n_attempts': 2, 'n_runs_opt': 1},
                'standard': {'n_samples': 80, 'maxiter': 15, 'n_attempts': 3, 'n_runs_opt': 2},
                'precise': {'n_samples': 100, 'maxiter': 25, 'n_attempts': 4, 'n_runs_opt': 2}
            }
            # Modify the number of samples for the selected mode
            import copy
            configs = copy.deepcopy(original_configs)
            if quality == 'quick':
                configs['quick']['n_samples'] = N_SAMPLES_CV
            elif quality == 'standard':
                configs['standard']['n_samples'] = N_SAMPLES_CV
            else:
                configs['precise']['n_samples'] = N_SAMPLES_CV

            # Call the function with cv_mode=False and the modified configs
            X, y, td_params, w_used = generate_training_data_anchored_to_td(
                td_prom, mode=quality,
                asd_base_params=asd_base_params,
                weight_kuramoto=weight_k if not auto_balance else 0.5,
                auto_balance=auto_balance,
                cv_mode=False
            )

        if auto_balance:
            print(f"   Kuramoto weight used: {w_used:.2f}")

        print("   Training AI with the average TD...")
        ai_model_cv = NeuronAI(input_dim=7, output_dim=9, complexity=complexity)
        ai_model_cv.train(X, y, validation_split=0.25)

        improvements = []
        n_total = len(asd_eeg_list)
        start_time = time.time()

        # Variables for diagnostics
        diagnostic_counter = 0
        max_diagnostic = min(5, n_total)  # Show diagnostics of the first 5 ASD

        for idx, eeg_asd in enumerate(asd_eeg_list):
            if idx % 5 == 0 or idx == n_total - 1:
                elapsed = time.time() - start_time
                if idx > 0:
                    estimated = (elapsed / idx) * (n_total - idx)
                    print(f"   Evaluating ASD {idx+1}/{n_total} (estimated: {estimated:.1f}s remaining)...")
                else:
                    print(f"   Evaluating ASD {idx+1}/{n_total}...")

            # Fit model to the ASD EEG
            g_ei_asd, g_ee_asd, lf_asd = fit_model_to_real(eeg_asd)
            sig_asd = np.mean(eeg_asd, axis=0)
            spec_asd, f, t = spectrogram_db(sig_asd)
            pow_asd = band_power_mean(spec_asd, f, t)
            vec_spec = [pow_asd.get(b, 0.0) for b in BANDS.keys()]
            plv_asd, r_asd = compute_plv_and_r(eeg_asd, FS)
            features_raw = np.concatenate([vec_spec, [plv_asd, r_asd]])

            # AI prediction
            deltas_opt = ai_model_cv.predict(features_raw)

            # DIAGNOSTICS: print features and deltas of the first ASD subjects
            if PRINT_DIAGNOSTICS and diagnostic_counter < max_diagnostic:
                print(f"\n   [DIAGNOSTICS ASD {idx+1}]")
                print(f"      Fitted parameters: g_ei={g_ei_asd:.3f}, g_ee={g_ee_asd:.3f}, lf={lf_asd:.3f}")
                print(f"      Spectrum (δ,θ,α,β,γ): {[f'{v:.2f}' for v in vec_spec]}")
                print(f"      PLV={plv_asd:.3f}, r={r_asd:.3f}")
                print(f"      Predicted deltas: {[f'{v:+.3f}' for v in deltas_opt]}")
                diagnostic_counter += 1

            # Base simulation (no modulation)
            model_base = NeuralMass(g_nmda_ei=g_ei_asd, g_nmda_ee=g_ee_asd, low_freq_amp=lf_asd, fs=FS)
            eeg_base = model_base.simulate(seed=42, n_runs=1)
            model_td = NeuralMass(g_nmda_ei=td_params[0], g_nmda_ee=td_params[1], low_freq_amp=td_params[2], fs=FS)
            eeg_td_sim = model_td.simulate(seed=44, n_runs=1)
            _, _, d_tot_base = compute_combined_distance(eeg_base, eeg_td_sim, FS, w_used)

            # Modulated simulation
            model_mod = NeuralMass(
                g_nmda_ei=g_ei_asd, g_nmda_ee=g_ee_asd, low_freq_amp=lf_asd,
                g_i_delta=deltas_opt[0], g_e_delta=deltas_opt[1],
                p_base_delta=deltas_opt[2], g_nmda_ee_delta=deltas_opt[3],
                gaba_b_delta=deltas_opt[4], nmda_nr2b_delta=deltas_opt[5],
                dopamine_mod=deltas_opt[6], serotonin_mod=deltas_opt[7],
                norepinephrine_mod=deltas_opt[8], fs=FS
            )
            eeg_mod = model_mod.simulate(seed=123, n_runs=1)
            _, _, d_tot_mod = compute_combined_distance(eeg_mod, eeg_td_sim, FS, w_used)
            improvement = (d_tot_base - d_tot_mod) / d_tot_base * 100 if d_tot_base > 0 else 0
            improvements.append(improvement)

        # Results
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        n_asd = len(improvements)
        ic_lower = mean_improvement - 1.96 * (std_improvement / np.sqrt(n_asd))
        ic_upper = mean_improvement + 1.96 * (std_improvement / np.sqrt(n_asd))

        print("\n📊 CROSS-VALIDATION RESULTS")
        print(f"   Number of ASD evaluated: {n_asd}")
        print(f"   Mean improvement: {mean_improvement:.1f}% ± {std_improvement:.1f}%")
        print(f"   95% CI: [{ic_lower:.1f}, {ic_upper:.1f}]%")

        if mean_improvement < -50:
            print("\n⚠️ WARNING: The mean improvement is very negative. Check:")
            print("   1. That the ASD and TD EEGs are truly different (not similar averages).")
            print("   2. That the preprocessing (filtering, channel selection) is correct.")
            print("   3. Try disabling automatic balancing and using a low Kuramoto weight (e.g., 0.2).")

        if n_asd >= 3:
            plt.figure(figsize=(8,4))
            plt.hist(improvements, bins=min(10, n_asd), edgecolor='black', alpha=0.7)
            plt.axvline(mean_improvement, color='red', linestyle='--', label=f'Mean: {mean_improvement:.1f}%')
            plt.xlabel('Improvement (%)')
            plt.ylabel('Frequency')
            plt.title('Distribution of improvements in cross-validation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        print("✅ Cross-validation completed.")

# =============================================================================
# 14. COMBINATORIAL SEARCH
# =============================================================================
def combinatorial_search(base_params, td_eeg, predicted_deltas, fs=FS, n_runs=2, max_combo_size=3, weight_kuramoto=0.5):
    names = ['g_i', 'g_e', 'p_base', 'g_nmda_ee', 'gaba_b', 'nmda_nr2b', 'dopamine', 'serotonin', 'norepinephrine']
    g_ei, g_ee, lf = base_params
    n_params = len(names)

    model_base = NeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf, fs=fs)
    eeg_base = model_base.simulate(seed=42, n_runs=n_runs)
    d_esp_base, d_kur_base, d_tot_base = compute_combined_distance(eeg_base, td_eeg, fs, weight_kuramoto)

    all_combos = []
    for r in range(1, min(max_combo_size, n_params) + 1):
        all_combos.extend(combinations(range(n_params), r))

    print(f"   Evaluating {len(all_combos)} combinations...")

    best_combo = None
    best_deltas = None
    best_distance = d_tot_base
    best_improvement = 0.0
    all_results = {}
    combo_counter = 0

    for combo_indices in all_combos:
        deltas_vec = np.zeros(n_params)
        for idx in combo_indices:
            deltas_vec[idx] = predicted_deltas[idx]

        model_mod = NeuralMass(
            g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf,
            g_i_delta=deltas_vec[0], g_e_delta=deltas_vec[1],
            p_base_delta=deltas_vec[2], g_nmda_ee_delta=deltas_vec[3],
            gaba_b_delta=deltas_vec[4], nmda_nr2b_delta=deltas_vec[5],
            dopamine_mod=deltas_vec[6], serotonin_mod=deltas_vec[7],
            norepinephrine_mod=deltas_vec[8], fs=fs
        )
        eeg_mod = model_mod.simulate(seed=123 + combo_counter, n_runs=n_runs)
        d_esp, d_kur, d_tot = compute_combined_distance(eeg_mod, td_eeg, fs, weight_kuramoto)

        combo_name = '+'.join([names[i] for i in combo_indices])
        deltas_dict = {names[i]: predicted_deltas[i] for i in combo_indices}
        all_results[combo_name] = {
            'deltas': deltas_vec.copy(),
            'deltas_dict': deltas_dict,
            'distance': d_tot,
            'improvement': (d_tot_base - d_tot) / d_tot_base * 100 if d_tot_base > 0 else 0
        }

        if d_tot < best_distance:
            best_distance = d_tot
            best_combo = combo_name
            best_deltas = deltas_vec.copy()
            best_improvement = (d_tot_base - d_tot) / d_tot_base * 100 if d_tot_base > 0 else 0

        combo_counter += 1
        if combo_counter % 10 == 0:
            print(f"      Processed {combo_counter}/{len(all_combos)} combinations...")

    return best_combo, best_deltas, best_distance, best_improvement, all_results, d_tot_base

# =============================================================================
# 15. SIMULATION AND OPTIMIZATION FUNCTIONS (with manual equivalent column)
# =============================================================================
def on_simulate_click(b):
    global td_params_fitted_global, last_ai_td_eeg, last_ai_weight_k
    with output:
        clear_output(wait=True)
        if eeg_asd is None:
            print("⚠️ First load a Problem EEG (ASD).")
            return

        # Determine the TD to use:
        # 1. If a saved TD from the last prediction exists, use it (maximum consistency)
        # 2. If not, but there are fitted TD parameters (real training), simulate it
        # 3. If not, use default synthetic TD
        if last_ai_td_eeg is not None:
            eeg_td_sim = last_ai_td_eeg
            print("   Using saved TD from last prediction (consistent initial distance).")
        elif td_params_fitted_global is not None:
            g_ei_td, g_ee_td, lf_td = td_params_fitted_global
            model_td = NeuralMass(g_nmda_ei=g_ei_td, g_nmda_ee=g_ee_td, low_freq_amp=lf_td, fs=FS)
            eeg_td_sim = model_td.simulate(seed=SEED_TD, n_runs=N_RUNS_EVAL)
            print("   Using TD simulated from real training.")
        else:
            print("⚠️ No reference TD. Using default synthetic TD (may be inconsistent).")
            eeg_td_sim = NeuralMass(g_nmda_ei=1.0, g_nmda_ee=1.0, low_freq_amp=0.0, fs=FS).simulate(seed=SEED_TD, n_runs=N_RUNS_EVAL)

        # Kuramoto weight
        if auto_balance_checkbox.value:
            weight_k = estimate_kuramoto_weight(eeg_asd, n_samples_est=10)
            print(f"   Estimated Kuramoto weight (dynamic): {weight_k:.2f}")
        else:
            weight_k = kuramoto_weight_slider.value

        display(HTML(f"<h3>🧠 Analyzing EEG: {asd_filename}</h3>"))
        print("   Fitting model to the Problem EEG...")
        params_asd_fit = fit_model_to_real(eeg_asd)
        g_ei, g_ee, lf = params_asd_fit
        print(f"   Fitted parameters: g_nmda_ei={g_ei:.3f}, g_nmda_ee={g_ee:.3f}, low_freq={lf:.3f}")

        selec = {n: w.value for n, w in prot_widgets.items()}
        neuromod_vals = {n: w.value for n, w in neuromod_widgets.items()}
        deltas, neuromod_deltas = get_deltas(selec, neuromod_vals)

        print("🔬 Simulating manual modulation on the fitted model...")
        model_base = NeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf, fs=FS)
        eeg_base = model_base.simulate(seed=SEED_BASE, n_runs=N_RUNS_EVAL)

        model_mod = NeuralMass(
            g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf,
            g_i_delta=deltas['g_i'], g_e_delta=deltas['g_e'],
            p_base_delta=deltas['p_base'], g_nmda_ee_delta=deltas['g_nmda_ee'],
            gaba_b_delta=deltas['gaba_b'], nmda_nr2b_delta=deltas['nmda_nr2b'],
            dopamine_mod=neuromod_deltas['dopamine'],
            serotonin_mod=neuromod_deltas['serotonin'],
            norepinephrine_mod=neuromod_deltas['norepinephrine'],
            fs=FS
        )
        eeg_mod = model_mod.simulate(seed=SEED_MOD, n_runs=N_RUNS_EVAL)

        d_esp_base, d_kur_base, d_tot_base = compute_combined_distance(eeg_base, eeg_td_sim, FS, weight_k)
        d_esp_mod, d_kur_mod, d_tot_mod = compute_combined_distance(eeg_mod, eeg_td_sim, FS, weight_k)

        print(f"Initial total distance (ASD-TD): {d_tot_base:.3f}")
        print(f"Total distance after modulation: {d_tot_mod:.3f}")
        improvement = (d_tot_base - d_tot_mod)/d_tot_base*100 if d_tot_base>0 else 0
        print(f"Improvement: {improvement:.1f}%")

        sig_base = np.mean(eeg_base, axis=0) if eeg_base.ndim==2 else eeg_base
        sig_mod = np.mean(eeg_mod, axis=0) if eeg_mod.ndim==2 else eeg_mod
        sig_td = np.mean(eeg_td_sim, axis=0) if eeg_td_sim.ndim==2 else eeg_td_sim
        spec_base, f, t = spectrogram_db(sig_base)
        spec_mod, _, _ = spectrogram_db(sig_mod)
        spec_td, _, _ = spectrogram_db(sig_td)

        plt.figure(figsize=(10,6))
        plt.plot(f, np.mean(spec_base, axis=1), label='Problem (fitted)')
        plt.plot(f, np.mean(spec_mod, axis=1), label='Modulated')
        plt.plot(f, np.mean(spec_td, axis=1), label='TD (reference)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.legend()
        plt.title('Average power spectrum (model fitted to real EEG)')
        plt.grid(True)
        plt.show()

def on_optimize_click(b):
    global ai_model, td_params_fitted_global, eeg_asd, asd_filename, prev_hash
    global last_ai_deltas, last_ai_combo, last_ai_base_params, last_ai_td_eeg, last_ai_d_tot_base, last_ai_weight_k
    with output:
        clear_output(wait=True)
        if ai_model is None or not ai_model.ready:
            print("⚠️ First train the AI (button 'Train AI').")
            return
        if eeg_asd is None:
            print("⚠️ Load a Problem EEG (ASD).")
            return
        if td_params_fitted_global is None:
            print("⚠️ No reference TD. Train the AI in real mode with a TD.")
            return

        if auto_balance_checkbox.value:
            weight_k = estimate_kuramoto_weight(eeg_asd, n_samples_est=10)
            print(f"   Estimated Kuramoto weight (dynamic): {weight_k:.2f}")
        else:
            weight_k = kuramoto_weight_slider.value

        flat = eeg_asd.flatten()[:100]
        current_hash = hashlib.md5(flat.tobytes()).hexdigest()[:8]
        display(HTML(f"<h3>🧠 Analyzing EEG: <span style='color:blue'>{asd_filename}</span></h3>"))
        print(f"   Channels: {eeg_asd.shape[0]}, Samples: {eeg_asd.shape[1]}, Hash: {current_hash}")

        if prev_hash is not None:
            if current_hash == prev_hash:
                print("⚠️ WARNING: This EEG has the same hash as the previous one. Data may be identical or very similar.")
            else:
                print("✅ Hash different from previous one, new EEG detected.")
        prev_hash = current_hash

        print("   Fitting model to the Problem EEG...")
        params_asd_fit = fit_model_to_real(eeg_asd)
        g_ei, g_ee, lf = params_asd_fit
        print(f"   Fitted parameters: g_nmda_ei={g_ei:.3f}, g_nmda_ee={g_ee:.3f}, low_freq={lf:.3f}")

        g_ei_td, g_ee_td, lf_td = td_params_fitted_global
        model_td = NeuralMass(g_nmda_ei=g_ei_td, g_nmda_ee=g_ee_td, low_freq_amp=lf_td, fs=FS)
        eeg_td_sim = model_td.simulate(seed=44, n_runs=3)

        sig_asd = np.mean(eeg_asd, axis=0)
        spec_asd, f, t = spectrogram_db(sig_asd)
        pow_asd = band_power_mean(spec_asd, f, t)
        vec_spec = [pow_asd.get(b, 0.0) for b in BANDS.keys()]
        plv_asd, r_asd = compute_plv_and_r(eeg_asd, FS)
        features_raw = np.concatenate([vec_spec, [plv_asd, r_asd]])

        print("\n📊 Features extracted from the EEG (unscaled):")
        print(f"   Spectrum (δ,θ,α,β,γ): {[f'{v:.3f}' for v in vec_spec]}")
        print(f"   PLV: {plv_asd:.3f}, r: {r_asd:.3f}")

        deltas_opt, scaled_features = ai_model.predict(features_raw, return_scaled=True)
        print("\n📊 Scaled features (AI input):")
        print(f"   {[f'{v:.3f}' for v in scaled_features]}")

        print("\n🔮 Optimal deltas predicted by the AI:")
        names = ['g_i', 'g_e', 'p_base', 'g_nmda_ee', 'gaba_b', 'nmda_nr2b', 'dopamine', 'serotonin', 'norepinephrine']
        # Also show manual equivalent
        print("   (Manual equivalent: Agonist/Inhibitor/None)")
        for n, v in zip(names, deltas_opt):
            eq, _ = get_manual_equivalent(v)
            print(f"   {n}: {v:+.3f} → Manual: {eq}")

        print("\n🔎 Performing combinatorial search...")
        best_combo, best_deltas, best_distance, best_improvement, all_results, d_tot_base = combinatorial_search(
            (g_ei, g_ee, lf),
            eeg_td_sim,
            deltas_opt,
            fs=FS,
            n_runs=N_RUNS_EVAL,
            max_combo_size=3,
            weight_kuramoto=weight_k
        )

        print(f"\n✅ Best combination found: {best_combo}")
        print(f"   Initial distance: {d_tot_base:.3f}")
        print(f"   Final distance: {best_distance:.3f} (improvement: {best_improvement:.1f}%)")

        # Save for consistency
        last_ai_deltas = best_deltas
        last_ai_combo = best_combo
        last_ai_base_params = (g_ei, g_ee, lf)
        last_ai_td_eeg = eeg_td_sim
        last_ai_d_tot_base = d_tot_base
        last_ai_weight_k = weight_k

        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['distance'])
        print("\n📊 Top 10 best combinations (includes individual ones):")

        df_data = []
        calc_ci = btn_calc_ci.value

        if calc_ci:
            print("⏳ Calculating confidence intervals (this may take ~30 seconds)...")
            start_time = time.time()

        for idx, (combo_name, res) in enumerate(sorted_results[:10]):
            deltas_dict = res['deltas_dict']
            interpretation = interpret_combo(combo_name, deltas_dict)

            # If computing CI, replace point improvement with CI mean
            if calc_ci:
                mean_imp, std_imp, ic_lower, ic_upper = compute_improvement_ci(
                    (g_ei, g_ee, lf),
                    eeg_td_sim,
                    res['deltas'],
                    d_tot_base,
                    n_reps=10,
                    n_runs_ci=N_RUNS_EVAL,
                    fs=FS,
                    weight_kuramoto=weight_k
                )
                ic_text = f"[{ic_lower:.1f}, {ic_upper:.1f}]"
                adjusted_ranking = ic_lower
                improvement_shown = mean_imp
            else:
                mean_imp = None
                ic_text = "N/A"
                adjusted_ranking = res['improvement']
                improvement_shown = res['improvement']

            # Generate manual equivalent for this combination
            manual_parts = []
            for p in combo_name.split('+'):
                if p in deltas_dict:
                    delta_val = deltas_dict[p]
                    eq, _ = get_manual_equivalent(delta_val)
                    manual_parts.append(f"{p} → {eq}")
            manual_str = "; ".join(manual_parts) if manual_parts else "None"

            df_data.append({
                'Combination': combo_name,
                'Distance': f"{res['distance']:.3f}",
                'Improvement (%)': f"{improvement_shown:.1f}%",
                '95% CI': ic_text,
                'Adjusted ranking': f"{adjusted_ranking:.1f}",
                'Manual equivalent': manual_str,
                'Interpretation': interpretation
            })

        if calc_ci:
            elapsed = time.time() - start_time
            print(f"   CI calculation time: {elapsed:.1f} seconds")

        df_results = pd.DataFrame(df_data)
        df_results['Ranking_num'] = df_results['Adjusted ranking'].astype(float)
        df_results['Improvement_num'] = df_results['Improvement (%)'].str.replace('%', '').astype(float)
        df_results = df_results.sort_values(['Ranking_num', 'Improvement_num'], ascending=[False, False])
        df_results = df_results.drop(columns=['Ranking_num', 'Improvement_num'])

        with pd.option_context('display.max_colwidth', None):
            display(df_results)

        individual_results = {k: v for k, v in all_results.items() if '+' not in k}
        if individual_results:
            best_individual_name, best_individual_data = sorted(individual_results.items(), key=lambda x: x[1]['distance'])[0]
            best_individual_dist = best_individual_data['distance']
            best_individual_improvement = best_individual_data['improvement']
        else:
            best_individual_name = None
            best_individual_dist = np.inf
            best_individual_improvement = -np.inf

        print("\n📌 COMPARISON: Best individual vs Best combination")
        if best_individual_name is not None:
            print(f"   Best individual: {best_individual_name} → Distance: {best_individual_dist:.3f} (improvement: {best_individual_improvement:.1f}%)")
        else:
            print("   No individual modulations available.")
        print(f"   Best combination: {best_combo} → Distance: {best_distance:.3f} (improvement: {best_improvement:.1f}%)")

        if best_individual_name is not None:
            if best_distance < best_individual_dist:
                print(f"\n✅ The best combination ({best_combo}) is BETTER than the best individual ({best_individual_name}) by {abs(best_distance - best_individual_dist):.3f} units and {best_improvement - best_individual_improvement:.1f}% additional improvement.")
                best_overall_name = best_combo
                best_overall_deltas = best_deltas
                best_overall_improvement = best_improvement
            else:
                print(f"\n✅ The best individual ({best_individual_name}) is BETTER than the best combination ({best_combo}) by {abs(best_distance - best_individual_dist):.3f} units and {best_individual_improvement - best_improvement:.1f}% additional improvement.")
                best_overall_name = best_individual_name
                best_overall_deltas = individual_results[best_individual_name]['deltas']
                best_overall_improvement = best_individual_improvement
        else:
            best_overall_name = best_combo
            best_overall_deltas = best_deltas
            best_overall_improvement = best_improvement

        print(f"\n🔧 Applying the best option: {best_overall_name}")
        model_mod = NeuralMass(
            g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf,
            g_i_delta=best_overall_deltas[0], g_e_delta=best_overall_deltas[1],
            p_base_delta=best_overall_deltas[2], g_nmda_ee_delta=best_overall_deltas[3],
            gaba_b_delta=best_overall_deltas[4], nmda_nr2b_delta=best_overall_deltas[5],
            dopamine_mod=best_overall_deltas[6], serotonin_mod=best_overall_deltas[7],
            norepinephrine_mod=best_overall_deltas[8], fs=FS
        )
        eeg_mod = model_mod.simulate(seed=123, n_runs=3)

        model_base = NeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf, fs=FS)
        eeg_base = model_base.simulate(seed=42, n_runs=3)

        sig_base_plot = np.mean(eeg_base, axis=0) if eeg_base.ndim == 2 else eeg_base
        sig_mod_plot = np.mean(eeg_mod, axis=0) if eeg_mod.ndim == 2 else eeg_mod
        sig_td_plot = np.mean(eeg_td_sim, axis=0) if eeg_td_sim.ndim == 2 else eeg_td_sim
        spec_base_p, _, _ = spectrogram_db(sig_base_plot)
        spec_mod_p, _, _ = spectrogram_db(sig_mod_plot)
        spec_td_p, _, _ = spectrogram_db(sig_td_plot)

        plt.figure(figsize=(10,6))
        plt.plot(f, np.mean(spec_base_p, axis=1), label='Problem (fitted)', linestyle='--', alpha=0.7)
        plt.plot(f, np.mean(spec_mod_p, axis=1), label=f'Best option: {best_overall_name}', linewidth=2)
        plt.plot(f, np.mean(spec_td_p, axis=1), label='TD (reference)', linestyle=':', alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.legend()
        plt.title(f'Spectrum - Best option: {best_overall_name} (improvement: {best_overall_improvement:.1f}%)')
        plt.grid(True)
        plt.show()

        print(f"✅ Suggested modulation: {best_overall_name} with improvement of {best_overall_improvement:.1f}%")

        if best_overall_improvement < 0:
            print("\n⚠️ The best option did not improve the distance. You can try a direct local optimization (without AI).")
            btn_local = widgets.Button(description='🔧 Run local optimization', button_style='warning')
            display(btn_local)

            def on_local_optimize(btn):
                with output:
                    print("   Performing local optimization (may take ~1 minute)...")
                    def local_cost(deltas_vec):
                        model_temp = NeuralMass(
                            g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf,
                            g_i_delta=deltas_vec[0], g_e_delta=deltas_vec[1],
                            p_base_delta=deltas_vec[2], g_nmda_ee_delta=deltas_vec[3],
                            gaba_b_delta=deltas_vec[4], nmda_nr2b_delta=deltas_vec[5],
                            dopamine_mod=deltas_vec[6], serotonin_mod=deltas_vec[7],
                            norepinephrine_mod=deltas_vec[8], fs=FS
                        )
                        eeg_temp = model_temp.simulate(seed=456, n_runs=2)
                        _, _, d_tot = compute_combined_distance(eeg_temp, eeg_td_sim, FS, weight_k)
                        penalty = 0.01 * np.sum(deltas_vec**2)
                        return d_tot + penalty

                    bounds_local = [(-0.8, 0.8)]*6 + [(-1.0, 1.0)]*3
                    best_local = None
                    best_cost = np.inf
                    for attempt in range(4):
                        x0 = np.random.uniform(-0.3, 0.3, 9)
                        res = minimize(local_cost, x0, method='COBYLA', bounds=bounds_local,
                                       options={'maxiter': 25, 'tol': 1e-4})
                        if res.fun < best_cost:
                            best_cost = res.fun
                            best_local = res.x
                    print("   Deltas found by local optimization:")
                    for n, v in zip(names, best_local):
                        print(f"      {n}: {v:+.3f}")
                    model_local = NeuralMass(
                        g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf,
                        g_i_delta=best_local[0], g_e_delta=best_local[1],
                        p_base_delta=best_local[2], g_nmda_ee_delta=best_local[3],
                        gaba_b_delta=best_local[4], nmda_nr2b_delta=best_local[5],
                        dopamine_mod=best_local[6], serotonin_mod=best_local[7],
                        norepinephrine_mod=best_local[8], fs=FS
                    )
                    eeg_local = model_local.simulate(seed=789, n_runs=3)
                    d_tot_loc, _, _ = compute_combined_distance(eeg_local, eeg_td_sim, FS, weight_k)
                    improvement_loc = (d_tot_base - d_tot_loc) / d_tot_base * 100 if d_tot_base > 0 else 0
                    print(f"Final distance with local optimization: {d_tot_loc:.3f} (improvement: {improvement_loc:.1f}%)")
                    sig_base_plot2 = np.mean(eeg_base, axis=0) if eeg_base.ndim == 2 else eeg_base
                    sig_mod_plot2 = np.mean(eeg_local, axis=0) if eeg_local.ndim == 2 else eeg_local
                    sig_td_plot2 = np.mean(eeg_td_sim, axis=0) if eeg_td_sim.ndim == 2 else eeg_td_sim
                    spec_base_p2, _, _ = spectrogram_db(sig_base_plot2)
                    spec_mod_p2, _, _ = spectrogram_db(sig_mod_plot2)
                    spec_td_p2, _, _ = spectrogram_db(sig_td_plot2)
                    plt.figure(figsize=(10,6))
                    plt.plot(f, np.mean(spec_base_p2, axis=1), label='Problem (fitted)', linestyle='--', alpha=0.7)
                    plt.plot(f, np.mean(spec_mod_p2, axis=1), label='Local optimization', linewidth=2)
                    plt.plot(f, np.mean(spec_td_p2, axis=1), label='TD (reference)', linestyle=':', alpha=0.7)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Power (dB)')
                    plt.legend()
                    plt.title(f'Spectrum - Local improvement: {improvement_loc:.1f}%')
                    plt.grid(True)
                    plt.show()
                    print("✅ Local optimization completed.")

            btn_local.on_click(on_local_optimize)
            display(btn_local)

# =============================================================================
# NEW BUTTON: Apply best AI combination in manual mode
# =============================================================================
btn_apply_ai = widgets.Button(
    description='📌 Apply best AI combination in manual mode',
    button_style='success',
    layout=widgets.Layout(width='400px')
)

def on_apply_ai_click(b):
    global last_ai_combo, last_ai_deltas, last_ai_base_params, last_ai_td_eeg, last_ai_weight_k
    global td_params_fitted_global, eeg_asd
    with output:
        clear_output(wait=True)
        if last_ai_combo is None:
            print("⚠️ First run the AI prediction (button 'PREDICT BEST MODULATION').")
            return
        if eeg_asd is None:
            print("⚠️ Load a Problem EEG (ASD).")
            return
        if td_params_fitted_global is None:
            print("⚠️ No reference TD. Train the AI in real mode with a TD.")
            return

        print(f"📌 Applying the best AI combination: **{last_ai_combo}**")
        print("   Setting manual widgets automatically...")

        # 1. Reset all widgets to 'None' and 0.0
        reset_widgets()

        # 2. Extract parameters from the combination
        combo_parts = last_ai_combo.split('+')
        # Mapping from parameter names to widget keys
        param_to_protein_name = {
            'g_i': 'GABA_A',
            'gaba_b': 'GABA_B',
            'g_e': 'NMDA (NR2A)',
            'nmda_nr2b': 'NMDA (NR2B)',
            'p_base': 'D2_dopamine',
            'g_nmda_ee': 'alpha7_nAChR'
        }
        # For neuromodulators, exact names
        neuromod_param_map = {
            'dopamine': 'Dopamine',
            'serotonin': 'Serotonin',
            'norepinephrine': 'Norepinephrine'
        }

        # Reset deltas to use
        deltas_dict = {}
        param_names = ['g_i','g_e','p_base','g_nmda_ee','gaba_b','nmda_nr2b','dopamine','serotonin','norepinephrine']
        for idx, pname in enumerate(param_names):
            deltas_dict[pname] = last_ai_deltas[idx]

        # Configure proteins
        for pname, protein_name in param_to_protein_name.items():
            if pname in deltas_dict:
                delta_val = deltas_dict[pname]
                eq, _ = get_manual_equivalent(delta_val)
                if protein_name in prot_widgets:
                    prot_widgets[protein_name].value = eq
                    print(f"   {protein_name} → {eq} (delta={delta_val:+.3f})")

        # Configure neuromodulators
        for pname, widget_key in neuromod_param_map.items():
            if pname in deltas_dict:
                delta_val = deltas_dict[pname]
                if widget_key in neuromod_widgets:
                    neuromod_widgets[widget_key].value = delta_val
                    print(f"   {widget_key} → {delta_val:+.3f}")

        print("\n✅ Widgets configured. Now simulating with these settings...")

        # 3. Perform manual simulation with the configured deltas
        selec = {n: w.value for n, w in prot_widgets.items()}
        neuromod_vals = {n: w.value for n, w in neuromod_widgets.items()}
        deltas_manual, neuromod_deltas = get_deltas(selec, neuromod_vals)

        # Build the 9-delta vector in standard order
        deltas_vec = [
            deltas_manual['g_i'],
            deltas_manual['g_e'],
            deltas_manual['p_base'],
            deltas_manual['g_nmda_ee'],
            deltas_manual['gaba_b'],
            deltas_manual['nmda_nr2b'],
            neuromod_deltas['dopamine'],
            neuromod_deltas['serotonin'],
            neuromod_deltas['norepinephrine']
        ]

        # Fit ASD and TD
        params_asd_fit = fit_model_to_real(eeg_asd)
        g_ei, g_ee, lf = params_asd_fit

        # Use the saved TD from the prediction
        eeg_td_sim = last_ai_td_eeg

        # Kuramoto weight
        weight_k = last_ai_weight_k if last_ai_weight_k is not None else 0.5

        # Simulate with the configured deltas
        d_tot_base, d_tot_mod, manual_improvement, eeg_base, eeg_mod = simulate_with_deltas(
            g_ei, g_ee, lf, deltas_vec, eeg_td_sim, weight_k, n_runs=N_RUNS_EVAL
        )

        # Show results
        print("\n📊 MANUAL SIMULATION RESULTS (with AI combination)")
        print(f"   Initial distance (ASD-TD): {d_tot_base:.3f}")
        print(f"   Final distance (modulated - TD): {d_tot_mod:.3f}")
        print(f"   Manual improvement: {manual_improvement:.1f}%")

        # Comparative plot
        sig_base_plot = np.mean(eeg_base, axis=0) if eeg_base.ndim == 2 else eeg_base
        sig_mod_plot = np.mean(eeg_mod, axis=0) if eeg_mod.ndim == 2 else eeg_mod
        sig_td_plot = np.mean(eeg_td_sim, axis=0) if eeg_td_sim.ndim == 2 else eeg_td_sim
        spec_base_p, f, t = spectrogram_db(sig_base_plot)
        spec_mod_p, _, _ = spectrogram_db(sig_mod_plot)
        spec_td_p, _, _ = spectrogram_db(sig_td_plot)

        plt.figure(figsize=(10,6))
        plt.plot(f, np.mean(spec_base_p, axis=1), label='Problem (fitted)', linestyle='--', alpha=0.7)
        plt.plot(f, np.mean(spec_mod_p, axis=1), label=f'Modulated (manual: {last_ai_combo})', linewidth=2)
        plt.plot(f, np.mean(spec_td_p, axis=1), label='TD (reference)', linestyle=':', alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.legend()
        plt.title(f'Spectrum - Manual application of AI suggestion (improvement: {manual_improvement:.1f}%)')
        plt.grid(True)
        plt.show()

        print("\n✅ Manual simulation completed. Verify that the improvement is consistent with the AI.")
        print("   If it differs significantly, it may be due to discretization or stochastic noise.")
        print("   Try increasing N_RUNS_EVAL (currently 3) to reduce variance.")

btn_apply_ai.on_click(on_apply_ai_click)

# =============================================================================
# 16. CALLBACK ASSIGNMENT AND DEPLOYMENT
# =============================================================================
btn_load_td_for_training.on_click(on_load_td_for_training_click)
btn_train.on_click(on_train_click)
btn_load_asd.on_click(on_load_asd_click)
btn_simulate.on_click(on_simulate_click)
btn_optimize.on_click(on_optimize_click)
btn_cross_val.on_click(on_cross_val_click)
btn_reset_analysis.on_click(on_reset_analysis_click)
btn_simulate_ai_deltas.on_click(on_simulate_ai_deltas_click)

display(HTML("<h1>🧠 NeuronAI v1.0</h1>"))
display(HTML("<p><b>New feature:</b> The Top 10 combinations table shows a 'Manual equivalent' column that translates the AI's continuous deltas to the discrete widget options (Agonist/Inhibitor/None). This facilitates the transition between AI and manual mode.</p>"))

display(HTML("<h2>1. AI Training</h2>"))
display(train_mode)
display(training_quality)
display(model_complexity)
display(auto_balance_checkbox)
display(kuramoto_weight_slider)
display(btn_load_td_for_training)
display(td_upload_status)
display(btn_train)

display(HTML("<h2>2. Load Problem EEG</h2>"))
display(btn_load_asd)
display(asd_upload_status)

display(HTML("<h2>3. Manual Adjustments</h2>"))
for prot in PROTEINS:
    display(prot_widgets[prot['name']])
for w in neuromod_widgets.values():
    display(w)

display(HTML("<h2>4. Actions</h2>"))
display(btn_calc_ci)
display(btn_simulate)
display(btn_optimize)
display(btn_simulate_ai_deltas)
display(btn_cross_val)
display(btn_reset_analysis)
display(btn_apply_ai)

display(progress_bar)
display(progress_text)
display(output)

print("✅ NeuronAI v1.0")
print("Workflow: 1) Train AI, 2) Load Problem, 3) Adjust, 4) Act.")